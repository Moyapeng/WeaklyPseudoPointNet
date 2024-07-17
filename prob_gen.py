import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from model import ResUNet34


from options import Options
from my_transforms import get_transforms
import imageio
import concurrent
from concurrent.futures import ThreadPoolExecutor

from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from utils import voronoi_finite_polygons_2d, poly2mask
from joblib import Parallel, delayed
import multiprocessing
from skimage.draw import polygon as skpolygon
from options import Options
import json


def main(option):
    opt = option
    opt.isTrain = False
    # opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpus'])

    img_dir = opt.test['img_dir']
    label_dir = opt.test['label_dir']
    save_dir = opt.test['save_dir']
    model_path = opt.test['model_path']
    save_flag = opt.test['save_flag']
    dbranch_num = len(opt.train['decoder_radius'])

    # data transforms
    test_transform = get_transforms(opt.transform['test'])
    
    model = ResUNet34(pretrained=opt.model['pretrained'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    cudnn.benchmark = True

    with open('./data/{:s}/train_val_test.json'.format(opt.dataset), 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    # ----- load trained model ----- #
    print("=> loading trained model")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    model = model.module

    # switch to evaluate mode
    model.eval()
    counter = 0
    print("=> Test begins:")

    img_names = os.listdir(img_dir)

    if save_flag:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        strs = img_dir.split('/')
        prob_maps_folder = '{:s}/{:s}_prob_maps'.format(save_dir, strs[-1])
        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)


    for img_name in img_names:
        # load test image
        if img_name not in train_list:
            continue
        print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        inst_name=img_name[:-4]+'_label.png'
        inst_path = './data/{:s}/labels_instance/'.format(opt.dataset)+inst_name
        gt_inst = np.array(Image.open(inst_path))
        gt_num = max(np.unique(gt_inst))

        ori_h = img.size[1]
        ori_w = img.size[0]
        name = os.path.splitext(img_name)[0]
        label_path = '{:s}/{:s}_label.png'.format(label_dir, name)
        gt = imageio.imread(label_path)
        
        input = test_transform((img,))[0].unsqueeze(0)

        print('\tComputing output probability maps...')
        prob_maps,gauss_maps = get_probmaps(input, model, opt)

        # save image
        if save_flag:
            print('\tSaving image results...')
            imageio.imsave('{:s}/{:s}_prob.png'.format(prob_maps_folder, name), prob_maps[1, :, :])


        counter += 1
        if counter % 10 == 0:
            print('\tProcessed {:d} images'.format(counter))

    print('=> Processed all {:d} images'.format(counter))
  


def get_probmaps(input, model, opt):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']
    dbranch_num = len(opt.train['decoder_radius'])

    if size == 0:
        with torch.no_grad():
            output = model(input.cuda())
    else:
        result = mysplit(model, input, size, overlap, dbranch_num)
        output = result[0]
        gauss_pred = torch.stack(result[1:], dim=0)
 
    output = output.squeeze(0)
    prob_maps = F.softmax(output, dim=0).cpu().numpy()
 
    gauss_maps = torch.sigmoid(gauss_pred).squeeze(1).cpu().numpy()

    return prob_maps, gauss_maps

def mysplit(model, input, size, overlap, dbranch_num, outchannel=2):
    '''
    split the input image for forward passes
    '''
    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    if h0 - size > 0 and (h0 - size) % (size - overlap) > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0 and (w0 - size) % (size - overlap) > 0:
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()
    output = torch.zeros((input.size(0), outchannel, h, w))

    g_out = torch.zeros((dbranch_num, input.size(0), 1, h, w))

    for i in range(0, h - overlap, size - overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w - overlap, size - overlap):
            c_end = j + size if j + size < w else w

            input_patch = input[:, :, i:r_end, j:c_end]
            input_var = input_patch.cuda()
            with torch.no_grad():
                result_patch = model(input_var)

                output_patch = result_patch[0]
                gauss_patch = torch.stack(result_patch[1:], dim=0)

            ind2_s = j + overlap // 2 if j > 0 else 0
            ind2_e = j + size - overlap // 2 if j + size < w else w
            output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:, :, ind1_s - i:ind1_e - i, ind2_s - j:ind2_e - j]
            
            g_out[:, :, :, ind1_s:ind1_e, ind2_s:ind2_e] = gauss_patch[:, :, :, ind1_s - i:ind1_e - i, ind2_s - j:ind2_e - j]

    output = output[:, :, :h0, :w0].cuda()
    g_out = g_out[:, :, :, :h0, :w0].cuda()

    result = [output] + [g for g in g_out]

    return result




if __name__ == '__main__':
    main()
