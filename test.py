import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
import scipy.ndimage.morphology as ndi_morph
from skimage import measure,morphology
from skimage.measure import label, regionprops
from scipy import misc

from model import ResUNet34
import utils
from accuracy import compute_metrics
import time

from options import Options
from my_transforms import get_transforms
import imageio
import concurrent
from concurrent.futures import ThreadPoolExecutor

from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from utils import voronoi_finite_polygons_2d, poly2mask
from options import Options



def get_point(img):
    a = np.where(img != 0)
    rmin, rmax, cmin, cmax = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return (rmin + rmax) // 2, (cmin + cmax) // 2


def create_Voronoi_label(label_point):
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon
    from utils import voronoi_finite_polygons_2d, poly2mask

    
    h, w = label_point.shape

    points = np.argwhere(label_point > 0)
    vor = Voronoi(points)

    regions, vertices = voronoi_finite_polygons_2d(vor)
    box = Polygon([[0, 0], [0, w], [h, w], [h, 0]])
    region_masks = np.zeros((h, w), dtype=np.int16)
    edges = np.zeros((h, w), dtype=np.bool)
    count = 1
    for region in regions:
        polygon = vertices[region]
        # Clipping polygon
        poly = Polygon(polygon)
        poly = poly.intersection(box)
        polygon = np.array([list(p) for p in poly.exterior.coords])

        mask = poly2mask(polygon[:, 0], polygon[:, 1], (h, w))
        edge = mask * (morphology.erosion(mask, morphology.disk(1)))
        edges += edge
        region_masks[mask] = count
        count += 1
    
    return region_masks

def process_label(image, i):
    nucleus = image == i
    if np.sum(nucleus) == 0:
        return None
    x, y = get_point(nucleus)
    return x, y

def create_point_label_from_instance(gauss_label):
    gauss_label  = measure.label(gauss_label)
    image = np.array(gauss_label)
    h, w = image.shape

    id_max = np.max(image)
    label_point = np.zeros((h, w), dtype=np.uint8)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda i: process_label(image, i), range(1, id_max + 1)))

    for result in results:
        if result:
            x, y = result
            label_point[x, y] = 255

    return label_point

def process_voronoi_region(region, vertices, box, h, w):
    polygon = vertices[region]
    poly = Polygon(polygon)
    poly = poly.intersection(box)
    polygon = np.array([list(p) for p in poly.exterior.coords])

    mask = poly2mask(polygon[:, 0], polygon[:, 1], (h, w))
    edge = mask * (morphology.erosion(mask, morphology.disk(1)))
    
    return mask, edge

def create_Voronoi_label(label_point):
    h, w = label_point.shape

    points = np.argwhere(label_point > 0)
    vor = Voronoi(points)

    regions, vertices = voronoi_finite_polygons_2d(vor)
    box = Polygon([[0, 0], [0, w], [h, w], [h, 0]])
    region_masks = np.zeros((h, w), dtype=np.int16)
    edges = np.zeros((h, w), dtype=np.bool)
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda region: process_voronoi_region(region, vertices, box, h, w), regions))

    count = 1
    for mask, edge in results:
        edges += edge
        region_masks[mask] = count
        count += 1

    return region_masks






def remove_background_points(gauss_label, pred_label):
    indices = np.argwhere(gauss_label == 1)
    for x, y in indices:
        if pred_label[x, y] == 0:
            gauss_label[x, y] = 0
    return gauss_label

def add_missing_centers(pred_label, gauss_label):
    regions = regionprops(pred_label)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        sub_gauss_label = gauss_label[minr:maxr, minc:maxc]
        sub_pred_label = pred_label[minr:maxr, minc:maxc]

        if np.sum(sub_gauss_label[sub_pred_label == region.label]) == 0:
            center = region.centroid
            gauss_label[int(center[0]), int(center[1])] = 1

    return gauss_label

def update_gauss_label(pred_label, gauss_label):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_remove_bg = executor.submit(remove_background_points, gauss_label.copy(), pred_label)
        future_add_centers = executor.submit(add_missing_centers, pred_label, gauss_label.copy())

        gauss_label = future_remove_bg.result()
        gauss_label = future_add_centers.result()

    return gauss_label

def get_centers(pred_label):
    labeled_image = label(pred_label)
    regions = regionprops(labeled_image)
    pred_c = np.zeros_like(pred_label, dtype=np.uint8)

    for region in regions:
        center = region.centroid
        center = (int(center[0]), int(center[1]))
        pred_c[center] = 255
    
    return pred_c


def merge_small_regions(pred_label, voronoi_mask, min_area_ratio=0.15):
    from scipy.ndimage import label,center_of_mass, distance_transform_edt
    labeled_array, num_features = label(pred_label)
    pred_label_refine = np.zeros_like(pred_label)

    for region_id in range(1, num_features + 1):
        region_mask = (labeled_array == region_id)
        total_area = np.sum(region_mask)
        
        if total_area == 0:
            continue 
        
        subregion_mask = region_mask & (voronoi_mask > 0)
        subregions = subregion_mask * voronoi_mask
        subregion_labels, subregion_counts = np.unique(subregions[subregions > 0], return_counts=True)
        
        large_subregion = np.zeros_like(pred_label)
        max_count = max_label = 0
        for sub_label, count in zip(subregion_labels, subregion_counts):
            if count >= total_area * min_area_ratio:
                # print(np.unique(large_subregion))
                large_subregion[subregions == sub_label] = sub_label
                if count > max_count:
                    max_count = count
                    max_label = sub_label


        small_subregion = region_mask & (large_subregion == 0)
        
        if np.any(small_subregion):
            large_subregion[small_subregion] = max_label

        pred_label_refine += large_subregion
        

    return pred_label_refine





def post_process(img, prob_maps, gauss_maps, threshold=0.05, min_area=20, img_name='image'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred = np.argmax(prob_maps, axis=0)  # prediction
    pred_labeled = measure.label(pred)
    pred_labeled = morph.remove_small_objects(pred_labeled, min_area)
    pred_labeled = ndi_morph.binary_fill_holes(pred_labeled > 0)
    pred_labeled = measure.label(pred_labeled)

    gauss_maps = [torch.tensor(gauss_map[0], device=device) for gauss_map in gauss_maps]
    gauss_list = []
    g1 = time.time()
    for gauss_map in gauss_maps:
        unique_values, counts = torch.unique(gauss_map, return_counts=True)
        max_count_index = torch.argmax(counts)
        mode_value = unique_values[max_count_index].item()
        g = gauss_map - mode_value
        g[g < 0] = 0
        gauss_list.append(g)
    

    gauss_refine_list = []
    for g in gauss_list:
        gauss_refine = (g > threshold).int().cpu().numpy()
        gauss_refine = morph.remove_small_objects(gauss_refine, min_area)
        gauss_refine = ndi_morph.binary_fill_holes(gauss_refine > 0)
        gauss_refine_list.append(gauss_refine)
        
    
    dbranch_num = len(gauss_maps)
    gauss_refine1 = np.zeros_like(gauss_refine_list[0]).astype(np.uint8)
    count_matrix = sum((gauss_refine > threshold).astype(np.uint8) for gauss_refine in gauss_refine_list)
    gauss_refine1[count_matrix >= dbranch_num/2] = 1
    gauss_refine1 = morph.remove_small_objects(gauss_refine1, min_area)
    gauss_refine1 = ndi_morph.binary_fill_holes(gauss_refine1 > 0)
    pred_gauss_labeled1, N = measure.label(gauss_refine1, return_num=True)


    if N > 20:
        large_area = morph.remove_small_objects(pred_gauss_labeled1, min_area)
        gauss_refine1 = gauss_refine1 * (large_area > 0)
        label_point = get_centers(gauss_refine1)
        updated_point_label = update_gauss_label(pred_labeled, label_point)
        voronoi_mask = create_Voronoi_label(updated_point_label)
        pred_label_refine = merge_small_regions(pred_labeled, voronoi_mask)

        return pred,pred_labeled,pred_label_refine,gauss_refine_list

    else:
        print('Instances fewer than 20!')
        return pred,pred_labeled,pred_labeled,gauss_refine_list


def main():
    opt = Options(isTrain=False)
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
        gauss_maps_folder = '{:s}/{:s}_gauss_maps'.format(save_dir, strs[-1])
        seg_folder = '{:s}/{:s}_segmentation'.format(save_dir, strs[-1])
        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)
        if not os.path.exists(seg_folder):
            os.mkdir(seg_folder)
        if not os.path.exists(gauss_maps_folder):
            os.mkdir(gauss_maps_folder)

    metric_names = ['acc', 'p_F1', 'p_recall', 'p_precision', 'dice_pixel', 'dice', 'aji','dq','sq','pq']
    test_results = dict()
    test_results_gauss = dict()
    all_result = utils.AverageMeter(len(metric_names))
    all_result_gauss = utils.AverageMeter(len(metric_names))

    total_time = 0
    post_time = 0
    for img_name in img_names:
        # load test image
        start_time = time.time()
        print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        time1 = time.time()
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
        pred,pred_labeled,pred_gauss_labeled_refine,gauss_refine_list = post_process(img, prob_maps, gauss_maps, threshold=0.07, min_area=20, img_name='image')

        print('\tComputing metrics...')
        metrics = compute_metrics(pred_labeled, gt, metric_names)
        metrics_gauss = compute_metrics(pred_gauss_labeled_refine, gt, metric_names)

        # save result for each image
        test_results[name] = [metrics['acc'], metrics['p_F1'], metrics['p_recall'], metrics['p_precision'], metrics['dice_pixel'],
                              metrics['dice'], metrics['aji'],metrics['dq'],metrics['sq'],metrics['pq']]

        # update the average result
        all_result.update([metrics['acc'], metrics['p_F1'], metrics['p_recall'], metrics['p_precision'], metrics['dice_pixel'],
                          metrics['dice'], metrics['aji'],metrics['dq'],metrics['sq'],metrics['pq']])
        
        test_results_gauss[name] = [metrics_gauss['acc'], metrics_gauss['p_F1'], metrics_gauss['p_recall'], metrics_gauss['p_precision'], metrics_gauss['dice_pixel'],
                              metrics_gauss['dice'], metrics_gauss['aji'],metrics_gauss['dq'],metrics_gauss['sq'],metrics_gauss['pq']]

        # update the average result
        all_result_gauss.update([metrics_gauss['acc'], metrics_gauss['p_F1'], metrics_gauss['p_recall'], metrics_gauss['p_precision'], metrics_gauss['dice_pixel'],
                          metrics_gauss['dice'], metrics_gauss['aji'],metrics_gauss['dq'],metrics_gauss['sq'],metrics_gauss['pq']])

        print('{} result:'.format(img_name))
        print('prob_map:')
        print('acc: {}, p_F1: {}, p_recall: {}'.format(metrics['acc'], metrics['p_F1'], metrics['p_recall']))
        print('dice_pixel: {}, dice: {}, aji: {}, dq: {}, sq: {}, pq: {}'.format(metrics['dice_pixel'],metrics['dice'],metrics['aji'],metrics['dq'],metrics['sq'],metrics['pq']))

        print('gauss_map:')
        print('acc: {}, p_F1: {}, p_recall: {}'.format(metrics_gauss['acc'], metrics_gauss['p_F1'], metrics_gauss['p_recall']))
        print('dice_pixel: {}, dice: {}, aji: {}, dq: {}, sq: {}, pq: {}'.format(metrics_gauss['dice_pixel'], metrics_gauss['dice'], metrics_gauss['aji'],metrics_gauss['dq'],metrics_gauss['sq'],metrics_gauss['pq']))

        # save image
        if save_flag:
            print('\tSaving image results...')
            imageio.imsave('{:s}/{:s}_pred.png'.format(prob_maps_folder, name), pred.astype(np.uint8) * 255)
            imageio.imsave('{:s}/{:s}_prob.png'.format(prob_maps_folder, name), prob_maps[1, :, :])
            # for i in range(dbranch_num):
            #     imageio.imsave('{:s}/{:s}_gauss_pred_radius{:s}.png'.format(gauss_maps_folder, name,str(3*(i+2))), gauss_refine_list[i].astype(np.uint8) * 255)
            #     imageio.imsave('{:s}/{:s}_gauss_prob_radius{:s}.png'.format(gauss_maps_folder, name,str(3*(i+2))), gauss_map[i][0, :, :])
            final_pred = Image.fromarray(pred_labeled.astype(np.uint16))
            final_pred.save('{:s}/{:s}_seg.tiff'.format(seg_folder, name))

            # save colored objects
            pred_colored_instance = np.zeros((ori_h, ori_w, 3))
            for k in range(1, pred_labeled.max() + 1):
                pred_colored_instance[pred_labeled == k, :] = np.array(utils.get_random_color())
            filename = '{:s}/{:s}_seg_colored.png'.format(seg_folder, name)
            imageio.imsave(filename, pred_colored_instance)    

            pred_colored_instance1 = np.zeros((ori_h, ori_w, 3))
            for k in range(1, pred_gauss_labeled_refine.max() + 1):
                pred_colored_instance1[pred_gauss_labeled_refine== k, :] = np.array(utils.get_random_color())
            filename = '{:s}/{:s}_seg_colored_refine.png'.format(seg_folder, name)
            imageio.imsave(filename, pred_colored_instance1)   

        counter += 1
        if counter % 10 == 0:
            print('\tProcessed {:d} images'.format(counter))

    print('=> Processed all {:d} images'.format(counter))
    print('Average Acc: {r[0]:.4f}\nF1: {r[1]:.4f}\nRecall: {r[2]:.4f}\n'
          'Precision: {r[3]:.4f}\nDice_pixel: {r[4]:.4f}\nDice: {r[5]:.4f}\nAJI: {r[6]:.4f}\ndq: {r[7]:.4f}\nsq: {r[8]:.4f}\npq: {r[9]:.4f}\n'.format(r=all_result.avg))

    avg_inference_time = total_time/len(img_names)
    print(f"Avg Inference Time: {avg_inference_time} seconds")

    avg_post_time = post_time/len(img_names)
    print(f"Avg post Time: {avg_post_time} seconds")
    header = metric_names
    utils.save_results(header, all_result.avg, test_results, '{:s}/test_results.txt'.format(save_dir))
    utils.save_results(header, all_result_gauss.avg, test_results_gauss, '{:s}/test_results_gauss.txt'.format(save_dir))


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
