import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
import numpy as np
import logging
from tensorboardX import SummaryWriter
from model import ResUNet34
import utils
from dataset import DataFolder
from my_transforms import get_transforms
import monai
import time
from tqdm import tqdm
from skimage import morphology
from scipy.ndimage import binary_fill_holes
from PIL import Image
import imageio


def myloss(pred,mask):
    coeff = (mask==0)*1.+(mask!=0)*10.0
    loss = (mask-pred)**2
    loss = coeff*loss

    return torch.mean(loss)

def mse_with_num_loss(pred,gt_map,gt_num,count_flag,a,b):
    # criterion = MSELoss().cuda()
    reg_loss = myloss(pred,gt_map)
    return reg_loss

    
 
    

def dice_loss_with_ignore_index(prediction, target, ignore_index=0, epsilon=1e-6):
    prediction = torch.argmax(prediction, dim=1)  # 假设prediction是一个one-hot编码的结果
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.one_hot(prediction, num_classes=target.shape[1])

    mask = target != ignore_index
    prediction = prediction.float()
    target = target.float()
    prediction = prediction[mask]
    target = target[mask]
    
    intersection = torch.sum(prediction * target)
    union = torch.sum(prediction) + torch.sum(target)
    dice = 1 - (2. * intersection + epsilon) / (union + epsilon)
    return dice

def dice_ce_loss(prediction, target, ignore_index=0, epsilon=1e-6):
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index).cuda()
    ce = ce_loss(prediction, target)

    dice = dice_loss_with_ignore_index(prediction, target, ignore_index=ignore_index, epsilon=epsilon)

    combined_loss = ce + dice
    return combined_loss




def main(option):
    global opt 
    global num_iter, tb_writer, logger, logger_results
    opt = option
    opt.isTrain = True

    tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

    # set up logger
    logger, logger_results = setup_logging(opt)

    # ----- create model ----- #
    model = ResUNet34(pretrained=opt.model['pretrained'])
    # if not opt.train['checkpoint']:
    #     logger.info(model)
    model = nn.DataParallel(model)
    model = model.cuda()
    cudnn.benchmark = True

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])

    # ----- define criterion ----- #
    # criterion = torch.nn.NLLLoss(ignore_index=2).cuda()
    criterion = monai.losses.DiceCELoss()
    # gauss_loss = mse_with_num_loss()
    if opt.train['crf_weight'] > 0:
        logger.info('=> Using CRF loss...')
        global criterion_crf
        # criterion_crf = CRFLoss(opt.train['sigmas'][0], opt.train['sigmas'][1])

    # ----- load data ----- #
    data_transforms = {'train': get_transforms(opt.transform['train']),
                       'test': get_transforms(opt.transform['test'])}

    img_dir = '{:s}/train'.format(opt.train['img_dir'])
    target_vor_dir = '{:s}/train'.format(opt.train['label_vor_dir'])
    target_cluster_dir = '{:s}/train'.format(opt.train['label_cluster_dir'])
    target_point_dir= '{:s}/train'.format(opt.train['label_point_dir'])
    target_gauss_dir = '{:s}/train'.format(opt.train['label_gauss_dir'])
    weight_pred_dir = os.path.join(opt.train['save_dir'], 'weight_pred')
    weight_prob0_dir = os.path.join(opt.train['save_dir'], 'weight_prob0')
    weight_prob1_dir = os.path.join(opt.train['save_dir'], 'weight_prob1')
    dir_list = [img_dir, target_vor_dir, target_cluster_dir, target_point_dir, weight_pred_dir]
    post_fix = ['label_vor.png', 'label_cluster.png', 'label_point.png','weight_pred.png']
    num_channels = [3, 3, 3, 1, 1]
    train_set = DataFolder( dir_list, post_fix, num_channels, data_transforms['train'])
    train_loader = DataLoader(train_set, batch_size=opt.train['batch_size'], shuffle=True,
                              num_workers=opt.train['workers'])

    # ----- optionally load from a checkpoint for validation or resuming training ----- #
    if opt.train['checkpoint']:
        if os.path.isfile(opt.train['checkpoint']):
            logger.info("=> loading checkpoint '{}'".format(opt.train['checkpoint']))
            checkpoint = torch.load(opt.train['checkpoint'])
            opt.train['start_epoch'] = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(opt.train['checkpoint'], checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.train['checkpoint']))

    # ----- training and validation ----- #
    num_epoch = opt.train['train_epochs'] + opt.train['finetune_epochs']
    num_iter = num_epoch * len(train_loader)
    # print training parameters
    logger.info("=> Initial learning rate: {:g}".format(opt.train['lr']))
    logger.info("=> Batch size: {:d}".format(opt.train['batch_size']))
    logger.info("=> Number of training iterations: {:d}".format(num_iter))
    logger.info("=> Training epochs: {:d}".format(opt.train['train_epochs']))
    logger.info("=> Fine-tune epochs using dense CRF loss: {:d}".format(opt.train['finetune_epochs']))
    logger.info("=> CRF loss weight: {:.2g}".format(opt.train['crf_weight']))
    start_update_epoch = opt.train['start_update_epoch']
    ema_weight = opt.train['ema_weight']
    total_time = 0
    for epoch in range(opt.train['start_epoch'], num_epoch):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch+1, num_epoch))
        finetune_flag = False if epoch < opt.train['train_epochs'] else True

        count_flag = False if epoch < 60 else True
        if epoch < 80:
            count_flag = 0
        elif epoch < 120:
            count_flag = 1
        else:
            count_flag = 2

        if epoch % opt.train['ema_interval'] == 0:
            if epoch == 0:
                if os.path.exists(os.path.join(opt.train['save_dir'], 'weight_prob0')):
                    shutil.rmtree(os.path.join(opt.train['save_dir'], 'weight_prob0'))
                if os.path.exists(os.path.join(opt.train['save_dir'], 'weight_prob1')):
                    shutil.rmtree(os.path.join(opt.train['save_dir'], 'weight_prob1'))
                if os.path.exists(os.path.join(opt.train['save_dir'], 'weight_pred')):
                    shutil.rmtree(os.path.join(opt.train['save_dir'], 'weight_pred'))
                update_prediction(img_dir, target_cluster_dir, weight_prob0_dir,weight_prob1_dir, weight_pred_dir, model,ema_weight)
        
            if epoch == start_update_epoch:
                if os.path.exists(os.path.join(opt.train['save_dir'], 'weight_prob0')):
                    shutil.rmtree(os.path.join(opt.train['save_dir'], 'weight_prob0'))
                if os.path.exists(os.path.join(opt.train['save_dir'], 'weight_prob1')):
                    shutil.rmtree(os.path.join(opt.train['save_dir'], 'weight_prob1'))
                if os.path.exists(os.path.join(opt.train['save_dir'], 'weight_pred')):
                    shutil.rmtree(os.path.join(opt.train['save_dir'], 'weight_pred'))
                update_prediction(img_dir, target_cluster_dir, weight_prob0_dir,weight_prob1_dir, weight_pred_dir, model,ema_weight)
        
            if epoch > start_update_epoch:
                update_prediction(img_dir, target_cluster_dir, weight_prob0_dir, weight_prob1_dir,weight_pred_dir, model,ema_weight)





        if epoch == opt.train['train_epochs']:
            logger.info("Fine-tune begins, lr = {:.2g}".format(opt.train['lr'] * 0.1))
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.train['lr'] * 0.1

        start_time = time.time()
        train_results = train(train_loader, model, optimizer, criterion, finetune_flag,count_flag,epoch)
        end_time = time.time()
        print('train_time:',end_time-start_time)
        total_time += end_time-start_time
        train_loss, train_loss_vor, train_loss_cluster, train_loss_crf = train_results

        cp_flag = (epoch+1) % opt.train['checkpoint_freq'] == 0
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, opt.train['save_dir'], cp_flag)

        # save the training results to txt files
        logger_results.info('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                            .format(epoch+1, train_loss, train_loss_vor, train_loss_cluster,
                                      train_loss_crf))
        # tensorboard logs
        tb_writer.add_scalars('epoch_losses',
                              {'train_loss': train_loss, 'train_loss_vor': train_loss_vor,
                               'train_loss_cluster': train_loss_cluster,
                               'train_loss_crf': train_loss_crf}, epoch)
    tb_writer.close()
    print('Average training time:',total_time/num_epoch)
    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()
        i.close()
    for i in list(logger_results.handlers):
        logger_results.removeHandler(i)
        i.flush()
        i.close()


def train(train_loader, model, optimizer, criterion, finetune_flag,count_flag,epoch):
    # list to store the average loss for this epoch
    results = utils.AverageMeter(4)

    # switch to train mode
    model.train()
    for i, sample in enumerate(train_loader):
        input, target1, target2, point,weight, gauss, num = sample

        if target1.dim() == 4:
            target1 = target1.squeeze(1)
 
        if target2.dim() == 4:
            target2 = target2.squeeze(1)

        input_var = input.cuda()

        result = model(input_var)
        output = result[0]
        gauss_pred = result[1:]


        model1 = ResUNet34(pretrained=opt.model['pretrained'])
        model1 = torch.nn.DataParallel(model1)
        model1 = model1.cuda()
        cudnn.benchmark = True

     
        log_prob_maps = F.log_softmax(output, dim=1)

        loss_vor = dice_ce_loss(log_prob_maps.cuda(), target1.cuda(),ignore_index=2)
        loss_cluster = dice_ce_loss(log_prob_maps.cuda(), target2.cuda(),ignore_index=2)
        if epoch<=opt.train['start_update_epoch']:
            loss_cluster = dice_ce_loss(log_prob_maps.cuda(), target2.cuda(), ignore_index=2)
        else:
            weight = weight.squeeze(1)
            weight[weight<127]=0
            weight[weight!=0]=1
            weight[target2==2]=2
            loss_cluster = dice_ce_loss(log_prob_maps.cuda(), weight.cuda(),ignore_index=2)


        loss_gauss = 0
        dbranch_num  = len(opt.train['decoder_radius'])
        for g in range(dbranch_num):
            log_gauss_maps = torch.sigmoid(gauss_pred[g])
            loss_gauss += mse_with_num_loss(log_gauss_maps,gauss[g][0].cuda().float(),num.cuda().float(),count_flag,gauss[g][1][0],gauss[g][2][0])
        loss_gauss = loss_gauss / dbranch_num

        loss = loss_vor + loss_cluster + loss_gauss

 



        if opt.train['crf_weight'] != 0 and finetune_flag:
            image = input.detach().clone().data.cpu()
            mean, std = np.load('{:s}/mean_std.npy'.format(opt.train['data_dir']))
            for k in range(image.size(0)):
                for t, m, s in zip(image[k], mean, std):
                    t.mul_(s).add_(m)




        
        if opt.train['crf_weight'] != 0 and finetune_flag:
            result = [loss.item(), loss_vor.item(), loss_cluster.item(),-1]
        else:
            result = [loss.item(), loss_vor.item(), loss_cluster.item(),-1]

        results.update(result, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_vor {r[1]:.4f}'
                        '\tLoss_cluster {r[2]:.4f}'
                        '\tLoss_CRF {r[3]:.4f}'.format(i, len(train_loader), r=results.avg))

    logger.info('\t=> Train Avg: Loss {r[0]:.4f}'
                '\tloss_vor {r[1]:.4f}'
                '\tloss_cluster {r[2]:.4f}'
                '\tloss_CRF {r[3]:.4f}'.format(r=results.avg))

    return results.avg


def update_prediction(img_dir, clu_dir, save_prob0_dir,save_prob1_dir,save_pred_dir, model, alpha):
    ## save pred into the experiment fold
    ## load all the training images
    img_names = os.listdir(img_dir)
    img_process = tqdm(img_names)
    test_transform = get_transforms(opt.transform['test'])
    print('[bold magenta]saving EMA weights ...[/bold magenta]')
    size = opt.test['patch_size']
    overlap = opt.test['overlap']
    for img_name in img_process:
        img = Image.open(os.path.join(img_dir, img_name))

        input = test_transform((img,))[0].unsqueeze(0).cpu()

        output = cal_pred(model, input, size, overlap, outchannel=2)
        log_prob_maps = F.softmax(output, dim=1)
        pred = log_prob_maps.squeeze(0).cpu().detach().numpy()

        try:
            weight0 = Image.open(os.path.join(save_prob0_dir, img_name[:-4])+ '_weight_prob.png')
            weight1 = Image.open(os.path.join(save_prob1_dir, img_name[:-4])+ '_weight_prob.png')
            weight0 = np.array(weight0)/255
            weight1 = np.array(weight1)/255
            weight = np.stack((weight0, weight1), axis=0)
            weight = alpha * pred + (1 - alpha) * weight
        except:
            clu_name = img_name[:-4] + '_label_cluster.png'
            clu_rgb = np.array(Image.open(os.path.join(clu_dir, clu_name)))

            clu = np.ones((clu_rgb.shape[0], clu_rgb.shape[1]), dtype=np.uint8) * 2  # ignored
            clu[clu_rgb[:, :, 0] > 255 * 0.3] = 0  # background
            clu[clu_rgb[:, :, 1] > 255 * 0.5] = 1  # nuclei

            weight = pred
            weight0=weight[0]
            weight1=weight[1]

            weight0[clu==0]= 1
            weight0[clu==1]= 0
            weight1[clu==0]= 0
            weight1[clu==1]= 1
            os.makedirs(save_prob0_dir, exist_ok=True)
            os.makedirs(save_prob1_dir, exist_ok=True)
            os.makedirs(save_pred_dir, exist_ok=True)


        prob_name  = img_name[:-4] + '_weight_prob.png'
        pred_name  = img_name[:-4] + '_weight_pred.png'
        prediction1 = np.argmax(weight,axis=0)

        pred_cleaned = morphology.remove_small_objects(prediction1.astype(bool), min_size=20)
        pred_filled = binary_fill_holes(pred_cleaned)
        # pred_smoothed = binary_closing(pred_filled, disk(3))
        pred_smoothed=pred_filled
        
        

        imageio.imsave(os.path.join(save_prob0_dir, prob_name), (weight0 * 255).astype(np.uint8))
        imageio.imsave(os.path.join(save_prob1_dir, prob_name), (weight1 * 255).astype(np.uint8))

        imageio.imsave(os.path.join(save_pred_dir, pred_name), (pred_smoothed* 255).astype(np.uint8))

def cal_pred(model, input, size, overlap, outchannel=2):
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

    for i in range(0, h-overlap, size-overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w-overlap, size-overlap):
            c_end = j+size if j+size < w else w

            input_patch = input[:,:,i:r_end,j:c_end]
            input_var = input_patch.cuda()
            with torch.no_grad():
                # output_patch = model(input_var)
                # print(output_patch)
                # quit()
                output_patch= model(input_var)[0]

            ind2_s = j+overlap//2 if j>0 else 0
            ind2_e = j+size-overlap//2 if j+size<w else w
            output[:,:,ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:,:,ind1_s-i:ind1_e-i, ind2_s-j:ind2_e-j]


    output = output[:,:,:h0,:w0].cuda()


    return output


def save_checkpoint(state, epoch, save_dir, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch+1))


def setup_logging(opt):
    mode = 'a' if opt.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train_log.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_vor\ttrain_loss_cluster')

    return logger, logger_results


if __name__ == '__main__':
    main()
