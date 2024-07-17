import os, json
import torch
import numpy as np
import random
from skimage import morphology, measure, io, util

from options import Options
from prepare_data import main as prepare_data
from train import main as train
from prob_gen import main as test
from cluster_regenerate import main as update


def main():
    opt = Options(isTrain=True)
    opt.parse()

    if opt.train['random_seed'] >= 0:
        print('=> Using random seed {:d}'.format(opt.train['random_seed']))
        torch.manual_seed(opt.train['random_seed'])
        torch.cuda.manual_seed(opt.train['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.train['random_seed'])
        random.seed(opt.train['random_seed'])
    else:
        torch.backends.cudnn.benchmark = True


    num_repeat = opt.round
    opt.test['test_epoch']=80
 
    # initial training
    print('=============== initial training ===============')
    opt.train['save_dir'] = '{:s}/{:d}'.format (opt.train['save_dir'], 0)
    print('=> Start training')
    train(opt)

    for i in range(1,num_repeat):
        opt.train['save_dir'] = './experiments/{:s}'.format(opt.dataset)
        # test model  
        print('=> Testing ...')
        opt.test['img_dir'] = './data_for_train/{:s}/images'.format(opt.dataset)
        opt.test['save_dir'] = './experiments/{:s}/{:d}/test_results'.format(opt.dataset,i-1)
        opt.test['model_path'] = '{:s}/{:d}/checkpoints/checkpoint_{:d}.pth.tar' \
            .format(opt.train['save_dir'],i-1, opt.test['test_epoch'])

        print('=> Inference ...')
        opt.test['img_dir'] = './data/{:s}/images'.format(opt.dataset)
        test(opt)

        print('=> Cluster Label regenerating ...')
        update(opt,i-1)

        print('=============== training round {:d} ==============='.format(i+1))
        opt.train['label_cluster_dir'] = '{:s}/labels_cluster_{:d}'.format(opt.train['data_dir'],i-1)
        opt.train['save_dir'] = '{:s}/{:d}'.format (opt.train['save_dir'], i)
        train(opt)





if __name__ == '__main__':
    main()
