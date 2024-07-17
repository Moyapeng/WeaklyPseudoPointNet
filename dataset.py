import torch
import torch.utils.data as data
import os
from PIL import Image
import numpy as np

from options import Options
from skimage import measure
from generate_gauss_kernel import create_detect_label_from_points,create_point_label_from_instance


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def img_loader(path, num_channels,is_array=False):
    if num_channels == 1:
        img = Image.open(path)
    else:
        img = Image.open(path).convert('RGB')
    if is_array:
        img = np.load(path)

    return img



def get_imgs_list(dir_list, post_fix=None):
    """
    :param dir_list: [img1_dir, img2_dir, ...]
    :param post_fix: e.g. ['label_vor.png', 'label_cluster.png',...]
    :return: e.g. [(img1.png, img1_label_vor.png, img1_label_cluster.png), ...]
    """
    img_list = []
    if len(dir_list) == 0:
        return img_list

    img_filename_list = os.listdir(dir_list[0])

    for img in img_filename_list:
        if not is_image_file(img):
            continue
        img1_name = os.path.splitext(img)[0]
        item = [os.path.join(dir_list[0], img), ]
        for i in range(1, len(dir_list)):
            img_name = '{:s}_{:s}'.format(img1_name, post_fix[i - 1])
            img_path = os.path.join(dir_list[i], img_name)
            item.append(img_path)

        if len(item) == len(dir_list):
            img_list.append(tuple(item))

    return img_list

# dataset that supports multiple images
class DataFolder(data.Dataset):
    def __init__(self, dir_list, post_fix, num_channels, data_transform=None, loader=img_loader):
        """
        :param dir_list: [img_dir, label_voronoi_dir, label_cluster_dir]
        :param post_fix:  ['label_vor.png', 'label_cluster.png']
        :param num_channels:  [3, 3, 3]
        :param data_transform: data transformations
        :param loader: image loader
        """
        super(DataFolder, self).__init__()
        if len(dir_list) != len(post_fix) + 1:
            raise (RuntimeError('Length of dir_list is different from length of post_fix + 1.'))
        if len(dir_list) != len(num_channels):
            raise (RuntimeError('Length of dir_list is different from length of num_channels.'))

        self.img_list = get_imgs_list(dir_list, post_fix)
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))

        self.data_transform = data_transform
        self.num_channels = num_channels
        self.loader = loader

    def __getitem__(self, index):
        img_paths = self.img_list[index]
        sample = [self.loader(img_paths[i], self.num_channels[i]) for i in range(len(img_paths))]
        # print(sample[0].shape,sample[1].shape,sample[2].shape)
        # quit()
        if self.data_transform is not None:
            sample = self.data_transform(sample)

        point = sample[3]
        sample=list(sample)
        p = np.array(point.squeeze())
        pp = p > 0
        inst_from_point = measure.label(pp, connectivity=2)

        gt_pointnum = max(np.unique(inst_from_point))
        inst_from_point = create_point_label_from_instance(inst_from_point)

        gauss = []
        
        opt = Options(isTrain=True)
        radius = opt.train['decoder_radius']
        dbranch_num = len(radius)
        for i in range(dbranch_num):
            # radius = 3*(i+2)
            density_map = create_detect_label_from_points(inst_from_point,radius[i])
            a = np.sum(density_map)/gt_pointnum
            b = len(np.where(density_map>0)[0])/gt_pointnum
            density_map=torch.from_numpy(density_map)
            gauss.append([density_map,a,b])

        sample.append(gauss)
        

        sample.append(torch.tensor(gt_pointnum))


        return sample

    def __len__(self):
        return len(self.img_list)

