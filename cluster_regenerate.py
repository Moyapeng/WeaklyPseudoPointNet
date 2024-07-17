import os
import shutil
from sklearn.cluster import KMeans
from scipy.ndimage.morphology import distance_transform_edt as dist_tranform
import glob
import json
import imageio
import numpy as np
from skimage import measure,morphology



def main(opt,i):
    global process_round 
    process_round = i
    dataset = opt.dataset
    data_dir = './data/{:s}'.format(dataset)
    img_dir = './data/{:s}/images'.format(dataset)
    label_point_dir = './data/{:s}/labels_point'.format(dataset)
    label_vor_dir = './data/{:s}/labels_voronoi'.format(dataset)
    label_cluster_dir = './data/{:s}/labels_cluster_{:d}'.format(dataset,process_round)

    # label_pred_dir = './experiments/{:s}/test_results/images_prob_maps'.format(dataset)
    label_pred_dir = './experiments/{:s}/{:d}/test_results/images_prob_maps'.format(dataset,process_round)


    patch_folder = './data/{:s}/patches'.format(dataset)
    train_data_dir = './data_for_train/{:s}'.format(dataset)
    create_folder('./data_for_train')
    create_folder(label_point_dir)
    create_folder(label_vor_dir)
    create_folder(label_cluster_dir)
    create_folder(label_pred_dir)
    create_folder(patch_folder)
    create_folder(train_data_dir)

    with open('{:s}/train_val_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list = data_list['train']

    
    # ------ create cluster label from point label and image
    create_cluster_label(img_dir, label_point_dir, label_vor_dir, label_cluster_dir, train_list,label_pred_dir)
    
   
    split_patches(label_cluster_dir, '{:s}/labels_cluster_{:d}'.format(patch_folder,process_round), 'label_cluster')

    organize_data_for_training(data_dir, train_data_dir)

  





def create_cluster_label(data_dir, label_point_dir, label_vor_dir, save_dir, train_list,label_pred_dir):
    from scipy.ndimage import morphology as ndi_morph

    img_list = os.listdir(data_dir)
    print("Generating cluster label from point label...")
    N_total = len(train_list)
    N_processed = 0
    for img_name in sorted(img_list):
        if img_name not in train_list:
            continue

        N_processed += 1
        flag = '' if N_processed < N_total else '\n'
        print('\r\t{:d}/{:d}'.format(N_processed, N_total), end=flag)

        # print('\t[{:d}/{:d}] Processing image {:s} ...'.format(count, len(img_list), img_name))
        ori_image = imageio.imread('{:s}/{:s}'.format(data_dir, img_name))
        h, w, _ = ori_image.shape
        label_point = imageio.imread('{:s}/{:s}_label_point.png'.format(label_point_dir, img_name[:-4]))

        # add probmap of the trained model
        prob_map = imageio.imread('{:s}/{:s}_prob.png'.format(label_pred_dir, img_name[:-4]))
        prob_map = prob_map/prob_map.mean()*20
        pred = prob_map.reshape(-1,1)
        # pred = np.zeros_like(prob_map)
        # pred[prob_map>=0.5]=20
        # pred = pred.reshape(-1,1)


        # k-means clustering
        dist_embeddings = dist_tranform(255 - label_point).reshape(-1, 1)
        clip_dist_embeddings = np.clip(dist_embeddings, a_min=0, a_max=20)
        color_embeddings = np.array(ori_image, dtype=np.float).reshape(-1, 3) / 10
        embeddings = np.concatenate((color_embeddings, clip_dist_embeddings), axis=1)
        embeddings = np.concatenate((embeddings, pred), axis=1)
        # embeddings = np.concatenate((clip_dist_embeddings, pred), axis=1)

        # print("\t\tPerforming k-means clustering...")
        kmeans = KMeans(n_clusters=3, random_state=0).fit(embeddings)
        clusters = np.reshape(kmeans.labels_, (h, w))

        # get nuclei and background clusters
        overlap_nums = [np.sum((clusters == i) * label_point) for i in range(3)]
        nuclei_idx = np.argmax(overlap_nums)
        remain_indices = np.delete(np.arange(3), nuclei_idx)
        dilated_label_point = morphology.binary_dilation(label_point, morphology.disk(5))
        overlap_nums = [np.sum((clusters == i) * dilated_label_point) for i in remain_indices]
        background_idx = remain_indices[np.argmin(overlap_nums)]

        nuclei_cluster = clusters == nuclei_idx
        background_cluster = clusters == background_idx

        overlap_nums = [np.sum((clusters == i) * label_point) for i in range(3)]
        nuclei_idx = np.argmax(overlap_nums)
        nuclei_cluster = clusters == nuclei_idx
        nuclei_cluster = ndi_morph.binary_fill_holes(nuclei_cluster)
        
        # get background cluster
        remain_indices = np.delete(np.arange(3), nuclei_idx)
        dilated_nuclei_cluster = morphology.binary_dilation(nuclei_cluster, morphology.disk(3))
        overlap_nums = [np.sum((clusters == i) * dilated_nuclei_cluster) for i in remain_indices]
        background_idx = remain_indices[np.argmin(overlap_nums)]
        background_cluster = clusters == background_idx

        #refine clustering results
        print("\t\tRefining clustering results...")

        nuclei_labeled = measure.label(nuclei_cluster)
        initial_nuclei = morphology.remove_small_objects(nuclei_labeled, 30)
        refined_nuclei = np.zeros(initial_nuclei.shape, dtype=np.bool)

        label_vor = imageio.imread('{:s}/{:s}_label_vor.png'.format(label_vor_dir, img_name[:-4]))
        voronoi_cells = measure.label(label_vor[:, :, 0] == 0)
        voronoi_cells = morphology.dilation(voronoi_cells, morphology.disk(2))

        unique_vals = np.unique(voronoi_cells)
        cell_indices = unique_vals[unique_vals != 0]
        N = len(cell_indices)
        for i in range(N):
            cell_i = voronoi_cells == cell_indices[i]
            nucleus_i = cell_i * initial_nuclei

            nucleus_i_dilated = morphology.binary_dilation(nucleus_i, morphology.disk(5))
            nucleus_i_dilated_filled = ndi_morph.binary_fill_holes(nucleus_i_dilated)
            nucleus_i_final = morphology.binary_erosion(nucleus_i_dilated_filled, morphology.disk(7))
            refined_nuclei += nucleus_i_final > 0

        refined_label = np.zeros((h, w, 3), dtype=np.uint8)
        label_point_dilated = morphology.dilation(label_point, morphology.disk(10))
        refined_label[:, :, 0] = (background_cluster * (refined_nuclei == 0) * (label_point_dilated == 0)).astype(np.uint8) * 255
        refined_label[:, :, 1] = refined_nuclei.astype(np.uint8) * 255

        imageio.imsave('{:s}/{:s}_label_cluster.png'.format(save_dir, img_name[:-4]), refined_label)


def split_patches(data_dir, save_dir, post_fix=None):
    import math
    """ split large image into small patches """
    create_folder(save_dir)
    print('split patches....')

    image_list = os.listdir(data_dir)
    for image_name in image_list:
        name = image_name.split('.')[0]
        if post_fix and name[-len(post_fix):] != post_fix:
            continue
        image_path = os.path.join(data_dir, image_name)
        image = imageio.imread(image_path)
        seg_imgs = []

        # split into 16 patches of size 250x250
        h, w = image.shape[0], image.shape[1]
        patch_size = 250
        h_overlap = math.ceil((4 * patch_size - h) / 3)
        w_overlap = math.ceil((4 * patch_size - w) / 3)
        for i in range(0, h-patch_size+1, patch_size-h_overlap):
            for j in range(0, w-patch_size+1, patch_size-w_overlap):
                if len(image.shape) == 3:
                    patch = image[i:i+patch_size, j:j+patch_size, :]
                else:
                    patch = image[i:i + patch_size, j:j + patch_size]
                seg_imgs.append(patch)

        for k in range(len(seg_imgs)):
            if post_fix:
                imageio.imsave('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(post_fix)-1], k, post_fix), seg_imgs[k])
            else:
                imageio.imsave('{:s}/{:s}_{:d}.png'.format(save_dir, name, k), seg_imgs[k])


def organize_data_for_training(data_dir, train_data_dir):
    # --- Step 1: create folders --- #
    create_folder('{:s}/labels_cluster_{:d}'.format(train_data_dir,process_round))
    create_folder('{:s}/labels_cluster_{:d}/train'.format(train_data_dir,process_round))
    # create_folder('{:s}/labels_point/train'.format(train_data_dir))
    # --- Step 2: move images and labels to each folder --- #
    print('Organizing data for training...')
    with open('{:s}/train_val_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    # train
    for img_name in train_list:
        name = img_name.split('.')[0]
   
        # label_cluster
        for file in glob.glob('{:s}/patches/labels_cluster_{:d}/{:s}*'.format(data_dir, process_round,name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/labels_cluster_{:d}/train/{:s}'.format(train_data_dir, process_round,file_name)
            shutil.copyfile(file, dst)



def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


if __name__ == '__main__':
    main()
