import numpy as np
import cv2
from scipy import spatial,ndimage
from scipy.ndimage import center_of_mass
from scipy.ndimage import gaussian_filter
from PIL import Image


def create_point_label_from_instance(image):
    def get_point(img):
        a = np.where(img != 0)
        rmin, rmax, cmin, cmax = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        return (rmin + rmax) // 2, (cmin + cmax) // 2

    # print("Generating point label from instance label...")
    h, w = image.shape
    # extract bbox
    id_max = np.max(image)
    label_point = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, id_max + 1):
        nucleus = image == i
        if np.sum(nucleus) == 0:
            continue
        x, y = get_point(nucleus)
        label_point[x, y] = 255
    
    return label_point







def generate_density_map(label_map, sigma):
    # Get the coordinates of people in the label map
    # y_coords, x_coords = np.where(label_map > 0)

    inst_list = np.unique(label_map)
    num_inst = max(inst_list)
    y_coords = []
    x_coords = []
    for inst_id in range(1,num_inst+1):
        instance_mask = label_map==inst_id
        y,x=center_of_mass(instance_mask)
        y_coords.append(int(y))
        x_coords.append(int(x))
    # print(y_coords,x_coords)

    # Create an empty density map
    density_map = np.zeros_like(label_map, dtype=np.float32)

    # Generate Gaussian kernel
    # kernel_size = int(6 * sigma) + 1
    kernel_size=9
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = gaussian_kernel * gaussian_kernel.transpose()
    gaussian_kernel = gaussian_kernel/(np.sum(gaussian_kernel))
    # print(np.sum(gaussian_kernel))

    # Add Gaussian kernel to density map for each person
    for y, x in zip(y_coords, x_coords):
        min_y = max(0, y - kernel_size // 2)
        max_y = min(label_map.shape[0], y + kernel_size // 2 + 1)
        min_x = max(0, x - kernel_size // 2)
        max_x = min(label_map.shape[1], x + kernel_size // 2 + 1)

        density_map[min_y:max_y, min_x:max_x] += gaussian_kernel[
            kernel_size // 2 - (y - min_y):kernel_size // 2 + (max_y - y),
            kernel_size // 2 - (x - min_x):kernel_size // 2 + (max_x - x)
        ]

    # Normalize the density map
    # density_map = cv2.normalize(density_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return density_map

def gaussian_filter_density(gt):
    print(gt.shape)

    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

    tree = spatial.KDTree(pts.copy(), leafsize=2048)
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density



def create_detect_label_from_points(points ,radius):



    if np.sum(points > 0):
        label_detect = gaussian_filter(points.astype(np.float), sigma=radius/3)
        val = np.min(label_detect[points > 0])
        label_detect = label_detect / val
        label_detect[label_detect < 0.05] = 0
        label_detect[label_detect > 1] = 1
    else:
        label_detect = np.zeros(points.shape)

    return  label_detect


# Example usage
# from PIL import Image
# import imageio
# img=Image.open('data/MO/labels_point/Breast_TCGA-AR-A1AK-01Z-00-DX1_label_point.png')
# label_map=np.array(img)
# # label_map = np.zeros((100, 100), dtype=np.uint8)  # Replace with your label map
# sigma = 5.0  # Replace with your desired sigma value
#
# density_map = generate_density_map(label_map, sigma)
# # density_map=gaussian_filter_density(label_map)
# print(np.sum(density_map))
# density_map = cv2.normalize(density_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
# print(np.sum(density_map))
# pred_map=np.zeros((1000,1000))
# pred_map[density_map>=0.5]=1
# cv2.imshow("Density Map", pred_map)
# cv2.waitKey(0)