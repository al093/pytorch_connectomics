import numpy as np
import torch
import torch.utils.data
import scipy
from scipy.ndimage import label as scipy_label
import scipy.ndimage.morphology as morphology
from scipy import spatial
from scipy import ndimage
import skimage
import warnings

from .misc import crop_volume, crop_volume_mul, rebalance_binary_class, rebalance_skeleton_weight
from torch_connectomics.utils.vis import save_data

class SkeletonGrowingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image, skeleton, flux, growing_data=None,
                 sample_input_size=(8, 64, 64),
                 augmentor=None,
                 mode='train'):
        if mode == 'test':
            raise Exception('Not Implemented')

        self.mode = mode
        self.image = image[0]
        self.skeleton = skeleton[0]
        self.growing_data = growing_data[0]
        self.flux = flux[0]
        self.augmentor = augmentor # data augmentation

        # samples, channels, depths, rows, cols
        self.image_size = np.array(self.image.shape)
        self.sample_input_size = np.array(sample_input_size)  # model input size
        self.half_input_sz = (sample_input_size//2)
        self.num_seeds = len(self.growing_data)
        print('Dataset size: ', self.num_seeds)

    def __len__(self):  # number of seed points
        return self.num_seeds

    def __getitem__(self, index):
        vol_size = self.sample_input_size

        if self.mode == 'train':
            #return the growing data and image, flux, skeleton
            path = self.growing_data[index]['path'][1:-1]
            start_pos = self.growing_data[index]['path'][0]
            stop_pos = self.growing_data[index]['path'][-1]
            start_sid = self.growing_data[index]['sids'][0]
            stop_sid = self.growing_data[index]['sids'][1]

            #reverse path augmentation
            reverse_path = np.random.randint(2)
            if reverse_path == 1:
                path = path[::-1]
                start_pos = self.growing_data[index]['path'][-1]
                stop_pos = self.growing_data[index]['path'][0]
                start_sid = self.growing_data[index]['sids'][1]
                stop_sid = self.growing_data[index]['sids'][0]

            #flip transpose augmentation
            ft_params = self.get_flip_transpose_params()
            # ft_params = None

            # TODO start_pos perturbation: shift the start point a bit inside the skeleton mask
            # z_sigma =
            # s = np.random.randint(0, z_sigma, 1000)

            return self.image, self.flux, self.skeleton, path, start_pos, stop_pos, start_sid, stop_sid, ft_params

        # Test Mode Specific Operations:
        elif self.mode == 'test':
            raise Exception('Not Implemented')

    def get_flip_transpose_params(self):
        xflip, yflip, zflip, xytranspose = np.random.randint(2, size=4)
        params = {'xflip':xflip, 'yflip':yflip, 'zflip':zflip, 'xytranspose':xytranspose }
        return params