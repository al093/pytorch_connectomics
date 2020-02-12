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
                 augmentor=None,
                 mode='train'):

        self.mode = mode
        self.image = image[0]
        self.skeleton = skeleton[0]
        self.growing_data = growing_data[0]
        self.flux = flux[0]
        self.augmentor = augmentor # data augmentation
        self.num_seeds = len(self.growing_data)
        print('Dataset size: ', self.num_seeds)

    def __len__(self):  # number of seed points
        return self.num_seeds

    def __getitem__(self, index):
        if self.mode == 'train':
            #return the growing data and image, flux, skeleton
            path = self.growing_data[index]['path'][1:-1]
            start_pos = self.growing_data[index]['path'][0]
            stop_pos = self.growing_data[index]['path'][-1]
            start_sid = self.growing_data[index]['sids'][0]
            stop_sid = self.growing_data[index]['sids'][1]
            if 'first_split_node' in self.growing_data[index].keys():
                first_split_node = self.growing_data[index]['first_split_node']
            else:
                first_split_node = -1

            # get the parameters here for flip transpose augmentation
            ft_params = self.get_flip_transpose_params()

            # calculate the approx class weights for the state preciction
            state_bce_weight = np.float32(5.0 / len (path)) # this is the loss weight which should be applied to all non-state predition positions
            return self.image, self.flux, self.skeleton, path, start_pos, stop_pos, start_sid, stop_sid, ft_params, state_bce_weight, first_split_node
        elif self.mode == 'test':
            start_pos = self.growing_data[index]['path'][0]
            start_sid = self.growing_data[index]['sids'][0]
            return self.image, self.flux, self.skeleton, start_pos, start_sid

    def get_flip_transpose_params(self):
        xflip, yflip, zflip, xytranspose = np.random.randint(2, size=4)
        params = {'xflip':xflip, 'yflip':yflip, 'zflip':zflip, 'xytranspose':xytranspose }
        return params