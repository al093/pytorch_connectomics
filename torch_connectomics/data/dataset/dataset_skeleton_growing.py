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
                 image, skeleton, flux, growing_data,
                 flux_gt=None, mode='train'):
        self.mode = mode
        self.image = image
        self.skeleton = skeleton
        self.growing_data = growing_data
        self.flux = flux
        self.flux_gt = flux_gt if flux_gt is not None else [None]*len(image)

        self.ids = []
        for gd in growing_data:
            self.ids.append(list(gd.keys()))

        self.sample_num = np.array([len(gd) for gd in self.growing_data])
        self.sample_num_a = int(np.sum(self.sample_num))
        self.sample_num_c = np.cumsum([0] + list(self.sample_num))
        print('Dataset size: ', self.sample_num_a)

    def __len__(self):  # number of seed points
        return self.sample_num_a

    def __getitem__(self, index):
        did = self.get_pos_dataset(index)
        index = self.get_dataset_index(did, index)
        k = self.ids[did][index]
        if self.mode == 'train':
            #return the growing data and image, flux, skeleton
            path = self.growing_data     [did][k]['path'][1:-1]
            start_pos = self.growing_data[did][k]['path'][0]
            stop_pos = self.growing_data [did][k]['path'][-1]
            start_sid = self.growing_data[did][k]['sids'][0]
            stop_sid = self.growing_data [did][k]['sids'][1]
            if 'first_split_node' in self.growing_data[did][k].keys():
                first_split_node = self.growing_data[did][k]['first_split_node']
            else:
                first_split_node = -1

            # get the parameters here for flip transpose augmentation
            ft_params = self.get_flip_transpose_params()

            # calculate the approx class weights for the state preciction
            state_bce_weight = np.float32(5.0 / len (path)) # this is the loss weight which should be applied to all non-state predition positions
            return self.image[did], self.flux[did], self.flux_gt[did], self.skeleton[did], path, start_pos, stop_pos, start_sid, stop_sid, ft_params, state_bce_weight, first_split_node, did, k
        elif self.mode == 'test':
            start_pos = self.growing_data[did][k]['path'][0]
            start_sid = self.growing_data[did][k]['sids'][0]
            return self.image[did], self.flux[did], self.skeleton[did], start_pos, start_sid, did, k

    def get_flip_transpose_params(self):
        xflip, yflip, zflip, xytranspose = np.random.randint(2, size=4)
        params = {'xflip':xflip, 'yflip':yflip, 'zflip':zflip, 'xytranspose':xytranspose }
        return params

    def get_pos_dataset(self, index):
        return np.argmax(index < self.sample_num_c) - 1  # which dataset

    def get_dataset_index(self, did, index):
        idx = index - self.sample_num_c[did]
        return idx