from __future__ import division

import random
import warnings
import numpy as np

from skimage.morphology import dilation,erosion
from skimage.filters import gaussian
from torch_connectomics.data.augmentation import swapz

class Compose(object):
    """Compose transforms

    Args:
        transforms (list): list of transformations to compose.
        input_size (tuple): input size of model in (z, y, x).
        keep_uncropped (bool): keep uncropped images and labels (default: False).
        keep_non_smooth (bool): return also the non-smoothed masks (default: False).
    """
    def __init__(self, 
                 transforms, 
                 input_size = (8,196,196),
                 keep_uncropped = False,
                 keep_non_smoothed = False):
        self.transforms = transforms
        self.input_size = np.array(input_size)
        self.sample_size = self.input_size.copy()
        self.set_sample_params()
        self.keep_uncropped = keep_uncropped
        self.keep_non_smoothed = keep_non_smoothed

    def set_sample_params(self):
        sym_sz_needed = False
        for _, t in enumerate(self.transforms):
            self.sample_size = np.ceil(self.sample_size * t.sample_params['ratio']).astype(int)
            self.sample_size = self.sample_size + (2 * np.array(t.sample_params['add']))
            # check if z swapping is used
            if isinstance(t, swapz.SwapZ):
                sym_sz_needed = True

        if sym_sz_needed is True:
            max_edge_size = np.max(self.sample_size)
            self.sample_size = np.array([max_edge_size, max_edge_size, max_edge_size])

        print('Sample size required for the augmentor:', self.sample_size)

    def smooth_edge(self, data):
        smoothed_label = data.copy()

        for z in range(smoothed_label.shape[0]):
            temp = smoothed_label[z].copy()
            for idx in np.unique(temp):
                if idx != 0:
                    binary = (temp==idx).astype(np.uint8)
                    for _ in range(2):
                        binary = dilation(binary)
                        binary = gaussian(binary, sigma=2, preserve_range=True)
                        binary = dilation(binary)
                        binary = (binary > 0.8).astype(np.uint8)
            
                    temp[np.where(temp==idx)]=0
                    temp[np.where(binary==1)]=idx
            smoothed_label[z] = temp

        return smoothed_label

    def get_crop_params(self, image):
        margin = (image.shape[1] - self.input_size[1]) // 2
        margin = int(margin)

        # whether need to crop z or not (missing section augmentation)
        if image.shape[0] > self.input_size[0]:
            # always crop keeping the center fixed
            z_low = (image.shape[0] - self.input_size[0]) // 2
        else:
            z_low = 0
        z_high = z_low + self.input_size[0]
        z_low, z_high = int(z_low), int(z_high)
        low = margin
        high = margin + self.input_size[1]

        return z_low, z_high, low, high

    def crop(self, image, crop_params):
        # assert image.shape[-3:] == label.shape
        assert image.ndim == 3 or image.ndim == 4
        (z_low, z_high, low, high) = crop_params
        if image.ndim == 3:
            return image[z_low:z_high, low:high, low:high]
        else:
            return image[:, z_low:z_high, low:high, low:high]

    def __call__(self, data, random_state=None):
        for t in reversed(self.transforms):
            if random.random() < t.p:
                data = t(data, random_state)

        # crop the data to input size
        # if self.keep_uncropped:
        #     data['uncropped_image'] = data['image']
        #     data['uncropped_label'] = data['label']
        #     if 'input_label' in data and data['input_label'] is not None:
        #         data['uncropped_input_label'] = data['input_label']

        output = {}
        crop_params = self.get_crop_params(data['image'])
        for key, val in data.items():
            output[key] = self.crop(val, crop_params)

        # if self.keep_non_smoothed:
        #     data['non_smoothed'] = data['label']
        # data['label'] = self.smooth_edge(data['label'])

        # if 'input_label' in data and data['input_label'] is not None:
        #     if self.keep_non_smoothed:
        #         data['input_label_non_smoothed'] = data['input_label']
        #     data['input_label'] = self.smooth_edge(data['input_label'])
        return output