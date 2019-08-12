from __future__ import print_function, division
import numpy as np
import scipy.spatial as sp
from scipy.ndimage.morphology import distance_transform_cdt
import torch
import torch.utils.data
import time
from .misc import crop_volume, rebalance_binary_class
from .dataset_mask import MaskDataset

class FFNStyleDataset(MaskDataset):
    """PyTorch ddataset class for affinity graph prediction.

    Args:
        volume: input image stacks.
        label: segmentation stacks.
        sample_input_size (tuple, int): model input size.
        sample_label_size (tuple, int): model output size.
        sample_stride (tuple, int): stride size for sampling.
        augmentor_1: data augmentor for the first sampling. This must not change the alignment of the data so that its easy to sample a second volume
        augmentor_2: Full fleged augmentation for the second sampling.
        mode (str): training or inference mode.
    """
    def __init__(self,
                 volume, label=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 sample_stride=(1, 1, 1),
                 augmentor_1=None,
                 augmentor_2=None,
                 mode='train',
                 seed_points=None,
                 pad_size=None):

        super(FFNStyleDataset, self).__init__(volume,
                                                    label,
                                                    sample_input_size,
                                                    sample_label_size,
                                                    sample_stride,
                                                    augmentor_2,
                                                    mode,
                                                    seed_points,
                                                    pad_size)

        self.return_distance_transform = False
        self.augmentor_1 = augmentor_1
        self.augmentor_2 = augmentor_2
        self.model_input_size = augmentor_1.input_size
        self.sphere_mask = (torch.from_numpy(self.get_sphere(10))).unsqueeze(0)
        self.seed_points_offset = pad_size - (self.model_input_size // 2) # override the base class variable data


    def get_sphere(self, radius):
        sphere = np.zeros(self.model_input_size, dtype=np.float32)
        z = np.arange(self.model_input_size[0], dtype=np.int16)
        y = np.arange(self.model_input_size[1], dtype=np.int16)
        x = np.arange(self.model_input_size[2], dtype=np.int16)
        center = self.model_input_size // 2
        Z, Y, X = np.meshgrid(z, y, x)
        data = np.vstack((Z.ravel(), Y.ravel(), X.ravel())).T
        distance = sp.distance.cdist(data, center.reshape(1, -1)).ravel()
        points_in_sphere = data[distance < radius]
        sphere[(tuple(points_in_sphere[:, 0]), tuple(points_in_sphere[:, 1]), tuple(points_in_sphere[:, 2]))] = 1
        return sphere


    def __getitem__(self, index, pos=None, past_pred=None):
        assert self.mode == 'train'
        seed = np.random.RandomState(index)
        # 1. get input volume
        if pos is None:
            vol_size = self.model_input_size
            pos = self.get_pos_seed(seed)
            augmentor = self.augmentor_1
        else:
            vol_size = self.sample_input_size
            augmentor = self.augmentor_2

        out_label = crop_volume(self.label[pos[0]], vol_size, pos[1:])
        out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])

        # 2. augmentation
        if augmentor is not None:  # augmentation
            if past_pred is None:
                data = {'image': out_input, 'label': out_label}
                augmented = augmentor(data, random_state=seed)
                out_input, out_label = augmented['image'], augmented['label']
            else:

                data = {'image': out_input, 'label': out_label, 'input_label': past_pred}
                # augmented = augmentor(data, random_state=seed)
                # out_input, out_label, out_label_input = augmented['image'], augmented['label'], augmented['input_label']
                out_label_input = past_pred.astype(np.float32)
                out_label_input = torch.from_numpy(out_label_input)
                out_label_input = out_label_input.unsqueeze(0)
                out_label_input = out_label_input.detach()


            out_input = out_input.astype(np.float32)
            out_label = out_label.astype(np.uint32)

        if self.return_distance_transform:
            out_distance_tx = distance_transform_cdt(out_label)
            out_distance_tx = torch.from_numpy(out_distance_tx.copy().astype(np.float32))
            out_distance_tx = (1 - out_distance_tx/out_distance_tx.max())*out_label
            out_distance_tx = out_distance_tx.unsqueeze(0)
            out_distance_tx = out_distance_tx.detach()

        #3. Turn input to Pytorch Tensor, unsqueeze once to include the channel dimension:
        out_label = torch.from_numpy(out_label.copy().astype(np.float32))
        out_label = out_label.unsqueeze(0)
        out_label = out_label.detach()

        if past_pred is None:
            out_label_input = self.sphere_mask * out_label
            out_label_input = out_label_input.detach()

        out_input = torch.from_numpy(out_input.copy())
        out_input = out_input.unsqueeze(0)
        out_input = out_input.detach()

        #4. Rebalancing
        temp = 1.0 - out_label.clone()
        weight_factor, weight = rebalance_binary_class(temp)
        weight_factor, weight = weight_factor.detach(), weight.detach()

        if self.return_distance_transform:
            return pos, out_input, out_label_input*out_distance_tx, out_distance_tx, weight, weight_factor
        else:
            return pos, out_input, out_label_input, out_label, weight, weight_factor
