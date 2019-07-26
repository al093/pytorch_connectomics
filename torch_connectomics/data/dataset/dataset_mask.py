from __future__ import print_function, division
import numpy as np
import random

import torch
import torch.utils.data

from torch_connectomics.utils.seg.aff_util import seg_to_affgraph, affinitize
from torch_connectomics.utils.seg.seg_util import mknhood3d, widen_border1, widen_border2

from .dataset import BaseDataset
from .misc import crop_volume, rebalance_binary_class, count_volume

class MaskDataset(BaseDataset):
    """PyTorch ddataset class for affinity graph prediction.

    Args:
        volume: input image stacks.
        label: segmentation stacks.
        sample_input_size (tuple, int): model input size.
        sample_label_size (tuple, int): model output size.
        sample_stride (tuple, int): stride size for sampling.
        augmentor: data augmentor.
        mode (str): training or inference mode.
    """
    def __init__(self,
                 volume, label=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 sample_stride=(1, 1, 1),
                 augmentor=None,
                 mode='train',
                 seed_points=None,
                 pad_size=None):

        super(MaskDataset, self).__init__(volume,
                                            label,
                                            sample_input_size,
                                            sample_label_size,
                                            sample_stride,
                                            augmentor,
                                            mode)
        self.seed_points = seed_points
        self.seed_points_offset = pad_size - (sample_input_size//2)
        if mode == 'test':
            self.sample_num = np.array([x.shape[0]*x.shape[1] for x in self.seed_points])
            self.sample_num_a = np.sum(self.sample_num)
            self.sample_num_c = np.cumsum([0] + list(self.sample_num))

    def __len__(self):  # number of seed points
        return self.sample_num_a

    def get_pos_seed(self, vol_size, seed):
        pos = [0, 0, 0, 0]
        # pick a dataset
        did = self.get_pos_dataset(seed.randint(self.sample_num_a))
        pos[0] = did

        # pick a mask bin
        size_bin = np.random.randint(len(self.seed_points[did]))
        # pick a index
        idx = np.random.randint(self.seed_points[did][size_bin].shape[0])
        # pick a position
        pos[1:] = self.seed_points[did][size_bin][idx] + self.seed_points_offset
        return pos

    def get_pos_test(self, index):
        did = self.get_pos_dataset(index)
        idx = index - self.sample_num_c[did]
        pos = self.seed_points[did][idx]
        pos = pos + self.seed_points_offset
        return np.concatenate(([did], pos))

    def __getitem__(self, index):
        vol_size = self.sample_input_size
        valid_mask = None

        # Train Mode Specific Operations:
        if self.mode == 'train':
            # 2. get input volume
            seed = np.random.RandomState(index)
            # if elastic deformation: need different receptive field
            # change vol_size first
            pos = self.get_pos_seed(vol_size, seed)
            out_label = crop_volume(self.label[pos[0]], vol_size, pos[1:])
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])

            if out_label.shape[1] == 0 or out_label.shape[2] == 0:
                import pdb;
                pdb.set_trace()
            # #debug
            # if out_label[49, 107, 107] != 1:
            #     import pdb; pdb.set_trace()
            #     print(pos[1:] + np.array([49, 107, 107], dtype=np.uint32))

            # 3. augmentation
            if self.augmentor is not None:  # augmentation
                data = {'image':out_input, 'label':out_label}
                augmented = self.augmentor(data, random_state=seed)
                out_input, out_label = augmented['image'], augmented['label']
                out_input = out_input.astype(np.float32)
                out_label = out_label.astype(np.uint32)

        # Test Mode Specific Operations:
        elif self.mode == 'test':
            # test mode
            pos = self.get_pos_test(index)
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])
            out_label = None if self.label is None else crop_volume(self.label[pos[0]], vol_size, pos[1:])
            
        # Turn segmentation label into affinity in Pytorch Tensor
        if out_label is not None:
            # # check for invalid region (-1)
            # seg_bad = np.array([-1]).astype(out_label.dtype)[0]
            # valid_mask = out_label!=seg_bad
            # out_label[out_label==seg_bad] = 0
            # out_label = widen_border1(out_label, 1)
            # #out_label = widen_border2(out_label, 1)
            # # replicate-pad the aff boundary
            # out_label = seg_to_affgraph(out_label, mknhood3d(1), pad='replicate').astype(np.float32)
            out_label = torch.from_numpy(out_label.copy().astype(np.float32))
            out_label = out_label.unsqueeze(0)

        # Turn input to Pytorch Tensor, unsqueeze once to include the channel dimension:
        out_input = torch.from_numpy(out_input.copy())
        out_input = out_input.unsqueeze(0)

        if self.mode == 'train':
            # Rebalancing
            temp = 1.0 - out_label.clone()
            weight_factor, weight = rebalance_binary_class(temp)
            return pos, out_input, out_label, weight, weight_factor

        else:
            return pos, out_input
