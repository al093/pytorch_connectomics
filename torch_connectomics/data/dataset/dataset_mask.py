from __future__ import print_function, division
import numpy as np

import torch
import torch.utils.data

from .misc import crop_volume, rebalance_binary_class

from scipy.ndimage import label as scipy_label
import scipy.ndimage.morphology as morphology

class MaskDataset(torch.utils.data.Dataset):
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
                 pad_size=None,
                 multisegment_gt=True):
        if mode == 'test':
            for x in seed_points:
                assert len(x) == 1

        self.mode = mode
        self.input = volume
        self.label = label
        self.augmentor = augmentor  # data augmentation

        # samples, channels, depths, rows, cols
        self.input_size = [np.array(x.shape) for x in self.input]  # volume size, could be multi-volume input
        self.sample_input_size = np.array(sample_input_size)  # model input size
        self.sample_label_size = np.array(sample_label_size)  # model label size

        self.seed_points = seed_points
        self.half_input_sz = (sample_input_size//2)
        self.seed_points_offset = pad_size - self.half_input_sz
        self.sample_num = np.array([(np.sum([y.shape[0] for y in x])) for x in self.seed_points])
        self.sample_num_a = np.sum(self.sample_num)
        self.sample_num_c = np.cumsum([0] + list(self.sample_num))

        # specifies if there are multiple segments in the GT, if yes then we need to keep only the central segment while calling get_item
        self.multisegment_gt = multisegment_gt

    def __len__(self):  # number of seed points
        return self.sample_num_a

    def __getitem__(self, index):
        vol_size = self.sample_input_size
        valid_mask = None

        # Train Mode Specific Operations:
        if self.mode == 'train':
            # 2. get input volume
            seed = np.random.RandomState(index)
            # if elastic deformation: need different receptive field
            # change vol_size first
            pos = self.get_pos_seed(seed)
            out_label = crop_volume(self.label[pos[0]], vol_size, pos[1:])
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])

            # select the center segment and delete the rest
            # this is needed only for parallel fibers, for the single neuron prediction only perform cc and remove
            # the non central segments
            if self.multisegment_gt:
                out_label = self.keep_seg(out_label, out_label[tuple(self.half_input_sz)])

            # Remove non central segment
            out_label = self.remove_non_central_seg(out_label)

            # 3. augmentation
            if self.augmentor is not None:  # augmentation
                data = {'image':out_input, 'label':out_label.astype(np.float32)}
                augmented = self.augmentor(data, random_state=seed)
                out_input, out_label = augmented['image'], augmented['label']
                out_input = out_input.astype(np.float32)
                out_label = out_label.astype(np.float32)

        # Test Mode Specific Operations:
        elif self.mode == 'test':
            # test mode
            pos = self.get_pos_test(index)
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])
            out_label = None if self.label is None else crop_volume(self.label[pos[0]], vol_size, pos[1:])

        if out_label is not None:
            out_label = torch.from_numpy(out_label) # did not create a copy because remove non central seg creates a copy
            out_label = out_label.unsqueeze(0)

        # Turn input to Pytorch Tensor, unsqueeze once to include the channel dimension:
        out_input = torch.from_numpy(out_input.copy())
        out_input = out_input.unsqueeze(0)

        if self.mode == 'train':
			# TODO if masked loss around center is needed use this mask for rebalancing
            # mask = morphology.binary_dilation(out_label[0].numpy(), structure=np.ones((5, 5, 5)))
            # mask = mask.astype(np.float32)

            # Rebalancing
            temp = 1.0 - out_label.clone()
            weight_factor, weight = rebalance_binary_class(temp, mask=None) # torch.from_numpy(mask)
            return pos, out_input, out_label, weight, weight_factor

        else:
            return pos, out_input

    def get_pos_dataset(self, index):
        return np.argmax(index < self.sample_num_c) - 1  # which dataset

    def get_pos_seed(self, seed, offset=None):
        pos = [0, 0, 0, 0]
        # pick a dataset
        did = self.get_pos_dataset(seed.randint(self.sample_num_a))
        pos[0] = did
        # pick a mask bin
        # p = [0.45, 0.15, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        size_bin = np.random.choice(len(self.seed_points[did]))
        # pick a index
        idx = np.random.randint(self.seed_points[did][size_bin].shape[0])
        # pick a position
        if offset is None:
            pos[1:] = self.seed_points[did][size_bin][idx] + self.seed_points_offset
        else:
            pos[1:] = self.seed_points[did][size_bin][idx] + offset
        return pos

    def get_pos_test(self, index):
        did = self.get_pos_dataset(index)
        idx = index - self.sample_num_c[did]
        pos = self.seed_points[did][0][idx]
        pos = pos + self.seed_points_offset
        return np.concatenate(([did], pos))

    def get_vol(self, pos):
        out_input = crop_volume(self.input[pos[0]], self.sample_input_size, pos[1:])
        out_input = torch.from_numpy(out_input.copy())
        out_input = out_input.unsqueeze(0)
        return out_input

    def keep_seg(self, label, seg_id_to_keep):
        return label == seg_id_to_keep

    def remove_non_central_seg(self, label):
        out_label, _ = scipy_label(label)

        if out_label[tuple(self.half_input_sz)] == 0:
            print('Center pixel is not inside 2nd inference\'s GT segmentation.')
            print('This probably happened due to augmentation')
            # Find nearby segment id and use that for now
            seg_ids = np.unique(out_label[self.half_input_sz[0]-5:self.half_input_sz[0]+6,
                                          self.half_input_sz[1]-5:self.half_input_sz[1]+6,
                                          self.half_input_sz[2]-5:self.half_input_sz[2]+6])
            seg_ids = seg_ids[seg_ids > 0]
            if seg_ids.shape[0] > 1:
                print('More than 1 disconnected segments near the center. This should have never happened!')
                print('Using the first segment')
            c_seg_id = seg_ids[0]
            out_label = (out_label == c_seg_id)
        else:
            out_label = (out_label == out_label[tuple(self.half_input_sz)])

        return out_label
