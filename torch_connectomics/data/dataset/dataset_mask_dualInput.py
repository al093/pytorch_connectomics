from __future__ import print_function, division
import numpy as np
from scipy.ndimage.interpolation import shift
import torch
import torch.utils.data

from .misc import crop_volume, rebalance_binary_class
from .dataset_mask import MaskDataset

class MaskDatasetDualInput(MaskDataset):
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

        super(MaskDatasetDualInput, self).__init__(volume,
                                                    label,
                                                    sample_input_size,
                                                    sample_label_size,
                                                    sample_stride,
                                                    augmentor,
                                                    mode,
                                                    seed_points,
                                                    pad_size)

        self.rim_width = np.array([4, 24, 24], dtype=np.uint32)

    def __getitem__(self, index):
        vol_size = self.sample_input_size
        valid_mask = None

        # Train Mode Specific Operations:
        if self.mode == 'train':
            # 2. get input volume
            seed = np.random.RandomState(index)

            # 3 First sampling, pos is the top left origin to start cropping from
            pos = self.get_pos_seed(vol_size, seed)
            out_label_input = crop_volume(self.label[pos[0]], vol_size, pos[1:])

            possible_pos = np.transpose(np.nonzero(out_label_input))
            possible_pos = possible_pos[(possible_pos[:, 0] < self.rim_width[0]) |
                                        (possible_pos[:, 0] > vol_size[0] - self.rim_width[0]) |
                                        (possible_pos[:, 1] < self.rim_width[1]) |
                                        (possible_pos[:, 1] > vol_size[1] - self.rim_width[1]) |
                                        (possible_pos[:, 2] < self.rim_width[2]) |
                                        (possible_pos[:, 2] > vol_size[2] - self.rim_width[2])]

            while True:
                choosen_idx = np.random.randint(possible_pos.shape[0], dtype=np.int64)
                pos_2 = possible_pos[choosen_idx] + pos[1:] - self.half_input_sz
                # check if inside bounds, such that a sample can be cropped around it
                if np.all(pos_2 + vol_size < self.label[pos[0]].shape) and np.all(pos_2 >= 0):
                    break

            # import pdb; pdb.set_trace()
            # get a partial label vol from the first sampling based on the int_pos
            out_label_input = crop_volume(self.label[pos[0]], vol_size, pos[1:])
            out_label_input = shift(out_label_input, -(possible_pos[choosen_idx].astype(np.int64) - self.half_input_sz.astype(np.int64)), order=0, prefilter=False)

            out_label = crop_volume(self.label[pos[0]], vol_size, pos_2)
            out_input = crop_volume(self.input[pos[0]], vol_size, pos_2)

            # 3. augmentation
            if self.augmentor is not None:  # augmentation
                data = {'image': out_input, 'label': out_label, 'input_label': out_label_input}
                augmented = self.augmentor(data, random_state=seed)
                out_input, out_label, out_label_input = augmented['image'], augmented['label'], augmented['input_label']
                out_input = out_input.astype(np.float32)
                out_label = out_label.astype(np.uint32)
                out_label_input = out_label_input.astype(np.uint32)

                # Test Mode Specific Operations:
        elif self.mode == 'test':
            # test mode
            pos = self.get_pos_test(index)
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])
            out_label = None if self.label is None else crop_volume(self.label[pos[0]], vol_size, pos[1:])

        if out_label is not None:
            out_label = torch.from_numpy(out_label.copy().astype(np.float32))
            out_label = out_label.unsqueeze(0)

        if out_label_input is not None:
            out_label_input = torch.from_numpy(out_label_input.copy().astype(np.float32))
            out_label_input = out_label_input.unsqueeze(0)

        # Turn input to Pytorch Tensor, unsqueeze once to include the channel dimension:
        out_input = torch.from_numpy(out_input.copy())
        out_input = out_input.unsqueeze(0)

        if self.mode == 'train':
            # Rebalancing
            temp = 1.0 - out_label.clone()
            weight_factor, weight = rebalance_binary_class(temp)
            return pos, out_input, out_label_input, out_label, weight, weight_factor

        else:
            return pos, out_input