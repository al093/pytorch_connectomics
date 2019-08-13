from __future__ import print_function, division
import numpy as np

from scipy.ndimage.morphology import distance_transform_cdt
from scipy.ndimage import label as scipy_label
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.interpolation import shift
import scipy.spatial as sp

import torch
import torch.utils.data
import torch.nn.functional as F

from .misc import crop_volume, rebalance_binary_class
from .dataset_mask import MaskDataset


class MaskDatasetDualInput(MaskDataset):

    def __init__(self,
                 volume, label=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 sample_stride=(1, 1, 1),
                 augmentor_pre=None,
                 augmentor=None,
                 mode='train',
                 seed_points=None,
                 pad_size=None,
                 model=None):

        assert mode == 'train'

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
        self.return_distance_transform = False
        self.model_input_size = augmentor.input_size
        self.model_half_isz = tuple(self.model_input_size // 2)
        self.sphere_mask = self.get_sphere(10)
        self.seed_points_offset_2 = pad_size - (self.model_input_size // 2)  # override the base class variable data
        self.augmentor_pre = augmentor_pre
        self.sel_cpu = np.ones((3, 3, 3), dtype=bool)
        self.sel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=torch.device('cpu'))
        self.model = model
        sz_diff = self.sample_input_size - self.model_input_size
        self.pad_sz = sz_diff // 2
        pad_adj = sz_diff % 2
        self.pad_sz[pad_adj > 0] += 1

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

    def __getitem__(self, index):
        seed = np.random.RandomState(index)

        # 1. get first volume
        pos = self.get_pos_seed(seed, offset=self.seed_points_offset_2)
        out_label = crop_volume(self.label[pos[0]], self.model_input_size, pos[1:])
        out_input = crop_volume(self.input[pos[0]], self.model_input_size, pos[1:])

        # 2. Apply augmentation for the first sample. alignment should not be changed here
        if self.augmentor_pre is not None:
            data = {'image': out_input, 'label': out_label}
            augmented = self.augmentor_pre(data, random_state=seed)
            out_input, out_label = augmented['image'], augmented['label']
            out_input = out_input.astype(np.float32)
            out_label = out_label.astype(np.uint32)

        # Turn input to Pytorch Tensor, unsqueeze twice once to include the channel dimension and batch size as 1:
        out_label_input = self.sphere_mask * out_label
        out_label_input = torch.from_numpy(np.array(out_label_input, dtype=np.float32))
        out_label_input = out_label_input.unsqueeze(0).unsqueeze(0).detach()
        out_input = torch.from_numpy(np.array(out_input, copy=True, dtype=np.float32))
        out_input = out_input.unsqueeze(0).unsqueeze(0).detach()

        # Run first inference
        with torch.no_grad():
            m = self.model.cpu()
            output = m(torch.cat((out_input, out_label_input), 1))
        output = output > 0.85

        # During initial training the output of the network would be bad, so use the out_label_input as second condition
        if output[(0, 0) + self.model_half_isz] == True:
            out_mask = output.cpu().detach().numpy().astype(bool)
        else:
            out_mask = out_label_input.cpu().detach().numpy().astype(bool)

        # edge detection to find the next sampling location
        cc_out_mask, _ = scipy_label(out_mask[0, 0])
        out_mask = (cc_out_mask == cc_out_mask[self.model_half_isz])
        out_mask = torch.from_numpy(np.array(out_mask, dtype=np.float32))
        edge = (F.conv3d(out_mask.unsqueeze(0).unsqueeze(0), self.sel, padding=1))
        edge = (edge > 0) * (edge < 9)
        edge = F.interpolate(edge.float(), scale_factor=1 / 4, mode='nearest')
        edge = edge > 0
        edge_pos = (torch.nonzero(edge[0, 0]) * 4).detach().numpy()

        if (edge_pos.shape[0] == 0):
            print(edge_pos.shape)

        int_pos = edge_pos[np.random.randint(edge_pos.shape[0], size=1)]
        int_pos = int_pos.astype(np.uint32)[0]
        global_pos = int_pos + pos[1:] - self.half_input_sz
        pos[1:] = global_pos
        # sample next location
        out_label = crop_volume(self.label[pos[0]], self.sample_input_size, pos[1:])
        out_input = crop_volume(self.input[pos[0]], self.sample_input_size, pos[1:])

        # align the first prediction with the second sampling
        out_mask = shift(out_mask,
                                -(int_pos.astype(np.int64) - self.model_half_isz),
                                order=0, prefilter=False)
        out_label_input = np.pad(out_mask, ((self.pad_sz[0], self.pad_sz[0]),
                                     (self.pad_sz[1], self.pad_sz[1]),
                                     (self.pad_sz[2], self.pad_sz[2])), 'constant')

        # Augment it now using the full fledged augmentor.
        if self.augmentor is not None:
            data = {'image': out_input, 'label': out_label, 'input_label': out_label_input}
            augmented = self.augmentor(data, random_state=seed)
            out_input, out_label, out_label_input = augmented['image'], augmented['label'], augmented['input_label']

        out_label_input = torch.from_numpy(np.array(out_label_input, dtype=np.float32))
        out_label_input = out_label_input.unsqueeze(0).detach()
        out_label = torch.from_numpy(np.array(out_label, copy=True, dtype=np.float32))
        out_label = out_label.unsqueeze(0).detach()
        out_input = torch.from_numpy(np.array(out_input, copy=True, dtype=np.float32))
        out_input = out_input.unsqueeze(0).detach()

        # Rebalancing
        temp = 1.0 - out_label.clone()
        weight_factor, weight = rebalance_binary_class(temp)
        return pos, out_input, out_label_input, out_label, weight, weight_factor