from __future__ import print_function, division
import numpy as np
import torch
import torch.utils.data
import scipy
from scipy.ndimage import label as scipy_label
import scipy.ndimage.morphology as morphology
from scipy import spatial
from scipy import ndimage
import skimage

from .misc import crop_volume, crop_volume_mul, rebalance_binary_class, rebalance_skeleton_weight
from torch_connectomics.utils.vis import save_data

class MatchSkeletonDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image, skeleton, flux, skeleton_p,
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
        self.image = image  # image
        self.skeleton_p = skeleton_p  # output from last network
        self.skeleton = skeleton  # output after CC
        self.flux = flux  # output of last layer
        self.augmentor = augmentor  # data augmentation

        # samples, channels, depths, rows, cols
        self.input_size = [np.array(x.shape) for x in self.image]  # volume size, could be multi-volume input
        self.sample_input_size = np.array(sample_input_size)  # model input size
        self.sample_label_size = np.array(sample_label_size)  # model label size

        self.seed_points = seed_points
        self.half_input_sz = (sample_input_size//2)

        self.seed_points_offset = pad_size #- self.half_input_sz
        self.sample_num = np.array([(np.sum([len(y) for y in x])) for x in self.seed_points])
        self.sample_num_a = np.sum(self.sample_num)
        self.sample_num_c = np.cumsum([0] + list(self.sample_num))

        self.dilation_sel = scipy.ndimage.generate_binary_structure(3, 1)
        self.minimum_seg_size = np.prod(self.sample_input_size) // 500

    def __len__(self):  # number of seed points
        return self.sample_num_a

    def __getitem__(self, index):
        vol_size = self.sample_input_size

        # Train Mode Specific Operations:
        if self.mode == 'train':
            # 2. get input volume
            seed = np.random.RandomState(index)
            pos, skel_id1, skel_id2, match = self.get_pos_seed(seed)
            out_image = crop_volume(self.image[pos[0]], vol_size, pos[1:])
            out_skeleton = crop_volume(self.skeleton[pos[0]], vol_size, pos[1:])
            out_skeleton_p = crop_volume(self.skeleton_p[pos[0]], vol_size, pos[1:])
            out_flux = crop_volume_mul(self.flux[pos[0]], vol_size, pos[1:])

            out_image = out_image.copy()
            out_skeleton = out_skeleton.copy()
            out_skeleton_p = out_skeleton_p.copy()
            out_flux = out_flux.copy()

            # Augmentations
            if self.augmentor is not None:  # augmentation
                data = {'image':out_image, 'flux':out_flux.astype(np.float32),
                        'skeleton':out_skeleton.astype(np.float32), 'skeleton_p':out_skeleton_p.astype(np.float32)}
                augmented = self.augmentor(data, random_state=seed)
                out_image, out_flux = augmented['image'], augmented['flux']
                out_skeleton, out_skeleton_p = augmented['skeleton'], augmented['skeleton_p']

                out_image = out_image.astype(np.float32)
                out_flux = out_flux.astype(np.float32)
                out_skeleton = out_skeleton.astype(np.float)
                out_skeleton_p = out_skeleton_p.astype(np.float)

            #keep the two selected skeletons into 2 separate volumes, erase rest
            out_skeleton_1 = out_skeleton.copy()
            out_skeleton_1[out_skeleton_1 != skel_id1] = 0
            out_skeleton_1 = (out_skeleton_1 > 0)
            out_skeleton_2 = out_skeleton
            out_skeleton_2[out_skeleton_2 != skel_id2] = 0
            out_skeleton_2 = (out_skeleton_2 > 0)
            match = np.array([match]).astype(np.float32)
            match = torch.from_numpy(match)

        # Test Mode Specific Operations:
        elif self.mode == 'test':
            pos, skel_id1, skel_id2, key = self.get_pos_test(index)
            out_image = crop_volume(self.image[pos[0]], vol_size, pos[1:])
            out_skeleton = crop_volume(self.skeleton[pos[0]], vol_size, pos[1:])
            out_skeleton_p = crop_volume(self.skeleton_p[pos[0]], vol_size, pos[1:])
            out_flux = crop_volume_mul(self.flux[pos[0]], vol_size, pos[1:])

            out_skeleton_1 = out_skeleton.copy()
            out_skeleton_1[out_skeleton_1 != skel_id1] = 0
            out_skeleton_1 = (out_skeleton_1 > 0)
            out_skeleton_2 = out_skeleton.copy()
            out_skeleton_2[out_skeleton_2 != skel_id2] = 0
            out_skeleton_2 = (out_skeleton_2 > 0)

            out_image = out_image.copy()
            out_skeleton_p = out_skeleton_p.copy()
            out_flux = out_flux.copy()

        out_image = torch.from_numpy(out_image.astype(np.float32, copy=False))
        out_image = out_image.unsqueeze(0)

        out_skeleton_1 = torch.from_numpy(out_skeleton_1.astype(np.float32, copy=False))
        out_skeleton_1 = out_skeleton_1.unsqueeze(0)

        out_skeleton_2 = torch.from_numpy(out_skeleton_2.astype(np.float32, copy=False))
        out_skeleton_2 = out_skeleton_2.unsqueeze(0)

        out_skeleton_p = torch.from_numpy(out_skeleton_p.astype(np.float32, copy=False))
        out_skeleton_p = out_skeleton_p.unsqueeze(0)

        out_flux = torch.from_numpy(out_flux.astype(np.float32, copy=False))

        if self.mode == 'train':
            return pos, out_image, out_skeleton_1, out_skeleton_2, out_skeleton_p, out_flux, match
        else:
            return key, out_image, out_skeleton_1, out_skeleton_2, out_skeleton_p, out_flux

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
        idx = np.random.randint(len(self.seed_points[did][size_bin]))
        # pick a position
        if offset is None:
            pos[1:] = self.seed_points[did][size_bin][idx][4] + self.seed_points_offset
        else:
            pos[1:] = self.seed_points[did][size_bin][idx][4] + offset
        return pos, self.seed_points[did][size_bin][idx][0], self.seed_points[did][size_bin][idx][1], float(size_bin == 0)

    def get_pos_test(self, index):
        did = self.get_pos_dataset(index)
        idx = index - self.sample_num_c[did]
        pos = self.seed_points[did][0][idx][4]
        pos = pos + self.seed_points_offset
        seg_id_1 = self.seed_points[did][0][idx][0]
        seg_id_2 = self.seed_points[did][0][idx][1]
        return np.concatenate(([did], pos)), seg_id_1, seg_id_2, self.seed_points[did][0][idx]

    def get_vol(self, pos):
        out_input = crop_volume(self.input[pos[0]], self.sample_input_size, pos[1:])
        out_input = torch.from_numpy(out_input.copy())
        out_input = out_input.unsqueeze(0)
        return out_input

