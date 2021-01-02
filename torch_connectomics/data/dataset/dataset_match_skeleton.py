import numpy as np
import torch
import torch.utils.data
import scipy
from scipy.ndimage import label as scipy_label

from .misc import crop_volume, crop_volume_mul, rebalance_binary_class, rebalance_skeleton_weight
# from torch_connectomics.utils.vis import save_data

class MatchSkeletonDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image, skeleton, flux,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 augmentor=None,
                 mode='train',
                 seed_points=None,
                 pad_size=None):
        self.mode = mode
        self.image = image  # image
        self.skeleton = skeleton  # output after first stage
        self.flux = flux  # output of last layer, or flux
        self.augmentor = augmentor  # data augmentation

        # samples, channels, depths, rows, cols
        self.input_size = [np.array(x.shape) for x in self.image]  # volume size, could be multi-volume input
        self.sample_input_size = np.array(sample_input_size)  # model input size
        self.sample_label_size = np.array(sample_label_size)  # model label size

        self.seed_points = seed_points  # seed points is a list of dict(match=[], no_match=[])
        self.half_input_sz = (sample_input_size//2)

        self.seed_points_offset = pad_size - self.half_input_sz
        self.sample_num = np.array([np.sum([len(y) for _, y in x.items()]) for x in self.seed_points])
        self.sample_num_a = int(np.sum(self.sample_num))
        self.sample_num_c = np.cumsum([0] + list(self.sample_num))

        self.dilation_sel = scipy.ndimage.generate_binary_structure(3, 1)
        self.minimum_seg_size = np.prod(self.sample_input_size) // 500

    def __len__(self):  # number of seed points
        return self.sample_num_a

    def __getitem__(self, index):
        vol_size = self.sample_input_size

        # Train Mode Specific Operations:
        if self.mode == 'train':
            # get input volume
            pos, skel_id1, skel_id2, match, sample = self.get_pos_seed()

            out_image = crop_volume(self.image[pos[0]], vol_size, pos[1:]).copy()
            out_skeleton = crop_volume(self.skeleton[pos[0]], vol_size, pos[1:]).astype(np.float32, copy=True)
            out_flux = crop_volume_mul(self.flux[pos[0]], vol_size, pos[1:]).copy()

            # Augmentations
            if self.augmentor is not None:  # augmentation
                data = {'image': out_image,
                        'flux': out_flux.astype(np.float32),
                        'skeleton': out_skeleton.astype(np.float32)}
                augmented = self.augmentor(data, random_state=None)
                out_image, out_flux = augmented['image'], augmented['flux']
                out_skeleton = augmented['skeleton']

        # Test Mode Specific Operations:
        elif self.mode == 'test':
            pos, skel_id1, skel_id2, match, sample = self.get_pos_test(index)

            out_image = crop_volume(self.image[pos[0]], vol_size, pos[1:]).copy()
            out_skeleton = crop_volume(self.skeleton[pos[0]], vol_size, pos[1:]).astype(np.float32, copy=True)
            out_flux = crop_volume_mul(self.flux[pos[0]], vol_size, pos[1:]).copy()

        # keep the two selected skeletons into 2 separate volumes, erase rest
        out_skeleton_1 = np.zeros_like(out_image)
        out_skeleton_1[out_skeleton == skel_id1] = 1
        out_skeleton_1 = torch.from_numpy(out_skeleton_1.astype(np.float32, copy=False))
        out_skeleton_1 = out_skeleton_1.unsqueeze(0)

        out_skeleton_2 = np.zeros_like(out_image)
        out_skeleton_2[out_skeleton == skel_id2] = 1
        out_skeleton_2 = torch.from_numpy(out_skeleton_2.astype(np.float32, copy=False))
        out_skeleton_2 = out_skeleton_2.unsqueeze(0)

        out_image = torch.from_numpy(out_image.astype(np.float32, copy=True))
        out_image = out_image.unsqueeze(0)

        out_flux = torch.from_numpy(out_flux.astype(np.float32, copy=True))

        match = np.array([match]).astype(np.float32)
        match = torch.from_numpy(match)

        return sample, out_image, out_skeleton_1, out_skeleton_2, out_flux, match

    def get_pos_dataset(self, index):
        return np.argmax(index < self.sample_num_c) - 1  # which dataset

    def get_pos_seed(self, offset=None) -> (np.array, int, int, bool):
        pos = [0, 0, 0, 0]

        # pick a dataset
        did = np.random.choice(len(self.seed_points))
        pos[0] = did

        # pick a match or no match bin
        if np.random.rand() >= 0.5:
            match = 1
            neg_pos_bin_name = "match"
        else:
            match = 0
            neg_pos_bin_name = "no_match"

        # pick a index, and sample
        idx = np.random.randint(len(self.seed_points[did][neg_pos_bin_name]))
        sample = self.seed_points[did][neg_pos_bin_name][idx]

        # pick a position
        if offset is None:
            pos[1:] = (sample[2] + sample[3]) // 2 + self.seed_points_offset
        else:
            raise NotImplementedError("Offset feature is not implemented.")
            # pos[1:] = sample[4] + offset

        skel_id_1, skel_id_2 = sample[0], sample[1]

        if np.random.rand() >= 0.5:
            skel_id_1, skel_id_2 = skel_id_2, skel_id_1

        return pos, skel_id_1, skel_id_2, match, sample

    def get_pos_test(self, index, offset=None):
        did = self.get_pos_dataset(index)
        idx = int(index - self.sample_num_c[did])

        # get sample from match or non match bin
        match_len = len(self.seed_points[did]['match'])
        if idx < match_len:
            sample = self.seed_points[did]['match'][idx]
            match = 1
        else:
            sample = self.seed_points[did]['no_match'][idx - match_len]
            match = 0

        pos = [did, 0, 0, 0]
        if offset is None:
            pos[1:] = (sample[2] + sample[3]) // 2 + self.seed_points_offset
        else:
            raise NotImplementedError("Offset feature is not implemented.")

        skel_id_1, skel_id_2 = sample[0], sample[1]

        return pos, skel_id_1, skel_id_2, match, sample

    def get_vol(self, pos):
        out_input = crop_volume(self.input[pos[0]], self.sample_input_size, pos[1:])
        out_input = torch.from_numpy(out_input.copy())
        out_input = out_input.unsqueeze(0)
        return out_input

