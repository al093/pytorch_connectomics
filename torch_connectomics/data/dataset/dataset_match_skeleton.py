import numpy as np

import scipy.ndimage
import torch
import edt
from .misc import crop_volume, crop_volume_mul

class MatchSkeletonDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image, skeleton, flux, skeleton_probability,
                 weight=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 dataset_resolution=None,
                 augmentor=None,
                 mode='train',
                 seed_points=None,
                 pad_size=None):
        self.mode = mode
        self.image = image  # image
        self.skeleton = skeleton  # output after first stage, i.e. after connected components
        self.flux = flux  # output of last layer, or flux
        self.skeleton_probability = skeleton_probability  # output of the network, or the divergence of flux network
        self.weight = weight
        self.augmentor = augmentor  # data augmentation

        # samples, channels, depths, rows, cols
        self.input_size = [np.array(x.shape, dtype=np.int) for x in self.image]  # volume size, could be multi-volume input
        self.sample_input_size = np.array(sample_input_size)  # model input size
        self.sample_label_size = np.array(sample_label_size)  # model label size
        self.dataset_resolution = dataset_resolution

        self.seed_points = seed_points  # seed points is a list of dict(match=[], no_match=[])
        self.half_input_sz = (sample_input_size//2)

        self.seed_points_offset = pad_size - self.half_input_sz
        self.sample_num = np.array([np.sum([len(y) for _, y in x.items()]) for x in self.seed_points])
        self.sample_num_a = int(np.sum(self.sample_num))
        self.sample_num_c = np.cumsum([0] + list(self.sample_num))

    def __len__(self):  # number of seed points
        return self.sample_num_a

    def __getitem__(self, index):
        vol_size = self.sample_input_size

        if self.mode == 'train':
            pos, skel_id1, skel_id2, match, sample = self.get_pos_seed()
        elif self.mode == 'test':
            pos, skel_id1, skel_id2, match, sample = self.get_pos_test(index)

        out_image = crop_volume(self.image[pos[0]], vol_size, pos[1:]).astype(np.float32, copy=True)
        out_skeleton = crop_volume(self.skeleton[pos[0]], vol_size, pos[1:]).astype(np.float32, copy=True)
        out_flux = crop_volume_mul(self.flux[pos[0]], vol_size, pos[1:]).astype(np.float32, copy=True)
        out_skeleton_probability = crop_volume(self.skeleton_probability[pos[0]], vol_size, pos[1:]).astype(np.float32, copy=True)

        if self.mode == 'train' and self.augmentor is not None:  # augmentation
            data = {'image': out_image,
                    'flux': out_flux,
                    'skeleton': out_skeleton,
                    'skeleton_probability': out_skeleton_probability}
            augmented = self.augmentor(data, random_state=None)
            out_image, out_flux = augmented['image'], augmented['flux']
            out_skeleton = augmented['skeleton']
            out_skeleton_probability = augmented['skeleton_probability']

            skeleton_distance_tx = edt.edt(out_skeleton == 0, anisotropy=self.dataset_resolution[::-1], black_border=False, order='C', parallel=1)
            weight_distance_th = 1.75 * self.dataset_resolution[0]
            out_weight = (skeleton_distance_tx <= weight_distance_th).astype(np.float32, copy=False)
            out_weight += 0.1
            out_weight /= 1.1
            out_weight = torch.from_numpy(out_weight).unsqueeze(0)

        # keep the two selected skeletons into 2 separate volumes, erase rest
        out_skeleton_1 = np.zeros_like(out_image)
        out_skeleton_1[out_skeleton == skel_id1] = 1
        out_skeleton_1 = torch.from_numpy(out_skeleton_1.astype(np.float32, copy=False))
        out_skeleton_1 = out_skeleton_1.unsqueeze(0)

        out_skeleton_2 = np.zeros_like(out_image)
        out_skeleton_2[out_skeleton == skel_id2] = 1
        out_skeleton_2 = torch.from_numpy(out_skeleton_2.astype(np.float32, copy=False))
        out_skeleton_2 = out_skeleton_2.unsqueeze(0)

        out_image = torch.from_numpy(out_image.astype(np.float32, copy=True)).unsqueeze(0)
        out_flux = torch.from_numpy(out_flux.astype(np.float32, copy=True))
        out_skeleton_probability = torch.from_numpy(out_skeleton_probability.astype(np.float32, copy=True)).unsqueeze(0)

        match = np.array([match]).astype(np.float32)
        match = torch.from_numpy(match)

        if self.mode == 'train':
            return sample, out_image, out_skeleton_1, out_skeleton_2, out_flux, out_skeleton_probability, out_weight, match
        else:
            return sample, out_image, out_skeleton_1, out_skeleton_2, out_flux, out_skeleton_probability, match

    def _dilate(self, skeleton):
        dilation_kernel = np.zeros((3,3,3), dtype=np.bool)
        dilation_kernel[1, ...] = True
        return scipy.ndimage.binary_dilation(skeleton, structure=dilation_kernel, iterations=2)

    def get_pos_dataset(self, index):
        return np.argmax(index < self.sample_num_c) - 1  # which dataset

    def get_pos_seed(self) -> (np.array, int, int, bool):
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

        pos[1:] = (sample[2] + sample[3]) // 2 + self.seed_points_offset # pos is the origin

        # TODO|NOTE shift augmentation is disabled
        if self.augmentor and False:
            # add some random offset but such that cropping can be still done
            pos_arr = np.array(pos[1:])

            max_positive_shift = self.input_size[pos[0]] - (pos_arr + self.sample_input_size)
            assert np.all(max_positive_shift >= 0)
            max_positive_shift = np.clip(max_positive_shift, [0, 0, 0], [2, 2, 2])

            min_negative_shift = -pos_arr
            assert np.all(min_negative_shift <= 0)
            min_negative_shift = np.clip(min_negative_shift, [-2, -2, -2], [0, 0, 0])

            random_shift = np.zeros((3, ), dtype=np.int)
            for dim in range(3):
                random_shift[dim] = np.random.choice(np.arange(min_negative_shift[dim],
                                                               max_positive_shift[dim]+1))

            pos[1:] += random_shift

        skel_id_1, skel_id_2 = sample[0], sample[1]

        if np.random.rand() >= 0.5:
            skel_id_1, skel_id_2 = skel_id_2, skel_id_1

        return pos, skel_id_1, skel_id_2, match, sample

    def get_pos_test(self, index):
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
        pos[1:] = (sample[2] + sample[3]) // 2 + self.seed_points_offset
        skel_id_1, skel_id_2 = sample[0], sample[1]

        return pos, skel_id_1, skel_id_2, match, sample

    def get_vol(self, pos):
        out_input = crop_volume(self.input[pos[0]], self.sample_input_size, pos[1:])
        out_input = torch.from_numpy(out_input.copy())
        out_input = out_input.unsqueeze(0)
        return out_input

