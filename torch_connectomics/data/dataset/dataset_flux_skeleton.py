from __future__ import print_function, division
import numpy as np
import torch
import torch.utils.data
import scipy
from scipy.ndimage import label as scipy_label
import scipy.ndimage.morphology as morphology
from scipy import spatial, ndimage
import skimage
import warnings
import h5py
import edt

from .misc import crop_volume, crop_volume_mul, rebalance_binary_class, rebalance_skeleton_weight
from torch_connectomics.utils.vis import save_data

class FluxAndSkeletonDataset(torch.utils.data.Dataset):
    def __init__(self,
                 volume, label=None, skeleton=None, flux=None, weight=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 dataset_resolution=None,
                 augmentor=None,
                 mode='train',
                 seed_points=None,
                 pad_size=None):
        if mode == 'test':
            for x in seed_points:
                assert len(x) == 1

        self.mode = mode

        if isinstance(volume[0], str):
            self.input_paths = volume
            self.label_paths = label
            self.skeleton_paths = skeleton
            self.flux_paths = flux
            self.weight_paths = weight
            self.h5_files_opened = False
        else:
            self.input = volume
            self.label = label
            self.skeleton = skeleton
            self.flux = flux
            self.weight = weight
            # samples, channels, depths, rows, cols
            self.input_size = [np.array(x.shape) for x in self.input]  # volume size, could be multi-volume input
            self.h5_files_opened = True

        self.augmentor = augmentor  # data augmentation

        self.sample_input_size = np.array(sample_input_size)  # model input size
        self.sample_label_size = np.array(sample_label_size)  # model label size
        self.dataset_resolution = dataset_resolution

        self.seed_points = seed_points
        self.half_input_sz = (sample_input_size // 2)
        self.seed_points_offset = pad_size - self.half_input_sz

        self.sample_num = np.array([(np.sum([y.shape[0] for y in x])) for x in self.seed_points])
        self.sample_num_a = int(np.sum(self.sample_num))
        self.sample_num_c = np.cumsum([0] + list(self.sample_num))

        self.dilation_sel = scipy.ndimage.generate_binary_structure(3, 1)
        self.minimum_seg_size = np.prod(self.sample_input_size) // 500

    def __len__(self):  # number of seed points
        return self.sample_num_a

    def __getitem__(self, index):
        self.open_files_to_read()

        vol_size = self.sample_input_size

        # Train Mode Specific Operations:
        if self.mode == 'train':
            # get input volume
            pos = self.get_pos_seed()
            out_label = crop_volume(self.label[pos[0]], vol_size, pos[1:])
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])
            # TODO(Alok) remove this check
            out_input = out_input.astype(np.float32, copy=False)
            out_skeleton = crop_volume(self.skeleton[pos[0]], vol_size, pos[1:])
            out_flux = crop_volume_mul(self.flux[pos[0]], vol_size, pos[1:])

            out_label = out_label.copy()
            out_skeleton = out_skeleton.copy()
            out_input = out_input.copy()
            out_flux = out_flux.copy()

            if self.weight[pos[0]]:
                pre_weight = crop_volume(self.weight[pos[0]], vol_size, pos[1:])
                pre_weight = pre_weight.astype(np.float32, copy=True)

            # Augmentations
            if self.augmentor is not None:  # augmentation
                if self.weight[pos[0]]:
                    data = {'image':out_input, 'flux':out_flux.astype(np.float32),
                            'skeleton':out_skeleton.astype(np.float32), 'context':out_label.astype(np.float32), 'weight':pre_weight}
                else:
                    data = {'image': out_input, 'flux': out_flux.astype(np.float32),
                            'skeleton': out_skeleton.astype(np.float32), 'context': out_label.astype(np.float32)}

                augmented = self.augmentor(data, random_state=None)
                out_input, out_flux = augmented['image'], augmented['flux']
                out_skeleton, out_label = augmented['skeleton'], augmented['context']
                if self.weight[pos[0]]:
                    pre_weight = augmented['weight']

            out_label_mask = out_label > 0

            # out_skeleton = ndimage.morphology.distance_transform_edt((out_skeleton == 0)).astype(np.float32)
            skeleton_distance_tx = edt.edt(out_skeleton == 0, anisotropy=self.dataset_resolution[::-1], black_border=False, order='C', parallel=1)
            distance_th = 1.0 * self.dataset_resolution[0]
            out_skeleton = ((skeleton_distance_tx <= distance_th) & out_label_mask).astype(np.float32, copy=False)

            # TODO | NOTE this is for distance transform of skeletons, the distance_th should be 4.0
            # out_skeleton /= distance_th

            out_input = out_input.astype(np.float32)
            out_flux = out_flux.astype(np.float32)

        # Test Mode Specific Operations:
        elif self.mode == 'test':
            # test mode
            pos = self.get_pos_test(index)
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])

        if self.mode == 'train':
            # Re-balancing weights for Flux and skeleton in a similar way
            # all_ones = np.ones_like(out_label_mask)
            # flux_weight = skeleton_weight_mask.astype(np.float32)
            # flux_weight = self.compute_flux_weights(all_ones, skeleton_weight_mask, alpha=1.0)
            weight_distance_th = 2.5 * self.dataset_resolution[0]
            out_weight = (skeleton_distance_tx <= weight_distance_th).astype(np.float32, copy=False)
            out_weight += 0.1
            out_weight /= 1.1

            if self.weight[pos[0]]:
                out_weight[pre_weight>0] *= 4

            out_weight = torch.from_numpy(out_weight).unsqueeze(0)

        out_input = torch.from_numpy(out_input)
        out_input = out_input.unsqueeze(0)

        if out_flux is not None:
            out_flux = torch.from_numpy(out_flux.astype(np.float32, copy=False))

        if out_skeleton is not None:
            out_skeleton = torch.from_numpy((out_skeleton).astype(np.float32, copy=False)).unsqueeze(0)

        if self.mode == 'train':
            out_label_mask = torch.from_numpy(out_label_mask.astype(np.float32)).unsqueeze(0)
            return pos, out_input, out_label_mask, out_flux, out_weight, out_skeleton
        else:
            return pos, out_input

    def get_pos_dataset(self, index):
        return np.argmax(index < self.sample_num_c) - 1  # which dataset

    def get_pos_seed(self, offset=None):
        pos = [0, 0, 0, 0]
        # pick a dataset
        did = np.random.choice(len(self.seed_points))  # sample from all datasets equally
        pos[0] = did

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

    def compute_2d_seg(self, seg_3d):
        seg_2d = (seg_3d > 0).astype(np.uint8)
        return seg_2d

    def compute_flux(self, segment, skeleton):
        skeleton_points = np.transpose(np.nonzero(skeleton))
        # Finding closest points to skeleton
        kdtree = spatial.KDTree(skeleton_points)
        points = np.transpose(np.nonzero(segment))
        _, idxs = kdtree.query(points)

        dir_vec = skeleton_points[idxs] - points
        factor = np.sqrt((np.sum(dir_vec**2, axis=1)) + np.finfo(np.float32).eps)
        dir_vec = dir_vec / np.expand_dims(factor, axis=1)

        # Creating direction field
        direction = np.zeros((3,) + segment.shape, dtype=np.float32)
        direction[0, tuple(points[:,0]), tuple(points[:,1]), tuple(points[:,2])] = dir_vec[:, 0]
        direction[1, tuple(points[:,0]), tuple(points[:,1]), tuple(points[:,2])] = dir_vec[:, 1]
        direction[2, tuple(points[:,0]), tuple(points[:,1]), tuple(points[:,2])] = dir_vec[:, 2]

        return direction

    def compute_flux_weights(self, label, skeleton, alpha=1.0):
        weight = np.zeros_like(label, dtype=np.float32)
        total_vol = float(label.sum())
        skl_vol = float((label*skeleton).sum())
        non_skl_vol = total_vol - skl_vol
        weight[skeleton] = alpha*non_skl_vol/(skl_vol + 1e-10)
        np.clip(weight, 0.0, 5e2, out=weight)
        weight[label & ~skeleton] = 1.0
        return weight

    def compute_close_context_weights(self, context, alpha=2.0):
        weight = np.ones_like(context, dtype=np.float32)
        context_padded = np.pad(context.astype(np.float32), ((1, 1), (5, 5), (5, 5)), mode='reflect')
        context_padded[context_padded==0] = np.nan

        sliding_view = skimage.util.view_as_windows(context_padded, (3, 11, 11), step=1)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            min_vals = np.nanmin(sliding_view, axis=(-1, -2, -3))
            max_vals = np.nanmax(sliding_view, axis=(-1, -2, -3))
            multi_mask = (min_vals != max_vals)
            multi_mask[np.isnan(min_vals)] = False
        multi_mask = morphology.binary_dilation(multi_mask, structure=np.ones((3, 5, 5)))
        weight[multi_mask] = alpha
        return weight

    def relabel_disconnected(self, label):
        if self.remove_small_seg(label) == False: # remove small segments first
            #all segments were removed
            return label, None, False
        seg_ids = np.unique(label)
        seg_ids = seg_ids[seg_ids > 0]  # find seg ids to check
        max_seg_id = seg_ids.max() + 1
        label_new = np.zeros_like(label, dtype=np.uint16)  # create a new volume to store the processed seg
        old_seg_ref = {} # keep a dictionary for each new seg id key have a ref to the old seg id
        for seg_id in seg_ids:  # for each seg perform cc to see if they are split
            mask = (label == seg_id)
            label_cc, num_cc = skimage.measure.label(mask, return_num=True)
            if num_cc == 1:  # this is good just leave them as it is
                label_new[mask] = seg_id
            elif num_cc > 0:
                if self.remove_small_seg(label_cc) == False:
                    return label, None, False # removing small noise like pixels in split segmentation
                split_seg_ids = np.unique(label_cc)
                split_seg_ids = split_seg_ids[split_seg_ids > 0]  # find all the remaining new segs id
                for split_id in split_seg_ids:
                    label_new[label_cc == split_id] = max_seg_id  # store the seg
                    old_seg_ref[max_seg_id] = seg_id  # reference to the old seg_id
                    max_seg_id += 1
        return label_new, old_seg_ref, True

    def remove_small_seg(self, label):
        seg_ids = np.unique(label)
        seg_ids = seg_ids[seg_ids>0]
        for seg_id in seg_ids:
            mask = (label == seg_id)
            if mask.sum() < self.minimum_seg_size:
                label[mask] = 0

        if np.all(label == 0):
            return False
        else:
            return True

    def open_files_to_read(self):
        if not self.h5_files_opened:
            self.input = []
            self.label = []
            self.skeleton = []
            self.flux = []
            self.weight = [] if self.weight_paths is not None else [None]*len(self.input_paths)

            for i in range(len(self.input_paths)):
                self.input.append(h5py.File(self.input_paths[i], 'r')['main'])
                # check if input is of float32 type
                if self.input[-1].dtype != np.float32:
                    pass
                    # raise RuntimeError(f"Input volume {self.input_paths[i]} is of type {self.input[-1].dtype.name},"
                    #                    f" it should be float32.")

                self.label.append(h5py.File(self.label_paths[i], 'r')['main'])
                self.skeleton.append(h5py.File(self.skeleton_paths[i], 'r')['main'])
                self.flux.append(h5py.File(self.flux_paths[i], 'r')['main'])

                if self.weight_paths:
                    self.weight.append(h5py.File(self.weight_paths[i], 'r')['main'])

            self.h5_files_opened = True

            # samples, channels, depths, rows, cols
            self.input_size = [np.array(x.shape) for x in self.input]  # volume size, could be multi-volume input
