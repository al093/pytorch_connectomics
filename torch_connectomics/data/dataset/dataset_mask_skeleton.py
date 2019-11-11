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
import warnings

from .misc import crop_volume, crop_volume_mul, rebalance_binary_class, rebalance_skeleton_weight
from torch_connectomics.utils.vis import save_data

class MaskAndSkeletonDataset(torch.utils.data.Dataset):
    def __init__(self,
                 volume, label=None, skeleton=None, flux=None, weight=None,
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
        self.skeleton = skeleton
        self.flux = flux
        self.weight = weight
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

        self.dilation_sel = scipy.ndimage.generate_binary_structure(3, 1)
        self.minimum_seg_size = np.prod(self.sample_input_size) // 500

    def __len__(self):  # number of seed points
        return self.sample_num_a

    def __getitem__(self, index):
        vol_size = self.sample_input_size
        # Train Mode Specific Operations:
        if self.mode == 'train':
            is_good = False
            while is_good is False:
                # 2. get input volume
                seed = np.random.RandomState(index)
                # if elastic deformation: need different receptive field
                # change vol_size first
                pos = self.get_pos_seed(seed)
                out_label = crop_volume(self.label[pos[0]], vol_size, pos[1:])
                out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])
                out_skeleton = crop_volume(self.skeleton[pos[0]], vol_size, pos[1:])
                out_flux = crop_volume_mul(self.flux[pos[0]], vol_size, pos[1:])

                out_label = out_label.copy()
                out_skeleton = out_skeleton.copy()
                out_input = out_input.copy()
                out_flux = out_flux.copy()

                if self.weight:
                    pre_weight = crop_volume(self.weight[pos[0]], vol_size, pos[1:])
                    pre_weight = pre_weight.astype(np.float32, copy=True)

                # Augmentations
                if self.augmentor is not None:  # augmentation
                    data = {'image':out_input, 'flux':out_flux.astype(np.float32),
                            'skeleton':out_skeleton.astype(np.float32), 'context':out_label.astype(np.float32), 'weight':pre_weight}
                    augmented = self.augmentor(data, random_state=seed)
                    out_input, out_flux = augmented['image'], augmented['flux']
                    out_skeleton, out_label = augmented['skeleton'], augmented['context']
                    pre_weight = augmented['weight']

                    out_input = out_input.astype(np.float32)
                    out_flux = out_flux.astype(np.float32)
                    out_label = out_label.astype(np.float)

                # see if small side segments need to be removed, because they dont have enough context for predicting flux
                out_label, old_ref_d, is_good = self.relabel_disconnected(out_label)

                #remove skeleton and flux of deleted segs
                mask = out_label > 0
                out_flux = mask*out_flux
                out_skeleton = mask*out_skeleton

                if out_skeleton.sum() == 0:
                    is_good = False

            #dilating skeletons for better learning
            out_skeleton = ((out_skeleton > 0) * mask)
            # out_skeleton_dilated = morphology.binary_dilation(out_skeleton, structure=np.ones((3, 3, 3)), iterations=1)
            out_skeleton_blurred = ndimage.morphology.distance_transform_cdt(~out_skeleton)
            distance_th = 4.0
            dis_mask = (out_skeleton_blurred <= distance_th)
            out_skeleton_blurred = (distance_th - out_skeleton_blurred).astype(np.int64)
            out_skeleton_blurred[~dis_mask] = 0.0
            out_skeleton_blurred[~mask] = 0.0

        # Test Mode Specific Operations:
        elif self.mode == 'test':
            # test mode
            pos = self.get_pos_test(index)
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])
            out_label = None if self.label is None else crop_volume(self.label[pos[0]], vol_size, pos[1:])

        if self.mode == 'train':
            # Rebalancing weights for Flux
            # TODO remove this
            # mask_dilated = morphology.binary_dilation(mask, structure=np.ones((1, 3, 3)), iterations=5)
            # flux_weight = self.compute_flux_weights(mask_dilated, mask, alpha=1.0)
            flux_weight = self.compute_flux_weights(np.ones_like(mask), mask, alpha=1.0)
            # close_context_weight = self.compute_close_context_weights(out_label, alpha=4.0)
            flux_weight *= pre_weight
            flux_weight = torch.from_numpy(flux_weight)
            flux_weight = flux_weight.unsqueeze(0)
            # boundary_weight = self.compute_flux_weights(mask|boundary_mask, boundary_mask, alpha=0.25)
            # boundary_weight = torch.from_numpy(boundary_weight)
            # boundary_weight = boundary_weight.unsqueeze(0)

        out_input = torch.from_numpy(out_input)
        out_input = out_input.unsqueeze(0)

        if out_skeleton is not None:
            out_skeleton_blurred = torch.from_numpy(out_skeleton_blurred.astype(np.int64))
            out_skeleton_blurred = out_skeleton_blurred

            # boundary_mask = torch.from_numpy(boundary_mask.astype(np.float32))
            # boundary_mask = boundary_mask.unsqueeze(0)

        if out_flux is not None:
            out_flux = torch.from_numpy(out_flux).float()
        if out_label is not None:
            out_label = torch.from_numpy(out_label.astype(np.float32))
            out_label = out_label.unsqueeze(0)

        if self.mode == 'train':
            # Rebalancing weights for skeleton and non skel pixels
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            # TODO may Remove this later. Added for uniform weighting outside context
            skeleton_weight = rebalance_skeleton_weight(skeleton_mask=mask, seg_mask=torch.ones_like(mask), alpha=1.0)
            skeleton_weight *= torch.from_numpy(pre_weight)
            #mask = morphology.binary_dilation(mask, structure=np.ones((1, 3, 3)), iterations=2)
            # skeleton_weight = torch.from_numpy(mask).unsqueeze(0).float()
            return pos, out_input, out_label, out_flux, out_skeleton_blurred, skeleton_weight, flux_weight

        else:
            return pos, out_input

    def get_pos_dataset(self, index):
        return np.argmax(index < self.sample_num_c) - 1  # which dataset

    def get_pos_seed(self, seed, offset=None):
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