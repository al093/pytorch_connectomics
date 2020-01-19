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

class SkeletonGrowingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image, skeleton, flux, growing_data=None,
                 sample_input_size=(8, 64, 64),
                 augmentor=None,
                 mode='train'):
        if mode == 'test':
            raise Exception('Not Implemented')

        self.mode = mode
        self.image = image[0]
        self.skeleton = skeleton[0]
        self.growing_data = growing_data[0]
        self.flux = flux[0]
        self.augmentor = augmentor  # data augmentation

        # samples, channels, depths, rows, cols
        self.image_size = np.array(self.image.shape)
        self.sample_input_size = np.array(sample_input_size)  # model input size
        self.half_input_sz = (sample_input_size//2)
        self.num_seeds = len(self.growing_data)

    def __len__(self):  # number of seed points
        return self.num_seeds

    def __getitem__(self, index):
        vol_size = self.sample_input_size

        if self.mode == 'train':
            #return the growing data and image, flux, skeleton
            path = self.growing_data[index]['path'][1:-1]
            start_pos = self.growing_data[index]['path'][0]
            stop_pos = self.growing_data[index]['path'][-1]
            start_sid = self.growing_data[index]['sids'][0]
            stop_sid = self.growing_data[index]['sids'][1]

            return self.image, self.flux, self.skeleton, path, start_pos, stop_pos, start_sid, stop_sid

        # Test Mode Specific Operations:
        elif self.mode == 'test':
            raise Exception('Not Implemented')

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