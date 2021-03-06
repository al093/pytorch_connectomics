from __future__ import print_function, division
import numpy as np
import random
import torch

####################################################################
## Process image stacks.
####################################################################

def count_volume(data_sz, vol_sz, stride):
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)

def crop_volume(data, sz, st=(0, 0, 0)):  # C*D*W*H, C=1
    return data[st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]

def crop_volume_mul(data, sz, st=(0, 0, 0)):  # C*D*W*H, for multi-channel input
    return data[:, st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]

def check_cropable(data, sz, st):
    # check if any postion is negative
    if np.any(st<0):
        return False

    # check if the crop exceeds image bounds
    if (st[0] + sz[0] <= data.shape[-3]) and (st[1] + sz[1] <= data.shape[-2]) and (st[2] + sz[2] <= data.shape[-1]):
        return True
    else:
        return False

####################################################################
## Rebalancing.
####################################################################

def rebalance_binary_class(label, mask=None, base_w=1.0):
    """Binary-class rebalancing."""
    weight_factor = label.float().sum() / torch.prod(torch.tensor(label.size()).float())
    if mask is not None:
        weight_factor = (mask*label).float().sum() / mask.sum()
    weight_factor = torch.clamp(weight_factor, min=1e-2)
    alpha = 1.0
    weight = alpha * label*(1-weight_factor)/weight_factor + (1-label)
    if mask is not None:
        weight = weight * mask
    return weight_factor, weight

def rebalance_skeleton_weight(skeleton_mask, seg_mask, alpha=1.0):
    num_skel_pixels = skeleton_mask.sum().float()
    num_seg_pixels = seg_mask.sum().float()
    weight_factor = alpha * num_seg_pixels / (num_skel_pixels + 1e-10)
    weight_factor = torch.clamp(weight_factor, max=1e3)
    weight = seg_mask.clone().float()
    weight[skeleton_mask == 1.0] = weight_factor
    return weight

####################################################################
## Affinitize.
####################################################################

def check_volume(data):
    """Ensure that data is a numpy 3D array."""
    assert isinstance(data, np.ndarray)

    if data.ndim == 2:
        data = data[np.newaxis,...]
    elif data.ndim == 3:
        pass
    elif data.ndim == 4:
        assert data.shape[0]==1
        data = np.reshape(data, data.shape[-3:])
    else:
        raise RuntimeError('data must be a numpy 3D array')

    assert data.ndim==3
    return data

# def affinitize(img, dst=(1,1,1), dtype=np.float32):
#     """
#     Transform segmentation to an affinity map.

#     Args:
#         img: 3D indexed image, with each index corresponding to each segment.

#     Returns:
#         ret: an affinity map (4D tensor).
#     """
#     img = check_volume(img)
#     if ret is None:
#         ret = np.zeros(img.shape, dtype=dtype)

#     # Sanity check.
#     (dz,dy,dx) = dst
#     assert abs(dx) < img.shape[-1]
#     assert abs(dy) < img.shape[-2]
#     assert abs(dz) < img.shape[-3]