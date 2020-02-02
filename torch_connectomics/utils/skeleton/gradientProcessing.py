import os, sys, time
import numpy as np
import scipy, skimage
import h5py
from tqdm import tqdm
from scipy import ndimage

import edt
import multiprocessing as mp
import networkx as nx

# add ibexHelper path
# https://github.com/donglaiw/ibexHelper
# sys.path.append('/n/home11/averma/repositories/ibexHelper')
# from ibexHelper.skel import CreateSkeleton,ReadSkeletons
# from ibexHelper.graph import GetNodeList, GetEdgeList
# from ibexHelper.skel2graph import GetGraphFromSkeleton

def divergence_3d(field):
    dz = np.gradient(field[0], axis=0)
    dy = np.gradient(field[1], axis=1)
    dx = np.gradient(field[2], axis=2)
    return dz + dy + dx

def normalize_vectors(field):
    norm = np.sqrt(np.sum(field ** 2, axis=0))
    mask = (norm > 1e-4)
    mask = np.stack([mask, mask, mask])
    field[mask] = (field / norm)[mask]

def normalize_scalar_field(field):
    max_v = field.max()
    min_v = field.min()
    field = (field - min_v) / (max_v - min_v)
    return field

def remove_ids(ar, ids_to_rem, max_id):
    ids = np.arange(max_id + 1, dtype=np.uint16)
    ids[ids_to_rem] = 0
    return ids[ar]

def compute_skeleton_from_gradient(gradient, params):
    threshold = params['adaptive_threshold']
    filter_sz = params['filter_size']
    absolute_div_th = params['absolute_threshold']
    min_skel_th = params['min_skel_threshold']
    block_size = params['block_size']
    shift = [3, 3, 3]

    skel_prob = divergence_3d(gradient)
    skel_binary = np.zeros_like(skel_prob, dtype=np.bool)

    pos = []
    for i in range(3):
        loc = np.arange(0, skel_prob.shape[i], step=(block_size[i] // shift[i]), dtype=np.uint16)
        loc[-1] = skel_prob.shape[i] - block_size[i]
        pos.append(loc)

    for z_idx, z_loc in enumerate(pos[0]):
        z_slice = slice(z_loc, z_loc + block_size[0], None)
        for y_idx, y_loc in enumerate(pos[1]):
            y_slice = slice(y_loc, y_loc + block_size[1], None)
            for x_idx, x_loc in enumerate(pos[2]):
                bounds = (z_slice, y_slice,
                          slice(x_loc, x_loc + block_size[2], None))

                vol = skel_prob[bounds]
                n_vol = normalize_scalar_field(vol)
                skel_prob_roi = skel_prob[bounds]
                skel_binary[bounds] |= ((skel_prob_roi > absolute_div_th) & (n_vol > float(threshold) / 100.0))

    skel = skel_binary
    if filter_sz > 0:
        skel = scipy.ndimage.morphology.binary_dilation(skel, structure=np.ones((1, filter_sz, filter_sz)),
                                                        iterations=1)
    label_cc, num_cc = skimage.measure.label(skel, return_num=True)

    if num_cc > np.iinfo(np.int32).max:
        print('cannot convert to uint32')
    else:
        label_cc = label_cc.astype(np.uint32)

    # Remove small skeletons
    ids, counts = np.unique(label_cc, return_counts=True)
    m = ((ids > 0) & (counts >= min_skel_th))
    skel_large = remove_ids(label_cc, ids[~m], np.max(ids))

    return skel_large


