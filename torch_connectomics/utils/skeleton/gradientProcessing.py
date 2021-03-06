import numpy as np
import scipy, skimage
from scipy import ndimage
from skimage.morphology import skeletonize_3d
import networkx as nx

from .computeParallel import compute_ibex_skeleton_graphs


def divergence_3d(field):
    dz = np.gradient(field[0], axis=0)
    dy = np.gradient(field[1], axis=1)
    dx = np.gradient(field[2], axis=2)
    return dz + dy + dx

def normalize_vectors(field):
    norm = np.sqrt(np.sum(field ** 2, axis=0)).astype(np.float32)
    mask = (norm > 1e-4)
    norm[~mask] = 1e-6 # so that numpy does not complain for zero division
    field_n = field.copy()
    field_n[:, mask] = (field_n / norm)[:, mask]
    return field_n, norm

def normalize_scalar_field(field):
    max_v = field.max()
    min_v = field.min()
    field = (field - min_v) / (max_v - min_v)
    return field

def remove_ids(ar, ids_to_rem, max_id):
    ids = np.arange(max_id + 1, dtype=np.uint16)
    ids[ids_to_rem] = 0
    return ids[ar]

def binarize_skeleton(skeleton_probability, params, remove_borders):
    absolute_div_th = params['absolute_threshold']
    block_size = params['block_size']
    threshold = params['adaptive_threshold']
    shift = [3, 3, 3]
    skel_binary = np.zeros_like(skeleton_probability, dtype=np.bool)
    pos = []
    for i in range(3):
        loc = np.arange(0, skeleton_probability.shape[i], step=(block_size[i] // shift[i]), dtype=np.uint16)
        loc[-1] = skeleton_probability.shape[i] - block_size[i]
        pos.append(loc)

    for z_idx, z_loc in enumerate(pos[0]):
        z_slice = slice(z_loc, z_loc + block_size[0], None)
        for y_idx, y_loc in enumerate(pos[1]):
            y_slice = slice(y_loc, y_loc + block_size[1], None)
            for x_idx, x_loc in enumerate(pos[2]):
                bounds = (z_slice, y_slice,
                          slice(x_loc, x_loc + block_size[2], None))

                vol = skeleton_probability[bounds]
                n_vol = normalize_scalar_field(vol)
                skel_binary[bounds] |= ((vol > absolute_div_th) & (n_vol > float(threshold) / 100.0))

    if remove_borders:
        if remove_borders[0][0]: skel_binary[0:remove_borders[0][0], :, :] = 0
        if remove_borders[0][1]: skel_binary[-remove_borders[0][1]:, :, :] = 0
        if remove_borders[1][0]: skel_binary[:, 0:remove_borders[1][0], :] = 0
        if remove_borders[1][1]: skel_binary[:, -remove_borders[1][0]:, :] = 0
        if remove_borders[2][0]: skel_binary[:, :, 0:remove_borders[2][0]] = 0
        if remove_borders[2][1]: skel_binary[:, :, -remove_borders[2][1]:] = 0

    return skel_binary


def compute_skeleton_from_probability(skeleton_probability, params, remove_borders=False, thin=True, debug=False):
    skel = binarize_skeleton(skeleton_probability, params, remove_borders)

    if debug:
        debug_data = {}
        debug_data['binary_raw_pred'] = skel.copy()

    filter_sz = params['filter_size']
    if filter_sz > 0:
        skel = scipy.ndimage.morphology.binary_dilation(skel, structure=np.ones((1, filter_sz, filter_sz)), iterations=1)

    if debug: debug_data['binary_dilated_pred'] = skel.copy()

    if thin: skel = thin_skeletons(skel)

    label_cc, num_cc = skimage.measure.label(skel, return_num=True)

    if num_cc > np.iinfo(np.uint32).max:
        raise Exception('Cannot convert volume to uint32, number of segments exceed range of uint32')
    else:
        label_cc = label_cc.astype(np.uint32)

    if debug:
        return label_cc, debug_data
    else:
        return label_cc


def compute_skeleton_like_deepflux(direction, lmd, k1, k2, binned_directions=None):
    '''
    :param direction:
    :param k1: dilation filter size
    :param k2: erosion filter size
    :return: instance skeletons seperated using CC
    '''
    binary_skel = np.full(direction.shape[1:], False, dtype=bool)

    if binned_directions is None:
        all_directions = []
        count = 0
        for pz in [-1, 0, 1]:
            for py in [-1, 0, 1]:
                for px in [-1, 0, 1]:
                    if px == 0 and py == 0 and pz == 0:
                        continue
                    d = np.array([pz, py, px], dtype=np.int16)
                    d = d.astype(np.float32) / np.sqrt((d ** 2).sum())
                    all_directions.append(d)
                    count += 1
        #bin directions
        all_directions = np.array(all_directions).astype(np.float32)[:, :, np.newaxis, np.newaxis, np.newaxis]
        binned_directions = np.squeeze(np.argmax((all_directions * direction).sum(axis=1), axis=0))

    direction, norm = normalize_vectors(direction)
    mask = norm > lmd
    mask_inv = ~mask

    count = 0
    for pz in [-1, 0, 1]:
        for py in [-1, 0, 1]:
            for px in [-1, 0, 1]:
                if px == 0 and py == 0 and pz == 0:
                    continue
                mask_val_neighbor = ndimage.affine_transform(mask_inv, matrix=np.diag([1, 1, 1]), offset=(pz, py, px), order=0)
                direction_mask = (binned_directions == count)
                binary_skel |= (mask_val_neighbor * mask * direction_mask)
                count += 1

    if binary_skel.sum() == 0:
            return None, binned_directions

    dilation_kernel = np.ones([k1, k1, k1], dtype=np.bool)
    binary_skel = ndimage.morphology.binary_dilation(binary_skel, structure=dilation_kernel)

    erosion_kernel = np.ones([k2, k2, k2], dtype=np.bool)
    binary_skel = ndimage.morphology.binary_erosion(binary_skel, structure=erosion_kernel)

    label_cc, num_cc = skimage.measure.label(binary_skel, return_num=True)
    if num_cc > np.iinfo(np.uint32).max:
        raise Exception('Cannot convert volume to uint32, number of segments exceed range of uint32 ')
    else:
        label_cc = label_cc.astype(np.uint32)

    return label_cc, binned_directions

def remove_small_skeletons(label_cc, min_skel_th):
    if label_cc is None:
        return None
    ids, counts = np.unique(label_cc, return_counts=True)
    m = ((ids > 0) & (counts >= min_skel_th))
    skel_large = remove_ids(label_cc, ids[~m], np.max(ids))
    return skel_large

def thin_skeletons(skel):
    return skeletonize_3d(skel > 0)

def compute_skeleton_from_scalar_field(skel_probability, method, threshold, k1, k2):
    binary_skel = skel_probability > threshold

    if binary_skel.sum() == 0:
        return None

    dilation_kernel = np.ones([k1, k1, k1], dtype=np.bool)
    binary_skel = ndimage.morphology.binary_dilation(binary_skel, structure=dilation_kernel)

    if k2 > 0:
        erosion_kernel = np.ones([k2, k2, k2], dtype=np.bool)
        binary_skel = ndimage.morphology.binary_erosion(binary_skel, structure=erosion_kernel)

    label_cc, num_cc = skimage.measure.label(binary_skel, return_num=True)
    if num_cc > np.iinfo(np.uint32).max:
        raise Exception('Cannot convert volume to uint32, number of segments exceed range of uint32 ')
    else:
        label_cc = label_cc.astype(np.uint32)

    return label_cc


def split_skeletons_using_graph(skeleton: np.ndarray, temp_folder, resolution):
    split_skeleton = np.zeros_like(skeleton)
    ibex_skeletons = compute_ibex_skeleton_graphs(skeleton, temp_folder, input_resolution=resolution, downsample_fac=(1, 1, 1))
    skeleton_id = 1
    for ibex_skeleton in ibex_skeletons:
        if len(ibex_skeleton.edges) == 0:
            continue
        if len(ibex_skeleton.endpoints) == 2:
            nodes = np.stack(ibex_skeleton.get_nodes())
            split_skeleton[nodes[:, 0], nodes[:, 1], nodes[:, 2]] = skeleton_id
            skeleton_id += 1
            continue

        nodes = np.stack(ibex_skeleton.get_nodes())
        junctions_idx = ibex_skeleton.get_junctions()
        junctions = [tuple(nodes[j, :]) for j in junctions_idx]
        edges = ibex_skeleton.get_edges()
        g_temp = nx.Graph()
        g_temp.add_edges_from([(tuple(edge[0]), tuple(edge[1]))
                               for edge in edges
                               if (tuple(edge[0]) not in junctions) and (tuple(edge[1]) not in junctions)])

        for sub_graph_nodes in nx.connected_components(g_temp):
            split_nodes = np.array(list(sub_graph_nodes))
            split_skeleton[split_nodes[:, 0], split_nodes[:, 1], split_nodes[:, 2]] = skeleton_id
            skeleton_id += 1

    return split_skeleton