import os, sys, scipy, skimage, h5py, traceback
import numpy as np
import multiprocessing as mp
import networkx as nx
from tqdm import tqdm
from scipy import ndimage

from .computeParallel import *

def split_at_junctions(input_skel, nodes):
    # Compute cutting plane for all junction points
    bb_rad = (6, 9, 9)
    idx_spacing_h = np.linspace(-bb_rad[1], bb_rad[1], 4 * bb_rad[1] + 1, endpoint=True, dtype=np.float32)
    idx_spacing_v = np.linspace(-bb_rad[0], bb_rad[0], 4 * bb_rad[0] + 1, endpoint=True, dtype=np.float32)
    k, j, i = np.meshgrid(idx_spacing_v, idx_spacing_h, idx_spacing_h, sparse=False, indexing='ij')
    output_skel = input_skel.copy()

    ar_sz = input_skel.shape
    for key in nodes.keys():
        if key[0] == 'j':
            skel_id = int(key[1:])
            junctions = nodes[key]
            for idx in range(junctions.shape[0]):
                jn = junctions[idx]
                bounds = (slice(jn[0] - bb_rad[0], jn[0] + bb_rad[0] + 1, None),
                          slice(jn[1] - bb_rad[1], jn[1] + bb_rad[1] + 1, None),
                          slice(jn[2] - bb_rad[2], jn[2] + bb_rad[2] + 1, None))
                X = (np.stack(np.nonzero(input_skel[bounds] == skel_id))).astype(np.float32)
                X -= X.sum(axis=1, keepdims=True)
                u, s, _ = np.linalg.svd(X)
                plane_v1, plane_v2, normal = u[:, 2], u[:, 1], u[:, 0]
                v1 = plane_v1[:, np.newaxis, np.newaxis, np.newaxis] * i
                v2 = plane_v2[:, np.newaxis, np.newaxis, np.newaxis] * j
                n = normal[:, np.newaxis, np.newaxis, np.newaxis] * k
                remove_pos = (v1 + v2 + n).astype(np.int32)
                remove_pos = remove_pos.reshape(3, -1).T + jn
                remove_pos = remove_pos[
                             (remove_pos[:, 0] >= 0.0) & (remove_pos[:, 1] >= 0.0) & (remove_pos[:, 2] >= 0.0), :]
                remove_pos = remove_pos[(remove_pos[:, 0] < ar_sz[0]) & (remove_pos[:, 1] < ar_sz[1]) & (
                            remove_pos[:, 2] < ar_sz[2]), :]
                remove_pos = remove_pos[output_skel[(remove_pos[:, 0], remove_pos[:, 1], remove_pos[:, 2])] == skel_id,
                             :]
                output_skel[(remove_pos[:, 0], remove_pos[:, 1], remove_pos[:, 2])] = 0
    return output_skel

def remove_small_skel_and_relabel(split_skel_large, threshold):
    split_skel_large = skimage.measure.label(split_skel_large)

    counts = np.bincount(np.ravel(split_skel_large))
    remove_list = np.nonzero(counts < threshold)[0]

    ids = np.unique(split_skel_large)
    ids = ids[ids > 0]
    relabel_ar = np.arange(ids.max() + 1).astype(split_skel_large.dtype)
    relabel_ar[remove_list] = 0
    return relabel_ar[split_skel_large].astype(np.uint16)

def load_dict(h5_path):
    f_dict = {}
    with h5py.File(h5_path, 'r') as h_file:
        for key in list(h_file):
            f_dict[key] = np.asarray(h_file[key])
    return f_dict

def get_skeleton_nodes(skeleton, input_resolution, downsample_factor, temp_folder, num_cpu=1, save_graph=False):
    sids = np.unique(skeleton)
    sids = sids[sids > 0]

    status = compute(fn=compute_skel_graph, num_proc=num_cpu, sids=sids, skel_vol_full=skeleton,
                              temp_folder=temp_folder, input_resolution=input_resolution,
                              downsample_fac=downsample_factor, output_file_name='nodes.h5', save_graph=save_graph)
    if status != True:
        raise Exception('Error while creating skeleton graph.')

    nodes = {}
    if num_cpu>0:
        for proc_id in range(0, num_cpu):
            with h5py.File(temp_folder + '/(' + str(proc_id) + ')nodes.h5', 'r') as hfile:
                for key in hfile.keys(): nodes[key] = np.asarray(hfile[key])
    else:
        with h5py.File(temp_folder + '/(1221)nodes.h5', 'r') as hfile:
            for key in hfile.keys(): nodes[key] = np.asarray(hfile[key])
    return nodes

def split(skeleton, min_skel_size, input_resolution, downsample_factor, temp_folder, num_cpu=1):
    nodes = get_skeleton_nodes(skeleton, input_resolution, downsample_factor, temp_folder, num_cpu)
    split_skeleton = split_at_junctions(skeleton, nodes)
    split_skeleton = remove_small_skel_and_relabel(split_skeleton, min_skel_size)
    return split_skeleton

def generate_skeleton_growing_data(skeleton, output_file, input_resolution, downsample_factor, temp_folder, num_cpu=1):
    nodes = get_skeleton_nodes(skeleton, input_resolution, downsample_factor, temp_folder, num_cpu)
    count = 0
    with h5py.File(output_file, 'w') as hfile:
        for key, val in nodes.items():
            if key[0] == 'e':
                for end in val:
                    count += 1
                    hg = hfile.create_group(str(count))
                    hg.create_dataset('vertices', data=np.array([end]).astype(np.uint16))
                    hg.create_dataset('sids', data=np.array([int(key[1:])]).astype(np.uint16))
    print('Total ends: ', count)


def interpolate_using_spline(nodes, limits, order=3, smoothing=10000):
    tck, u = interpolate.splprep([nodes[:, 0], nodes[:, 1], nodes[:, 2]], k=order, s=smoothing)
    u_fine = np.linspace(0, 1, 100)
    z_fine, y_fine, x_fine = interpolate.splev(u_fine, tck)
    new_nodes = np.vstack((z_fine, y_fine, x_fine)).T
    if new_nodes.shape[0] > 0:
        new_nodes = new_nodes[(new_nodes[:, 0])>=0 & (new_nodes[:,1]>=0) & (new_nodes[:,2]>=0), :]
        new_nodes = new_nodes[(new_nodes[:, 0]<limits[0]-1.0) & (new_nodes[:, 1]<limits[1]-1.0) & (new_nodes[:, 2]<limits[2]-1.0), :]
    return np.concatenate([nodes[0:1,:], new_nodes, nodes[-1:-2:-1]], axis=0)

def merge(split_skeleton, merge_data):
    split_skeleton = split_skeleton.copy()
    merge_ids = []
    graph = nx.Graph()

    for g in merge_data.keys():
        match_ids = merge_data[g]['sids']
        # average_path_score = merge_data[g]['states'][:-1].sum()/(merge_data[g]['states'].shape[0] - 1)
        # print('Skel_id: {}, path_score: {}'.format(g, average_path_score))
        if np.all(match_ids > 0):
            graph.add_edge(match_ids[0], match_ids[1])

    for c in nx.connected_components(graph):
        merge_ids.append(list(graph.subgraph(c).nodes(data=False)))

    # merge graphs
    merged = []
    gg = np.mgrid[-1:2, -2:3, -2:3]
    grid = np.transpose((gg[0].flatten(), gg[1].flatten(), gg[2].flatten())).astype(np.int32)
    for g in merge_data.keys():
        match_ids = merge_data[g]['sids']
        if np.all(match_ids > 0) and tuple(match_ids) not in merged:
            merged.append(tuple(match_ids))
            i_vertices = interpolate_using_spline(merge_data[g]['vertices'], np.array(split_skeleton.shape), order=2, smoothing=100)
            i_vertices = np.unique(i_vertices.astype(np.int32), axis=0)
            i_vertices = (i_vertices[:, np.newaxis, :] + grid).reshape(-1, 3)
            i_vertices = i_vertices[np.all(i_vertices >= 0, axis=1) & np.all(i_vertices < np.array(split_skeleton.shape), axis=1), :]
            split_skeleton[i_vertices[:,0], i_vertices[:,1], i_vertices[:,2]] = match_ids[0]

    # relable merged skeletons
    labels = np.arange(split_skeleton.max() + 1, dtype=split_skeleton.dtype)
    for ids in merge_ids:
        labels[ids] = ids[0]
    merged = labels[split_skeleton]
    return merged

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout