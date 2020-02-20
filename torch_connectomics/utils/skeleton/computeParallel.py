import os, sys, scipy, skimage, h5py, traceback, pickle
import numpy as np
import multiprocessing as mp
import networkx as nx
from scipy import ndimage, interpolate
from skimage.morphology import skeletonize_3d

# add ibexHelper path
# https://github.com/donglaiw/ibexHelper
sys.path.append('/n/home11/averma/repositories/ibexHelper')
from ibexHelper.skel import CreateSkeleton, ReadSkeletons
from ibexHelper.graph import GetEdgeList
from ibexHelper.skel2graph import GetGraphFromSkeleton

def interpolate_using_spline(nodes, limits, order=3, smoothing=10000):
    tck, u = interpolate.splprep([nodes[:, 0], nodes[:, 1], nodes[:, 2]], k=order, s=smoothing)
    u_fine = np.linspace(0, 1, 500)
    z_fine, y_fine, x_fine = interpolate.splev(u_fine, tck)
    new_nodes = np.vstack((z_fine, y_fine, x_fine)).T
    if new_nodes.shape[0] > 0:
        new_nodes = new_nodes[(new_nodes[:, 0])>=0 & (new_nodes[:,1]>=0) & (new_nodes[:,2]>=0), :]
        new_nodes = new_nodes[(new_nodes[:, 0]<limits[0]-1.0) & (new_nodes[:, 1]<limits[1]-1.0) & (new_nodes[:, 2]<limits[2]-1.0), :]
    return np.concatenate([nodes[0:1,:], new_nodes, nodes[-1:-2:-1]], axis=0)

def upsample_skeleton_using_splines(edge_list, nodes, limits, return_graph, start_pos):
    nodes = nodes.astype(np.float32)
    upsampled_nodes = []
    g = nx.Graph()

    sampling = np.linspace(0.0, 1.0, num=20, endpoint=True).astype(np.float32)

    for long_sec in edge_list:
        path = long_sec[2]['path']
        if len(path) > 3:
            new_nodes = interpolate_using_spline(nodes[path, :], limits, order=3, smoothing=400)

        else:
            new_nodes = []
            for i in range(len(path) - 1):
                start = nodes[path[i]]
                end = nodes[path[i + 1]]
                direction = end - start
                new_nodes_ar = start + sampling[:, np.newaxis] * direction
                new_nodes.append(np.unique(new_nodes_ar, axis=0))
            new_nodes = np.vstack(new_nodes)

        if new_nodes.shape[0] > 1:
            new_nodes_adj = new_nodes + start_pos
            upsampled_nodes.append(new_nodes)
            if return_graph == True:
                g.add_edges_from([ (tuple(new_nodes_adj[i]), tuple(new_nodes_adj[i+1])) for i in range(new_nodes_adj.shape[0]-1) ])

    upsampled_nodes = np.vstack(upsampled_nodes)

    if return_graph is True:
        return upsampled_nodes, g
    else:
        return upsampled_nodes

def compute_skel_graph(process_id, seg_ids, skel_vol_full, temp_folder, input_resolution, downsample_fac, output_file_name, save_graph):
    process_id = str(process_id)

    out_folder = temp_folder + '/' + process_id
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    input_resolution = np.array(input_resolution).astype(np.uint8)
    downsample_fac = np.array(downsample_fac).astype(np.uint8)
    graphs = {}

    with h5py.File(temp_folder + '/(' + process_id + ')' + output_file_name , 'w') as hf_nodes:
        locations = scipy.ndimage.find_objects(skel_vol_full)
        for idx, seg_id in enumerate(seg_ids):
            loc = locations[int(seg_id) - 1]
            start_pos = np.array([loc[0].start, loc[1].start, loc[2].start], dtype=np.uint16)
            skel_mask = (skel_vol_full[loc] == int(seg_id))
            try:
                CreateSkeleton(skel_mask, out_folder, input_resolution, input_resolution*downsample_fac)
                skel_obj = ReadSkeletons(out_folder, skeleton_algorithm='thinning', downsample_resolution=input_resolution*downsample_fac, read_edges=True)[1]
                nodes = np.stack(skel_obj.get_nodes()).astype(np.uint16)

                if nodes.shape[0] < 10:
                    # print('skipped skel: {} (too small!)'.format(seg_id))
                    continue

                graph, wt_dict, th_dict, ph_dict = GetGraphFromSkeleton(skel_obj, modified_bfs=False)
                edge_list = GetEdgeList(graph, wt_dict, th_dict, ph_dict)
            except:
                # print('Catched exp in skel: ', seg_id)
                #traceback.print_exc(file=sys.stdout)
                continue

            if save_graph is True:
                _, graph = upsample_skeleton_using_splines(edge_list, nodes, skel_mask.shape, return_graph=True, start_pos=start_pos)
                graphs[seg_id] = graph

            g = nx.Graph()
            g.add_edges_from(edge_list)
            j_ids = [x for x in g.nodes() if g.degree(x) > 2]
            e_ids = [x for x in g.nodes() if g.degree(x) == 1]
            nodes = nodes + start_pos
            if len(j_ids) > 0:
                junctions = nodes[j_ids]
                hf_nodes.create_dataset('j' + str(seg_id), data=junctions, compression='gzip')

            if len(e_ids) > 0:
                end_points = nodes[e_ids]
                hf_nodes.create_dataset('e' + str(seg_id), data=end_points, compression='gzip')
            hf_nodes.create_dataset('allNodes' + str(seg_id), data=nodes, compression='gzip')

        if save_graph is True:
            with open(temp_folder + '/(' + process_id + ')graph.h5', 'wb') as pfile:
                pickle.dump(graphs, pfile, protocol=pickle.HIGHEST_PROTOCOL)


def compute_thinned_nodes(process_id, seg_ids, skel_vol_full, temp_folder, input_resolution, downsample_fac, output_file_name):
    process_id = str(process_id)
    out_folder = temp_folder + '/' + process_id
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    input_resolution = np.array(input_resolution).astype(np.uint8)
    downsample_fac = np.array(downsample_fac).astype(np.uint8)
    graphs = {}

    with h5py.File(temp_folder + '/(' + process_id + ')' + output_file_name , 'w') as hf_nodes:
        locations = scipy.ndimage.find_objects(skel_vol_full)
        for idx, seg_id in enumerate(seg_ids):
            loc = locations[int(seg_id) - 1]
            start_pos = np.array([loc[0].start, loc[1].start, loc[2].start], dtype=np.uint16)
            skel_mask = (skel_vol_full[loc] == int(seg_id))
            try:
                CreateSkeleton(skel_mask, out_folder, input_resolution, input_resolution*downsample_fac)
                skel_obj = ReadSkeletons(out_folder, skeleton_algorithm='thinning', downsample_resolution=input_resolution*downsample_fac, read_edges=True)[1]
                nodes = start_pos + np.stack(skel_obj.get_nodes()).astype(np.uint16)
            except:
                continue
            hf_nodes.create_dataset('allNodes' + str(seg_id), data=nodes, compression='gzip')

def compute_thinned_nodes_skimage_skeletonize(process_id, seg_ids, skel_vol_full, temp_folder, output_file_name):
    process_id = str(process_id)
    out_folder = temp_folder + '/' + process_id
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    with h5py.File(temp_folder + '/(' + process_id + ')' + output_file_name , 'w') as hf_nodes:
        locations = scipy.ndimage.find_objects(skel_vol_full)
        for seg_id in seg_ids:
            loc = locations[int(seg_id) - 1]
            start_pos = np.array([loc[0].start, loc[1].start, loc[2].start], dtype=np.uint16)
            skel_mask = (skel_vol_full[loc] == int(seg_id))
            thinned_skel = skeletonize_3d(skel_mask)
            nodes = start_pos + np.transpose(np.nonzero(thinned_skel)).astype(np.uint16)
            hf_nodes.create_dataset('allNodes' + str(seg_id), data=nodes, compression='gzip')

def compute(fn, num_proc, sids, **kwargs):
    if num_proc == 0:
        fn('1221', sids, **kwargs)
        return True
    else:
        seg_per_proc = len(sids) // num_proc
        seg_ids_proc = {}
        extras = len(sids) - seg_per_proc*num_proc
        top = 0
        for i in range(num_proc):
            if i < extras:
                seg_ids_proc[i] = sids[top:top+seg_per_proc+1]
                top += seg_per_proc+1
            else:
                seg_ids_proc[i] = sids[top:top+seg_per_proc]
                top += seg_per_proc

        processes = [mp.Process(target=fn, args=(x, seg_ids_proc[x]), kwargs=kwargs) for x in range(num_proc)]

        for i, p in enumerate(processes): p.start()
        for p in processes: p.join()

        exit_code = np.array([p.exitcode for p in processes])
        if np.all(exit_code == 0):
            return True
        else:
            print('\x1b[31mProcesses: {} exited with non-zero code\x1b[0m'.format(np.arange(num_proc)[exit_code > 0]))
            return False