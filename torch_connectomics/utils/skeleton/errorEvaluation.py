import numpy as np
import h5py
import edt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from .computeParallel import *

def write_hf(data, name):
    with h5py.File(name, 'w') as hf:
        hf.create_dataset('main', data=data, compression='gzip')
def read_hf(name):
    with h5py.File(name, 'r') as hf:
        data = np.asarray(hf['main'])
    return data

def calculate_error_metric(pred_skel, gt_skel, gt_context, gt_skel_ids, anisotropy, gt_save_data=False):
    bloat_radius = np.float32(np.max(resolution) * 2.0)
    anisotropy_factor = np.float32(anisotropy[0])/np.float32(anisotropy[1])
    pred_skel_ids = np.unique(pred_skel)
    pred_skel_ids = pred_skel_ids[pred_skel_ids > 0]

    # find GT skeleton id for each predicted Skeleton id
    max_p_id = np.max(pred_skel_ids)
    matches = np.arange(max_p_id + 1, dtype=np.uint16)

    matches_gt2p = {}
    for i in gt_skel_ids: matches_gt2p[i] = []
    matches_gt2p[0] = []

    for p_id in pred_skel_ids:
        mask = (pred_skel == p_id)
        counts = np.bincount(gt_context[mask])

        if np.all(counts < (mask.sum() * 0.05)):
            matches[p_id] = 0
        else:
            g_id = np.argmax(counts)
            matches[p_id] = g_id
            matches_gt2p[g_id].append(p_id)

    relabelled_p_skel = matches[pred_skel]

    # find the distance transform of the predicted skeleton and inflate (contex) them.
    # use this inflated prediction to find all the FN points
    skel_dtx = edt.edt(relabelled_p_skel == 0, anisotropy=anisotropy,
                       black_border=False, order='C',
                       parallel=0)

    skel_context = skel_dtx < bloat_radius
    skel_points = np.nonzero(relabelled_p_skel)
    skel_points = np.transpose((anisotropy_factor * skel_points[0], skel_points[1], skel_points[2]))

    context_points = np.nonzero(skel_context)
    context_points = np.transpose((anisotropy_factor * context_points[0], context_points[1], context_points[2]))

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean', n_jobs=-1).fit(skel_points)
    _, indices = nbrs.kneighbors(context_points)

    context_points[:, 0] = context_points[:, 0] / anisotropy_factor
    context_points = context_points.astype(np.uint16)

    skel_points[:, 0] = skel_points[:, 0] / anisotropy_factor
    skel_points = skel_points.astype(np.uint16)

    relabelled_skel_context = np.zeros_like(pred_skel)
    relabelled_skel_context[(context_points[:, 0], context_points[:, 1], context_points[:, 2])] \
        = relabelled_p_skel[skel_points[indices[:, 0], 0], skel_points[indices[:, 0], 1], skel_points[indices[:, 0], 2]]

    nzero_mask_pred_skel = (relabelled_p_skel > 0)
    nzero_mask_gt_skel = (gt_skel > 0)
    tp = float(sum((relabelled_p_skel == gt_context)[nzero_mask_pred_skel]))
    fp = float(sum((relabelled_p_skel != gt_context)[nzero_mask_pred_skel]))
    fn = float(sum((gt_skel != relabelled_skel_context)[nzero_mask_gt_skel]))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)

    connectivity = {}
    mean_connectivity = 0.0
    count = 0
    for g_id, p_ids in matches_gt2p.items():
        if len(p_ids) > 0:
            count += 1
            connectivity[g_id] = 1.0 / len(p_ids)
            mean_connectivity += connectivity[g_id]
        else:
            connectivity[g_id] = 0
    mean_connectivity /= count

    print('Precision:    {:10.4f}'.format(precision))
    print('Recall:       {:10.4f}'.format(recall))
    print('F Score:      {:10.4f}'.format(f_score))
    print('Connectivity: {:10.4f}'.format(mean_connectivity))

    return precision, recall, f_score, mean_connectivity

def calculate_error_metric_2(pred_skel, gt_skel_graphs, gt_skel_ids, gt_context, resolution, temp_folder, num_cpu, debug=False):
    matching_radius = np.float32(np.max(resolution) * 2.0)
    pred_skel_ids = np.unique(pred_skel)
    pred_skel_ids = pred_skel_ids[pred_skel_ids > 0]

    # find GT skeleton id for each predicted Skeleton id
    max_p_id = np.max(pred_skel_ids)
    matches = np.arange(max_p_id + 1, dtype=np.uint16)

    matches_gt2p = {}
    for i in gt_skel_ids: matches_gt2p[i] = []
    matches_gt2p[0] = []
    rois = scipy.ndimage.find_objects(pred_skel)
    for p_id in pred_skel_ids:
        roi = rois[p_id - 1]
        mask = (pred_skel[roi] == p_id)
        counts = np.bincount(gt_context[roi][mask])[1:] # remove counts for the zero id

        if np.all(counts < (mask.sum() * 0.05)):
            matches[p_id] = 0
        else:
            g_id = np.argmax(counts) + 1
            matches[p_id] = g_id
            matches_gt2p[g_id].append(p_id)

    relabelled_p_skel = matches[pred_skel]

    p_sids = np.unique(relabelled_p_skel)
    p_sids = p_sids[p_sids > 0]

    p_nodes_dict = get_thin_skeletons_nodes(relabelled_p_skel, p_sids, resolution, np.array([1, 1, 1]), temp_folder, num_cpu)

    tp, fp, fn = 0.0, 0.0, 0.0
    if debug is True:
        debug_vol = np.zeros(pred_skel.shape, dtype=np.uint16)
        debug_vol_e = np.zeros(pred_skel.shape, dtype=np.uint16)
        debug_vol_prediction = np.zeros(pred_skel.shape, dtype=np.uint16)

    for gt_id, p_nodes in p_nodes_dict.items():
        gt_nodes = np.array(gt_skel_graphs[str(gt_id)].nodes()).astype(np.int32)
        gt_nodes = np.unique(gt_nodes, axis=0)
        distance = np.sqrt(((resolution*(gt_nodes[:, np.newaxis, :] - p_nodes[np.newaxis, :, :]))**2).sum(2))
        matched_points = (np.any(distance <= matching_radius, axis=0)).sum()
        tp += matched_points
        fp += p_nodes.shape[0] - matched_points
        fn += np.all(distance > matching_radius, axis=1).sum()
        if debug is True:
            tp_i = matched_points
            fp_i = p_nodes.shape[0] - matched_points
            fn_i = np.all(distance > matching_radius, axis=1).sum()
            precision_i = tp_i / (tp_i + fp_i)
            recall_i = tp_i / (tp_i + fn_i)
            print('GT: {}, TP: {}, FP: {}, FN: {}, P: {:1.4f}, R: {:1.4f}'.format(gt_id, tp_i, fp_i, fn_i, precision_i, recall_i))
            if g_id == 350:
                import pdb; pdb.set_trace()
            debug_vol_e[tuple(np.hsplit(gt_nodes[np.all(distance > matching_radius, axis=1), :], 3))] = 3 #fn
            debug_vol_e[tuple(np.hsplit(p_nodes[np.any(distance <= matching_radius, axis=0), :], 3))] = 1 #tp
            debug_vol_e[tuple(np.hsplit(p_nodes[np.any(distance > matching_radius, axis=0), :], 3))] = 2 #fp
            debug_vol[tuple(np.hsplit(gt_nodes, 3))] = np.uint16(gt_id)
            debug_vol_prediction[tuple(np.hsplit(p_nodes.astype(np.int32), 3))] = np.uint16(gt_id)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)

    connectivity = {}
    mean_connectivity = 0.0
    count = 0
    for g_id, p_ids in matches_gt2p.items():
        if len(p_ids) > 0:
            count += 1
            connectivity[g_id] = 1.0 / len(p_ids)
            mean_connectivity += connectivity[g_id]
        else:
            connectivity[g_id] = 0
    mean_connectivity /= count

    print('Precision:    {:10.4f}'.format(precision))
    print('Recall:       {:10.4f}'.format(recall))
    print('F Score:      {:10.4f}'.format(f_score))
    print('Connectivity: {:10.4f}'.format(mean_connectivity))

    if debug is True:
        return precision, recall, f_score, mean_connectivity, (debug_vol, debug_vol_prediction, debug_vol_e)
    else:
        return precision, recall, f_score, mean_connectivity

def get_thin_skeletons_nodes(skeleton, sids, input_resolution, downsample_factor, temp_folder, num_cpu):
    output_filename = 'nodes.h5'
    status = compute(fn=compute_skel_graph, num_proc=num_cpu, sids=sids, skel_vol_full=skeleton,
                              temp_folder=temp_folder, input_resolution=input_resolution,
                              downsample_fac=downsample_factor, output_file_name=output_filename, save_graph=False)
    if status != True:
        raise Exception('Error while creating skeleton graph.')

    nodes = {}
    if num_cpu>0:
        for proc_id in range(0, num_cpu):
            with h5py.File(temp_folder + '/(' + str(proc_id) + ')nodes.h5', 'r') as hfile:
                for key in hfile.keys():
                    if key[:8] == 'allNodes':
                        nodes[int(key[8:])] = np.asarray(hfile[key])
    else:
        with h5py.File(temp_folder + '/(1221)nodes.h5', 'r') as hfile:
            for key in hfile.keys():
                if key[:8] == 'allNodes':
                    nodes[int(key[8:])] = np.asarray(hfile[key])
    return nodes