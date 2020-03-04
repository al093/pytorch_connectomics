import numpy as np
import h5py
import edt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from torch_connectomics.utils.vis import save_data, read_data
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

def calculate_error_metric_2(pred_skel, gt_skel, gt_context, resolution, temp_folder, num_cpu, debug=False, matching_radius=None):
    if matching_radius is None:
        matching_radius = np.float32(np.max(resolution) * 2.0)
    pred_skel_ids = np.unique(pred_skel)
    pred_skel_ids = pred_skel_ids[pred_skel_ids > 0]

    # get skeletons points for each gt id
    gt_idx_sort = np.argsort(gt_skel.ravel())
    gt_skel_ids, idx_start, count = np.unique(gt_skel.ravel()[gt_idx_sort], return_counts=True, return_index=True)
    all_gt_nodes = {}
    for i, gt_id in enumerate(gt_skel_ids):
        if gt_id > 0:
            all_gt_nodes[gt_id] = np.transpose(
                np.unravel_index(gt_idx_sort[idx_start[i]:idx_start[i] + count[i]], gt_skel.shape)).astype(np.int16)
    gt_skel_ids=gt_skel_ids[gt_skel_ids>0]

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

        if np.all(counts < (mask.sum() * 0.001)):
            matches[p_id] = 0
        else:
            g_id = np.argmax(counts) + 1
            matches[p_id] = g_id
            matches_gt2p[g_id].append(p_id)

    relabelled_p_skel = matches[pred_skel]

    p_sids = np.unique(relabelled_p_skel)
    p_sids = p_sids[p_sids > 0]

    if p_sids.size == 0:
        if debug is True:
            return -1, -1, -1, -1, (None, None, None)
        else:
            return -1, -1, -1, -1

    p_nodes_dict = get_thin_skeletons_nodes(relabelled_p_skel, p_sids, resolution, np.array([1, 1, 1]), temp_folder,
                                            num_cpu, method='skimage_skeletonize')

    tp, fp, fn = 0.0, 0.0, 0.0
    if debug is True:
        debug_vol = np.zeros(pred_skel.shape, dtype=np.uint16)
        debug_vol_e = np.zeros(pred_skel.shape, dtype=np.uint16)
        debug_vol_prediction = np.zeros(pred_skel.shape, dtype=np.uint16)

    # all gt ids which were not matched with any predicted ids are false negative
    for gt_id in list(set(gt_skel_ids).difference(set(p_sids))):
        gt_nodes = all_gt_nodes[gt_id]
        fn += gt_nodes.shape[0]
        if debug is True:
            debug_vol_e[gt_nodes[:, 0], gt_nodes[:, 1], gt_nodes[:, 2]] = 3

    for gt_id, p_nodes in p_nodes_dict.items():
        gt_nodes = all_gt_nodes[gt_id]

        # there may be no points because thinning really small segments results in no points
        if p_nodes.shape[0] == 0:
            fn += gt_nodes.shape[0]
            continue

        gt_nodes_s = resolution*gt_nodes
        p_nodes_s = resolution*p_nodes
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean', n_jobs=-1).fit(gt_nodes_s)
        p_distance, p_indices = nbrs.kneighbors(p_nodes_s)
        p_matched_mask = p_distance <= matching_radius
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean', n_jobs=-1).fit(p_nodes_s)
        g_distance, g_indices = nbrs.kneighbors(gt_nodes_s)
        g_matched_mask = g_distance <= matching_radius

        tp += p_matched_mask.sum()
        fp += p_nodes.shape[0] - p_matched_mask.sum()
        fn += gt_nodes.shape[0] - g_matched_mask.sum()

        if debug is True:
            tp_i = p_matched_mask.sum()
            fp_i = p_nodes.shape[0] - p_matched_mask.sum()
            fn_i = gt_nodes.shape[0] - g_matched_mask.sum()
            precision_i = tp_i / (tp_i + fp_i)
            recall_i = tp_i / (tp_i + fn_i)
            print('GT: {}, TP: {}, FP: {}, FN: {}, P: {:1.4f}, R: {:1.4f}'.format(gt_id, tp_i, fp_i, fn_i, precision_i, recall_i))
            debug_vol_e[tuple(np.hsplit(gt_nodes[~g_matched_mask, :], 3))] = 3 #fn
            debug_vol_e[tuple(np.hsplit(p_nodes[p_matched_mask, :], 3))] = 1 #tp
            debug_vol_e[tuple(np.hsplit(p_nodes[~p_matched_mask, :], 3))] = 2 #fp
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

    # print('Precision:    {:10.4f}'.format(precision))
    # print('Recall:       {:10.4f}'.format(recall))
    # print('F Score:      {:10.4f}'.format(f_score))
    # print('Connectivity: {:10.4f}'.format(mean_connectivity))

    prc_hmean = (3*precision*recall*mean_connectivity)/(precision*recall + precision*mean_connectivity + recall*mean_connectivity)

    if debug is True:
        return precision, recall, f_score, mean_connectivity, prc_hmean, (debug_vol, debug_vol_prediction, debug_vol_e)
    else:
        return precision, recall, f_score, mean_connectivity, prc_hmean

def calculate_error_metric_binary_overlap(pred_skel, gt_skel, resolution, temp_folder, num_cpu, debug=False):
    pred_skel_ids = np.unique(pred_skel)
    pred_skel_ids = pred_skel_ids[pred_skel_ids > 0]
    if pred_skel_ids.size == 0:
        return -1, -1, -1
    p_nodes_dict = get_thin_skeletons_nodes(pred_skel, pred_skel_ids, resolution, np.array([1, 1, 1]), temp_folder,
                                            num_cpu, method='ibex') # method='skimage_skeletonize'
    gt_nodes = np.transpose(np.nonzero(gt_skel)).astype(np.int16)

    # collect all predicted skeleton points
    p_nodes = []
    for _, pn in p_nodes_dict.items():
        p_nodes.append(pn)
    p_nodes = np.concatenate(p_nodes)

    matched_points_mask = np.all(np.abs(gt_nodes[:, np.newaxis, :] - p_nodes[np.newaxis, :, :]) <= 1, axis=2)

    tp = np.any(matched_points_mask, axis=0).sum()
    fp = p_nodes.shape[0] - tp
    fn = np.all(~matched_points_mask, axis=1).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)

    if debug is True:
        p_skel = np.zeros_like(pred_skel, dtype=np.uint16)
        g_skel = np.zeros_like(pred_skel, dtype=np.uint16)
        e_skel = np.zeros_like(pred_skel, dtype=np.uint16)
        tp_mask = np.any(matched_points_mask, axis=0)
        e_skel[p_nodes[tp_mask, 0], p_nodes[tp_mask, 1], p_nodes[tp_mask, 2]] = 1
        e_skel[p_nodes[~tp_mask, 0], p_nodes[~tp_mask, 1], p_nodes[~tp_mask, 2]] = 2
        fn_mask = np.all(~matched_points_mask, axis=1)
        e_skel[gt_nodes[fn_mask, 0], gt_nodes[fn_mask, 1], gt_nodes[fn_mask, 2]] = 3
        p_skel[p_nodes[:, 0], p_nodes[:, 1], p_nodes[:, 2]] = 1
        g_skel[gt_nodes[:, 0], gt_nodes[:, 1], gt_nodes[:, 2]] = 1
        return precision, recall, f_score, p_skel, g_skel, e_skel

    return precision, recall, f_score

def calculate_error_metric_binary_overlap_like_l3dfrom2d(pred_skel, gt_skel, dilated_gt_skel):
    pred_skel_ids = pred_skel_ids[pred_skel_ids > 0]
    if pred_skel_ids.size == 0:
        return -1, -1, -1

    # ignore all voxels in the dilated region
    valid_mask = (~dilated_gt_skel | gt_skel)
    pred = pred_skel[valid_mask]
    gt = gt_skel[valid_mask]

    # collect all predicted skeleton points
    tp = (pred == True)[gt].sum()
    fp = (pred == True)[~gt].sum()
    fn = (pred == False)[gt].sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score

def get_thin_skeletons_nodes(skeleton, sids, input_resolution, downsample_factor, temp_folder, num_cpu, method='ibex'):
    output_filename = 'nodes.h5'

    if method == 'ibex':
        status = compute(fn=compute_thinned_nodes, num_proc=num_cpu, sids=sids, skel_vol_full=skeleton,
                          temp_folder=temp_folder, input_resolution=input_resolution,
                          downsample_fac=downsample_factor, output_file_name=output_filename)
    elif method == 'skimage_skeletonize':
        status = compute(fn=compute_thinned_nodes_skimage_skeletonize, num_proc=num_cpu, sids=sids, skel_vol_full=skeleton,
                          temp_folder=temp_folder, output_file_name=output_filename)
    else:
        raise Exception('specfied method: {} not found'.format(method))

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

def calculate_binary_errors_batch(pred_skeletons, gt_skeleton_paths, resolution, temp_folder, num_cpu, like_l3dfrom2d=False, dilation_kernel_sz=5):
    errors = [None] * len(pred_skeletons)
    for i, pred_skeleton_all_steps in enumerate(tqdm(pred_skeletons)):
        gt_skeleton = read_data(gt_skeleton_paths[i])
        if like_l3dfrom2d is True:
            gt_skeleton = gt_skeleton > 0
            sel = np.ones([dilation_kernel_sz, dilation_kernel_sz, dilation_kernel_sz])
            gt_skeleton_dilated = ndimage.morphology.binary_dilation(gt_skeleton, sel)
        errors[i] = []
        for pred_skeleton in pred_skeleton_all_steps:
            if like_l3dfrom2d is True:
                p, r, f = calculate_error_metric_binary_overlap_like_l3dfrom2d(pred_skeleton > 0, gt_skeleton, gt_skeleton_dilated)
            else:
                p, r, f = calculate_error_metric_binary_overlap(pred_skeleton, gt_skeleton, resolution, temp_folder, num_cpu)

            errors[i].append({'p':p, 'r':r, 'f':f})

    p_avg, r_avg, f_avg = [0.0]*len(pred_skeletons[0]), [0.0]*len(pred_skeletons[0]), [0.0]*len(pred_skeletons[0])
    # calculate avg error
    for j in range(len(errors[0])):
        for i in range(len(errors)):
            p_avg[j] += errors[i][j]['p']
            r_avg[j] += errors[i][j]['r']
            f_avg[j] += errors[i][j]['f']

    # print results
    n_vol = len(pred_skeletons)
    print('Precision: ' + ' '.join([('{:3.4f}'.format(x/n_vol)) for x in p_avg]))
    print('Recall:    ' + ' '.join([('{:3.4f}'.format(x/n_vol)) for x in p_avg]))
    print('F score:   ' + ' '.join([('{:3.4f}'.format(x/n_vol)) for x in f_avg]))
    return errors

def calculate_errors_batch(pred_skeletons, gt_skeleton_paths, gt_skeleton_ctx_paths, resolution, temp_folder, num_cpu, matching_radius=None):
    errors = [None] * len(pred_skeletons)
    for i, pred_skeleton_all_steps in enumerate(tqdm(pred_skeletons)):
        gt_skeleton = read_data(gt_skeleton_paths[i])
        gt_context = read_data(gt_skeleton_ctx_paths[i])
        errors[i] = []
        for pred_skeleton in pred_skeleton_all_steps:
            if pred_skeleton.max() == 0:
                p, r, f, c, hm = 0, 0, 0, 0, 0
            else:
                p, r, f, c, hm = calculate_error_metric_2(pred_skeleton, gt_skeleton, gt_context,
                                                          resolution, temp_folder, num_cpu, matching_radius)
            errors[i].append({'p':p, 'r':r, 'f':f, 'c':c, 'hm':hm})

    p_avg, r_avg, f_avg, c_avg, hm_avg = [0.0]*len(pred_skeletons[0]), [0.0]*len(pred_skeletons[0]), [0.0]*len(pred_skeletons[0]), [0.0]*len(pred_skeletons[0]), [0.0]*len(pred_skeletons[0])
    # calculate avg error
    for j in range(len(errors[0])):
        for i in range(len(errors)):
            p_avg[j] += errors[i][j]['p']
            r_avg[j] += errors[i][j]['r']
            f_avg[j] += errors[i][j]['f']
            c_avg[j] += errors[i][j]['c']
            hm_avg[j] += errors[i][j]['hm']

    # print results
    n_vol = len(pred_skeletons)
    print('P:    ' + ' '.join([('{:3.4f}'.format(x/n_vol)) for x in p_avg]))
    print('R:    ' + ' '.join([('{:3.4f}'.format(x/n_vol)) for x in r_avg]))
    print('PR:   ' + ' '.join([('{:3.4f}'.format(x/n_vol)) for x in f_avg]))
    print('C:    ' + ' '.join([('{:3.4f}'.format(x / n_vol)) for x in c_avg]))
    print('PRC:  ' + ' '.join([('{:3.4f}'.format(x / n_vol)) for x in hm_avg]))

    return errors