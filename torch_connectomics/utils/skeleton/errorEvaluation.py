import numpy as np
import h5py
import edt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage
from torch_connectomics.utils.vis import save_data, read_data
from skimage.morphology import skeletonize_3d
from .computeParallel import *
from .gradientProcessing import remove_small_skeletons

sys.path.append('/n/home11/averma/repositories/cerebellum/')
from cerebellum.error_analysis.skel_methods import SkeletonEvaluation

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

def calculate_error_metric_2(pred_skel, gt_skel, gt_context, resolution, temp_folder, num_cpu, matching_radius=None, evaluate_thinned=False, debug=False):
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
            return -1, -1, -1, -1, -1, (None, None, None)
        else:
            return -1, -1, -1, -1, -1

    if evaluate_thinned is True:
        p_nodes_dict = get_thin_skeletons_nodes(relabelled_p_skel, p_sids, resolution, np.array([1, 1, 1]), temp_folder,
                                                num_cpu, method='skimage_skeletonize')
    else:
        p_nodes_dict = get_skeletons_nodes(relabelled_p_skel)

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
        p_matched_mask = (p_distance <= matching_radius)[:,0]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean', n_jobs=-1).fit(p_nodes_s)
        g_distance, g_indices = nbrs.kneighbors(gt_nodes_s)
        g_matched_mask = (g_distance <= matching_radius)[:,0]

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

def calculate_error_metric_binary_overlap(pred_skel, gt_skel, resolution, temp_folder, num_cpu, matching_radius, debug=False):
    if (pred_skel==0).sum() == 0:
        return -1, -1, -1

    gt_nodes = np.transpose(np.nonzero(gt_skel))
    p_nodes = np.transpose(np.nonzero(pred_skel))

    gt_nodes_s = resolution * gt_nodes
    p_nodes_s = resolution * p_nodes

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean', n_jobs=-1).fit(gt_nodes_s)
    p_distance, p_indices = nbrs.kneighbors(p_nodes_s)
    p_matched_mask = (p_distance <= matching_radius)[:, 0]

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean', n_jobs=-1).fit(p_nodes_s)
    g_distance, g_indices = nbrs.kneighbors(gt_nodes_s)
    g_matched_mask = (g_distance <= matching_radius)[:, 0]

    tp = p_matched_mask.sum()
    fp = p_nodes.shape[0] - p_matched_mask.sum()
    fn = gt_nodes.shape[0] - g_matched_mask.sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)

    if debug is True:
        p_skel = np.zeros_like(pred_skel, dtype=np.uint8)
        g_skel = np.zeros_like(pred_skel, dtype=np.uint8)
        e_skel = np.zeros_like(pred_skel, dtype=np.uint8)
        # in e_skel: 1 is TP, 2 is FP, 3 is FN
        e_skel[p_nodes[p_matched_mask, 0], p_nodes[p_matched_mask, 1], p_nodes[p_matched_mask, 2]] = 1
        e_skel[p_nodes[~p_matched_mask, 0], p_nodes[~p_matched_mask, 1], p_nodes[~p_matched_mask, 2]] = 2

        # fn_mask = list(set(range(gt_nodes.shape[0])) - set(list(p_indices.flatten())))
        # e_skel[gt_nodes[fn_mask, 0], gt_nodes[fn_mask, 1], gt_nodes[fn_mask, 2]] = 3

        e_skel[gt_nodes[~g_matched_mask, 0], gt_nodes[~g_matched_mask, 1], gt_nodes[~g_matched_mask, 2]] = 3

        p_skel[p_nodes[:, 0], p_nodes[:, 1], p_nodes[:, 2]] = 1
        g_skel[gt_nodes[:, 0], gt_nodes[:, 1], gt_nodes[:, 2]] = 1
        return precision, recall, f_score, p_skel, g_skel, e_skel

    return precision, recall, f_score

def calculate_error_metric_binary_overlap_like_l3dfrom2d(pred_skel, gt_skel, maxpooled_gt_skel, debug=False):
    if pred_skel is None or (pred_skel > 0).sum() == 0:
        if debug is True:
            return -1, -1, -1, None
        else:
            return -1, -1, -1

    # ignore all voxels in the maxpooled region
    valid_mask = (~maxpooled_gt_skel) | gt_skel
    pred = pred_skel[valid_mask]
    gt = gt_skel[valid_mask]

    # collect all predicted skeleton points
    tp = (pred == True)[gt].sum()
    fp = (pred == True)[~gt].sum()
    fn = (pred == False)[gt].sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)

    if debug is True:
        error_volume = np.zeros_like(pred_skel, dtype=np.uint16)
        error_volume[((pred_skel == True) & (gt_skel == True) & valid_mask)] = 1  # TRUE POSITIVE
        error_volume[((pred_skel == True) & (gt_skel == False) & valid_mask)] = 2 # False Positive
        error_volume[((pred_skel == False) & (gt_skel == True) & valid_mask)] = 3 # False Negative

        # save_data(error_volume.astype(np.uint16), '/n/home11/averma/temp/error_volume.h5')
        # save_data(pred_skel.astype(np.uint16), '/n/home11/averma/temp/prediction.h5')
        # save_data(gt_skel.astype(np.uint16), '/n/home11/averma/temp/gt.h5')
        # save_data(maxpooled_gt_skel.astype(np.uint16), '/n/home11/averma/temp/gt_mp.h5')
        return precision, recall, f_score, error_volume

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

def get_skeletons_nodes(skeleton):
    idx_sorted = np.argsort(skeleton.ravel())
    skel_ids, idx_start, count = np.unique(skeleton.ravel()[idx_sorted], return_counts=True, return_index=True)
    nodes = {}
    for i, id in enumerate(skel_ids):
        if id > 0:
            nodes[id] = np.transpose(
                np.unravel_index(idx_sorted[idx_start[i]:idx_start[i] + count[i]], skeleton.shape)).astype(np.uint16)
    return nodes

def calculate_binary_errors_batch(pred_skeletons, gt_skeleton_paths, resolution, temp_folder, num_cpu, matching_radius, like_l3dfrom2d=False, maxpool_kernel_sz=5):
    errors = [None] * len(pred_skeletons)
    for i, pred_skeleton_all_steps in enumerate(pred_skeletons):
        gt_skeleton = read_data(gt_skeleton_paths[i])
        if like_l3dfrom2d is True:
            gt_skeleton = gt_skeleton > 0
            gt_skeleton_maxpooled = ndimage.filters.maximum_filter(gt_skeleton, size=maxpool_kernel_sz, mode='constant', cval=0)
        errors[i] = []
        for pred_skeleton in pred_skeleton_all_steps:
            if pred_skeleton is None or pred_skeleton.max() == 0:
                p, r, f = -1, -1, -1
                print('Predicion was empty/None')
            else:
                if like_l3dfrom2d is True:
                    # pred_skeleton = remove_small_skeletons(pred_skeleton, 200)
                    # binary_pred = skeletonize_3d(pred_skeleton > 0)
                    # dilation_kernel = np.ones([4, 4, 4], dtype=np.bool)
                    # binary_pred = ndimage.morphology.binary_dilation(binary_pred, structure=dilation_kernel)
                    p, r, f, _ = calculate_error_metric_binary_overlap_like_l3dfrom2d(pred_skeleton > 0, gt_skeleton, gt_skeleton_maxpooled, debug=True)
                else:
                    p, r, f = calculate_error_metric_binary_overlap(pred_skeleton, gt_skeleton, resolution, temp_folder, num_cpu, matching_radius)

            errors[i].append({'p':p, 'r':r, 'pr':f})

    p_avg, r_avg, f_avg = [0.0]*len(pred_skeletons[0]), [0.0]*len(pred_skeletons[0]), [0.0]*len(pred_skeletons[0])
    # calculate avg error
    for j in range(len(errors[0])):
        for i in range(len(errors)):
            p_avg[j] += errors[i][j]['p']
            r_avg[j] += errors[i][j]['r']
            f_avg[j] += errors[i][j]['pr']

    # print results
    n_vol = len(pred_skeletons)
    for i in range(len(p_avg)):
        p_avg[i] = p_avg[i] / n_vol
        r_avg[i] = r_avg[i] / n_vol
        f_avg[i] = f_avg[i] / n_vol

    print('Precision: ' + ' '.join([('{:3.4f}'.format(x)) for x in p_avg]))
    print('Recall:    ' + ' '.join([('{:3.4f}'.format(x)) for x in r_avg]))
    print('F score:   ' + ' '.join([('{:3.4f}'.format(x)) for x in f_avg]))

    avg_errors = []
    for i, _ in enumerate(p_avg):
        avg_errors.append({'p':p_avg[i], 'r':r_avg[i], 'pr':f_avg[i]})

    return avg_errors

def calculate_errors_batch(pred_skeletons, gt_skeletons, gt_skeleton_ctxs, resolution, temp_folder, num_cpu,
                           matching_radius, ibex_downsample_fac, erl_overlap_allowance):
    errors = [None] * len(pred_skeletons)
    for i, pred_skeleton_all_steps in enumerate(pred_skeletons):
        gt_skeleton = read_data(gt_skeletons[i]) if gt_skeletons[i] is str else gt_skeletons[i]
        gt_context = read_data(gt_skeleton_ctxs[i]) if gt_skeleton_ctxs[i] is str else gt_skeleton_ctxs[i]

        ibex_skeletons = compute_ibex_skeleton_graphs(gt_context, temp_folder + '/ibex_graphs/', resolution, ibex_downsample_fac)
        errors[i] = []
        for pred_skeleton in pred_skeleton_all_steps:
            if pred_skeleton is None or pred_skeleton.max() == 0:
                p, r, f, c, hm, erl = -1, -1, -1, -1, -1, -1
                print('Predicion was empty/None')
            else:
                # p, r, f, c, hm = calculate_error_metric_2(pred_skeleton, gt_skeleton, gt_context,
                #                                           resolution, temp_folder, num_cpu, matching_radius)
                p, r, f = calculate_error_metric_binary_overlap(pred_skeleton, gt_skeleton, resolution,
                                                                temp_folder, num_cpu, matching_radius)
                c, hm = -1, -1
                erl = calculate_erl(pred_skeleton, ibex_skeletons, erl_overlap_allowance)
            errors[i].append({'p':p, 'r':r, 'f':f, 'c':c, 'hm':hm, 'erl':erl})

    steps = len(pred_skeletons[0])
    p_avg, r_avg, f_avg, c_avg, hm_avg, erl_avg = [0.0]*steps, [0.0]*steps, [0.0]*steps, [0.0]*steps, [0.0]*steps, [0.0]*steps
    # calculate avg error
    for j in range(len(errors[0])):
        for i in range(len(errors)):
            p_avg[j] += errors[i][j]['p']
            r_avg[j] += errors[i][j]['r']
            f_avg[j] += errors[i][j]['f']
            c_avg[j] += errors[i][j]['c']
            hm_avg[j] += errors[i][j]['hm']
            erl_avg[j] += errors[i][j]['erl']

    n_vol = len(pred_skeletons)
    for i in range(len(p_avg)):
        p_avg[i] = p_avg[i] / n_vol
        r_avg[i] = r_avg[i] / n_vol
        f_avg[i] = f_avg[i] / n_vol
        c_avg[i] = c_avg[i] / n_vol
        hm_avg[i] = hm_avg[i] / n_vol
        erl_avg[i] = erl_avg[i] / n_vol

    # print results
    print('P:    ' + ' '.join(['{:3.4f}'.format(x) for x in p_avg]))
    print('R:    ' + ' '.join(['{:3.4f}'.format(x) for x in r_avg]))
    print('PR:   ' + ' '.join(['{:3.4f}'.format(x) for x in f_avg]))
    # print('C:    ' + ' '.join(['{:3.4f}'.format(x) for x in c_avg]))
    # print('PRC:  ' + ' '.join(['{:3.4f}'.format(x) for x in hm_avg]))
    print('ERL:  ' + ' '.join(['{:3.4f}'.format(x) for x in erl_avg]))

    avg_errors = []
    for i, _ in enumerate(p_avg):
        avg_errors.append({'p':p_avg[i], 'r':r_avg[i], 'pr':f_avg[i], 'c':c_avg[i], 'hm':hm_avg[i], 'erl':erl_avg[i]})
    return avg_errors

def calculate_erl(pred_skeleton, gt_ibex_skels, overlap_allowance):
    eval = SkeletonEvaluation("syn", gt_ibex_skels, pred_skeleton, t_om=0.9, t_m=0.2, t_s=0.8, include_zero_split=False,
                              include_zero_merge=False, calc_erl=True, overlap_allowance=overlap_allowance)
    eval.summary()
    return eval.erl_pred