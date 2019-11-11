import os, sys, time
import numpy as np
import h5py
from pprint import pprint
from scipy.spatial.distance import directed_hausdorff
import edt
# from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def write_hf(data, name):
    with h5py.File(name, 'w') as hf:
        hf.create_dataset('main', data=data, compression='gzip')
def read_hf(name):
    with h5py.File(name, 'r') as hf:
        data = np.asarray(hf['main'])
    return data

def calculate_error_metric(pred_skel, gt_skel, gt_context, save_data=False):
# def calculateErrorMetric(skel_pred_f, skel_gt_f, gt_context_f, save_data=False):
#     gt_skel = read_hf(skel_gt_f)
    gt_skel_ids = np.unique(gt_skel)
    gt_skel_ids = gt_skel_ids[gt_skel_ids > 0]
    # gt_context = read_hf(gt_context_f)

    # pred_skel = read_hf(skel_pred_f)
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
    skel_dtx = edt.edt(relabelled_p_skel == 0, anisotropy=(30, 6, 6),
                       black_border=False, order='C',
                       parallel=12)

    skel_context = skel_dtx < 15.0
    skel_points = np.nonzero(relabelled_p_skel)
    skel_points = np.transpose((5.0 * skel_points[0], skel_points[1], skel_points[2]))

    context_points = np.nonzero(skel_context)
    context_points = np.transpose((5.0 * context_points[0], context_points[1], context_points[2]))

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean', n_jobs=-1).fit(skel_points)
    _, indices = nbrs.kneighbors(context_points)

    # assign labels
    context_points[:, 0] = context_points[:, 0] / 5.0
    context_points = context_points.astype(np.uint16)

    skel_points[:, 0] = skel_points[:, 0] / 5.0
    skel_points = skel_points.astype(np.uint16)

    relabelled_skel_context = np.zeros_like(pred_skel)
    relabelled_skel_context[(context_points[:, 0], context_points[:, 1], context_points[:, 2])] \
        = relabelled_p_skel[
        (skel_points[indices[:, 0], 0], skel_points[indices[:, 0], 1], skel_points[indices[:, 0], 2])]

    # calculate TP
    # TP : All points on the predicted skeleton which are close (or inside the context area) to the matched GT
    # FP : All points on the predicted skeleton which are far (outside the context area) to the matched GT
    # FN : All Points on the matched GT Skeleton which were not identified
    pr = {}
    tp_global = np.zeros_like(pred_skel, dtype=np.bool)
    fp_global = np.zeros_like(pred_skel, dtype=np.bool)
    fn_global = np.zeros_like(pred_skel, dtype=np.bool)

    for g_id, p_ids in matches_gt2p.items():
        if g_id == 0:
            continue

        if len(p_ids) is 0:
            pr[g_id] = (0, 0)
            mask_gt = (gt_skel == g_id)
            fn_global |= mask_gt

        else:
            mask_p = (relabelled_p_skel == g_id)
            mask_p_context = (relabelled_skel_context == g_id)
            mask_gt = (gt_skel == g_id)
            mask_gt_context = (gt_context == g_id)

            tp = mask_p & mask_gt_context
            fp = mask_p & ~mask_gt_context
            fn = ~mask_p_context & mask_gt

            tp_global |= tp
            fp_global |= fp
            fn_global |= fn

            tp_s, fp_s, fn_s = float(tp.sum()), float(fp.sum()), float(fn.sum())
            precision = tp_s / (tp_s + fp_s)
            recall = tp_s / (tp_s + fn_s)

            pr[g_id] = (precision, recall)

    tp_s, fp_s, fn_s = float(tp_global.sum()), float(fp_global.sum()), float(fn_global.sum())
    precision = tp_s / (tp_s + fp_s)
    recall = tp_s / (tp_s + fn_s)
    f_score = 2 * precision * recall / (precision + recall)
    print('Overall Precision: ', precision)
    print('Overall Recall: ', recall)
    print('F Score: ', f_score)

    # debug,
    # write_hf(tp_global, path + 'tp.h5')
    # write_hf(fp_global, path + 'fp.h5')
    # write_hf(fn_global, path + 'fn.h5')
    # write_hf(np.concatenate((tp_global, fp_global, fn_global)), path + 'error_metric_mask.h5')

    # Get Connectivity score
    # For each ground truth how many skeletons was it spread into
    # Report average score also
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
    print('Mean Connectivity: ', mean_connectivity)
    return (precision, recall, f_score, mean_connectivity)

# if __name__ == '__main__':
#     calculateErrorMetric()