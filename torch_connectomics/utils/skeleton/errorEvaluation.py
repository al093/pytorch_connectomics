import numpy as np
import h5py
import edt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def write_hf(data, name):
    with h5py.File(name, 'w') as hf:
        hf.create_dataset('main', data=data, compression='gzip')
def read_hf(name):
    with h5py.File(name, 'r') as hf:
        data = np.asarray(hf['main'])
    return data

def calculate_error_metric(pred_skel, gt_skel, gt_context, gt_skel_ids, anisotropy, gt_save_data=False):
    bloat_radius = np.float32(35.0)
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

