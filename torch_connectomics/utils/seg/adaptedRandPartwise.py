import numpy as np
import scipy.sparse as sparse

def adapted_rand_partwise(seg, gt):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]

    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.

    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error

    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)

    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n,int)

    p_ij = sparse.csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A,:]
    b = p_ij[1:n_labels_A,1:n_labels_B]
    c = p_ij[1:n_labels_A,0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / sumB
    recall = sumAB / sumA

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    areImprovement = {}
    fScore_new = {}
    precision = {}
    recall = {}
    for i in range(1, n_labels_A):
        print('\nComputing improvement score for gt segment {} / {}'.format(i, n_labels_A-1))
        zero_col = sparse.csr_matrix((n_labels_A - 1, 1), dtype=a.dtype)
        a_new = sparse.lil_matrix(sparse.hstack((a, zero_col)))
        b_new = sparse.lil_matrix(sparse.hstack((b, zero_col)))
        c_new = c
        a_new[i-1, :] = 0
        b_new[i-1, :] = 0
        c_new[i-1, 0] = 0
        d_new = b_new.multiply(b_new)

        new_seg_count = sparse.csr_matrix.sum(a[i-1, :])
        a_new[i-1, -1] = new_seg_count
        b_new[i-1, -1] = new_seg_count

        a_i = np.array(a_new.sum(1))
        b_i = np.array(b_new.sum(0))

        sumA = np.sum(a_i * a_i)
        sumB = np.sum(b_i * b_i) + (np.sum(c_new) / n)
        sumAB = np.sum(d_new) + (np.sum(c_new) / n)

        precision[i] = sumAB / sumB
        recall[i] = sumAB / sumA

        fScore_new[i] = 1.0 - (2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
        areImprovement[i] = are - fScore_new[i]

    #find corresponding segments for gt and output
    corr_seg = find_corresponding(seg, gt)

    return are, areImprovement, fScore_new, precision, recall, corr_seg

def calculate_segmentwise_score(i, a, b, c, seg, gt, are, n_labels_A):
        print('\nComputing improvement score for gt segment {} / {}'.format(i, n_labels_A-1))
        zero_col = sparse.csr_matrix((n_labels_A - 1, 1), dtype=a.dtype)
        a_new = sparse.lil_matrix(sparse.hstack((a, zero_col)))
        b_new = sparse.lil_matrix(sparse.hstack((b, zero_col)))
        c_new = c
        a_new[i-1, :] = 0
        b_new[i-1, :] = 0
        c_new[i-1, 0] = 0
        d_new = b_new.multiply(b_new)

        new_seg_count = sparse.csr_matrix.sum(a[i-1, :])
        a_new[i-1, -1] = new_seg_count
        b_new[i-1, -1] = new_seg_count

        a_i = np.array(a_new.sum(1))
        b_i = np.array(b_new.sum(0))

        sumA = np.sum(a_i * a_i)
        sumB = np.sum(b_i * b_i) + (np.sum(c_new) / n)
        sumAB = np.sum(d_new) + (np.sum(c_new) / n)

        precision = sumAB / sumB
        recall = sumAB / sumA

        fScore_new = 1.0 - (2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
        areImprovement = are - fScore_new[i]

        #find corresponding segments for gt and output
        corr_seg = find_corresponding(seg, gt)

        return areImprovement, fScore_new, precision, recall, corr_seg

def find_corresponding(seg, gt):
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    corr_segments = {}
    for i in range(1, n_labels_A):
        idx, counts = np.unique(segB[segA == i], return_counts=True)
        order = np.flip(np.argsort(counts))
        corr_segments[i] = idx[order]
    return corr_segments