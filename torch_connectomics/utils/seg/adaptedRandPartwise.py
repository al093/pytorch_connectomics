import numpy as np
import scipy.sparse as sparse

def adapted_rand_partwise(seg, gt):
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
    are_new = {}
    precision_new = {}
    recall_new = {}
    delta_precision = {}
    delta_recall = {}
    for i in range(1, n_labels_A):
        print('\nComputing improvement score for gt segment {} / {}'.format(i, n_labels_A-1))
        zero_col = sparse.csr_matrix((n_labels_A - 1, 1), dtype=a.dtype)
        a_new = sparse.lil_matrix(sparse.hstack((a, zero_col)))
        b_new = sparse.lil_matrix(sparse.hstack((b, zero_col)))
        c_new = c.copy()

        a_new[i-1, :] = 0
        b_new[i-1, :] = 0
        c_new[i-1, 0] = 0

        new_seg_count = sparse.csr_matrix.sum(a[i-1, :])
        a_new[i-1, -1] = new_seg_count
        b_new[i-1, -1] = new_seg_count
        d_new = b_new.multiply(b_new)

        a_i = np.array(a_new.sum(1))
        b_i = np.array(b_new.sum(0))

        sumA = np.sum(a_i * a_i)
        sumB = np.sum(b_i * b_i) + (np.sum(c_new) / n)
        sumAB = np.sum(d_new) + (np.sum(c_new) / n)

        precision_new[i] = sumAB / sumB
        recall_new[i] = sumAB / sumA
        delta_precision[i] = precision_new[i] - precision
        delta_recall[i] = recall_new[i] - recall

        are_new[i] = 1.0 - (2.0 * precision_new[i] * recall_new[i] / (precision_new[i] + recall_new[i]))
        areImprovement[i] = are - are_new[i]
        print('Delta A-RAND: ', areImprovement[i])
    #find corresponding segments for gt and output
    corr_seg = find_corresponding(seg, gt)

    return are, areImprovement, are_new, precision_new, recall_new, delta_precision, delta_recall, corr_seg


def adapted_rand_groupwise(seg, gt, top50, groups):
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
    are_new = {}
    precision_new = {}
    recall_new = {}
    delta_precision = {}
    delta_recall = {}
    group_idx = 0

    segments_group = []
    for i in groups:
        segments_group.append(top50[:i])

    print(segments_group)

    for group in segments_group:
        print('\nComputing improvement score for group {} / {}'.format(group_idx + 1, len(groups)))
        zero_col = sparse.csr_matrix((n_labels_A - 1, len(group)), dtype=a.dtype)
        a_new = sparse.lil_matrix(sparse.hstack((a, zero_col)))
        b_new = sparse.lil_matrix(sparse.hstack((b, zero_col)))
        c_new = c.copy()

        sub_idx = [i - 1 for i in group]
        a_new[sub_idx, :] = 0
        b_new[sub_idx, :] = 0
        c_new[sub_idx, 0] = 0

        new_seg_count = a[sub_idx, :].sum(axis=1)

        a_new[sub_idx, np.arange(-1, -len(group)-1, -1)] = new_seg_count.transpose()[0]
        b_new[sub_idx, np.arange(-1, -len(group)-1, -1)] = new_seg_count.transpose()[0]
        d_new = b_new.multiply(b_new)

        a_i = np.array(a_new.sum(1))
        b_i = np.array(b_new.sum(0))

        sumA = np.sum(a_i * a_i)
        sumB = np.sum(b_i * b_i) + (np.sum(c_new) / n)
        sumAB = np.sum(d_new) + (np.sum(c_new) / n)

        precision_new[group_idx] = sumAB / sumB
        recall_new[group_idx] = sumAB / sumA
        delta_precision[group_idx] = precision_new[group_idx] - precision
        delta_recall[group_idx] = recall_new[group_idx] - recall

        are_new[group_idx] = 1.0 - (2.0 * precision_new[group_idx] * recall_new[group_idx] / (precision_new[group_idx] + recall_new[group_idx]))
        areImprovement[group_idx] = are - are_new[group_idx]
        print('Delta A-RAND: ', areImprovement[group_idx])
        group_idx += 1

    return areImprovement, are_new

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