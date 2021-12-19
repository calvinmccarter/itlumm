import numpy as np
import numpy.linalg as npla
import scipy.optimize as spop

from collections import defaultdict

def balanced_partition(eigVals, M):
    dim = eigVals.size
    dim_subspace = dim / M
    assert dim % M == 0  # pq start vs end not yet implemented
    dim_tables = defaultdict(list)
    fvals = np.log(eigVals + 1e-20)
    fvals = fvals - np.min(fvals) + 1
    sum_list = np.zeros(M)
    big_number = 1e10 + np.sum(fvals)

    cur_subidx = 0
    for d in range(dim):
        dim_tables[cur_subidx].append(d)
        sum_list[cur_subidx] += fvals[d]
        if len(dim_tables[cur_subidx]) == dim_subspace:
            sum_list[cur_subidx] = big_number
        cur_subidx = np.argmin(sum_list)

    dim_ordered = []
    for m in range(M):
        dim_ordered.extend(dim_tables[m])
    return dim_ordered

def eigenvalue_allocation(X, M):
    #X = X - np.mean(X, axis=0, keepdims=True)
    dim = X.shape[1]
    dim_pca = dim
    covX = np.cov(X, rowvar=False)
    w, v = npla.eig(covX)
    #rint(w)
    sort_ix = np.argsort(np.abs(w))[::-1]
    #rint(w[sort_ix])
    eigVal = w[sort_ix]
    eigVec = v[:,sort_ix]
    dim_ordered = balanced_partition(eigVal, M)
    R = eigVec[:, dim_ordered]
    return R

def R_to_P(R):
    """Rotation matrix to permutation matrix"""
    row_ind, col_ind = spop.linear_sum_assignment(R, maximize=True)
    P = np.zeros_like(R)
    P[row_ind, col_ind] = 1

    return P

def R_to_reordering(R):
    """Rotation matrix to reordering"""
    row_ind, col_ind = spop.linear_sum_assignment(R, maximize=True)
    qq = np.zeros(R.shape[0], dtype=int)
    qq[col_ind] = row_ind.astype(int)  # reverse the sort
    qq = qq.tolist()
    return qq

def eigenvalue_permutation(X, M):
    R = eigenvalue_allocation(X, M)
    return R_to_P(R)

def opq_reordering(X, M):
    R = eigenvalue_allocation(X, M)
    reord = R_to_reordering(R)
    return reord
    
