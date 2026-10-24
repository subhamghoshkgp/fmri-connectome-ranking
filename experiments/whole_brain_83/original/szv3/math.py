"""Incidence geometry and exact intercept-only activation boundary."""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def edge_map(n_rois):
    u, v = np.triu_indices(n_rois, 1)
    p = len(u)
    H = sparse.csc_matrix((np.ones(2*p),
                          (np.r_[u, v], np.tile(np.arange(p), 2))),
                         shape=(n_rois, p))
    return u, v, H


def lambda_max(X, y, H, alpha):
    if not 0 <= alpha <= 1 or len(np.unique(y)) != 2:
        raise ValueError("alpha must be in [0,1]; training data need both classes")
    g = X.T @ (np.mean(y) - y) / len(y)
    if alpha == 1:
        return float(np.max(np.abs(g)))
    if alpha == 0:
        return float(np.max(H @ np.abs(g)))
    lo, hi = 0., float(np.max(np.abs(g)) / alpha)
    for _ in range(80):
        mid = (lo + hi)/2
        if np.max(H @ np.maximum(np.abs(g)-alpha*mid, 0)) <= (1-alpha)*mid:
            hi = mid
        else:
            lo = mid
    return hi


def psi(beta, H):
    result = linprog(np.ones(H.shape[0]), A_ub=-H.T,
                     b_ub=-np.abs(beta), bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun), result.x


def dual_norm(s, H):
    return float(np.max(H @ np.abs(s)))
