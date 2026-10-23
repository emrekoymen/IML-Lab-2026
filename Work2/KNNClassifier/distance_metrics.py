"""Distance metrics for kNN."""

import numpy as np


def euclidean_distance(X_query, X_base):
    diff = X_query[:, np.newaxis, :] - X_base[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def manhattan_distance(X_query, X_base):
    diff = np.abs(X_query[:, np.newaxis, :] - X_base[np.newaxis, :, :])
    return np.sum(diff, axis=-1)


def canberra_distance(X_query, X_base, eps=1e-12):
    q_exp = X_query[:, np.newaxis, :]
    b_exp = X_base[np.newaxis, :, :]
    num = np.abs(q_exp - b_exp)
    denom = np.abs(q_exp) + np.abs(b_exp) + eps
    return np.sum(num / denom, axis=-1)



def heom_distance(X_query, X_base):
    """Heterogeneous Euclidean Overlap Metric (HEOM) by Wilson & Martinez (1997).

    - For binary/nominal (one-hot encoded {0, 1}) columns: uses the overlap
      metric (0 if equal, 1 if different), weighted by 0.5 per one-hot column
      so a categorical mismatch across a one-hot pair (1->0 and 0->1) contributes
      exactly 0.5 + 0.5 = 1.0 to the squared distance sum.
    - For continuous numeric columns (scaled to [0, 1]): uses the range-normalized
      squared difference (q_f - x_f)^2.
    """
    is_binary = np.all((X_base == 0.0) | (X_base == 1.0), axis=0)

    diff_sq = (X_query[:, np.newaxis, :] - X_base[np.newaxis, :, :]) ** 2
    if np.any(is_binary):
        diff_sq[:, :, is_binary] *= 0.5

    return np.sqrt(np.sum(diff_sq, axis=-1))


DISTANCES = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "canberra": canberra_distance,
    "heom": heom_distance,
}


def compute_distance_matrix(X_query, X_base, metric="euclidean"):
    key = metric.lower()
    if key not in DISTANCES:
        raise ValueError(f"Unknown metric '{metric}'. Choices: {list(DISTANCES.keys())}")
    return DISTANCES[key](X_query, X_base)

