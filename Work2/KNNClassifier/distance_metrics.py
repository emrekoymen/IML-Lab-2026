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


def hamming_distance(X_query, X_base):
    # Proportion of features that differ
    return np.mean(X_query[:, np.newaxis, :] != X_base[np.newaxis, :, :], axis=-1)


DISTANCES = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "canberra": canberra_distance,
    "hamming": hamming_distance,
}


def compute_distance_matrix(X_query, X_base, metric="euclidean"):
    key = metric.lower()
    if key not in DISTANCES:
        raise ValueError(f"Unknown metric '{metric}'. Choices: {list(DISTANCES.keys())}")
    return DISTANCES[key](X_query, X_base)
