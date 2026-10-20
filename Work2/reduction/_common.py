"""Helpers shared by the reduction methods."""

import numpy as np

from KNNClassifier.distance_metrics import compute_distance_matrix
from KNNClassifier.knn_classifier import k_nearest
from KNNClassifier.voting_schemes import resolve_votes


def pairwise_distances(X, distance_metric="euclidean", chunk_size=64):
    """n x n distance matrix, in row chunks to bound memory."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    D = np.empty((n, n), dtype=float)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        D[start:stop] = compute_distance_matrix(X[start:stop], X, metric=distance_metric)
    np.fill_diagonal(D, 0.0)
    return np.minimum(D, D.T)


def knn_predict(D, y_train, k, voting_scheme="majority", p=1.0):
    """kNN predictions from a query-to-training distance block, as KNNClassifier does."""
    indices, distances = k_nearest(D, k)
    predictions = [
        resolve_votes(y_train[row_indices], row_distances, scheme=voting_scheme, p=p)
        for row_indices, row_distances in zip(indices, distances)
    ]
    return np.array(predictions)


def stratified_folds(y, n_folds, seed=0):
    """Split indices into folds that keep the class proportions."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    offset = 0
    for label in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == label))
        for position, index in enumerate(members):
            folds[(offset + position) % n_folds].append(index)
        offset += len(members)
    return [np.sort(np.array(fold, dtype=int)) for fold in folds]
