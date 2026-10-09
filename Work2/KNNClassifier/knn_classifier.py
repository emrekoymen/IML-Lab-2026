"""kNN Classifier implementation from scratch."""

import numpy as np

from .distance_metrics import compute_distance_matrix
from .voting_schemes import resolve_votes
from .retention_policies import RetentionManager


class KNNClassifier:
    """k-Nearest Neighbors (kNN) classifier from scratch.

    Parameters
    ----------
    k : int, default=3
        Number of nearest neighbors to retrieve.
    distance_metric : str, default='euclidean'
        Distance metric to use ('euclidean', 'manhattan', 'canberra', 'hamming').
    voting_scheme : str, default='majority'
        Voting method ('majority', 'inverse_distance').
    retention_policy : str, default='never_retain'
        Retention/forgetting policy ('never_retain', 'always_retain',
        'different_class', 'oblivion_by_goodness').
    alpha : float, default=0.2
        Learning rate for the oblivion by goodness policy.
    p : float, default=1.0
        Power exponent for inverse distance weighting.
    """

    def __init__(
        self,
        k=3,
        distance_metric="euclidean",
        voting_scheme="majority",
        retention_policy="never_retain",
        alpha=0.2,
        p=1.0,
    ):
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")

        self.k = int(k)
        self.distance_metric = distance_metric.lower().strip()
        self.voting_scheme = voting_scheme.lower().strip()
        self.retention_policy = retention_policy.lower().strip()
        self.alpha = float(alpha)
        self.p = float(p)

        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X_test, y_test=None):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call fit(X, y) before predict.")

        X_test_arr = np.asarray(X_test, dtype=float)
        y_test_arr = np.asarray(y_test) if y_test is not None else None

        # Static case base: compute all pairwise distances at once
        if self.retention_policy in ("never_retain", "nr"):
            return self._predict_batch(X_test_arr)

        # Dynamic retention: sequential loop to update instance base
        return self._predict_sequential(X_test_arr, y_test_arr)

    def _predict_batch(self, X_test):
        dist_matrix = compute_distance_matrix(X_test, self.X_train, metric=self.distance_metric)
        n_test = X_test.shape[0]
        n_train = self.X_train.shape[0]
        k = min(self.k, n_train)

        predictions = []

        # Find top k nearest neighbors per query
        if k < n_train:
            k_part = np.argpartition(dist_matrix, k - 1, axis=1)[:, :k]
            rows = np.arange(n_test)[:, np.newaxis]
            k_dists = dist_matrix[rows, k_part]
            
            sort_order = np.argsort(k_dists, axis=1)
            sorted_indices = np.take_along_axis(k_part, sort_order, axis=1)
            sorted_dists = np.take_along_axis(k_dists, sort_order, axis=1)
        else:
            sorted_indices = np.argsort(dist_matrix, axis=1)
            sorted_dists = np.take_along_axis(dist_matrix, sorted_indices, axis=1)

        for i in range(n_test):
            winner = resolve_votes(
                neighbor_labels=self.y_train[sorted_indices[i]],
                neighbor_distances=sorted_dists[i],
                scheme=self.voting_scheme,
                p=self.p,
            )
            predictions.append(winner)

        return np.array(predictions)

    def _predict_sequential(self, X_test, y_test):
        manager = RetentionManager(
            X_train=self.X_train,
            y_train=self.y_train,
            policy=self.retention_policy,
            alpha=self.alpha,
        )

        predictions = []
        for i in range(len(X_test)):
            x_query = X_test[i : i + 1]
            k = min(self.k, manager.size)

            dist_vec = compute_distance_matrix(x_query, manager.X_base, metric=self.distance_metric)[0]

            if k < manager.size:
                top_k = np.argpartition(dist_vec, k - 1)[:k]
                sorted_k = top_k[np.argsort(dist_vec[top_k])]
            else:
                sorted_k = np.argsort(dist_vec)

            sorted_dists = dist_vec[sorted_k]
            neighbor_labels = manager.y_base[sorted_k]

            pred_label = resolve_votes(
                neighbor_labels=neighbor_labels,
                neighbor_distances=sorted_dists,
                scheme=self.voting_scheme,
                p=self.p,
            )
            predictions.append(pred_label)

            # Update base online
            true_label = y_test[i] if y_test is not None else None
            manager.update(X_test[i], true_label, pred_label, sorted_k)

        return np.array(predictions)
