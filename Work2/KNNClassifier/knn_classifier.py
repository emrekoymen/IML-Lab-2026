"""kNN Classifier implementation from scratch."""

import numpy as np

from .distance_metrics import compute_distance_matrix
from .voting_schemes import resolve_votes
from .retention_policies import RetentionManager


def k_nearest(dist_matrix, k):
    """Indices and distances of the k nearest base rows of each query row, nearest first."""
    n_query, n_base = dist_matrix.shape
    k = min(k, n_base)
    if k < n_base:
        k_part = np.argpartition(dist_matrix, k - 1, axis=1)[:, :k]
        rows = np.arange(n_query)[:, np.newaxis]
        k_dists = dist_matrix[rows, k_part]

        sort_order = np.argsort(k_dists, axis=1)
        sorted_indices = np.take_along_axis(k_part, sort_order, axis=1)
        sorted_dists = np.take_along_axis(k_dists, sort_order, axis=1)
    else:
        sorted_indices = np.argsort(dist_matrix, axis=1)
        sorted_dists = np.take_along_axis(dist_matrix, sorted_indices, axis=1)
    return sorted_indices, sorted_dists


class KNNClassifier:
    """k-Nearest Neighbors (kNN) classifier from scratch.

    Parameters
    ----------
    k : int, default=3
        Number of nearest neighbors to retrieve.
    distance_metric : str, default='euclidean'
        Distance metric to use ('euclidean', 'manhattan', 'canberra', 'heom', 'hamming').
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
        self.n_stored_ = None
        self.storage_pct_ = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.n_stored_ = len(self.X_train)
        self.storage_pct_ = 100.0
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

        predictions = []

        # Find top k nearest neighbors per query
        sorted_indices, sorted_dists = k_nearest(dist_matrix, self.k)

        for i in range(n_test):
            winner = resolve_votes(
                neighbor_labels=self.y_train[sorted_indices[i]],
                neighbor_distances=sorted_dists[i],
                scheme=self.voting_scheme,
                p=self.p,
            )
            predictions.append(winner)

        self.n_stored_ = len(self.X_train)
        self.storage_pct_ = 100.0
        return np.array(predictions)

    def _predict_sequential(self, X_test, y_test):
        manager = RetentionManager(
            X_train=self.X_train,
            y_train=self.y_train,
            policy=self.retention_policy,
            alpha=self.alpha,
        )

        # Precompute distances from all X_test queries to the initial X_train case base.
        # For AR/DC: newly retained test points are appended and distances computed on-the-fly.
        # For OG: X_base is never physically shrunk (soft-deletion via active_mask), so
        #         dist_matrix column indices always align with the full X_base.
        dist_matrix = compute_distance_matrix(X_test, self.X_train, metric=self.distance_metric)
        n_test = len(X_test)

        # Buffer for test points added by AR/DC policies; grows as instances are retained.
        # Pre-allocating avoids O(n^2) vstack inside the loop.
        added_rows = []        # list of 1-D arrays (one per added point)
        n_added = 0

        predictions = []
        for i in range(n_test):
            # --- Build the full distance vector for query i ---
            # Columns 0..len(X_train)-1  → pre-computed (always valid; OG never deletes)
            # Columns len(X_train)..     → on-the-fly for AR/DC added points
            if n_added:
                added_arr = np.array(added_rows)          # shape (n_added, n_features)
                dist_added = compute_distance_matrix(
                    X_test[i : i + 1], added_arr, metric=self.distance_metric
                )[0]
                dist_vec = np.concatenate([dist_matrix[i], dist_added])
            else:
                dist_vec = dist_matrix[i]

            # For OG: apply the soft-deletion mask to expose only active distances.
            # active_mask has the same length as dist_vec (both cover the full X_base).
            if manager.active_mask is not None:
                active_dist = dist_vec[manager.active_mask]
                active_labels = manager.y_base[manager.active_mask]
            else:
                active_dist = dist_vec
                active_labels = manager.y_base

            n_active = len(active_dist)
            k = min(self.k, n_active)
            if k < n_active:
                top_k = np.argpartition(active_dist, k - 1)[:k]
                sorted_k = top_k[np.argsort(active_dist[top_k])]
            else:
                sorted_k = np.argsort(active_dist)

            sorted_dists = active_dist[sorted_k]
            neighbor_labels = active_labels[sorted_k]

            pred_label = resolve_votes(
                neighbor_labels=neighbor_labels,
                neighbor_distances=sorted_dists,
                scheme=self.voting_scheme,
                p=self.p,
            )
            predictions.append(pred_label)

            # --- Update case base ---
            true_label = y_test[i] if y_test is not None else None
            prev_size = manager.size

            # For OG, retrieved_indices must refer to positions in the *full* (unmasked)
            # X_base, not in the masked view.  Convert back via the active positions.
            if manager.active_mask is not None:
                active_positions = np.flatnonzero(manager.active_mask)
                og_retrieved = active_positions[sorted_k]
                manager.update(X_test[i], true_label, pred_label, og_retrieved)
            else:
                manager.update(X_test[i], true_label, pred_label, sorted_k)

            # Track newly appended AR/DC points for subsequent on-the-fly distances.
            if manager.size > prev_size:
                added_rows.append(X_test[i])
                n_added += 1

        self.n_stored_ = manager.size
        n_initial = len(self.X_train)
        self.storage_pct_ = (manager.size / n_initial * 100.0) if n_initial > 0 else 100.0
        return np.array(predictions)


