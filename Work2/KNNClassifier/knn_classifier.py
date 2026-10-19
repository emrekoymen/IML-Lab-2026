"""kNN Classifier implementation from scratch."""

from time import perf_counter
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

        # Precompute distances from all X_test queries to the initial X_train case base
        # and maintain an active distance row per query as cases are added or forgotten
        dist_matrix = compute_distance_matrix(X_test, self.X_train, metric=self.distance_metric)
        n_test = len(X_test)
        added_X = []

        predictions = []
        for i in range(n_test):
            # Distance vector from query i to current base:
            # base consists of surviving initial points + any newly added test points
            if added_X:
                x_query = X_test[i : i + 1]
                added_arr = np.vstack(added_X)
                dist_added = compute_distance_matrix(x_query, added_arr, metric=self.distance_metric)[0]
                dist_vec = np.concatenate([dist_matrix[i], dist_added])
            else:
                dist_vec = dist_matrix[i]

            if manager.active_mask is not None:
                dist_vec = dist_vec[manager.active_mask]

            k = min(self.k, manager.size)
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
            prev_size = manager.size
            manager.update(X_test[i], true_label, pred_label, sorted_k)
            if manager.policy in ("always_retain", "ar") or (
                manager.policy in ("different_class", "dc", "df") and manager.size > prev_size
            ):
                added_X.append(X_test[i : i + 1])

        self.n_stored_ = manager.size
        self.storage_pct_ = (manager.size / len(self.X_train)) * 100.0 if len(self.X_train) > 0 else 100.0
        return np.array(predictions)


def kNNAlgorithm(
    X_train,
    y_train,
    X_test,
    y_test=None,
    k=3,
    distance_metric="euclidean",
    voting_scheme="majority",
    retention_policy="never_retain",
    alpha=0.2,
    p=1.0,
):
    """Classify each instance from TestMatrix (X_test) using TrainMatrix (X_train, y_train).

    Returns
    -------
    predictions : np.ndarray
        Predicted class labels for X_test.
    training_time : float
        Time in seconds spent fitting the training matrix.
    prediction_time : float
        Problem-solving time in seconds spent classifying X_test.
    """
    model = KNNClassifier(
        k=k,
        distance_metric=distance_metric,
        voting_scheme=voting_scheme,
        retention_policy=retention_policy,
        alpha=alpha,
        p=p,
    )

    start = perf_counter()
    model.fit(X_train, y_train)
    training_time = perf_counter() - start

    start = perf_counter()
    predictions = model.predict(X_test, y_test=y_test)
    prediction_time = perf_counter() - start

    return predictions, training_time, prediction_time

