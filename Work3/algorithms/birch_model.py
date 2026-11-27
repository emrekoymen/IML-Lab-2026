"""
BIRCH wrapper (sklearn.cluster.Birch).

fit_predict(X) returns integer cluster labels (no noise — BIRCH assigns all points).
extra_info_ exposes n_subclusters (len of subcluster_centers_).

If threshold is too large, the CF-tree may produce fewer subclusters than n_clusters,
causing a ConvergenceWarning — this is logged in extra_info_["subcluster_warning"].
"""

import warnings

import numpy as np
from sklearn.cluster import Birch
from sklearn.exceptions import ConvergenceWarning


class BIRCHModel:
    def __init__(
        self,
        threshold: float = 0.5,
        branching_factor: int = 50,
        n_clusters: int | None = 3,
    ):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.extra_info_: dict = {}

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        model = Birch(
            threshold=self.threshold,
            branching_factor=self.branching_factor,
            n_clusters=self.n_clusters,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            labels = model.fit_predict(X)

        n_sub = len(model.subcluster_centers_)
        self.extra_info_ = {
            "n_subclusters": n_sub,
            "subcluster_warning": len(caught) > 0,
        }
        return labels.astype(int)
