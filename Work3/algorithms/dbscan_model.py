"""
DBSCAN wrapper (sklearn.cluster.DBSCAN).

fit_predict(X) returns labels where -1 = noise.
extra_info_ exposes n_core_samples for diagnostics.

sklearn note:
  ball_tree and kd_tree do NOT support cosine metric (raises ValueError).
  Use algorithm="auto" or "brute" when metric="cosine".
"""

import numpy as np
from sklearn.cluster import DBSCAN


class DBSCANModel:
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.extra_info_: dict = {}

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
            n_jobs=-1,
        )
        labels = model.fit_predict(X)
        self.extra_info_ = {
            "n_core_samples": int(len(model.core_sample_indices_)),
        }
        return labels.astype(int)
