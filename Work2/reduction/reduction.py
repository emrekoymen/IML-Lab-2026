"""Entry points: reduce a training set, and the reduction_kNNAlgorithm."""

from time import perf_counter

import numpy as np

from KNNClassifier import KNNClassifier

from ._common import pairwise_distances
from .del_reduction import del_reduction
from .gcnn import gcnn, select_rho
from .rnge import rnge

REDUCTION_METHODS = ("none", "gcnn", "rnge", "del")


def reduce(X, y, method, distance_metric="euclidean", k=3, voting_scheme="majority", p=1.0,
           return_info=False, **method_params):
    """Sorted indices of the training instances to keep, usable with any model.

    k, voting_scheme, p : the kNN used inside DEL and GCNN with rho="auto".
    method_params       : gcnn: rho (0.5 or "auto"), rho_grid, n_folds, seed
                          rnge: order (1 or 2)
    """
    method = method.lower().strip()
    if method not in REDUCTION_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choices: {list(REDUCTION_METHODS)}")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    info = {"method": method, "distance_metric": distance_metric}

    start = perf_counter()
    if method == "none":
        kept = np.arange(len(y))
    else:
        D = pairwise_distances(X, distance_metric)
        if method == "gcnn":
            rho = method_params.pop("rho", 0.5)
            if rho == "auto":
                rho, info["rho_scores"] = select_rho(
                    D, y, k=k, voting_scheme=voting_scheme, p=p, **method_params
                )
            elif method_params:
                raise TypeError(f"{sorted(method_params)} only apply with rho='auto'")
            info["rho"] = rho
            kept = gcnn(D, y, rho)
        elif method == "rnge":
            info["order"] = method_params.get("order", 1)
            kept = rnge(D, y, **method_params)
        else:
            info.update(k=k, voting_scheme=voting_scheme)
            kept = del_reduction(D, y, k=k, voting_scheme=voting_scheme, p=p, **method_params)
    info["reduction_time"] = perf_counter() - start
    if len(kept) == 0:
        raise ValueError(f"'{method}' removed every training instance; nothing left to train on")
    info["n_kept"] = len(kept)
    info["storage"] = len(kept) / len(y)

    return (kept, info) if return_info else kept


def reduction_kNNAlgorithm(X_train, y_train, X_test, y_test=None, method="gcnn", k=3,
                           distance_metric="euclidean", voting_scheme="majority",
                           retention_policy="never_retain", alpha=0.2, p=1.0, **method_params):
    """Reduce the training set, then classify X_test with the kNN. Returns (predictions, info)."""
    kept, info = reduce(
        X_train, y_train, method, distance_metric=distance_metric, k=k,
        voting_scheme=voting_scheme, p=p, return_info=True, **method_params,
    )

    model = KNNClassifier(
        k=k, distance_metric=distance_metric, voting_scheme=voting_scheme,
        retention_policy=retention_policy, alpha=alpha, p=p,
    )
    model.fit(np.asarray(X_train, dtype=float)[kept], np.asarray(y_train)[kept])

    start = perf_counter()
    predictions = model.predict(np.asarray(X_test, dtype=float), y_test)
    info["prediction_time"] = perf_counter() - start
    info["kept_indices"] = kept
    info["n_final_stored"] = getattr(model, "n_stored_", len(kept))
    info["final_storage"] = info["n_final_stored"] / len(y_train) if len(y_train) > 0 else 1.0
    return predictions, info

