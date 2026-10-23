"""Classification algorithm interfaces for Work 2.

Unified interface for all algorithm functions used in experiments:
    kNNAlgorithm            — kNN from scratch (unified metric dict)
    reduction_kNNAlgorithm  — kNN on reduced training set (unified metric dict)
    svmAlgorithm            — SVM via scikit-learn (unified metric dict)
    rfAlgorithm             — Random Forest via scikit-learn (unified metric dict)

All functions return a dict with the unified results schema:
    predictions, accuracy, f1_macro,
    n_correct, n_incorrect,
    training_time, prediction_time, peak_memory_mb,
    n_stored, storage_pct
"""

import tracemalloc
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from KNNClassifier import KNNClassifier
from reduction import REDUCTION_METHODS, reduce


def _make_result(y_test, preds, training_time, prediction_time, peak_bytes, n_stored, n_train):
    """Assemble the unified metric dict shared by all classifiers."""
    if y_test is not None:
        y_arr = np.asarray(y_test)
        preds_arr = np.asarray(preds)
        n_correct = int(np.sum(preds_arr == y_arr))
        n_incorrect = int(len(y_arr) - n_correct)
        acc = float(accuracy_score(y_arr, preds_arr))
        f1 = float(f1_score(y_arr, preds_arr, average="macro", zero_division=0))
    else:
        n_correct = None
        n_incorrect = None
        acc = None
        f1 = None

    return {
        "predictions": preds,
        "accuracy": acc,
        "f1_macro": f1,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "training_time": float(training_time),
        "prediction_time": float(prediction_time),
        "peak_memory_mb": float(peak_bytes / (1024 ** 2)),
        "n_stored": int(n_stored),
        "storage_pct": float(n_stored / n_train * 100) if n_train > 0 else 100.0,
    }


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
    """Train a kNN classifier and classify X_test.

    Returns a dict matching the unified results schema:
        predictions, accuracy, f1_macro,
        n_correct, n_incorrect,
        training_time, prediction_time, peak_memory_mb,
        n_stored, storage_pct
    """
    model = KNNClassifier(
        k=k,
        distance_metric=distance_metric,
        voting_scheme=voting_scheme,
        retention_policy=retention_policy,
        alpha=alpha,
        p=p,
    )

    tracemalloc.start()
    try:
        t0 = perf_counter()
        model.fit(X_train, y_train)
        training_time = perf_counter() - t0

        t0 = perf_counter()
        if y_test is not None:
            preds = model.predict(X_test, y_test=y_test)
        else:
            preds = model.predict(X_test)
        prediction_time = perf_counter() - t0

        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    n_stored = getattr(model, "n_stored_", len(X_train))
    storage_pct = getattr(model, "storage_pct_", 100.0)

    res = _make_result(
        y_test, preds, training_time, prediction_time, peak,
        n_stored=n_stored, n_train=len(X_train),
    )
    res["storage_pct"] = float(storage_pct)
    return res


def reduction_kNNAlgorithm(
    X_train,
    y_train,
    X_test,
    y_test=None,
    method="gcnn",
    k=3,
    distance_metric="euclidean",
    voting_scheme="majority",
    retention_policy="never_retain",
    alpha=0.2,
    p=1.0,
    **method_params,
):
    """Reduce training set and train a kNN classifier on the kept subset,
    then classify X_test.

    Returns a dict matching the unified results schema with reduction metrics:
        predictions, accuracy, f1_macro,
        n_correct, n_incorrect,
        reduction_time, training_time, prediction_time, peak_memory_mb,
        n_train_initial, n_train_kept, n_stored, storage_pct, final_storage_pct,
        kept_indices, reduction_method, reduction_info
    """
    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train)
    X_te = np.asarray(X_test, dtype=float)

    tracemalloc.start()
    try:
        t_red = perf_counter()
        kept, red_info = reduce(
            X_tr,
            y_tr,
            method=method,
            distance_metric=distance_metric,
            k=k,
            voting_scheme=voting_scheme,
            p=p,
            return_info=True,
            **method_params,
        )
        reduction_time = perf_counter() - t_red

        model = KNNClassifier(
            k=k,
            distance_metric=distance_metric,
            voting_scheme=voting_scheme,
            retention_policy=retention_policy,
            alpha=alpha,
            p=p,
        )

        t_fit = perf_counter()
        model.fit(X_tr[kept], y_tr[kept])
        training_time = perf_counter() - t_fit

        t_pred = perf_counter()
        if y_test is not None:
            preds = model.predict(X_te, y_test=y_test)
        else:
            preds = model.predict(X_te)
        prediction_time = perf_counter() - t_pred

        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    n_stored = int(getattr(model, "n_stored_", len(kept)))
    storage_pct = float(len(kept) / len(y_tr) * 100.0) if len(y_tr) > 0 else 100.0
    final_storage_pct = float(n_stored / len(y_tr) * 100.0) if len(y_tr) > 0 else 100.0

    res = _make_result(
        y_test, preds, training_time, prediction_time, peak,
        n_stored=n_stored, n_train=len(y_tr),
    )
    res.update({
        "reduction_method": method,
        "reduction_time": float(reduction_time),
        "n_train_initial": len(y_tr),
        "n_train_kept": len(kept),
        "storage_pct": float(storage_pct),
        "final_storage_pct": float(final_storage_pct),
        "kept_indices": kept,
        "reduction_info": red_info,
    })
    return res


def svmAlgorithm(
    X_train,
    y_train,
    X_test,
    y_test=None,
    kernel="rbf",
    C=1.0,
    gamma="scale",
    degree=3,
):
    """Train an SVM and classify X_test.

    Returns a dict matching the unified results schema.
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)

    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=42)

    tracemalloc.start()
    try:
        t0 = perf_counter()
        model.fit(X_tr, y_train)
        training_time = perf_counter() - t0

        t0 = perf_counter()
        preds = model.predict(X_te)
        prediction_time = perf_counter() - t0

        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return _make_result(
        y_test, preds, training_time, prediction_time, peak,
        n_stored=len(X_tr), n_train=len(X_tr),
    )


def rfAlgorithm(
    X_train,
    y_train,
    X_test,
    y_test=None,
    n_estimators=100,
    criterion="gini",
    max_depth=None,
):
    """Train a Random Forest and classify X_test.

    Returns a dict matching the unified results schema.
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        random_state=42,
    )

    tracemalloc.start()
    try:
        t0 = perf_counter()
        model.fit(X_tr, y_train)
        training_time = perf_counter() - t0

        t0 = perf_counter()
        preds = model.predict(X_te)
        prediction_time = perf_counter() - t0

        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return _make_result(
        y_test, preds, training_time, prediction_time, peak,
        n_stored=len(X_tr), n_train=len(X_tr),
    )


__all__ = [
    "kNNAlgorithm",
    "reduction_kNNAlgorithm",
    "svmAlgorithm",
    "rfAlgorithm",
    "KNNClassifier",
    "REDUCTION_METHODS",
]
