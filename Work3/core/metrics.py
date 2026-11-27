"""
Clustering evaluation metrics — internal (unsupervised) and external (vs y_true).

Internal metrics are computed on non-noise points only (labels != -1).
External metrics are post-hoc only — never use y_true for hyperparameter selection.
"""

import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
)


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """
    Compute all internal and external clustering metrics.

    Internal scores use only X and labels (safe for parameter selection).
    External scores compare labels against y_true (post-hoc only).

    Silhouette  : [-1, 1]   higher is better; +1=well-separated, 0=overlapping, <0=misassigned
    DBI         : [0, inf)  lower is better;  0=ideal compact clusters
    CH          : [0, inf)  higher is better; between/within variance ratio
    ARI         : [-0.5, 1] higher is better; 1=exact match, ~0=random, <0=worse than random
    NMI         : [0, 1]    higher is better; 1=perfect mutual info with true classes
    Purity      : [0, 1]    higher is better; inflates with large k; noise hurts purity
    F1-macro    : [0, 1]    higher is better; Hungarian-matched macro-F1
    """
    labels = np.asarray(labels, dtype=int)
    mask = labels != -1
    valid_labels = labels[mask]
    n_clusters = len(np.unique(valid_labels)) if mask.any() else 0
    noise_count = int((~mask).sum())
    noise_pct = float(noise_count / len(labels) * 100)

    # --- Internal -------------------------------------------------------
    sil, dbi, ch = np.nan, np.nan, np.nan
    if n_clusters >= 2 and mask.sum() >= 2:
        try:
            sil = float(silhouette_score(X[mask], valid_labels))
        except Exception:
            pass
        try:
            dbi = float(davies_bouldin_score(X[mask], valid_labels))
        except Exception:
            pass
        try:
            ch = float(calinski_harabasz_score(X[mask], valid_labels))
        except Exception:
            pass

    # --- External -------------------------------------------------------
    ari = float(adjusted_rand_score(y_true, labels))
    nmi = float(normalized_mutual_info_score(y_true, labels, average_method="arithmetic"))
    purity = _purity(y_true, labels)
    f1 = _hungarian_f1(y_true, labels)

    return {
        "n_clusters_found": n_clusters,
        "noise_count": noise_count,
        "noise_pct": round(noise_pct, 2),
        "silhouette": round(sil, 4) if not np.isnan(sil) else np.nan,
        "davies_bouldin": round(dbi, 4) if not np.isnan(dbi) else np.nan,
        "calinski_harabasz": round(ch, 4) if not np.isnan(ch) else np.nan,
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
        "purity": round(purity, 4),
        "f1_macro": round(f1, 4),
    }


def _purity(y_true: np.ndarray, labels: np.ndarray) -> float:
    """Purity = sum of per-cluster majority-class counts / N. Noise counted as wrong."""
    n = len(y_true)
    total = 0
    for cid in np.unique(labels):
        if cid == -1:
            continue  # noise = all wrong
        mask = labels == cid
        _, counts = np.unique(y_true[mask], return_counts=True)
        total += counts.max()
    return total / n


def _hungarian_f1(y_true: np.ndarray, labels: np.ndarray) -> float:
    """Macro-F1 after optimal cluster↔class assignment via Hungarian algorithm."""
    classes = np.unique(y_true)
    cluster_ids = [c for c in np.unique(labels) if c != -1]
    if not cluster_ids:
        return 0.0

    # Build contingency matrix: rows=classes, cols=clusters
    contingency = np.zeros((len(classes), len(cluster_ids)), dtype=int)
    class_idx = {c: i for i, c in enumerate(classes)}
    cluster_idx = {c: i for i, c in enumerate(cluster_ids)}
    for yt, lbl in zip(y_true, labels):
        if lbl != -1:
            contingency[class_idx[yt], cluster_idx[lbl]] += 1

    # Optimal assignment maximises total overlap
    row_ind, col_ind = linear_sum_assignment(-contingency)

    # Build mapped predictions (noise → "NOISE", unmatched → "UNMATCHED")
    cluster_to_class = {cluster_ids[c]: classes[r] for r, c in zip(row_ind, col_ind)}
    mapped = np.array(
        [cluster_to_class.get(lbl, "__noise__") for lbl in labels]
    )
    # Treat noise/unmatched as a separate class so they count as wrong
    all_classes = np.append(classes, "__noise__")
    return float(f1_score(y_true, mapped, labels=classes, average="macro", zero_division=0))


def build_confusion_matrix(y_true: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Return a contingency DataFrame: rows=true classes, cols=cluster IDs."""
    df = pd.DataFrame({"true": y_true, "cluster": labels})
    return pd.crosstab(df["true"], df["cluster"])
