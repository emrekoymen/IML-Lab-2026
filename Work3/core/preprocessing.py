"""
Preprocessing pipeline. Entry point for all teammates: load_preprocessed(dataset_name).

Scaling: StandardScaler for all numeric columns (locked — do not change).
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from core.data_loader import load_arff, DatasetBundle


# ---------------------------------------------------------------------------
# Dataset-specific cleaning applied before generic transforms
# ---------------------------------------------------------------------------

def _clean(bundle: DatasetBundle) -> DatasetBundle:
    if bundle.name == "hypothyroid":
        # TBG is 100% missing; TBG_measured is constant "f" — both carry no info
        drop = [c for c in ["TBG", "TBG_measured"] if c in bundle.X_raw.columns]
        bundle.X_raw = bundle.X_raw.drop(columns=drop)
        # age=455 is a known typo
        if "age" in bundle.X_raw.columns:
            bundle.X_raw.loc[bundle.X_raw["age"] == 455, "age"] = np.nan
        bundle.numeric_cols = bundle.X_raw.select_dtypes(include="number").columns.tolist()
        bundle.nominal_cols = bundle.X_raw.select_dtypes(exclude="number").columns.tolist()
    return bundle


# ---------------------------------------------------------------------------
# Generic pipeline: impute → scale (numeric) | impute → OHE (nominal)
# ---------------------------------------------------------------------------

def _build_X(bundle: DatasetBundle) -> tuple[np.ndarray, list[str]]:
    parts, feature_names = [], []

    if bundle.numeric_cols:
        num_df = bundle.X_raw[bundle.numeric_cols]
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        num_arr = scaler.fit_transform(imputer.fit_transform(num_df))
        parts.append(num_arr)
        feature_names.extend(bundle.numeric_cols)

    if bundle.nominal_cols:
        cat_df = bundle.X_raw[bundle.nominal_cols]
        imputer = SimpleImputer(strategy="most_frequent")
        encoder = OneHotEncoder(sparse_output=False, dtype=np.float64, handle_unknown="ignore")
        cat_arr = encoder.fit_transform(imputer.fit_transform(cat_df))
        parts.append(cat_arr)
        feature_names.extend(encoder.get_feature_names_out(bundle.nominal_cols).tolist())

    X = np.hstack(parts) if parts else np.empty((len(bundle.X_raw), 0))
    return X.astype(np.float64), feature_names


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_preprocessed(dataset_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load, clean, and preprocess a dataset.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Imputed, scaled, and one-hot encoded feature matrix.
    y_true : np.ndarray, shape (n_samples,)
        Class labels — for post-hoc external evaluation only.
    meta : dict
        Dataset statistics and feature names.
    """
    bundle = _clean(load_arff(dataset_name))
    X, feature_names = _build_X(bundle)
    classes, counts = np.unique(bundle.y_true, return_counts=True)

    meta = {
        "dataset": dataset_name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(classes),
        "class_names": classes.tolist(),
        "class_counts": dict(zip(classes.tolist(), counts.tolist())),
        "feature_names": feature_names,
    }
    return X, bundle.y_true, meta


# ---------------------------------------------------------------------------
# Optional helper (Task 1 note)
# ---------------------------------------------------------------------------

def discretize_numeric(df: pd.DataFrame, q: int = 4) -> pd.DataFrame:
    """Bin numeric columns into q equal-frequency bins using pandas.qcut."""
    result = df.copy()
    for col in df.select_dtypes(include="number").columns:
        result[col] = pd.qcut(df[col], q=q, labels=False, duplicates="drop")
    return result
