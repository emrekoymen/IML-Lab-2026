"""
Raw ARFF loader. Do not use directly — call load_preprocessed() from core.preprocessing.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class DatasetBundle:
    name: str
    X_raw: pd.DataFrame
    y_true: np.ndarray          # string class labels — post-hoc use only
    numeric_cols: list = field(default_factory=list)
    nominal_cols: list = field(default_factory=list)


def load_arff(dataset_name: str) -> DatasetBundle:
    """Load a single-file .arff dataset and separate features from class label."""
    path = DATA_DIR / f"{dataset_name}.arff"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw, meta = arff.loadarff(path)
    df = pd.DataFrame(raw)

    # Decode bytes → str and mark "?" as NaN
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8").replace("?", np.nan)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Last column is always the class label
    X_raw = df.iloc[:, :-1].copy()
    y_true = df.iloc[:, -1].to_numpy(dtype=str)

    numeric_cols = X_raw.select_dtypes(include="number").columns.tolist()
    nominal_cols = X_raw.select_dtypes(exclude="number").columns.tolist()

    return DatasetBundle(
        name=dataset_name,
        X_raw=X_raw,
        y_true=y_true,
        numeric_cols=numeric_cols,
        nominal_cols=nominal_cols,
    )
