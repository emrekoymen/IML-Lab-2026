"""
Shared plotting helpers. All figures use consistent styling and are saved as high-DPI PNGs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global style defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

SAVE_KWARGS = dict(bbox_inches="tight", dpi=150)


# ---------------------------------------------------------------------------
# Confusion / contingency matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: pd.DataFrame, title: str, save_path: Path) -> None:
    """Heatmap of a contingency table (rows=true classes, cols=cluster IDs)."""
    fig, ax = plt.subplots(figsize=(max(6, cm.shape[1] * 0.7), max(4, cm.shape[0] * 0.5)))
    im = ax.imshow(cm.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("True Class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Count")

    # Annotate cells
    thresh = cm.values.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm.values[i, j],
                    ha="center", va="center", fontsize=8,
                    color="white" if cm.values[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, **SAVE_KWARGS)
    plt.close(fig)


# ---------------------------------------------------------------------------
# k-distance plot (DBSCAN eps selection)
# ---------------------------------------------------------------------------

def plot_k_distance(distances: np.ndarray, k: int, dataset: str,
                    metric: str, save_path: Path) -> None:
    """Sorted k-th nearest-neighbour distance plot for DBSCAN eps selection."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.sort(distances)[::-1], linewidth=1.2)
    ax.set_xlabel("Points (sorted by distance)")
    ax.set_ylabel(f"{k}-NN distance ({metric})")
    ax.set_title(f"k-distance plot — {dataset} (k={k}, metric={metric})")
    plt.tight_layout()
    plt.savefig(save_path, **SAVE_KWARGS)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metric curve (line plot over a 1D parameter)
# ---------------------------------------------------------------------------

def plot_metric_curve(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    save_path: Path,
    x_label: str | None = None,
) -> None:
    """Multi-line plot: x_col on x-axis, one line per metric in y_cols."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for y in y_cols:
        ax.plot(df[x_col], df[y], marker="o", label=y)
    ax.set_xlabel(x_label or x_col)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, **SAVE_KWARGS)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parameter heatmap (2D grid)
# ---------------------------------------------------------------------------

def plot_param_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    title: str,
    save_path: Path,
) -> None:
    """Heatmap showing a metric across two hyperparameters."""
    pivot = df.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(5, pivot.shape[1] * 0.8), max(4, pivot.shape[0] * 0.6)))
    im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    ax.set_title(f"{title} ({value_col})")
    fig.colorbar(im, ax=ax, label=value_col)
    plt.tight_layout()
    plt.savefig(save_path, **SAVE_KWARGS)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bar chart (e.g. runtime by algorithm)
# ---------------------------------------------------------------------------

def plot_bar(
    labels: list,
    values: list,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="steelblue", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, **SAVE_KWARGS)
    plt.close(fig)
