"""
BIRCH experiment sweep — Task 3.

Runs for each dataset (glass, hypothyroid, soybean):
  1. Full threshold × branching_factor × n_clusters parameter sweep via runner
  2. Post-sweep plots: threshold×n_clusters heatmaps, branching_factor sensitivity

Outputs → results/birch/<dataset>/runs.csv
           results/birch/<dataset>/summary.csv
           results/birch/<dataset>/best_labels.npy
           results/birch/<dataset>/plots/

Notes
-----
- BIRCH is deterministic — no random seeds needed.
- n_clusters=None uses raw CF-tree leaf subclusters (no second-stage clustering).
- When threshold is too large, n_subclusters < n_clusters → ConvergenceWarning.
  This is caught by BIRCHModel and logged in extra_info_["subcluster_warning"].
- Soybean has ~100+ dimensions after OHE. BIRCH threshold values are Euclidean
  radii in that space — calibrate independently (larger values needed).
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.birch_model import BIRCHModel
from core.preprocessing import load_preprocessed
from core.runner import run_experiment_suite
from core.plotting import plot_metric_curve, plot_param_heatmap

DATASETS = ["glass", "hypothyroid", "soybean"]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "birch"


def _plot_dir(dataset: str) -> Path:
    d = RESULTS_DIR / dataset / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

# threshold: Euclidean radius of CF-tree subclusters (scale-dependent after StandardScaler)
# Ranges differ across datasets due to dimensionality:
#   glass:        9 dims — lower thresholds meaningful
#   hypothyroid:  ~50 dims — moderate range
#   soybean:      ~100+ dims — large thresholds needed; also test None to skip second stage
THRESHOLD_GRIDS = {
    "glass":       [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "hypothyroid": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    "soybean":     [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0],
}

BRANCHING_FACTORS = [10, 25, 50, 100]

# n_clusters: None = raw CF-tree subclusters; integers tailored to dataset class counts
N_CLUSTERS_GRID = {
    "glass":       [None, 2, 3, 4, 5, 6, 8],
    "hypothyroid": [None, 2, 3, 4, 6, 8],
    "soybean":     [None, 4, 6, 8, 10, 15, 19],
}

SEEDS = None  # BIRCH is deterministic


# ---------------------------------------------------------------------------
# Parameter grid builders
# ---------------------------------------------------------------------------

def build_main_grid(dataset: str) -> list[dict]:
    """
    Primary grid: threshold × n_clusters with branching_factor=50 (default).
    Used to study the most important interactions.
    """
    configs = []
    for threshold in THRESHOLD_GRIDS[dataset]:
        for n_clusters in N_CLUSTERS_GRID[dataset]:
            configs.append({
                "threshold": threshold,
                "branching_factor": 50,
                "n_clusters": n_clusters,
            })
    return configs


def build_branching_grid(dataset: str) -> list[dict]:
    """
    Secondary grid: branching_factor sweep at fixed threshold and n_clusters.
    Skips bf=50 since build_main_grid already covers it.
    """
    fixed_threshold = {"glass": 0.5, "hypothyroid": 1.5, "soybean": 4.0}[dataset]
    true_k = {"glass": 6, "hypothyroid": 4, "soybean": 19}[dataset]
    return [
        {"threshold": fixed_threshold, "branching_factor": bf, "n_clusters": true_k}
        for bf in BRANCHING_FACTORS
        if bf != 50
    ]


def make_model(params: dict, seed) -> BIRCHModel:
    return BIRCHModel(**params)


# ---------------------------------------------------------------------------
# Post-sweep plots
# ---------------------------------------------------------------------------

def plot_sweep_results(dataset: str) -> None:
    """Generate heatmaps and sensitivity plots from the saved summary CSV."""
    import pandas as pd

    summary_path = RESULTS_DIR / dataset / "summary.csv"
    if not summary_path.exists():
        return

    plot_dir = _plot_dir(dataset)
    df = pd.read_csv(summary_path)
    df["params"] = df["params_json"].map(json.loads)
    df["threshold"] = df["params"].map(lambda p: p["threshold"])
    df["branching_factor"] = df["params"].map(lambda p: p["branching_factor"])
    df["n_clusters_param"] = df["params"].map(lambda p: str(p["n_clusters"]))

    # 1. threshold × n_clusters heatmaps for key metrics (branching_factor=50)
    sub = df[df["branching_factor"] == 50].copy()
    for metric in ["silhouette_mean", "davies_bouldin_mean", "ari_mean", "n_clusters_found_mean"]:
        save_path = plot_dir / f"thresh_nclusters_{metric}.png"
        plot_param_heatmap(
            df=sub.assign(
                threshold=sub["threshold"].astype(str),
                n_clusters_param=sub["n_clusters_param"],
            ),
            row_col="n_clusters_param",
            col_col="threshold",
            value_col=metric,
            title=f"BIRCH — {dataset}",
            save_path=save_path,
        )

    # 2. branching_factor sensitivity (fixed threshold & true_k)
    fixed_threshold = {"glass": 0.5, "hypothyroid": 1.5, "soybean": 4.0}[dataset]
    true_k = str({"glass": 6, "hypothyroid": 4, "soybean": 19}[dataset])
    bf_sub = df[(df["threshold"] == fixed_threshold) & (df["n_clusters_param"] == true_k)].copy()
    if not bf_sub.empty:
        bf_sub = bf_sub.sort_values("branching_factor")
        save_path = plot_dir / "branching_factor_sensitivity.png"
        plot_metric_curve(
            df=bf_sub,
            x_col="branching_factor",
            y_cols=["silhouette_mean", "runtime_sec_mean"],
            title=f"BIRCH branching_factor sensitivity — {dataset}",
            save_path=save_path,
            x_label="branching_factor",
        )

    # 3. n_subclusters diagnostic (shows CF-tree collapse)
    runs_path = RESULTS_DIR / dataset / "runs.csv"
    if runs_path.exists():
        runs_df = pd.read_csv(runs_path)
        if "extra_json" in runs_df.columns:
            runs_df["n_subclusters"] = runs_df["extra_json"].map(
                lambda s: json.loads(s).get("n_subclusters", None)
            )
            runs_df["params"] = runs_df["params_json"].map(json.loads)
            runs_df["threshold"] = runs_df["params"].map(lambda p: p["threshold"])
            runs_df["branching_factor"] = runs_df["params"].map(lambda p: p["branching_factor"])
            runs_df["n_clusters_param"] = runs_df["params"].map(lambda p: str(p["n_clusters"]))
            sub2 = runs_df[runs_df["branching_factor"] == 50].dropna(subset=["n_subclusters"])
            if not sub2.empty:
                save_path = plot_dir / "nsubclusters_heatmap.png"
                plot_param_heatmap(
                    df=sub2.assign(
                        threshold=sub2["threshold"].astype(str),
                        n_clusters_param=sub2["n_clusters_param"],
                    ),
                    row_col="n_clusters_param",
                    col_col="threshold",
                    value_col="n_subclusters",
                    title=f"BIRCH n_subclusters (CF-tree) — {dataset}",
                    save_path=save_path,
                )

    print(f"  [plots] Sweep plots saved for {dataset}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"BIRCH — {dataset}")
        print(f"{'='*60}")

        X, y_true, meta = load_preprocessed(dataset)
        print(f"  Shape: {X.shape}  |  Classes: {meta['n_classes']}")

        # Combine main grid + branching_factor sweep into one suite so summary.csv has all configs
        full_grid = build_main_grid(dataset) + build_branching_grid(dataset)
        print(f"\n[Stage 1] Running sweep: {len(full_grid)} configurations...")
        run_experiment_suite(
            algo_name="birch",
            factory_fn=make_model,
            dataset_name=dataset,
            param_grid=full_grid,
            seeds=SEEDS,
        )

        # Post-sweep plots
        print("\n[Stage 2] Generating plots...")
        plot_sweep_results(dataset)

    print("\nDone. Results in results/")
