"""
DBSCAN experiment sweep — Task 2.

Runs three stages for each dataset (glass, hypothyroid, soybean):
  1. k-distance diagnostic plots (eps selection aid)
  2. Full eps × min_samples × metric parameter sweep via runner
  3. Algorithm speed benchmark (auto / ball_tree / kd_tree / brute)

Outputs → results/dbscan/<dataset>/runs.csv
           results/dbscan/<dataset>/summary.csv
           results/dbscan/<dataset>/best_labels.npy
           results/dbscan/<dataset>/plots/
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.dbscan_model import DBSCANModel
from core.preprocessing import load_preprocessed
from core.runner import run_experiment_suite
from core.plotting import plot_k_distance, plot_metric_curve, plot_param_heatmap, plot_bar

DATASETS = ["glass", "hypothyroid", "soybean"]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "dbscan"


def _plot_dir(dataset: str) -> Path:
    d = RESULTS_DIR / dataset / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---------------------------------------------------------------------------
# Parameter grids — dataset-specific eps ranges calibrated to data scale
# (StandardScaler → all features ~unit variance; adjust eps per dataset)
# ---------------------------------------------------------------------------

# min_samples to use for k-distance diagnostic
K_VALUES = [5, 10]

# Metrics to sweep (ball_tree/kd_tree do NOT support cosine)
METRICS = ["euclidean", "manhattan", "cosine"]

# eps ranges tuned per dataset:
#   glass:        9 numeric dims after scaling, dense structure
#   hypothyroid:  ~50 dims (7 num + OHE), moderate density
#   soybean:      ~100+ dims (35 nominal → OHE), high-dimensional — larger eps expected
EPS_GRIDS = {
    "glass":       [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0],
    "hypothyroid": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
    "soybean":     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
}

MIN_SAMPLES_VALUES = [3, 5, 10, 15, 20]

# DBSCAN is deterministic — no seeds needed
SEEDS = None


# ---------------------------------------------------------------------------
# Stage 1: k-distance plots
# ---------------------------------------------------------------------------

def run_k_distance(dataset: str, X: np.ndarray) -> None:
    """Compute and plot sorted k-th NN distances to help choose eps."""
    from sklearn.neighbors import NearestNeighbors

    for metric in ["euclidean", "manhattan"]:  # cosine omitted for speed
        for k in K_VALUES:
            nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            kth_dist = distances[:, -1]  # distance to the k-th neighbour

            save_path = _plot_dir(dataset) / f"kdist_k{k}_{metric}.png"
            plot_k_distance(kth_dist, k=k, dataset=dataset, metric=metric,
                            save_path=save_path)
            print(f"  [k-dist] {dataset} k={k} {metric}: "
                  f"median={np.median(kth_dist):.3f}  "
                  f"90th={np.percentile(kth_dist, 90):.3f}  → {save_path.name}")


# ---------------------------------------------------------------------------
# Stage 2: Full parameter sweep
# ---------------------------------------------------------------------------

def build_param_grid(dataset: str) -> list[dict]:
    """
    Build configs for all metric × eps × min_samples combinations.

    cosine is excluded from ball_tree/kd_tree — we always use algorithm="auto"
    so sklearn picks brute for cosine automatically.
    """
    configs = []
    for metric in METRICS:
        for eps in EPS_GRIDS[dataset]:
            for ms in MIN_SAMPLES_VALUES:
                configs.append({
                    "eps": eps,
                    "min_samples": ms,
                    "metric": metric,
                    "algorithm": "auto",   # auto handles cosine correctly
                })
    return configs


def make_model(params: dict, seed) -> DBSCANModel:
    return DBSCANModel(**params)


# ---------------------------------------------------------------------------
# Stage 3: Algorithm benchmark (auto / ball_tree / kd_tree / brute)
# ---------------------------------------------------------------------------

def run_algorithm_benchmark(dataset: str, X: np.ndarray) -> None:
    """
    Compare sklearn DBSCAN algorithm implementations on runtime.
    Uses a single representative (eps, min_samples) per dataset.
    cosine is not tested with ball_tree/kd_tree (sklearn raises ValueError).
    """
    # Pick a mid-range eps for each dataset as the benchmark point
    benchmark_eps = {"glass": 1.0, "hypothyroid": 2.5, "soybean": 5.0}[dataset]
    benchmark_ms = 5
    metric = "euclidean"

    algo_params = [
        ("auto",       dict(algorithm="auto",       metric=metric)),
        ("ball_tree",  dict(algorithm="ball_tree",  metric=metric)),
        ("kd_tree",    dict(algorithm="kd_tree",    metric=metric)),
        ("brute",      dict(algorithm="brute",      metric=metric)),
        ("brute+cosine", dict(algorithm="brute",    metric="cosine")),
    ]

    labels_ref = None
    algo_names, runtimes = [], []

    for name, kw in algo_params:
        model = DBSCANModel(eps=benchmark_eps, min_samples=benchmark_ms, **kw)
        t0 = time.perf_counter()
        try:
            labels = model.fit_predict(X)
            rt = time.perf_counter() - t0
            algo_names.append(name)
            runtimes.append(round(rt, 4))

            # Verify label equivalence for euclidean variants
            if name == "auto":
                labels_ref = labels
            elif "cosine" not in name and labels_ref is not None:
                match = np.array_equal(labels, labels_ref)
                print(f"  [bench] {dataset} {name}: {rt:.3f}s  labels_match={match}")
            else:
                print(f"  [bench] {dataset} {name}: {rt:.3f}s")
        except Exception as e:
            print(f"  [bench] {dataset} {name}: SKIPPED ({e})")

    save_path = _plot_dir(dataset) / "algorithm_runtime.png"
    plot_bar(
        labels=algo_names,
        values=runtimes,
        title=f"DBSCAN algorithm runtime — {dataset} (eps={benchmark_eps}, min_samples={benchmark_ms})",
        xlabel="algorithm",
        ylabel="Runtime (s)",
        save_path=save_path,
    )
    print(f"  [bench] Saved → {save_path.name}")


# ---------------------------------------------------------------------------
# Post-sweep plots from summary CSV
# ---------------------------------------------------------------------------

def plot_sweep_results(dataset: str) -> None:
    """Generate sensitivity plots from the saved summary CSV."""
    import pandas as pd
    summary_path = RESULTS_DIR / dataset / "summary.csv"
    if not summary_path.exists():
        return

    plot_dir = _plot_dir(dataset)
    import json
    df = pd.read_csv(summary_path)
    df["params"] = df["params_json"].map(json.loads)
    df["eps"] = df["params"].map(lambda p: p["eps"])
    df["min_samples"] = df["params"].map(lambda p: p["min_samples"])
    df["metric"] = df["params"].map(lambda p: p["metric"])

    for metric in METRICS:
        sub = df[df["metric"] == metric].copy()
        if sub.empty:
            continue

        for ms in MIN_SAMPLES_VALUES:
            msub = sub[sub["min_samples"] == ms].sort_values("eps")
            if msub.empty:
                continue

            # eps vs silhouette, noise_pct, n_clusters_found, ari
            save_path = plot_dir / f"{metric}_ms{ms}_sensitivity.png"
            plot_metric_curve(
                df=msub,
                x_col="eps",
                y_cols=["silhouette_mean", "ari_mean", "noise_pct_mean"],
                title=f"DBSCAN sensitivity — {dataset} {metric} min_samples={ms}",
                save_path=save_path,
                x_label="eps",
            )

        # Heatmap: eps × min_samples for silhouette (per metric)
        save_path = plot_dir / f"{metric}_sil_heatmap.png"
        plot_param_heatmap(
            df=sub.assign(eps=sub["eps"].astype(str), ms=sub["min_samples"].astype(str)),
            row_col="ms",
            col_col="eps",
            value_col="silhouette_mean",
            title=f"DBSCAN Silhouette — {dataset} {metric}",
            save_path=save_path,
        )
    print(f"  [plots] Sweep plots saved for {dataset}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"DBSCAN — {dataset}")
        print(f"{'='*60}")

        X, y_true, meta = load_preprocessed(dataset)
        print(f"  Shape: {X.shape}  |  Classes: {meta['n_classes']}")

        # Stage 1: k-distance plots
        print("\n[Stage 1] k-distance diagnostic...")
        run_k_distance(dataset, X)

        # Stage 2: Full parameter sweep
        print("\n[Stage 2] Parameter sweep...")
        param_grid = build_param_grid(dataset)
        print(f"  {len(param_grid)} configurations")
        run_experiment_suite(
            algo_name="dbscan",
            factory_fn=make_model,
            dataset_name=dataset,
            param_grid=param_grid,
            seeds=SEEDS,
        )

        # Stage 3: Algorithm benchmark
        print("\n[Stage 3] Algorithm benchmark...")
        run_algorithm_benchmark(dataset, X)

        # Post-sweep plots
        print("\n[Stage 4] Generating sweep plots...")
        plot_sweep_results(dataset)

    print("\nDone. Results in results/")
