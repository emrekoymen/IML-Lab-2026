"""
Experiment runner. Handles the full loop: load → sweep configs → log → aggregate → save best.

Usage (in experiments/run_<algo>.py):
    run_experiment_suite(algo_name, factory_fn, dataset_name, param_grid, seeds=None)
"""

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from core.preprocessing import load_preprocessed
from core.metrics import evaluate_clustering, build_confusion_matrix
from core.plotting import plot_confusion_matrix

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

INTERNAL_METRICS = ["silhouette", "davies_bouldin", "calinski_harabasz"]
ALL_METRICS = [
    "n_clusters_found", "noise_count", "noise_pct",
    "silhouette", "davies_bouldin", "calinski_harabasz",
    "ari", "nmi", "purity", "f1_macro",
]


def _algo_dataset_dir(algo_name: str, dataset_name: str) -> Path:
    """Base output dir for one algorithm + dataset: results/<algo>/<dataset>/"""
    d = RESULTS_DIR / algo_name / dataset_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_experiment_suite(
    algo_name: str,
    factory_fn,
    dataset_name: str,
    param_grid: list[dict],
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """
    Run all configs (and seeds) for one algorithm on one dataset.

    Parameters
    ----------
    algo_name   : short name used in filenames, e.g. "dbscan"
    factory_fn  : callable(params: dict, seed: int | None) -> model with fit_predict(X)
    dataset_name: "glass", "hypothyroid", or "soybean"
    param_grid  : list of param dicts to sweep
    seeds       : list of random seeds for stochastic algorithms; None = run once

    Returns
    -------
    summary_df : aggregated summary DataFrame (also saved to results/<algo>/<dataset>/summary.csv)
    """
    X, y_true, meta = load_preprocessed(dataset_name)
    seed_list = seeds if seeds is not None else [None]

    out_dir = _algo_dataset_dir(algo_name, dataset_name)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    runs_csv = out_dir / "runs.csv"
    runs_csv.unlink(missing_ok=True)

    run_rows = []
    for config_id, params in enumerate(param_grid):
        params_json = json.dumps(params, sort_keys=True)
        for seed in seed_list:
            row = _run_single(
                algo_name, dataset_name, config_id, params, params_json,
                seed, factory_fn, X, y_true,
            )
            run_rows.append(row)
            _append_csv(runs_csv, row)

    runs_df = pd.DataFrame(run_rows)
    summary_df = _aggregate(runs_df, algo_name, dataset_name)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    # Save best config artifacts
    best = summary_df[summary_df["is_best"]].iloc[0]
    best_params = json.loads(best["params_json"])
    best_model = factory_fn(best_params, seeds[0] if seeds else None)
    best_labels = best_model.fit_predict(X)

    np.save(out_dir / "best_labels.npy", best_labels)
    cm = build_confusion_matrix(y_true, best_labels)
    plot_confusion_matrix(
        cm,
        title=f"{algo_name.upper()} — {dataset_name} (best config)",
        save_path=plot_dir / "cm.png",
    )
    print(f"[{algo_name}/{dataset_name}] Done. Best config_id={best['config_id']}  "
          f"silhouette={best['silhouette_mean']:.3f}  ari={best['ari_mean']:.3f}")

    return summary_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_single(algo_name, dataset_name, config_id, params, params_json,
                seed, factory_fn, X, y_true) -> dict:
    model = factory_fn(params, seed)

    tracemalloc.start()
    t0 = time.perf_counter()
    labels = model.fit_predict(X)
    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    extra = getattr(model, "extra_info_", {})
    metrics = evaluate_clustering(X, np.asarray(labels, dtype=int), y_true)

    return {
        "dataset": dataset_name,
        "algorithm": algo_name,
        "config_id": config_id,
        "params_json": params_json,
        "seed": seed,
        "runtime_sec": round(runtime, 4),
        "peak_memory_mb": round(peak / 1024 ** 2, 3),
        "extra_json": json.dumps(extra) if extra else "{}",
        **metrics,
    }


def _aggregate(runs_df: pd.DataFrame, algo_name: str, dataset_name: str) -> pd.DataFrame:
    group_cols = ["dataset", "algorithm", "config_id", "params_json"]
    agg = {}
    for m in ALL_METRICS + ["runtime_sec", "peak_memory_mb"]:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
    agg["n_runs"] = ("seed", "count")

    summary = (
        runs_df.groupby(group_cols, as_index=False)
        .agg(**{k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in agg.items()})
    )
    summary["is_best"] = False
    summary = _mark_best(summary)
    return summary


def _mark_best(df: pd.DataFrame) -> pd.DataFrame:
    """Select best config using internal metrics only (no y_true)."""
    valid = df[
        (df["n_clusters_found_mean"] >= 2) &
        (df["noise_pct_mean"] <= 30.0) &
        df["silhouette_mean"].notna()
    ]
    if valid.empty:
        # Fallback: best ARI as last resort (should not happen in normal runs)
        idx = df["ari_mean"].idxmax()
    else:
        idx = valid["silhouette_mean"].idxmax()
    df.loc[idx, "is_best"] = True
    return df


def _append_csv(path: Path, row: dict) -> None:
    write_header = not path.exists()
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=write_header, index=False)
