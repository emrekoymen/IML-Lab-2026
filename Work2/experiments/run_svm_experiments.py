"""Automated 10-fold SVM experiments with Demšar (2006) statistical tests.

Grid search: kernel × C × gamma × degree
Datasets: soybean, hypothyroid
Outputs → results/svm/{dataset}_fold_results.csv
           results/svm/{dataset}_summary_results.csv
           results/svm/{dataset}_stats_top_configs.csv
           results/svm/{dataset}_cd_top_configs.png
"""

import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── path setup (works from any working directory) ──────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # → Work2/
sys.path.insert(0, str(ROOT))

from models import svmAlgorithm
from parser import load_fold
from preprocessing import preprocess, preprocess_hypothyroid
from KNNClassifier import wilcoxon_test, friedman_test, plot_critical_difference_diagram

OUTPUT_DIR = ROOT / "results" / "svm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── parameter grid ─────────────────────────────────────────────────────────
KERNELS  = ["linear", "rbf", "poly"]
C_VALUES = [0.1, 1.0, 10.0, 100.0]
GAMMAS   = ["scale", 0.01, 0.1, 1.0]   # ignored for linear
DEGREES  = [2, 3]                        # only for poly


def _build_configs():
    configs = []
    for kernel in KERNELS:
        if kernel == "linear":
            for C in C_VALUES:
                name = f"SVM(kernel={kernel}, C={C})"
                configs.append((name, dict(kernel=kernel, C=C, gamma="scale", degree=3)))
        elif kernel == "rbf":
            for C, gamma in itertools.product(C_VALUES, GAMMAS):
                name = f"SVM(kernel={kernel}, C={C}, gamma={gamma})"
                configs.append((name, dict(kernel=kernel, C=C, gamma=gamma, degree=3)))
        elif kernel == "poly":
            for C, gamma, degree in itertools.product(C_VALUES, GAMMAS, DEGREES):
                name = f"SVM(kernel={kernel}, C={C}, gamma={gamma}, degree={degree})"
                configs.append((name, dict(kernel=kernel, C=C, gamma=gamma, degree=degree)))
    return configs


METRIC_COLS = [
    "accuracy", "f1_macro", "n_correct", "n_incorrect",
    "training_time", "prediction_time", "peak_memory_mb",
    "n_stored", "storage_pct",
]
PARAM_COLS = ["kernel", "C", "gamma", "degree"]


def run_svm_experiments(dataset_name, n_folds=10):
    preprocess_fn = preprocess_hypothyroid if dataset_name == "hypothyroid" else preprocess
    configs = _build_configs()
    print(f"\nSVM on '{dataset_name}': {len(configs)} configs × {n_folds} folds = {len(configs)*n_folds} runs")

    fold_rows = []
    for fold in range(n_folds):
        print(f"  fold {fold+1}/{n_folds} …")
        X_train, y_train, X_test, y_test = load_fold(dataset_name, fold)
        X_train, X_test = preprocess_fn(X_train, X_test)

        for name, params in configs:
            res = svmAlgorithm(X_train, y_train, X_test, y_test, **params)
            fold_rows.append({
                "dataset": dataset_name, "fold": fold,
                "algorithm": "SVM", "model": name,
                **params,
                "n_test": len(y_test),
                **{k: res[k] for k in METRIC_COLS},
            })

    df = pd.DataFrame(fold_rows)
    df["prediction_time_ms"] = df["prediction_time"] * 1000.0
    fold_cols = [
        "fold",
        "model",
        "correct_classification_count",
        "misclassification_count",
        "accuracy",
        "prediction_time_ms",
    ]
    df_save = df.rename(columns={
        "n_correct": "correct_classification_count",
        "n_incorrect": "misclassification_count",
    })[fold_cols]
    fold_csv = OUTPUT_DIR / f"{dataset_name}_fold_results.csv"
    df_save.to_csv(fold_csv, index=False)
    print(f"[Saved] {fold_csv}")

    mean_df = df.groupby("model", as_index=False)[["accuracy", "prediction_time_ms"]].mean()
    std_df  = df.groupby("model", as_index=False)["accuracy"].std()
    mean_df["accuracy_std"] = std_df["accuracy"]
    summary_cols = [
        "model",
        "accuracy",
        "accuracy_std",
        "prediction_time_ms",
    ]
    mean_df = mean_df.sort_values("accuracy", ascending=False)[summary_cols]

    summary_csv = OUTPUT_DIR / f"{dataset_name}_summary_results.csv"
    mean_df.to_csv(summary_csv, index=False)
    print(f"[Saved] {summary_csv}")

    # ── Statistical analysis: top configs ─────────────────────────────────
    top_n = min(8, len(mean_df))
    top_models = list(mean_df["model"].head(top_n))
    pivot = df[df["model"].isin(top_models)].pivot(
        index="fold", columns="model", values="accuracy"
    )[top_models]

    if top_n >= 3:
        fr = friedman_test(pivot, higher_is_better=True)
        fr["summary_table"].to_csv(
            OUTPUT_DIR / f"{dataset_name}_stats_top_configs.csv", index=False
        )
        plot_critical_difference_diagram(
            fr,
            title=f"{dataset_name.capitalize()}: Top SVM Configs — Accuracy",
            save_path=str(OUTPUT_DIR / f"{dataset_name}_cd_top_configs.png"),
        )
        print(f"Friedman chi2={fr['friedman_chi2']:.4f} p={fr['friedman_p_value']:.4f}")

    # Wilcoxon: #1 vs #2
    if top_n >= 2:
        s1 = df[df["model"] == top_models[0]].sort_values("fold")["accuracy"].values
        s2 = df[df["model"] == top_models[1]].sort_values("fold")["accuracy"].values
        w = wilcoxon_test(s1, s2, name_a=top_models[0], name_b=top_models[1])
        pd.DataFrame([w]).to_csv(
            OUTPUT_DIR / f"{dataset_name}_wilcoxon_best_vs_second.csv", index=False
        )
        print(f"Wilcoxon #1 vs #2: W={w['wilcoxon_stat']:.4f} p={w['p_value']:.4f} "
              f"significant={w['significant']}")

    return df, mean_df


if __name__ == "__main__":
    for ds in ("soybean", "hypothyroid"):
        run_svm_experiments(ds)
