"""Automated 10-fold kNN experiments, statistical tests (Demšar 2006), and plots.

Grid search: k - distance_metric - voting_scheme - retention_policy
Outputs → results/knn/{dataset}_fold_results.csv
           results/knn/{dataset}_summary_results.csv
           results/knn/{dataset}_stats_*.csv
           results/knn/{dataset}_cd_*.png
"""

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # → Work2/
sys.path.insert(0, str(ROOT))

from models import kNNAlgorithm
from KNNClassifier import (
    KNNClassifier,
    friedman_test,
    plot_critical_difference_diagram,
    wilcoxon_test,
)
from parser import load_fold
from preprocessing import preprocess, preprocess_hypothyroid

DEFAULT_OUTPUT_DIR = ROOT / "results" / "knn"


def run_knn_experiments(
    dataset_name="soybean",
    k_values=(3, 5, 7),
    distance_metrics=("euclidean", "manhattan", "canberra", "heom"),
    voting_schemes=("majority", "inverse_distance"),
    retention_policies=("never_retain", "always_retain", "different_class", "oblivion_by_goodness"),
    n_folds=10,
    output_dir=None,
):
    """Run all grid combinations across all folds and save CSV results."""
    output_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    preprocess_fn = preprocess_hypothyroid if dataset_name == "hypothyroid" else preprocess

    configs = []
    for k, metric, voting, retention in itertools.product(
        k_values, distance_metrics, voting_schemes, retention_policies
    ):
        name = f"KNN(k={k}, metric={metric}, voting={voting}, retention={retention})"
        configs.append((name, {
            "k": k,
            "distance_metric": metric,
            "voting_scheme": voting,
            "retention_policy": retention,
        }))

    print(f"Starting evaluation on '{dataset_name}': {len(configs)} configurations x {n_folds} folds = {len(configs) * n_folds} runs")

    fold_rows = []
    for fold in range(n_folds):
        print(f"  -> Running Fold {fold + 1}/{n_folds}...")
        X_train, y_train, X_test, y_test = load_fold(dataset_name, fold)
        X_train, X_test = preprocess_fn(X_train, X_test)

        for name, params in configs:
            res = kNNAlgorithm(X_train, y_train, X_test, y_test, **params)
            fold_rows.append({
                "dataset": dataset_name,
                "fold": fold,
                "algorithm": "KNN",
                "model": name,
                **params,
                "n_test": len(y_test),
                "accuracy": res["accuracy"],
                "f1_macro": res["f1_macro"],
                "n_correct": res["n_correct"],
                "n_incorrect": res["n_incorrect"],
                "n_stored": res["n_stored"],
                "storage_pct": res["storage_pct"],
                "training_time": res["training_time"],
                "prediction_time": res["prediction_time"],
                "peak_memory_mb": res["peak_memory_mb"],
            })

    # 1. Detailed per-fold DataFrame
    df_folds = pd.DataFrame(fold_rows)
    df_folds["prediction_time_ms"] = df_folds["prediction_time"] * 1000.0
    fold_cols = [
        "fold",
        "model",
        "correct_classification_count",
        "misclassification_count",
        "accuracy",
        "prediction_time_ms",
    ]
    df_folds_save = df_folds.rename(columns={
        "n_correct": "correct_classification_count",
        "n_incorrect": "misclassification_count",
    })[fold_cols]
    fold_csv = output_path / f"{dataset_name}_fold_results.csv"
    df_folds_save.to_csv(fold_csv, index=False)
    print(f"[Saved] Detailed fold results: {fold_csv}")

    # 2. Aggregated summary DataFrame across folds
    summary_mean = df_folds.groupby("model", as_index=False)[["accuracy", "prediction_time_ms"]].mean()
    summary_std = df_folds.groupby("model", as_index=False)["accuracy"].std()
    summary_mean["accuracy_std"] = summary_std["accuracy"]

    summary_cols = [
        "model",
        "accuracy",
        "accuracy_std",
        "prediction_time_ms",
    ]
    summary_mean = summary_mean.sort_values(by="accuracy", ascending=False)[summary_cols]
    summary_csv = output_path / f"{dataset_name}_summary_results.csv"
    summary_mean.to_csv(summary_csv, index=False)
    print(f"[Saved] Summary results: {summary_csv}")

    return df_folds, summary_mean


def _ensure_knn_columns(df):
    df = df.copy()
    if "storage_pct" not in df.columns:
        df["storage_pct"] = 100.0
    if "distance_metric" not in df.columns or "k" not in df.columns or "voting_scheme" not in df.columns or "retention_policy" not in df.columns:
        pattern = r"k=(\d+),\s*metric=([^,]+),\s*voting=([^,]+),\s*retention=([^,\)]+)"
        extracted = df["model"].str.extract(pattern)
        extracted.columns = ["k", "distance_metric", "voting_scheme", "retention_policy"]
        extracted["k"] = extracted["k"].astype(int)
        for col in extracted.columns:
            if col not in df.columns:
                df[col] = extracted[col]
    return df


def run_statistical_analysis(df_folds, df_summary, dataset_name="soybean", output_dir=None):
    """Run Friedman, Nemenyi, Bonferroni-Dunn, and Wilcoxon tests across kNN parameters."""
    df_folds = _ensure_knn_columns(df_folds)
    df_summary = _ensure_knn_columns(df_summary)
    output_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Statistical Analysis (Demšar 2006) for {dataset_name.upper()} ===")

    # 1. Distance Metric Comparison
    pivot_metric = df_folds.groupby(["fold", "distance_metric"])["accuracy"].mean().unstack()
    fr_metric = friedman_test(pivot_metric, higher_is_better=True)
    print("\n1. Distance Metrics (Friedman Test):")
    print(f"   Chi2 = {fr_metric['friedman_chi2']:.4f}, p = {fr_metric['friedman_p_value']:.4f}")
    fr_metric["summary_table"].to_csv(output_path / f"{dataset_name}_stats_distance_metrics.csv", index=False)
    plot_critical_difference_diagram(
        fr_metric,
        title=f"{dataset_name.capitalize()}: Distance Metrics CD Diagram",
        save_path=str(output_path / f"{dataset_name}_cd_distance_metrics.png"),
    )

    # 2. K Parameter Comparison (k = 3, 5, 7)
    pivot_k = df_folds.groupby(["fold", "k"])["accuracy"].mean().unstack()
    pivot_k.columns = [f"k={col}" for col in pivot_k.columns]
    fr_k = friedman_test(pivot_k, higher_is_better=True)
    print("\n2. Hyperparameter K (Friedman Test):")
    print(f"   Chi2 = {fr_k['friedman_chi2']:.4f}, p = {fr_k['friedman_p_value']:.4f}")
    fr_k["summary_table"].to_csv(output_path / f"{dataset_name}_stats_k_values.csv", index=False)
    plot_critical_difference_diagram(
        fr_k,
        title=f"{dataset_name.capitalize()}: K Parameter CD Diagram",
        save_path=str(output_path / f"{dataset_name}_cd_k_values.png"),
    )

    # 3. Voting Scheme Comparison (Wilcoxon Signed-Rank Test)
    pivot_voting = df_folds.groupby(["fold", "voting_scheme"])["accuracy"].mean().unstack()
    wil_voting = wilcoxon_test(
        pivot_voting["inverse_distance"],
        pivot_voting["majority"],
        name_a="inverse_distance",
        name_b="majority",
    )
    print("\n3. Voting Schemes (Wilcoxon Signed-Rank Test):")
    print(f"   W = {wil_voting['wilcoxon_stat']:.4f}, p = {wil_voting['p_value']:.4f}")
    pd.DataFrame([wil_voting]).to_csv(output_path / f"{dataset_name}_stats_voting_schemes.csv", index=False)

    # 4. Retention Policy Comparison (NR, AR, DC, OG)
    pivot_ret = df_folds.groupby(["fold", "retention_policy"])["accuracy"].mean().unstack()
    fr_ret = friedman_test(pivot_ret, higher_is_better=True)
    print("\n4. Retention Policies (Friedman Test - Accuracy):")
    print(f"   Chi2 = {fr_ret['friedman_chi2']:.4f}, p = {fr_ret['friedman_p_value']:.4f}")
    fr_ret["summary_table"].to_csv(output_path / f"{dataset_name}_stats_retention_policies.csv", index=False)
    plot_critical_difference_diagram(
        fr_ret,
        title=f"{dataset_name.capitalize()}: Retention Policies CD Diagram",
        save_path=str(output_path / f"{dataset_name}_cd_retention_policies.png"),
    )

    # 5. Top 8 Overall Configurations Comparison
    top8_models = list(df_summary["model"].head(8))
    pivot_top8 = df_folds[df_folds["model"].isin(top8_models)].pivot(index="fold", columns="model", values="accuracy")[top8_models]
    short_names = {}
    for m in top8_models:
        row = df_summary[df_summary["model"] == m].iloc[0]
        ret_short = {"never_retain": "NR", "always_retain": "AR", "different_class": "DC", "oblivion_by_goodness": "OG"}[row["retention_policy"]]
        vot_short = "maj" if row["voting_scheme"] == "majority" else "invD"
        met_short = row["distance_metric"][:4]
        short_names[m] = f"k{row['k']}-{met_short}-{vot_short}-{ret_short}"

    pivot_top8_short = pivot_top8.rename(columns=short_names)
    fr_top8 = friedman_test(pivot_top8_short, higher_is_better=True)
    print("\n5. Top 8 Overall kNN Configurations (Friedman Test):")
    print(f"   Chi2 = {fr_top8['friedman_chi2']:.4f}, p = {fr_top8['friedman_p_value']:.4f}")
    fr_top8["summary_table"].to_csv(output_path / f"{dataset_name}_stats_top8_models.csv", index=False)
    plot_critical_difference_diagram(
        fr_top8,
        title=f"{dataset_name.capitalize()}: Top 8 kNN Configs CD Diagram",
        save_path=str(output_path / f"{dataset_name}_cd_top8_models.png"),
    )

    # Pairwise Wilcoxon #1 vs #2, #1 vs Best NR
    best_model = top8_models[0]
    second_model = top8_models[1]
    best_nr_model = df_summary[df_summary["retention_policy"] == "never_retain"].iloc[0]["model"]

    s1 = df_folds[df_folds["model"] == best_model].sort_values("fold")["accuracy"].values
    s2 = df_folds[df_folds["model"] == second_model].sort_values("fold")["accuracy"].values
    s_nr = df_folds[df_folds["model"] == best_nr_model].sort_values("fold")["accuracy"].values

    w_1_2 = wilcoxon_test(s1, s2, name_a=best_model, name_b=second_model)
    w_1_nr = wilcoxon_test(s1, s_nr, name_a=best_model, name_b=best_nr_model)
    pd.DataFrame([w_1_2, w_1_nr]).to_csv(output_path / f"{dataset_name}_stats_wilcoxon_best.csv", index=False)

    # 6. Summary Bar Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"k-NN Parameter Analysis on {dataset_name.capitalize()} (10-Fold CV)", fontsize=14, fontweight="bold")

    # (a) By Distance Metric
    m_grp = df_folds.groupby("distance_metric")["accuracy"].agg(["mean", "std"]).reindex(["euclidean", "manhattan", "canberra", "heom"])
    axes[0, 0].bar(m_grp.index, m_grp["mean"] * 100, yerr=m_grp["std"] * 100, capsize=4, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Accuracy (%) by Distance Metric")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(max(0, (m_grp["mean"].min() - 0.08) * 100), min(100, (m_grp["mean"].max() + 0.05) * 100))
    axes[0, 0].grid(axis="y", ls="--", alpha=0.5)

    # (b) By K and Voting Scheme
    kv_grp = df_folds.groupby(["k", "voting_scheme"])["accuracy"].mean().unstack() * 100
    kv_grp.plot(kind="bar", ax=axes[0, 1], edgecolor="black", rot=0)
    axes[0, 1].set_title("Accuracy (%) by K and Voting Scheme")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_ylim(max(0, kv_grp.values.min() - 5), min(100, kv_grp.values.max() + 3))
    axes[0, 1].grid(axis="y", ls="--", alpha=0.5)

    # (c) Accuracy & Storage (%) by Retention Policy
    r_grp = df_folds.groupby("retention_policy")[["accuracy", "storage_pct"]].mean().reindex(
        ["never_retain", "always_retain", "different_class", "oblivion_by_goodness"]
    )
    x_pos = np.arange(len(r_grp))
    width = 0.35
    ax_acc = axes[1, 0]
    ax_sto = ax_acc.twinx()
    ax_acc.bar(x_pos - width / 2, r_grp["accuracy"] * 100, width, label="Accuracy (%)", color="seagreen", edgecolor="black")
    ax_sto.bar(x_pos + width / 2, r_grp["storage_pct"], width, label="Storage (%)", color="sandybrown", edgecolor="black")
    ax_acc.set_xticks(x_pos)
    ax_acc.set_xticklabels(["NR", "AR", "DC", "OG"])
    ax_acc.set_ylabel("Accuracy (%)", color="seagreen")
    ax_sto.set_ylabel("Final Storage (%)", color="chocolate")
    ax_acc.set_ylim(max(0, r_grp["accuracy"].min() * 100 - 5), min(100, r_grp["accuracy"].max() * 100 + 3))
    ax_sto.set_ylim(0, max(120, r_grp["storage_pct"].max() + 10))
    ax_acc.set_title("Accuracy (%) vs Storage (%) by Retention Policy")
    ax_acc.grid(axis="y", ls="--", alpha=0.5)

    # (d) Prediction Time (ms) by Retention Policy & Metric
    t_grp = (df_folds.groupby(["retention_policy", "distance_metric"])["prediction_time"].mean().unstack() * 1000).reindex(
        ["never_retain", "always_retain", "different_class", "oblivion_by_goodness"]
    )
    t_grp.index = ["NR", "AR", "DC", "OG"]
    t_grp.plot(kind="bar", ax=axes[1, 1], edgecolor="black", rot=0)
    axes[1, 1].set_title("Average Prediction Time (ms) by Retention & Metric")
    axes[1, 1].set_ylabel("Prediction Time (ms)")
    axes[1, 1].grid(axis="y", ls="--", alpha=0.5)

    plt.tight_layout()
    summary_plot_path = output_path / f"{dataset_name}_parameter_analysis.png"
    plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Parameter analysis plot: {summary_plot_path}")


if __name__ == "__main__":
    for ds in ("soybean", "hypothyroid"):
        df_folds, df_summary = run_knn_experiments(dataset_name=ds)
        run_statistical_analysis(df_folds, df_summary, dataset_name=ds)
