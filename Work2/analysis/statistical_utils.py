"""Statistical analysis utilities and helpers for model selection and evaluation notebooks.

Provides unified loaders, Demšar (2006) test wrappers, and plotting helpers
for analyzing 10-fold CV results of kNN, SVM, Random Forest, and Instance Reduction.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Path anchoring
ROOT = Path(__file__).resolve().parent.parent   # → Work2/
RESULTS_DIR = ROOT / "results"

from KNNClassifier import (
    friedman_test,
    plot_critical_difference_diagram,
    wilcoxon_test,
)


# ── 1. Result Loaders ────────────────────────────────────────────────────────

def get_results_dir() -> Path:
    """Return path to results/ directory."""
    return RESULTS_DIR


def load_fold_results(algo: str, dataset: str) -> pd.DataFrame:
    """Load per-fold results CSV for a given algorithm and dataset.
    
    algo: 'knn' | 'svm' | 'rf' | 'reduction'
    dataset: 'soybean' | 'hypothyroid'
    """
    algo_lower = algo.lower()
    if algo_lower == "reduction":
        path = RESULTS_DIR / "reduction" / f"{dataset}_reduction_fold_results.csv"
    else:
        path = RESULTS_DIR / algo_lower / f"{dataset}_fold_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fold results not found at: {path}")
    df = pd.read_csv(path)
    if "correct_classification_count" in df.columns and "n_correct" not in df.columns:
        df["n_correct"] = df["correct_classification_count"]
    if "misclassification_count" in df.columns and "n_incorrect" not in df.columns:
        df["n_incorrect"] = df["misclassification_count"]
    if "prediction_time_ms" in df.columns and "prediction_time" not in df.columns:
        df["prediction_time"] = df["prediction_time_ms"] / 1000.0
    elif "prediction_time" in df.columns and "prediction_time_ms" not in df.columns:
        df["prediction_time_ms"] = df["prediction_time"] * 1000.0
    return df


def load_summary_results(algo: str, dataset: str) -> pd.DataFrame:
    """Load summary results CSV for a given algorithm and dataset."""
    algo_lower = algo.lower()
    if algo_lower == "reduction":
        path = RESULTS_DIR / "reduction" / f"{dataset}_reduction_summary_results.csv"
    else:
        path = RESULTS_DIR / algo_lower / f"{dataset}_summary_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Summary results not found at: {path}")
    df = pd.read_csv(path)
    if "prediction_time_ms" in df.columns and "prediction_time" not in df.columns:
        df["prediction_time"] = df["prediction_time_ms"] / 1000.0
    elif "prediction_time" in df.columns and "prediction_time_ms" not in df.columns:
        df["prediction_time_ms"] = df["prediction_time"] * 1000.0
    if algo_lower == "knn" and "k" not in df.columns:
        pattern = r"k=(\d+),\s*metric=([^,]+),\s*voting=([^,]+),\s*retention=([^,\)]+)"
        extracted = df["model"].str.extract(pattern)
        extracted.columns = ["k", "distance_metric", "voting_scheme", "retention_policy"]
        extracted["k"] = extracted["k"].astype(int)
        for col in extracted.columns:
            if col not in df.columns:
                df[col] = extracted[col]
    elif algo_lower == "rf" and "n_estimators" not in df.columns:
        pattern = r"n=(\d+),\s*crit=([^,]+),\s*depth=([^,\)]+)"
        extracted = df["model"].str.extract(pattern)
        extracted.columns = ["n_estimators", "criterion", "max_depth"]
        extracted["n_estimators"] = extracted["n_estimators"].astype(int)
        for col in extracted.columns:
            if col not in df.columns:
                df[col] = extracted[col]
    elif algo_lower == "svm" and "kernel" not in df.columns:
        pattern = r"kernel=([^,]+),\s*C=([^,\)]+)"
        extracted = df["model"].str.extract(pattern)
        extracted.columns = ["kernel", "C"]
        extracted["C"] = extracted["C"].astype(float)
        for col in extracted.columns:
            if col not in df.columns:
                df[col] = extracted[col]
    elif algo_lower == "reduction":
        if "reduction_method" not in df.columns:
            pattern = r"reduction=([a-zA-Z0-9_]+)"
            df["reduction_method"] = df["model"].str.extract(pattern)[0]
        if "algorithm" not in df.columns:
            def _parse_algo(m):
                m_str = str(m)
                if m_str.startswith("SVM"):
                    return "Best SVM"
                elif m_str.startswith("RF"):
                    return "Best RF"
                elif m_str.startswith("KNN"):
                    if "retention=always_retain" in m_str:
                        return "Best kNN (AR)"
                    return "Best kNN (NR)"
                return "Unknown"
            df["algorithm"] = df["model"].apply(_parse_algo)
        if "reduction_time" not in df.columns:
            df["reduction_time"] = 0.0
        if "f1_macro" not in df.columns:
            df["f1_macro"] = df["accuracy"]
    return df


# ── 2. Demšar (2006) Statistical Analysis Helpers ────────────────────────────

def run_friedman_analysis(
    pivot_df: pd.DataFrame,
    higher_is_better: bool = True,
    title: str = "Critical Difference Diagram",
    save_path: Optional[str] = None,
) -> Dict:
    """Execute Friedman test, Iman-Davenport correction, CD calculation, and optional CD plot."""
    res = friedman_test(pivot_df, higher_is_better=higher_is_better)
    if save_path:
        plot_critical_difference_diagram(res, title=title, save_path=save_path)
    return res


def run_holm_adjusted_wilcoxon(
    pivot_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run all pairwise Wilcoxon signed-rank tests across models with Holm step-down correction.
    
    Parameters
    ----------
    pivot_df : pd.DataFrame
        Rows = folds (10), Columns = model names, Values = metric scores.
    """
    models = list(pivot_df.columns)
    k = len(models)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            m_a = models[i]
            m_b = models[j]
            s_a = pivot_df[m_a].values
            s_b = pivot_df[m_b].values
            w = wilcoxon_test(s_a, s_b, name_a=m_a, name_b=m_b, alpha=alpha)
            pairs.append(w)

    df_pairs = pd.DataFrame(pairs)
    if df_pairs.empty:
        return df_pairs

    # Holm-Bonferroni step-down correction
    df_pairs = df_pairs.sort_values("p_value").reset_index(drop=True)
    m = len(df_pairs)
    adjusted_p = []
    significant_holm = []
    current_max = 0.0

    for idx, row in df_pairs.iterrows():
        p_val = row["p_value"]
        # Holm adjusted p-value = min(1, max(previous_adj, (m - idx) * p_val))
        adj_p = min(1.0, max(current_max, (m - idx) * p_val))
        current_max = adj_p
        adjusted_p.append(adj_p)
        significant_holm.append(adj_p < alpha)

    df_pairs["p_value_holm"] = adjusted_p
    df_pairs["significant_holm"] = significant_holm
    return df_pairs


def select_best_configuration(
    df_folds: pd.DataFrame,
    df_summary: pd.DataFrame,
    dataset: str,
    algo_name: str,
    top_n: Optional[int] = None,
    metric: str = "accuracy",
    secondary_metric: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Select the best configuration for an algorithm on a dataset using Demšar (2006).
    
    Parameters
    ----------
    df_folds : pd.DataFrame
        Per-fold evaluations.
    df_summary : pd.DataFrame
        Summary evaluations across folds.
    dataset : str
        'soybean' or 'hypothyroid'.
    algo_name : str
        'KNN', 'SVM', or 'RF'.
    top_n : int, optional
        Number of top candidate models to include in statistical tests. Default None (all).
    metric : str
        'accuracy' or 'prediction_time' (or 'prediction_time_ms').
    secondary_metric : str, optional
        Tie-breaking metric. If metric is accuracy, defaults to prediction_time.
        If metric is prediction_time, defaults to accuracy.
    output_dir : Path, optional
        Target directory for artifacts.
    """
    out_dir = Path(output_dir) if output_dir else (RESULTS_DIR / algo_name.lower())
    out_dir.mkdir(parents=True, exist_ok=True)

    is_efficiency = metric in ["prediction_time", "prediction_time_ms", "training_time"]
    if secondary_metric is None:
        secondary_metric = "accuracy" if is_efficiency else "prediction_time"

    metric_higher_is_better = not is_efficiency
    sec_is_time = secondary_metric in ["prediction_time", "prediction_time_ms", "training_time"]
    sec_higher_is_better = not sec_is_time

    # Primary and secondary sorting directions
    ranked_summary = df_summary.sort_values(
        by=[metric, secondary_metric],
        ascending=[not metric_higher_is_better, not sec_higher_is_better]
    )

    if top_n is not None and top_n > 0:
        candidate_models = list(ranked_summary["model"].head(top_n))
        title_scope = f"Top {top_n} Configs"
    else:
        candidate_models = list(ranked_summary["model"].unique())
        title_scope = f"All {len(candidate_models)} Configs"

    # Primary metric pivot and Friedman test
    pivot_primary = df_folds[df_folds["model"].isin(candidate_models)].pivot(
        index="fold", columns="model", values=metric
    )[candidate_models]

    criterion_tag = "efficiency" if is_efficiency else "accuracy"
    fr_primary = friedman_test(pivot_primary, higher_is_better=metric_higher_is_better)
    cd_prim_path = out_dir / f"{dataset}_cd_{criterion_tag}.png"
    metric_label = "Prediction Time (s)" if is_efficiency else "Accuracy"
    plot_critical_difference_diagram(
        fr_primary,
        title=f"{dataset.capitalize()} {algo_name}: {title_scope} — {metric_label}",
        save_path=str(cd_prim_path),
    )
    fr_primary["summary_table"].to_csv(out_dir / f"{dataset}_friedman_{criterion_tag}.csv", index=False)

    # Secondary metric pivot and Friedman test
    pivot_sec = df_folds[df_folds["model"].isin(candidate_models)].pivot(
        index="fold", columns="model", values=secondary_metric
    )[candidate_models]

    fr_sec = friedman_test(pivot_sec, higher_is_better=sec_higher_is_better)

    # Wilcoxon #1 vs #2 on primary metric
    best_m = candidate_models[0]
    sec_m = candidate_models[1] if len(candidate_models) > 1 else candidate_models[0]
    w_test = wilcoxon_test(
        pivot_primary[best_m].values,
        pivot_primary[sec_m].values,
        name_a=best_m,
        name_b=sec_m,
    )
    pd.DataFrame([w_test]).to_csv(out_dir / f"{dataset}_wilcoxon_{criterion_tag}_best_vs_second.csv", index=False)

    # Best config record
    best_row = ranked_summary.iloc[0].to_dict()
    best_config_meta = {
        "dataset": dataset,
        "algorithm": algo_name,
        "criterion": criterion_tag,
        "best_model": best_m,
        "n_configurations_compared": len(candidate_models),
        "mean_accuracy": float(best_row.get("accuracy", 0.0)),
        "std_accuracy": float(best_row.get("accuracy_std", 0.0)),
        "mean_f1_macro": float(best_row.get("f1_macro", 0.0)),
        "mean_prediction_time": float(best_row.get("prediction_time", 0.0)),
        "mean_prediction_time_ms": float(best_row.get("prediction_time_ms", best_row.get("prediction_time", 0.0) * 1000.0)),
        "mean_training_time": float(best_row.get("training_time", 0.0)),
        "mean_storage_pct": float(best_row.get("storage_pct", 100.0)),
        "friedman_chi2": fr_primary["friedman_chi2"],
        "friedman_p": fr_primary["friedman_p_value"],
        "friedman_significant": fr_primary["significant"],
        "wilcoxon_best_vs_second_p": w_test["p_value"],
        "wilcoxon_best_vs_second_significant": w_test["significant"],
        # Backward-compatible fields
        "friedman_chi2_accuracy": fr_primary["friedman_chi2"] if not is_efficiency else fr_sec["friedman_chi2"],
        "friedman_p_accuracy": fr_primary["friedman_p_value"] if not is_efficiency else fr_sec["friedman_p_value"],
        "friedman_significant_accuracy": fr_primary["significant"] if not is_efficiency else fr_sec["significant"],
    }

    # Save JSON files
    json_path = out_dir / f"{dataset}_best_config_{criterion_tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_config_meta, f, indent=2)
    if not is_efficiency:
        with open(out_dir / f"{dataset}_best_config.json", "w", encoding="utf-8") as f:
            json.dump(best_config_meta, f, indent=2)

    return {
        "best_model": best_m,
        "second_model": sec_m,
        "criterion": criterion_tag,
        "meta": best_config_meta,
        "friedman_primary": fr_primary,
        "friedman_secondary": fr_sec,
        "wilcoxon_best_vs_second": w_test,
        # Backward-compatible references
        "friedman_accuracy": fr_primary if not is_efficiency else fr_sec,
        "friedman_time": fr_sec if not is_efficiency else fr_primary,
    }

