"""Automated 10-fold Instance Reduction experiments (Points 6.d & 6.e) and Demšar (2006) statistical tests.

Evaluates:
1. Best kNN (`reduction_kNNAlgorithm`) with 'none' (100% storage), 'gcnn', 'rnge', and 'del'
   (including both the best static Never Retain configuration and the best dynamic retention configuration).
2. Best SVM (`svmAlgorithm`) trained on full training set ('none') vs reduced training sets ('gcnn', 'rnge', 'del').
3. Best Random Forest (`rfAlgorithm`) trained on full training set ('none') vs reduced training sets ('gcnn', 'rnge', 'del').

Outputs → results/reduction/{dataset}_reduction_fold_results.csv
           results/reduction/{dataset}_reduction_summary_results.csv
           results/reduction/{dataset}_stats_*.csv
           results/reduction/{dataset}_cd_*.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # → Work2/
sys.path.insert(0, str(ROOT))

from KNNClassifier import (
    friedman_test,
    plot_critical_difference_diagram,
    wilcoxon_test,
)
from models import rfAlgorithm, svmAlgorithm, reduction_kNNAlgorithm
from parser import load_fold
from preprocessing import preprocess, preprocess_hypothyroid
from reduction import REDUCTION_METHODS, reduce

DEFAULT_OUTPUT_DIR = ROOT / "results" / "reduction"

# Best configurations per dataset determined from the 10-fold parameter searches
BEST_CONFIGS = {
    "soybean": {
        "knn_best_overall": {
            "k": 3,
            "distance_metric": "canberra",
            "voting_scheme": "majority",
            "retention_policy": "always_retain",
        },
        "knn_best_static": {
            "k": 7,
            "distance_metric": "manhattan",  # ties Canberra on accuracy, wins on prediction time
            "voting_scheme": "inverse_distance",
            "retention_policy": "never_retain",
        },
        "svm": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": 0.1,
            "degree": 3,
        },
        "rf": {
            "n_estimators": 200,
            "criterion": "log_loss",
            "max_depth": 10,
        },
    },
    "hypothyroid": {
        "knn_best_overall": {
            "k": 5,
            "distance_metric": "canberra",
            "voting_scheme": "inverse_distance",
            "retention_policy": "always_retain",
        },
        "knn_best_static": {
            "k": 5,
            "distance_metric": "canberra",
            "voting_scheme": "inverse_distance",
            "retention_policy": "never_retain",
        },
        "svm": {
            "kernel": "poly",
            "C": 100.0,
            "gamma": 1.0,
            "degree": 2,
        },
        "rf": {
            "n_estimators": 200,
            "criterion": "gini",
            "max_depth": None,
        },
    },
}



def run_reduction_experiments_for_dataset(dataset_name="soybean", n_folds=10, output_dir=None):
    output_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    preprocess_fn = preprocess_hypothyroid if dataset_name == "hypothyroid" else preprocess
    cfg = BEST_CONFIGS[dataset_name]

    knn_overall_cfg = cfg["knn_best_overall"]
    knn_static_cfg = cfg["knn_best_static"]
    svm_cfg = cfg["svm"]
    rf_cfg = cfg["rf"]

    print(f"\n==================================================================")
    print(f"Running 10-Fold Instance Reduction Experiments on '{dataset_name.upper()}'")
    print(f"==================================================================")

    fold_rows = []
    for fold in range(n_folds):
        print(f"  -> Fold {fold + 1}/{n_folds}...")
        X_train, y_train, X_test, y_test = load_fold(dataset_name, fold)
        X_train, X_test = preprocess_fn(X_train, X_test)
        X_tr_arr = np.asarray(X_train, dtype=float)
        y_tr_arr = np.asarray(y_train)
        X_te_arr = np.asarray(X_test, dtype=float)
        y_te_arr = np.asarray(y_test)

        for method in REDUCTION_METHODS:
            # 1. Best kNN (Static NR baseline - pure effect of training set reduction)
            res_knn_nr = reduction_kNNAlgorithm(
                X_tr_arr,
                y_tr_arr,
                X_te_arr,
                y_te_arr,
                method=method,
                **knn_static_cfg,
            )
            kept = res_knn_nr["kept_indices"]
            red_time = res_knn_nr["reduction_time"]
            storage_pct = res_knn_nr["storage_pct"]

            m_knn_nr = f"KNN(reduction={method}, k={knn_static_cfg['k']}, metric={knn_static_cfg['distance_metric']}, voting={knn_static_cfg['voting_scheme']}, retention={knn_static_cfg['retention_policy']})"
            fold_rows.append({
                "dataset": dataset_name,
                "fold": fold,
                "model": m_knn_nr,
                "algorithm": "Best kNN (NR)",
                "reduction_method": method,
                "accuracy": res_knn_nr["accuracy"],
                "f1_macro": res_knn_nr["f1_macro"],
                "n_correct": res_knn_nr["n_correct"],
                "n_incorrect": res_knn_nr["n_incorrect"],
                "n_test": len(y_te_arr),
                "n_stored": len(kept),
                "storage_pct": res_knn_nr["storage_pct"],
                "reduction_time": red_time,
                "prediction_time": res_knn_nr["prediction_time"],
                "training_time": res_knn_nr.get("training_time", 0.0),
                "peak_memory_mb": res_knn_nr["peak_memory_mb"],
            })

            # Subsample for SVM, RF, and kNN (AR) when the AR reduction would be identical
            X_tr_red = X_tr_arr[kept]
            y_tr_red = y_tr_arr[kept]

            # 2. Best kNN (Dynamic AR policy on reduced initial set)
            # Reduce again when the AR configuration changes what the reduction depends on
            # (the distance metric for every method; also k and voting for DEL)
            if method != "none" and (
                knn_overall_cfg["distance_metric"] != knn_static_cfg["distance_metric"]
                or (method == "del" and (
                    knn_overall_cfg["k"] != knn_static_cfg["k"]
                    or knn_overall_cfg["voting_scheme"] != knn_static_cfg["voting_scheme"]
                ))
            ):
                res_knn_ar = reduction_kNNAlgorithm(
                    X_tr_arr,
                    y_tr_arr,
                    X_te_arr,
                    y_te_arr,
                    method=method,
                    **knn_overall_cfg,
                )
            else:
                from models import kNNAlgorithm
                res_knn_ar = kNNAlgorithm(
                    X_tr_red,
                    y_tr_red,
                    X_te_arr,
                    y_te_arr,
                    **knn_overall_cfg,
                )
                res_knn_ar["reduction_time"] = red_time
                res_knn_ar["final_storage_pct"] = (res_knn_ar["n_stored"] / len(y_tr_arr) * 100.0) if len(y_tr_arr) > 0 else 100.0

            m_knn_ar = f"KNN(reduction={method}, k={knn_overall_cfg['k']}, metric={knn_overall_cfg['distance_metric']}, voting={knn_overall_cfg['voting_scheme']}, retention={knn_overall_cfg['retention_policy']})"
            fold_rows.append({
                "dataset": dataset_name,
                "fold": fold,
                "model": m_knn_ar,
                "algorithm": "Best kNN (AR)",
                "reduction_method": method,
                "accuracy": res_knn_ar["accuracy"],
                "f1_macro": res_knn_ar["f1_macro"],
                "n_correct": res_knn_ar["n_correct"],
                "n_incorrect": res_knn_ar["n_incorrect"],
                "n_test": len(y_te_arr),
                "n_stored": res_knn_ar["n_stored"],
                "storage_pct": res_knn_ar["final_storage_pct"],
                "reduction_time": res_knn_ar.get("reduction_time", red_time),
                "prediction_time": res_knn_ar["prediction_time"],
                "training_time": res_knn_ar.get("training_time", 0.0),
                "peak_memory_mb": res_knn_ar["peak_memory_mb"],
            })

            # 3. SVM trained on reduced set
            res_svm = svmAlgorithm(
                X_tr_red,
                y_tr_red,
                X_te_arr,
                y_te_arr,
                **svm_cfg,
            )
            gamma_val = svm_cfg.get("gamma")
            gamma_str = f", gamma={gamma_val}" if gamma_val is not None else ""
            m_svm = f"SVM(reduction={method}, kernel={svm_cfg['kernel']}, C={svm_cfg['C']}{gamma_str})"
            fold_rows.append({
                "dataset": dataset_name,
                "fold": fold,
                "model": m_svm,
                "algorithm": "Best SVM",
                "reduction_method": method,
                "accuracy": res_svm["accuracy"],
                "f1_macro": res_svm["f1_macro"],
                "n_correct": res_svm["n_correct"],
                "n_incorrect": res_svm["n_incorrect"],
                "n_test": len(y_te_arr),
                "n_stored": len(kept),
                "storage_pct": storage_pct,
                "reduction_time": red_time,
                "prediction_time": res_svm["prediction_time"],
                "training_time": res_svm["training_time"],
                "peak_memory_mb": res_svm["peak_memory_mb"],
            })

            # 4. Random Forest trained on reduced set
            res_rf = rfAlgorithm(
                X_tr_red,
                y_tr_red,
                X_te_arr,
                y_te_arr,
                **rf_cfg,
            )
            m_rf = f"RF(reduction={method}, n={rf_cfg['n_estimators']}, crit={rf_cfg['criterion']}, depth={rf_cfg['max_depth']})"
            fold_rows.append({
                "dataset": dataset_name,
                "fold": fold,
                "model": m_rf,
                "algorithm": "Best RF",
                "reduction_method": method,
                "accuracy": res_rf["accuracy"],
                "f1_macro": res_rf["f1_macro"],
                "n_correct": res_rf["n_correct"],
                "n_incorrect": res_rf["n_incorrect"],
                "n_test": len(y_te_arr),
                "n_stored": len(kept),
                "storage_pct": storage_pct,
                "reduction_time": red_time,
                "prediction_time": res_rf["prediction_time"],
                "training_time": res_rf["training_time"],
                "peak_memory_mb": res_rf["peak_memory_mb"],
            })

    # Detailed fold results
    df_folds = pd.DataFrame(fold_rows)
    df_folds["prediction_time_ms"] = df_folds["prediction_time"] * 1000.0
    fold_cols = [
        "fold",
        "model",
        "correct_classification_count",
        "misclassification_count",
        "accuracy",
        "storage_pct",
        "prediction_time_ms",
    ]
    df_folds_save = df_folds.rename(columns={
        "n_correct": "correct_classification_count",
        "n_incorrect": "misclassification_count",
    })[fold_cols]
    fold_csv = output_path / f"{dataset_name}_reduction_fold_results.csv"
    df_folds_save.to_csv(fold_csv, index=False)
    print(f"[Saved] Detailed fold results: {fold_csv}")

    # Aggregated summary results
    group_cols = ["model", "algorithm", "reduction_method"]
    num_cols = ["accuracy", "f1_macro", "storage_pct", "reduction_time", "prediction_time", "prediction_time_ms"]
    mean_df = df_folds.groupby(group_cols, as_index=False)[num_cols].mean()
    std_df = df_folds.groupby(group_cols, as_index=False)["accuracy"].std()
    mean_df["accuracy_std"] = std_df["accuracy"]

    summary_cols = [
        "model",
        "algorithm",
        "reduction_method",
        "accuracy",
        "accuracy_std",
        "f1_macro",
        "storage_pct",
        "reduction_time",
        "prediction_time",
        "prediction_time_ms",
    ]
    summary_mean = mean_df.sort_values(by="accuracy", ascending=False)
    summary_csv = output_path / f"{dataset_name}_reduction_summary_results.csv"
    summary_mean[summary_cols].to_csv(summary_csv, index=False)
    print(f"[Saved] Summary results: {summary_csv}")

    # Demšar (2006) Statistical Analysis
    print(f"\n=== Statistical Significance of Instance Reduction on {dataset_name.upper()} ===")

    # 1. Friedman Test & CD diagram for each classifier across reduction methods
    algorithms = ["Best kNN (NR)", "Best kNN (AR)", "Best SVM", "Best RF"]
    for algo in algorithms:
        df_sub = df_folds[df_folds["algorithm"] == algo]
        piv = df_sub.pivot(index="fold", columns="reduction_method", values="accuracy")[list(REDUCTION_METHODS)]

        if piv.shape[1] >= 3 and len(piv) >= 2:
            fr = friedman_test(piv, higher_is_better=True)
            print(f"\nFriedman test for '{algo}':")
            print(f"  Chi2 = {fr['friedman_chi2']:.4f}, p = {fr['friedman_p_value']:.4f}, "
                  f"Iman-Davenport F = {fr['iman_davenport_f']:.4f} (p = {fr['iman_davenport_p_value']:.4f})")

            safe_name = algo.lower().replace(" ", "_").replace("(", "").replace(")", "")
            fr["summary_table"].to_csv(output_path / f"{dataset_name}_stats_{safe_name}_reduction.csv", index=False)
            plot_critical_difference_diagram(
                fr,
                title=f"{dataset_name.capitalize()}: {algo} across Reduction Methods",
                save_path=str(output_path / f"{dataset_name}_cd_{safe_name}_reduction.png"),
            )

    # 2. Pairwise Wilcoxon Signed-Rank Test: Each reduction method vs 'none'
    wilcoxon_rows = []
    for algo in algorithms:
        df_sub = df_folds[df_folds["algorithm"] == algo]
        baseline = df_sub[df_sub["reduction_method"] == "none"].sort_values("fold")["accuracy"].values

        for method in ["gcnn", "rnge", "del"]:
            reduced = df_sub[df_sub["reduction_method"] == method].sort_values("fold")["accuracy"].values
            if len(reduced) >= 2:
                w_res = wilcoxon_test(
                    reduced,
                    baseline,
                    name_a=f"{algo}_{method}",
                    name_b=f"{algo}_none",
                )
                w_res["algorithm"] = algo
                w_res["reduction_method"] = method
                wilcoxon_rows.append(w_res)

    df_wilcoxon = pd.DataFrame(wilcoxon_rows)
    if not df_wilcoxon.empty:
        wilc_csv = output_path / f"{dataset_name}_stats_wilcoxon_vs_none.csv"
        df_wilcoxon.to_csv(wilc_csv, index=False)
        print(f"\n[Saved] Wilcoxon tests vs 'none': {wilc_csv}")

    # Generate 2x2 Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Instance Reduction Impact on {dataset_name.capitalize()} (10-Fold CV)", fontsize=14, fontweight="bold")

    # (a) Accuracy (%) across algorithms and reduction methods
    acc_piv = (summary_mean.pivot(index="reduction_method", columns="algorithm", values="accuracy") * 100).reindex(list(REDUCTION_METHODS))
    acc_piv.index = [m.upper() for m in acc_piv.index]
    acc_piv.plot(kind="bar", ax=axes[0, 0], edgecolor="black", rot=0)
    axes[0, 0].set_title("Classification Accuracy (%) by Reduction Method")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(max(0, acc_piv.values.min() - 8), min(102, acc_piv.values.max() + 3))
    axes[0, 0].grid(axis="y", ls="--", alpha=0.5)

    # (b) Storage (%) retained
    sto_piv = summary_mean[summary_mean["algorithm"] == "Best kNN (NR)"].set_index("reduction_method")["storage_pct"].reindex(list(REDUCTION_METHODS))
    bars = axes[0, 1].bar([m.upper() for m in sto_piv.index], sto_piv.values, color=["slategray", "steelblue", "mediumseagreen", "coral"], edgecolor="black")
    axes[0, 1].set_title("Training Storage Retained (%)")
    axes[0, 1].set_ylabel("Storage (%)")
    axes[0, 1].set_ylim(0, 115)
    axes[0, 1].grid(axis="y", ls="--", alpha=0.5)
    for b in bars:
        axes[0, 1].text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")

    # (c) kNN Prediction Time (ms)
    pred_piv = (summary_mean[summary_mean["algorithm"].isin(["Best kNN (NR)", "Best kNN (AR)"])].pivot(
        index="reduction_method", columns="algorithm", values="prediction_time"
    ) * 1000).reindex(list(REDUCTION_METHODS))
    pred_piv.index = [m.upper() for m in pred_piv.index]
    pred_piv.plot(kind="bar", ax=axes[1, 0], edgecolor="black", rot=0)
    axes[1, 0].set_title("k-NN Prediction / Problem-Solving Time (ms)")
    axes[1, 0].set_ylabel("Prediction Time (ms)")
    axes[1, 0].grid(axis="y", ls="--", alpha=0.5)

    # (d) Preprocessing Reduction Time (s)
    red_piv = summary_mean[summary_mean["algorithm"] == "Best kNN (NR)"].set_index("reduction_method")["reduction_time"].reindex(["gcnn", "rnge", "del"])
    bars_r = axes[1, 1].bar([m.upper() for m in red_piv.index], red_piv.values, color=["steelblue", "mediumseagreen", "coral"], edgecolor="black")
    axes[1, 1].set_title("Instance Reduction Preprocessing Time (s)")
    axes[1, 1].set_ylabel("Reduction Time (s)")
    axes[1, 1].grid(axis="y", ls="--", alpha=0.5)
    for b in bars_r:
        axes[1, 1].text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02, f"{b.get_height():.3f}s", ha="center", fontsize=9)

    plt.tight_layout()
    plot_path = output_path / f"{dataset_name}_reduction_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Reduction analysis plot: {plot_path}")

    return df_folds, summary_mean, df_wilcoxon


if __name__ == "__main__":
    for ds in ("soybean", "hypothyroid"):
        run_reduction_experiments_for_dataset(dataset_name=ds)
