"""Automated 10-fold Instance Reduction experiments (Points 6.d & 6.e) and Demšar (2006) statistical tests.

Evaluates:
1. Best kNN (`reduction_kNNAlgorithm`) with 'none' (100% storage), 'gcnn', 'rnge', and 'del'
   (including both the best static Never Retain configuration and the best dynamic retention configuration).
2. Best SVM (`svmAlgorithm`) trained on full training set ('none') vs reduced training sets ('gcnn', 'rnge', 'del').
3. Best Random Forest (`rfAlgorithm`) trained on full training set ('none') vs reduced training sets ('gcnn', 'rnge', 'del').
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from parser import load_fold
from preprocessing import preprocess, preprocess_hypothyroid
from models import svmAlgorithm, rfAlgorithm
from reduction import REDUCTION_METHODS, reduce, reduction_kNNAlgorithm
from KNNClassifier import (
    wilcoxon_test,
    friedman_test,
    plot_critical_difference_diagram,
)


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
            "distance_metric": "canberra",
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


def evaluate_predictions(y_true, y_pred):
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n_correct = int(np.sum(y_true_arr == y_pred_arr))
    n_incorrect = int(len(y_true_arr) - n_correct)
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    return acc, f1, n_correct, n_incorrect


def run_reduction_experiments_for_dataset(dataset_name="soybean", n_folds=10, output_dir="reduction_experiments"):
    os.makedirs(output_dir, exist_ok=True)
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
            # Also computes `kept` and `red_time` once per fold!
            preds_knn_nr, info_nr = reduction_kNNAlgorithm(
                X_tr_arr,
                y_tr_arr,
                X_te_arr,
                y_te_arr,
                method=method,
                **knn_static_cfg,
            )
            acc, f1, n_corr, n_incorr = evaluate_predictions(y_te_arr, preds_knn_nr)
            kept = info_nr["kept_indices"]
            red_time = info_nr["reduction_time"]
            storage_pct = info_nr["storage"] * 100.0

            fold_rows.append({
                "dataset": dataset_name,
                "fold": fold,
                "algorithm": "Best kNN (NR)",
                "reduction_method": method,
                "accuracy": acc,
                "f1_macro": f1,
                "n_correct": n_corr,
                "n_incorrect": n_incorr,
                "n_train_initial": len(y_tr_arr),
                "n_train_kept": info_nr["n_kept"],
                "storage_pct": storage_pct,
                "final_storage_pct": info_nr["final_storage"] * 100.0,
                "reduction_time": red_time,
                "training_time": 0.0,
                "prediction_time": info_nr["prediction_time"],
                "total_time": red_time + info_nr["prediction_time"],
            })

            # Reduced training set for Best kNN (AR), Best SVM, and Best RF
            X_tr_reduced = X_tr_arr[kept]
            y_tr_reduced = y_tr_arr[kept]

            # 2. Best kNN (Overall best with dynamic retention AR) trained on X_tr_reduced
            # Note: For DEL on Soybean (where knn_overall uses k=3, majority and knn_static uses k=7, inverse_distance),
            # if k or voting_scheme differs and method == 'del', run reduction_kNNAlgorithm directly; otherwise reuse X_tr_reduced
            if method == "del" and (
                knn_overall_cfg["k"] != knn_static_cfg["k"]
                or knn_overall_cfg["voting_scheme"] != knn_static_cfg["voting_scheme"]
            ):
                preds_knn_ar, info_ar = reduction_kNNAlgorithm(
                    X_tr_arr,
                    y_tr_arr,
                    X_te_arr,
                    y_te_arr,
                    method=method,
                    **knn_overall_cfg,
                )
            else:
                from time import perf_counter
                from KNNClassifier import KNNClassifier
                model_ar = KNNClassifier(**knn_overall_cfg)
                model_ar.fit(X_tr_reduced, y_tr_reduced)
                t0 = perf_counter()
                preds_knn_ar = model_ar.predict(X_te_arr, y_te_arr)
                pred_t_ar = perf_counter() - t0
                n_final_ar = getattr(model_ar, "n_stored_", len(kept))
                info_ar = {
                    "n_kept": len(kept),
                    "storage": len(kept) / len(y_tr_arr),
                    "final_storage": n_final_ar / len(y_tr_arr),
                    "reduction_time": red_time,
                    "prediction_time": pred_t_ar,
                }

            acc_ar, f1_ar, n_corr_ar, n_incorr_ar = evaluate_predictions(y_te_arr, preds_knn_ar)
            fold_rows.append({
                "dataset": dataset_name,
                "fold": fold,
                "algorithm": "Best kNN (AR)",
                "reduction_method": method,
                "accuracy": acc_ar,
                "f1_macro": f1_ar,
                "n_correct": n_corr_ar,
                "n_incorrect": n_incorr_ar,
                "n_train_initial": len(y_tr_arr),
                "n_train_kept": info_ar["n_kept"],
                "storage_pct": info_ar["storage"] * 100.0,
                "final_storage_pct": info_ar["final_storage"] * 100.0,
                "reduction_time": info_ar["reduction_time"],
                "training_time": 0.0,
                "prediction_time": info_ar["prediction_time"],
                "total_time": info_ar["reduction_time"] + info_ar["prediction_time"],
            })

            # 3. Best SVM on reduced training set
            preds_svm, svm_train_t, svm_pred_t = svmAlgorithm(
                X_tr_reduced, y_tr_reduced, X_te_arr, **svm_cfg
            )
            acc_svm, f1_svm, n_corr_svm, n_incorr_svm = evaluate_predictions(y_te_arr, preds_svm)
            fold_rows.append({
                "dataset": dataset_name,
                "fold": fold,
                "algorithm": "Best SVM",
                "reduction_method": method,
                "accuracy": acc_svm,
                "f1_macro": f1_svm,
                "n_correct": n_corr_svm,
                "n_incorrect": n_incorr_svm,
                "n_train_initial": len(y_tr_arr),
                "n_train_kept": len(kept),
                "storage_pct": storage_pct,
                "final_storage_pct": storage_pct,
                "reduction_time": red_time,
                "training_time": svm_train_t,
                "prediction_time": svm_pred_t,
                "total_time": red_time + svm_train_t + svm_pred_t,
            })

            # 4. Best RF on reduced training set
            preds_rf, rf_train_t, rf_pred_t = rfAlgorithm(
                X_tr_reduced, y_tr_reduced, X_te_arr, **rf_cfg
            )
            acc_rf, f1_rf, n_corr_rf, n_incorr_rf = evaluate_predictions(y_te_arr, preds_rf)
            fold_rows.append({
                "dataset": dataset_name,
                "fold": fold,
                "algorithm": "Best RF",
                "reduction_method": method,
                "accuracy": acc_rf,
                "f1_macro": f1_rf,
                "n_correct": n_corr_rf,
                "n_incorrect": n_incorr_rf,
                "n_train_initial": len(y_tr_arr),
                "n_train_kept": len(kept),
                "storage_pct": storage_pct,
                "final_storage_pct": storage_pct,
                "reduction_time": red_time,
                "training_time": rf_train_t,
                "prediction_time": rf_pred_t,
                "total_time": red_time + rf_train_t + rf_pred_t,
            })

    df_folds = pd.DataFrame(fold_rows)
    fold_csv = os.path.join(output_dir, f"{dataset_name}_reduction_fold_results.csv")
    df_folds.to_csv(fold_csv, index=False)
    print(f"[Saved] {fold_csv}")

    metric_cols = [
        "accuracy",
        "f1_macro",
        "n_correct",
        "n_incorrect",
        "n_train_kept",
        "storage_pct",
        "final_storage_pct",
        "reduction_time",
        "training_time",
        "prediction_time",
        "total_time",
    ]
    summary_mean = df_folds.groupby(["dataset", "algorithm", "reduction_method"], as_index=False, sort=False)[metric_cols].mean()
    summary_std = df_folds.groupby(["dataset", "algorithm", "reduction_method"], as_index=False, sort=False)[metric_cols].std()
    for col in metric_cols:
        summary_mean[f"{col}_std"] = summary_std[col]

    summary_csv = os.path.join(output_dir, f"{dataset_name}_reduction_summary_results.csv")
    summary_mean.to_csv(summary_csv, index=False)
    print(f"[Saved] {summary_csv}")
    print("\nSummary Table:")
    print(summary_mean[["algorithm", "reduction_method", "accuracy", "accuracy_std", "storage_pct", "reduction_time", "prediction_time"]].to_string(index=False))

    # Statistical Analysis per Algorithm across the 4 reduction methods (none, gcnn, rnge, del)
    stats_rows = []
    wilcoxon_rows = []
    for alg in ("Best kNN (NR)", "Best kNN (AR)", "Best SVM", "Best RF"):
        sub = df_folds[df_folds["algorithm"] == alg]
        pivot_acc = sub.pivot(index="fold", columns="reduction_method", values="accuracy")[list(REDUCTION_METHODS)]
        fr = friedman_test(pivot_acc, higher_is_better=True)

        alg_slug = alg.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fr["summary_table"].to_csv(
            os.path.join(output_dir, f"{dataset_name}_stats_{alg_slug}_reduction.csv"), index=False
        )
        plot_critical_difference_diagram(
            fr,
            title=f"{dataset_name.capitalize()} - {alg}: Reduction Methods CD Diagram",
            save_path=os.path.join(output_dir, f"{dataset_name}_cd_{alg_slug}_reduction.png"),
        )

        print(f"\n--- Friedman Test for {alg} on {dataset_name.upper()} ---")
        print(f"Chi2 = {fr['friedman_chi2']:.4f}, p = {fr['friedman_p_value']:.4f}, "
              f"Iman-Davenport F = {fr['iman_davenport_f']:.4f} (p = {fr['iman_davenport_p_value']:.4f})")
        print(fr["summary_table"].to_string(index=False))

        # Pairwise Wilcoxon comparing each reduction method ('gcnn', 'rnge', 'del') against unreduced 'none'
        base_scores = pivot_acc["none"].values
        for m in ("gcnn", "rnge", "del"):
            w = wilcoxon_test(pivot_acc[m].values, base_scores, name_a=f"{alg} + {m.upper()}", name_b=f"{alg} + NONE")
            w["dataset"] = dataset_name
            w["algorithm"] = alg
            w["reduction_method"] = m
            wilcoxon_rows.append(w)

    df_wilcoxon = pd.DataFrame(wilcoxon_rows)
    wilcoxon_csv = os.path.join(output_dir, f"{dataset_name}_stats_wilcoxon_vs_none.csv")
    df_wilcoxon.to_csv(wilcoxon_csv, index=False)
    print(f"\n[Saved] Pairwise Wilcoxon vs Full Training Set ('none'): {wilcoxon_csv}")
    print(df_wilcoxon[["algorithm", "reduction_method", "mean_a", "mean_b", "mean_diff", "p_value", "significant"]].to_string(index=False))

    # Generate 2x2 Comparison Plot (Accuracy, Storage %, Prediction Time, Reduction Time)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Instance Reduction Impact on {dataset_name.capitalize()} (10-Fold CV)", fontsize=14, fontweight="bold")

    # (a) Accuracy (%) across algorithms and reduction methods
    acc_piv = (summary_mean.pivot(index="reduction_method", columns="algorithm", values="accuracy") * 100).reindex(REDUCTION_METHODS)
    acc_piv.index = [m.upper() for m in acc_piv.index]
    acc_piv.plot(kind="bar", ax=axes[0, 0], edgecolor="black", rot=0)
    axes[0, 0].set_title("Classification Accuracy (%) by Reduction Method")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(max(0, acc_piv.values.min() - 8), min(102, acc_piv.values.max() + 3))
    axes[0, 0].grid(axis="y", ls="--", alpha=0.5)

    # (b) Storage (%) retained
    sto_piv = summary_mean[summary_mean["algorithm"] == "Best kNN (NR)"].set_index("reduction_method")["storage_pct"].reindex(REDUCTION_METHODS)
    bars = axes[0, 1].bar([m.upper() for m in sto_piv.index], sto_piv.values, color=["slategray", "steelblue", "mediumseagreen", "coral"], edgecolor="black")
    axes[0, 1].set_title("Training Storage Retained (%)")
    axes[0, 1].set_ylabel("Storage (%)")
    axes[0, 1].set_ylim(0, 115)
    axes[0, 1].grid(axis="y", ls="--", alpha=0.5)
    for b in bars:
        axes[0, 1].text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")

    # (c) kNN Prediction Time (ms) before vs after reduction
    pred_piv = (summary_mean[summary_mean["algorithm"].isin(["Best kNN (NR)", "Best kNN (AR)"])].pivot(
        index="reduction_method", columns="algorithm", values="prediction_time"
    ) * 1000).reindex(REDUCTION_METHODS)
    pred_piv.index = [m.upper() for m in pred_piv.index]
    pred_piv.plot(kind="bar", ax=axes[1, 0], edgecolor="black", rot=0)
    axes[1, 0].set_title("k-NN Prediction / Problem-Solving Time (ms)")
    axes[1, 0].set_ylabel("Prediction Time (ms)")
    axes[1, 0].grid(axis="y", ls="--", alpha=0.5)

    # (d) Preprocessing Reduction Time (s) + SVM/RF Training Time (ms)
    red_piv = summary_mean[summary_mean["algorithm"] == "Best kNN (NR)"].set_index("reduction_method")["reduction_time"].reindex(["gcnn", "rnge", "del"])
    bars_r = axes[1, 1].bar([m.upper() for m in red_piv.index], red_piv.values, color=["steelblue", "mediumseagreen", "coral"], edgecolor="black")
    axes[1, 1].set_title("Instance Reduction Preprocessing Time (s)")
    axes[1, 1].set_ylabel("Reduction Time (s)")
    axes[1, 1].grid(axis="y", ls="--", alpha=0.5)
    for b in bars_r:
        axes[1, 1].text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02, f"{b.get_height():.3f}s", ha="center", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{dataset_name}_reduction_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Reduction analysis plot: {plot_path}")

    return df_folds, summary_mean, df_wilcoxon


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reduction_experiments")
    for ds in ("soybean", "hypothyroid"):
        run_reduction_experiments_for_dataset(dataset_name=ds, output_dir=out_dir)

