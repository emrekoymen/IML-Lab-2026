import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import t

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
KNN_DIR = BASE_DIR / "knn_experiments"

svm = pd.read_csv(RESULTS_DIR / "svm_grid_results.csv")
rf = pd.read_csv(RESULTS_DIR / "rf_grid_results.csv")

for dataset in ["soybean", "hypothyroid"]:
    knn = pd.read_csv(
        KNN_DIR / f"{dataset}_fold_results.csv"
    )

    print(f"\n{dataset.upper()}")

    # Find the best kNN configuration
    knn_params = [
        "k", "distance_metric",
        "voting_scheme", "retention_policy"
    ]

    knn_summary = (
        knn.groupby(knn_params)["accuracy"]
        .mean()
        .reset_index()
    )

    print("\nBest kNN:")
    print(knn_summary.nlargest(1, "accuracy").to_string(index=False))

    # Find the best SVM configuration
    svm_data = svm[svm["dataset"].str.lower() == dataset]

    svm_summary = (
        svm_data.groupby(
            ["kernel", "C", "gamma", "degree"],
            dropna=False
        )["accuracy"]
        .mean()
        .reset_index()
    )

    print("\nBest SVM:")
    print(svm_summary.nlargest(1, "accuracy").to_string(index=False))

    # Find the best RF configuration
    rf_data = rf[rf["dataset"].str.lower() == dataset]

    rf_summary = (
        rf_data.groupby(
            ["n_estimators", "criterion", "max_depth"],
            dropna=False
        )["accuracy"]
        .mean()
        .reset_index()
    )

    print("\nBest RF:")
    print(rf_summary.nlargest(1, "accuracy").to_string(index=False))



    # Get the best configuration for each algorithm
    best_knn = knn_summary.nlargest(1, "accuracy").iloc[0]
    best_svm = svm_summary.nlargest(1, "accuracy").iloc[0]
    best_rf = rf_summary.nlargest(1, "accuracy").iloc[0]

    # Function to get fold accuracies for one configuration
    def get_scores(data, config, parameters):
        mask = pd.Series(True, index=data.index)

        for parameter in parameters:
            if pd.isna(config[parameter]):
                mask &= data[parameter].isna()
            else:
                mask &= data[parameter] == config[parameter]

        selected = data.loc[mask].sort_values("fold")

        if len(selected) != 10 or selected["fold"].nunique() != 10:
            raise ValueError("Expected exactly 10 distinct folds")

        return selected.set_index("fold")["accuracy"]

    # Retrieve the ten accuracies for each algorithm
    knn_scores = get_scores(knn, best_knn, knn_params)

    svm_scores = get_scores(
        svm_data, best_svm,
        ["kernel", "C", "gamma", "degree"]
    )

    rf_scores = get_scores(
        rf_data, best_rf,
        ["n_estimators", "criterion", "max_depth"]
    )

    # Store all three algorithms
    scores = {
        "kNN": knn_scores,
        "SVM": svm_scores,
        "RF": rf_scores
    }

    # Pairwise comparisons
    pairs = [
        ("kNN", "SVM"),
        ("kNN", "RF"),
        ("SVM", "RF")
    ]

    results = []

    for algorithm_a, algorithm_b in pairs:
        a = scores[algorithm_a]
        b = scores[algorithm_b].loc[a.index]

        differences = a.values - b.values
        mean_difference = np.mean(differences)
        variance = np.var(differences, ddof=1)

        if variance == 0:
            p_value = 1.0 if mean_difference == 0 else np.nan
        else:
            # Approximate corrected paired t-test for 10-fold CV
            correction = 1 / 10 + 1 / 9
            standard_error = np.sqrt(correction * variance)

            t_statistic = mean_difference / standard_error
            p_value = 2 * t.sf(abs(t_statistic), df=9)

        results.append({
            "algorithm_a": algorithm_a,
            "algorithm_b": algorithm_b,
            "mean_accuracy_difference": mean_difference,
            "p_value": p_value
        })

    comparison_df = pd.DataFrame(results)

    # Holm adjustment for three comparisons
    comparison_df["adjusted_p_value"] = np.nan

    valid = comparison_df["p_value"].dropna().sort_values()
    previous = 0.0
    m = len(valid)

    for rank, (index, p_value) in enumerate(valid.items()):
        adjusted = min(1.0, (m - rank) * p_value)
        previous = max(previous, adjusted)
        comparison_df.loc[index, "adjusted_p_value"] = previous

    comparison_df["significant"] = (
        comparison_df["adjusted_p_value"] < 0.05
    )

    print(f"\nAlgorithm statistical comparisons for {dataset}:")
    print(comparison_df.to_string(index=False))

    # Save results
    comparison_df.to_csv(
        RESULTS_DIR / f"{dataset}_algorithm_statistical.csv",
        index=False
    )