import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import t


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Read existing experiment results
svm = pd.read_csv(RESULTS_DIR / "svm_grid_results.csv")
rf = pd.read_csv(RESULTS_DIR / "rf_grid_results.csv")

# SVM parameters
svm_parameters = ["kernel", "C", "gamma", "degree"]

# Analyze SVM on each dataset
for dataset_name in ["soybean", "hypothyroid"]:

    # Select results for this dataset
    df = svm[
        svm["dataset"].str.lower() == dataset_name
    ]

    # Calculate mean accuracy, standard deviation,
    # and number of folds for each configuration
    summary = (
        df.groupby(svm_parameters, dropna=False)["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Select the five configurations with highest accuracy
    top5 = summary.sort_values(
        by="mean",
        ascending=False
    ).head(5)

    print(f"\nTop 5 SVM configurations for {dataset_name}:")
    print(top5.to_string(index=False))



    # Get the 10 fold accuracies for one SVM configuration
    def get_fold_accuracies(config):
        mask = pd.Series(True, index=df.index)

        for parameter in svm_parameters:
            if pd.isna(config[parameter]):
                mask &= df[parameter].isna()
            else:
                mask &= df[parameter] == config[parameter]

        selected = df.loc[mask].sort_values("fold")

        if len(selected) != 10 or selected["fold"].nunique() != 10:
            raise ValueError("Expected exactly 10 distinct folds")

        return selected.set_index("fold")["accuracy"]

    best = top5.iloc[0]
    best_scores = get_fold_accuracies(best)

    comparisons = []

    # Compare the best configuration with the other four
    for i in range(1, len(top5)):
        competitor = top5.iloc[i]
        competitor_scores = get_fold_accuracies(competitor)

        # Match the same folds
        competitor_scores = competitor_scores.loc[best_scores.index]

        differences = best_scores.values - competitor_scores.values

        mean_difference = np.mean(differences)
        variance = np.var(differences, ddof=1)

        if variance == 0:
            p_value = 1.0 if mean_difference == 0 else np.nan
        else:
            # Approximate corrected paired t-test
            correction = 1 / 10 + 1 / 9
            standard_error = np.sqrt(correction * variance)

            t_statistic = mean_difference / standard_error
            p_value = 2 * t.sf(abs(t_statistic), df=9)

        comparisons.append({
            "competitor_rank": i + 1,
            "mean_accuracy_difference": mean_difference,
            "p_value": p_value
        })

    comparison_df = pd.DataFrame(comparisons)

    # Holm adjustment for multiple comparisons
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

    print(f"\nSVM statistical comparisons for {dataset_name}:")
    print(comparison_df.to_string(index=False))

    # Save results
    comparison_df.to_csv(
        RESULTS_DIR / f"{dataset_name}_svm_statistical.csv",
        index=False
    )




# RANDOM FOREST STATISTICAL ANALYSIS
rf_parameters = [
    "n_estimators",
    "criterion",
    "max_depth"
]

for dataset_name in ["soybean", "hypothyroid"]:

    # Select Random Forest results for this dataset
    df = rf[rf["dataset"].str.lower() == dataset_name]

    # Calculate statistics for each configuration
    summary = (
        df.groupby(rf_parameters, dropna=False)["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Select the five configurations with highest mean accuracy
    top5 = summary.sort_values(
        by="mean",
        ascending=False
    ).head(5)

    print(f"\nTop 5 RF configurations for {dataset_name}:")
    print(top5.to_string(index=False))



    # Get fold accuracies for one RF configuration
    def get_rf_fold_accuracies(config):
        mask = pd.Series(True, index=df.index)

        for parameter in rf_parameters:
            if pd.isna(config[parameter]):
                mask &= df[parameter].isna()
            else:
                mask &= df[parameter] == config[parameter]

        selected = df.loc[mask].sort_values("fold")

        if len(selected) != 10 or selected["fold"].nunique() != 10:
            raise ValueError("Expected exactly 10 distinct folds")

        return selected.set_index("fold")["accuracy"]

    best = top5.iloc[0]
    best_scores = get_rf_fold_accuracies(best)

    comparisons = []

    for i in range(1, len(top5)):
        competitor = top5.iloc[i]
        competitor_scores = get_rf_fold_accuracies(competitor)

        competitor_scores = competitor_scores.loc[best_scores.index]

        differences = best_scores.values - competitor_scores.values

        mean_difference = np.mean(differences)
        variance = np.var(differences, ddof=1)

        if variance == 0:
            p_value = 1.0 if mean_difference == 0 else np.nan
        else:
            correction = 1 / 10 + 1 / 9
            standard_error = np.sqrt(correction * variance)

            t_statistic = mean_difference / standard_error
            p_value = 2 * t.sf(abs(t_statistic), df=9)

        comparisons.append({
            "competitor_rank": i + 1,
            "mean_accuracy_difference": mean_difference,
            "p_value": p_value
        })

    comparison_df = pd.DataFrame(comparisons)

    # Holm correction
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

    print(f"\nRF statistical comparisons for {dataset_name}:")
    print(comparison_df.to_string(index=False))

    # Save statistical results
    comparison_df.to_csv(
        RESULTS_DIR / f"{dataset_name}_rf_statistical.csv",
        index=False
    )
