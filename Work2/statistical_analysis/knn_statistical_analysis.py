import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import t

# Find the Work2 folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Folder containing the kNN experiment results
INPUT_DIR = BASE_DIR / "knn_experiments"

# Read the 10-fold results
soybean = pd.read_csv(
    INPUT_DIR / "soybean_fold_results.csv"
)

hypothyroid = pd.read_csv(
    INPUT_DIR / "hypothyroid_fold_results.csv"
)

# Check the datasets
print("Soybean shape:", soybean.shape)
print("Hypothyroid shape:", hypothyroid.shape)

print("\nSoybean columns:")
print(soybean.columns.tolist())

print("\nHypothyroid columns:")
print(hypothyroid.columns.tolist())





# Parameters that define each kNN configuration
parameters = [
    "k",
    "distance_metric",
    "voting_scheme",
    "retention_policy"
]

# Analyze both datasets
for dataset_name, df in [
    ("Soybean", soybean),
    ("Hypothyroid", hypothyroid)
]:

    # Calculate mean accuracy across the 10 folds
    summary = (
        df.groupby(parameters)["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Sort configurations by highest mean accuracy
    summary = summary.sort_values(
        by="mean",
        ascending=False
    )

    # Select the five best configurations
    top5 = summary.head(5)

    print(f"\nTop 5 kNN configurations for {dataset_name}:")
    print(top5.to_string(index=False))



    # Get the 10 accuracy values for a configuration
    def get_fold_accuracies(config):
        mask = pd.Series(True, index=df.index)

        for parameter in parameters:
            mask &= df[parameter] == config[parameter]

        selected = df.loc[mask].sort_values("fold")

        if len(selected) != 10 or selected["fold"].nunique() != 10:
            raise ValueError("Expected exactly 10 different folds")

        return selected.set_index("fold")["accuracy"]

    # Best configuration
    best = top5.iloc[0]
    best_scores = get_fold_accuracies(best)

    comparisons = []

    # Compare best configuration against the next four
    for i in range(1, len(top5)):
        competitor = top5.iloc[i]
        competitor_scores = get_fold_accuracies(competitor)

        # Ensure the same folds are compared
        competitor_scores = competitor_scores.loc[best_scores.index]

        differences = best_scores.values - competitor_scores.values

        mean_difference = np.mean(differences)
        variance = np.var(differences, ddof=1)

        # Corrected paired t-test approximation for 10-fold CV
        if np.isclose(variance, 0):
            p_value = np.nan
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

    # Holm adjustment for four comparisons
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

    print(f"\nStatistical comparisons for {dataset_name}:")
    print(comparison_df.to_string(index=False))

    # Save results
    output_dir = BASE_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    comparison_df.to_csv(
        output_dir / f"{dataset_name.lower()}_knn_statistical.csv",
        index=False
    )




