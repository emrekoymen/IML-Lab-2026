import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find the results folder
BASE_DIR = Path(__file__).resolve().parent.parent
results_dir = BASE_DIR / "results"


# Read the existing experiment results
experiment_df = pd.read_csv(
    results_dir / "rf_grid_results.csv"
)

# Calculate summary statistics
summary = experiment_df.groupby(
    ["dataset", "n_estimators", "criterion", "max_depth"],
    dropna=False
).agg(
    mean_accuracy=("accuracy", "mean"),
    std_accuracy=("accuracy", "std"),
    mean_train_time=("train_time", "mean"),
    mean_predict_time=("predict_time", "mean")
).reset_index()

# Sort by accuracy
summary = summary.sort_values(
    ["dataset", "mean_accuracy"],
    ascending=[True, False]
)

# Save CSV
summary_path = results_dir / "rf_summary.csv"
summary.to_csv(summary_path, index=False)

print("Summary saved successfully!")
print("Location:", summary_path)
print("Number of rows:", len(summary))





# Display the 5 best configurations for each dataset
for dataset in ["soybean", "hypothyroid"]:

    dataset_results = summary[
        summary["dataset"] == dataset
    ]

    top_5 = dataset_results.head(5)

    print(f"\nTop 5 Random Forest configurations for {dataset}:")
    print(top_5.to_string(index=False))





# Compare two RF configurations on Soybean
soybean = experiment_df[
    experiment_df["dataset"] == "soybean"
]

rf_100 = soybean[
    (soybean["n_estimators"] == 100) &
    (soybean["criterion"] == "entropy") &
    (soybean["max_depth"] == 10)
].sort_values("fold")

rf_200 = soybean[
    (soybean["n_estimators"] == 200) &
    (soybean["criterion"] == "entropy") &
    (soybean["max_depth"] == 10)
].sort_values("fold")

comparison = pd.DataFrame({
    "fold": rf_100["fold"].to_numpy(),
    "RF_100_accuracy": rf_100["accuracy"].to_numpy(),
    "RF_200_accuracy": rf_200["accuracy"].to_numpy()
})

comparison["difference"] = (
    comparison["RF_200_accuracy"] -
    comparison["RF_100_accuracy"]
)

print("\nComparison of RF 100 vs RF 200 on Soybean:")
print(comparison.to_string(index=False))



# Create combined RF plots
for dataset in ["soybean", "hypothyroid"]:

    data = summary[summary["dataset"] == dataset]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Effect of number of trees
    trees_data = data[
        (data["criterion"] == "gini") &
        (data["max_depth"].isna())
    ].sort_values("n_estimators")

    axes[0, 0].plot(
        trees_data["n_estimators"],
        trees_data["mean_accuracy"] * 100,
        marker="o"
    )
    axes[0, 0].set_title("Effect of Number of Trees")
    axes[0, 0].set_xlabel("Number of Trees")
    axes[0, 0].set_ylabel("Mean Accuracy (%)")

    # 2. Effect of splitting criterion
    criterion_data = data[
        (data["n_estimators"] == 100) &
        (data["max_depth"].isna())
    ]

    axes[0, 1].bar(
        criterion_data["criterion"],
        criterion_data["mean_accuracy"] * 100
    )
    axes[0, 1].set_title("Effect of Splitting Criterion")
    axes[0, 1].set_ylabel("Mean Accuracy (%)")

    # 3. Effect of maximum depth
    depth_data = data[
        (data["n_estimators"] == 100) &
        (data["criterion"] == "gini") &
        (data["max_depth"].notna())
    ].sort_values("max_depth")

    axes[1, 0].plot(
        depth_data["max_depth"],
        depth_data["mean_accuracy"] * 100,
        marker="o"
    )
    axes[1, 0].set_title("Effect of Maximum Depth")
    axes[1, 0].set_xlabel("Maximum Depth")
    axes[1, 0].set_ylabel("Mean Accuracy (%)")

    # 4. Accuracy versus training time
    for criterion, group in data.groupby("criterion"):
        axes[1, 1].scatter(
            group["mean_train_time"],
            group["mean_accuracy"] * 100,
            label=criterion,
            alpha=0.7
        )

    axes[1, 1].set_title("Accuracy vs Training Time")
    axes[1, 1].set_xlabel("Mean Training Time (seconds)")
    axes[1, 1].set_ylabel("Mean Accuracy (%)")
    axes[1, 1].legend()

    # Apply grid lines
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Random Forest Parameter Analysis - {dataset.capitalize()}",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        results_dir / f"{dataset}_rf_grid.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("Random Forest grid plots saved successfully!")

