import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Find the results folder
BASE_DIR = Path(__file__).resolve().parent.parent
results_dir = BASE_DIR / "results"


# Load the results of all 1680 SVM experiments
experiment_df = pd.read_csv(
    results_dir / "svm_grid_results.csv"
)

# Calculate statistics for each configuration
summary = experiment_df.groupby(
    ["dataset", "kernel", "C", "gamma", "degree"],
    dropna=False
).agg(
    mean_accuracy=("accuracy", "mean"),
    std_accuracy=("accuracy", "std"),
    mean_train_time=("train_time", "mean"),
    mean_predict_time=("predict_time", "mean")
).reset_index()

# Sort configurations by accuracy
summary = summary.sort_values(
    ["dataset", "mean_accuracy"],
    ascending=[True, False]
)

# Save summary
summary_path = results_dir / "svm_summary.csv"
summary.to_csv(summary_path, index=False)

print("SVM summary saved successfully!")
print("Location:", summary_path)
print("Number of rows:", len(summary))

# Display top 5 configurations for each dataset
for dataset in ["soybean", "hypothyroid"]:
    dataset_results = summary[
        summary["dataset"] == dataset
    ]

    top_5 = dataset_results.head(5)

    print(f"\nTop 5 SVM configurations for {dataset}:")
    print(top_5.to_string(index=False))



for dataset in ["soybean", "hypothyroid"]:

    data = summary[summary["dataset"] == dataset]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Best accuracy for each kernel
    kernel_results = (
        data.groupby("kernel")["mean_accuracy"].max() * 100
    )

    kernel_results.plot(
        kind="bar",
        ax=axes[0, 0],
        title="Kernel Comparison",
        ylabel="Accuracy (%)",
        rot=0
    )

    # 2. Effect of C
    c_data = data[
        (data["gamma"] == "scale") &
        (data["degree"] == 3)
    ]

    c_results = c_data.pivot(
        index="C",
        columns="kernel",
        values="mean_accuracy"
    )

    (c_results * 100).plot(
        ax=axes[0, 1],
        marker="o",
        title="Effect of C",
        ylabel="Accuracy (%)"
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("C")

    # 3. Effect of gamma for RBF
    gamma_data = data[
        (data["kernel"] == "rbf") &
        (data["gamma"].isin(["0.01", "0.1", "1"]))
    ].copy()

    gamma_data["gamma"] = gamma_data["gamma"].astype(float)

    gamma_results = gamma_data.pivot(
        index="gamma",
        columns="C",
        values="mean_accuracy"
    )

    (gamma_results * 100).plot(
        ax=axes[1, 0],
        marker="o",
        title="Effect of Gamma (RBF)",
        ylabel="Accuracy (%)"
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Gamma")

    # 4. Effect of polynomial degree
    degree_data = data[
        (data["kernel"] == "poly") &
        (data["gamma"] == "scale")
    ]

    degree_results = degree_data.pivot(
        index="degree",
        columns="C",
        values="mean_accuracy"
    )

    (degree_results * 100).plot(
        ax=axes[1, 1],
        marker="o",
        title="Effect of Polynomial Degree",
        ylabel="Accuracy (%)"
    )
    axes[1, 1].set_xticks([2, 3, 4])

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"SVM Parameter Analysis - {dataset.capitalize()}",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        results_dir / f"{dataset}_svm_grid.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("SVM grid plots saved successfully!")

