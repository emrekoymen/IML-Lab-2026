
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "knn_experiments"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)



datasets = ["soybean", "hypothyroid"]
best_rows = []

for dataset in datasets:
    df = pd.read_csv(
        RESULTS_DIR / f"{dataset}_summary_results.csv"
    )

    # Sort configurations by accuracy
    df = df.sort_values(
        ["accuracy", "f1_macro"],
        ascending=False
    ).reset_index(drop=True)

    # Save top 10 configurations
    top10 = df.head(10)
    top10.to_csv(
        OUTPUT_DIR / f"{dataset}_knn_top10.csv",
        index=False
    )

    best = df.iloc[0].copy()
    best["dataset"] = dataset
    best_rows.append(best)

    print(f"\n{'=' * 55}")
    print(f"Dataset: {dataset.upper()}")
    print(f"{'=' * 55}")

    print("\nBest kNN configuration:")
    print(best[[
        "k", "distance_metric", "voting_scheme",
        "retention_policy", "accuracy",
        "accuracy_std", "f1_macro",
        "prediction_time", "peak_memory_mb"
    ]])

    # FIGURE 1: Parameter analysis
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    parameters = [
        ("k", "Effect of K"),
        ("distance_metric", "Effect of Distance Metric"),
        ("voting_scheme", "Effect of Voting Scheme"),
        ("retention_policy", "Effect of Retention Policy")
    ]

    for ax, (column, title) in zip(axes.flat, parameters):

        grouped = df.groupby(column)["accuracy"].mean()

        if column == "k":
            grouped = grouped.sort_index()
            ax.plot(
                grouped.index,
                grouped.values,
                marker="o"
            )
            ax.set_xticks(grouped.index)
        else:
            ax.bar(
                grouped.index.astype(str),
                grouped.values
            )
            ax.tick_params(axis="x", rotation=20)

        ax.set_title(title)
        ax.set_ylabel("Mean Accuracy")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{dataset.capitalize()} - kNN Parameter Analysis",
        fontsize=15
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{dataset}_knn_parameters.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # FIGURE 2: Performance and efficiency
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Accuracy and F1 by distance metric
    performance = df.groupby("distance_metric")[
        ["accuracy", "f1_macro"]
    ].mean()

    performance.plot(
        kind="bar",
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("Accuracy and Macro-F1 by Distance")
    axes[0, 0].set_ylabel("Mean Score")
    axes[0, 0].tick_params(axis="x", rotation=20)

    # Prediction time by distance metric
    time_results = df.groupby(
        "distance_metric"
    )["prediction_time"].mean()

    axes[0, 1].bar(
        time_results.index,
        time_results.values
    )
    axes[0, 1].set_title("Prediction Time by Distance")
    axes[0, 1].set_ylabel("Mean Prediction Time (s)")
    axes[0, 1].tick_params(axis="x", rotation=20)

    # Memory usage by retention policy
    memory_results = df.groupby(
        "retention_policy"
    )["peak_memory_mb"].mean()

    axes[1, 0].bar(
        memory_results.index,
        memory_results.values
    )
    axes[1, 0].set_title("Peak Memory by Retention Policy")
    axes[1, 0].set_ylabel("Mean Peak Memory (MB)")
    axes[1, 0].tick_params(axis="x", rotation=20)

    # Accuracy vs prediction time
    axes[1, 1].scatter(
        df["prediction_time"],
        df["accuracy"],
        alpha=0.6
    )
    axes[1, 1].set_title("Accuracy vs Prediction Time")
    axes[1, 1].set_xlabel("Prediction Time (s)")
    axes[1, 1].set_ylabel("Accuracy")

    fig.suptitle(
        f"{dataset.capitalize()} - kNN Performance and Efficiency",
        fontsize=15
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{dataset}_knn_performance.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\nSaved parameter and performance graphs.")

# Best configuration comparison
best_df = pd.DataFrame(best_rows)

columns = [
    "dataset",
    "k",
    "distance_metric",
    "voting_scheme",
    "retention_policy",
    "accuracy",
    "accuracy_std",
    "f1_macro",
    "prediction_time",
    "peak_memory_mb"
]

best_df[columns].to_csv(
    OUTPUT_DIR / "knn_best_comparison.csv",
    index=False
)

print("\nBest configurations comparison:")
print(best_df[columns].to_string(index=False))

print("\nAll kNN analysis files saved successfully.")
