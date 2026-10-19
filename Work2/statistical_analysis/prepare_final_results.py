import pandas as pd
from pathlib import Path

# Find the Work2 folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Existing results folder
RESULTS_DIR = BASE_DIR / "results"

# New folder for organized results
FINAL_DIR = BASE_DIR / "final_report_results"
FINAL_DIR.mkdir(exist_ok=True)

# Collect all statistical comparisons
all_statistics = []

for dataset in ["soybean", "hypothyroid"]:

    for analysis in ["knn", "svm", "rf", "algorithm"]:

        file_path = (
            RESULTS_DIR / f"{dataset}_{analysis}_statistical.csv"
        )

        df = pd.read_csv(file_path)

        # Identify which dataset and analysis each row belongs to
        df["dataset"] = dataset
        df["analysis"] = analysis

        all_statistics.append(df)

# Combine into one table
combined_statistics = pd.concat(
    all_statistics,
    ignore_index=True
)

# Save one CSV file
output_file = FINAL_DIR / "all_statistical_results.csv"

combined_statistics.to_csv(output_file, index=False)

print("Statistical results combined successfully!")
print("Saved to:", output_file)
print("Total comparisons:", len(combined_statistics))




# COMBINE PARAMETER RESULTS
all_parameters = []

# kNN parameter results
for dataset in ["soybean", "hypothyroid"]:

    knn_file = (
        BASE_DIR / "knn_experiments"
        / f"{dataset}_summary_results.csv"
    )

    df = pd.read_csv(knn_file)

    df["dataset"] = dataset
    df["algorithm"] = "kNN"

    df = df.rename(columns={
        "accuracy": "mean_accuracy",
        "accuracy_std": "std_accuracy",
        "training_time": "mean_train_time",
        "prediction_time": "mean_predict_time"
    })

    all_parameters.append(df)

# SVM and RF parameter results
for algorithm in ["svm", "rf"]:

    df = pd.read_csv(
        RESULTS_DIR / f"{algorithm}_summary.csv"
    )

    df["algorithm"] = algorithm.upper()

    all_parameters.append(df)

# Combine all parameter results
combined_parameters = pd.concat(
    all_parameters,
    ignore_index=True
)

# Save combined CSV
parameter_file = FINAL_DIR / "all_parameter_results.csv"

combined_parameters.to_csv(
    parameter_file,
    index=False
)

print("\nParameter results combined successfully!")
print("Saved to:", parameter_file)
print("Total configurations:", len(combined_parameters))




# BEST MODEL SUMMARY
best_models = (
    combined_parameters
    .sort_values("mean_accuracy", ascending=False)
    .groupby(["dataset", "algorithm"])
    .head(1)
    .copy()
)

# Keep useful columns
columns = [
    "dataset",
    "algorithm",
    "k",
    "distance_metric",
    "voting_scheme",
    "retention_policy",
    "kernel",
    "C",
    "gamma",
    "degree",
    "n_estimators",
    "criterion",
    "max_depth",
    "mean_accuracy",
    "std_accuracy",
    "mean_train_time",
    "mean_predict_time"
]

best_models = best_models[
    [col for col in columns if col in best_models.columns]
]

best_models = best_models.sort_values(
    ["dataset", "algorithm"]
)

best_file = FINAL_DIR / "best_models_summary.csv"

best_models.to_csv(best_file, index=False)

print("\nBest model summary created successfully!")
print("Total best models:", len(best_models))
print(best_models[
    ["dataset", "algorithm", "mean_accuracy"]
].to_string(index=False))



# BEST MODEL ACCURACY VISUALIZATION

import matplotlib.pyplot as plt

# Create a folder for final figures
FIGURES_DIR = FINAL_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Prepare data for the graph
plot_data = best_models.pivot(
    index="dataset",
    columns="algorithm",
    values="mean_accuracy"
)

# Convert accuracy to percentages
plot_data = plot_data * 100

# Arrange algorithms consistently
plot_data = plot_data[["kNN", "SVM", "RF"]]

# Create grouped bar chart
ax = plot_data.plot(
    kind="bar",
    figsize=(10, 6),
    rot=0
)

ax.set_title("Comparison of Best Model Accuracies")
ax.set_xlabel("Dataset")
ax.set_ylabel("Mean Accuracy (%)")
ax.set_ylim(0, 105)
ax.legend(title="Algorithm")

# Display accuracy values above bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%", padding=3)

plt.tight_layout()

# Save figure
figure_file = FIGURES_DIR / "model_accuracy_comparison.png"
plt.savefig(figure_file, dpi=300)
plt.close()

print("\nAccuracy comparison figure created!")
print("Saved to:", figure_file)





import shutil

# Existing figures to include in the final report
figures_to_keep = [
    "soybean_knn_parameters.png",
    "hypothyroid_knn_parameters.png",
    "soybean_svm_grid.png",
    "hypothyroid_svm_grid.png",
    "soybean_rf_grid.png",
    "hypothyroid_rf_grid.png"
]

for filename in figures_to_keep:
    source = RESULTS_DIR / filename
    destination = FIGURES_DIR / filename

    shutil.copy2(source, destination)

print("\nParameter figures organized successfully!")
print("Total parameter figures:", len(figures_to_keep))