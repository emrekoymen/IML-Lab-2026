from itertools import product
from sklearn.metrics import accuracy_score
from models import rfAlgorithm
from parser import load_fold
from preprocessing import preprocess, preprocess_hypothyroid
import pandas as pd
from pathlib import Path



# Different Random Forest parameters to test
RF_PARAMETERS = {
    "n_estimators": [10, 50, 100, 200],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 5, 10]
}

# Generate all combinations of parameters
combinations = list(product(
    RF_PARAMETERS["n_estimators"],
    RF_PARAMETERS["criterion"],
    RF_PARAMETERS["max_depth"]
))

print("Number of combinations:", len(combinations))

for combination in combinations:
    print(combination)





# Create the results folder
BASE_DIR = Path(__file__).resolve().parent.parent
results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

# Convert parameter combinations into a DataFrame
df = pd.DataFrame(
    combinations,
    columns=["n_estimators", "criterion", "max_depth"]
)

# Save the combinations to a CSV file
csv_path = results_dir / "rf_parameters.csv"

df.to_csv(csv_path, index=False)

print("\nCSV file saved successfully!")
print("File location:", csv_path)







# Store results of every experiment
experiment_rows = []

# Run on both datasets
for folder, preprocess_function in [
    ("hypothyroid", preprocess_hypothyroid),
    ("soybean", preprocess)
]:

    print(f"\nRunning experiments on {folder}...")

    # Use the 10 predefined folds
    for fold in range(10):

        # Load training and testing data
        X_train, y_train, X_test, y_test = load_fold(
            folder, fold
        )

        # Apply the same preprocessing as teammates
        X_train, X_test = preprocess_function(
            X_train, X_test
        )

        # Test all 36 parameter combinations
        for n_trees, criterion, depth in combinations:

            predictions, train_time, predict_time = rfAlgorithm(
                X_train,
                y_train,
                X_test,
                n_estimators=n_trees,
                criterion=criterion,
                max_depth=depth
            )

            accuracy = accuracy_score(y_test, predictions)

            experiment_rows.append({
                "dataset": folder,
                "fold": fold,
                "n_estimators": n_trees,
                "criterion": criterion,
                "max_depth": depth,
                "accuracy": accuracy,
                "train_time": train_time,
                "predict_time": predict_time
            })

            print(
                f"{folder} | Fold {fold} | "
                f"Trees {n_trees} | {criterion} | "
                f"Depth {depth} | Accuracy {accuracy:.4f}"
            )




# Convert experiment results into a DataFrame
experiment_df = pd.DataFrame(experiment_rows)

# Save results for every fold and configuration
output_path = results_dir / "rf_grid_results.csv"

experiment_df.to_csv(output_path, index=False)

print("\nAll experiments completed!")
print("Results saved to:", output_path)




# Calculate summary statistics for each configuration
summary = experiment_df.groupby(
    ["dataset", "n_estimators", "criterion", "max_depth"],
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

# Save summary to CSV
summary_path = results_dir / "rf_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nRandom Forest summary:")
print(summary.to_string(index=False))

print("\nSummary saved to:", summary_path)
