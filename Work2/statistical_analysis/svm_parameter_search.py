
from itertools import product
import pandas as pd
from itertools import product
from pathlib import Path
from sklearn.metrics import accuracy_score
from models import svmAlgorithm
from parser import load_fold
from preprocessing import preprocess, preprocess_hypothyroid
import sys

# Find the Work2 folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Allow Python to import modules from Work2
sys.path.insert(0, str(BASE_DIR))

C_VALUES = [0.1, 1, 10, 100]
GAMMA_VALUES = ["scale", 0.01, 0.1, 1]
DEGREES = [2, 3, 4]

# Linear kernel
linear_combinations = [
    ("linear", C, "scale", 3)
    for C in C_VALUES
]

# RBF kernel
rbf_combinations = [
    ("rbf", C, gamma, 3)
    for C, gamma in product(C_VALUES, GAMMA_VALUES)
]

# Polynomial kernel
poly_combinations = [
    ("poly", C, gamma, degree)
    for C, gamma, degree in product(
        C_VALUES, GAMMA_VALUES, DEGREES
    )
]

# Sigmoid kernel
sigmoid_combinations = [
    ("sigmoid", C, gamma, 3)
    for C, gamma in product(C_VALUES, GAMMA_VALUES)
]

combinations = (
    linear_combinations
    + rbf_combinations
    + poly_combinations
    + sigmoid_combinations
)

print("Number of SVM configurations:", len(combinations))

for combination in combinations:
    print(combination)


results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)


# Save SVM parameter combinations to CSV
df = pd.DataFrame(
    combinations,
    columns=["kernel", "C", "gamma", "degree"]
)

csv_path = results_dir / "svm_parameters.csv"
df.to_csv(csv_path, index=False)

print("\nParameters saved to:", csv_path)




# Store experiment results
experiment_rows = []

# Test only 2 configurations first
test_combinations = combinations

for folder, preprocess_function in [
    ("hypothyroid", preprocess_hypothyroid),
    ("soybean", preprocess)
]:
    print(f"\nRunning SVM experiments on {folder}...")

    for fold in range(10):
        X_train, y_train, X_test, y_test = load_fold(folder, fold)

        X_train, X_test = preprocess_function(X_train, X_test)

        for kernel, C, gamma, degree in test_combinations:
            predictions, train_time, predict_time = svmAlgorithm(
                X_train,
                y_train,
                X_test,
                kernel=kernel,
                C=C,
                gamma=gamma,
                degree=degree
            )

            accuracy = accuracy_score(y_test, predictions)

            experiment_rows.append({
                "dataset": folder,
                "fold": fold,
                "kernel": kernel,
                "C": C,
                "gamma": gamma,
                "degree": degree,
                "accuracy": accuracy,
                "train_time": train_time,
                "predict_time": predict_time
            })

            print(
                f"{folder} | Fold {fold} | "
                f"Kernel {kernel} | C {C} | "
                f"Accuracy {accuracy:.4f}"
            )

# Save experiment results
experiment_df = pd.DataFrame(experiment_rows)

output_path = results_dir / "svm_grid_results.csv"
experiment_df.to_csv(output_path, index=False)

print("\nSVM experiments completed!")
print("Number of results:", len(experiment_df))
print("Results saved to:", output_path)


