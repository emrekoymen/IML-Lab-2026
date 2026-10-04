"""Run SVM and Random Forest on the 10 folds of a dataset."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from models import rfAlgorithm, svmAlgorithm
from parser import load_fold
from preprocessing import preprocess, preprocess_hypothyroid

MODELS = [
    ("SVM linear", svmAlgorithm, {"kernel": "linear"}),
    ("SVM rbf", svmAlgorithm, {"kernel": "rbf"}),
    ("RF 100 gini", rfAlgorithm, {"n_estimators": 100, "criterion": "gini"}),
    ("RF 200 gini", rfAlgorithm, {"n_estimators": 200, "criterion": "gini"}),
    ("RF 100 entropy", rfAlgorithm, {"n_estimators": 100, "criterion": "entropy"}),
    ("RF 200 entropy", rfAlgorithm, {"n_estimators": 200, "criterion": "entropy"}),
]


def run_experiments(folder, preprocess):
    rows = []
    for fold in range(10):
        X_train, y_train, X_test, y_test = load_fold(folder, fold)
        X_train, X_test = preprocess(X_train, X_test)

        for name, algorithm, parameters in MODELS:
            predictions, train_time, predict_time = algorithm(X_train, y_train, X_test, **parameters)
            accuracy = accuracy_score(y_test, predictions)
            rows.append([fold, name, accuracy, train_time, predict_time])

    results = pd.DataFrame(rows, columns=["fold", "model", "accuracy", "train_time", "predict_time"])
    summary = results.groupby("model", sort=False)[["accuracy", "train_time", "predict_time"]].mean()
    summary["accuracy_std"] = results.groupby("model", sort=False)["accuracy"].std()
    return results, summary


if __name__ == "__main__":
    for folder, preprocess_function in [("hypothyroid", preprocess_hypothyroid), ("soybean", preprocess)]:
        results, summary = run_experiments(folder, preprocess_function)
        print(folder)
        print(summary.round(4), "\n")

        results.to_csv(f"results/{folder}_results.csv", index=False)
        summary["accuracy"].plot.bar(yerr=summary["accuracy_std"], title=f"{folder}: mean accuracy")
        plt.tight_layout()
        plt.savefig(f"results/{folder}_accuracy.png")
        plt.close()
