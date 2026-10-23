"""Run one selected IML model configuration on the supplied 10 folds.

Run from the repository root, for example:
    python Work2/main.py --dataset soybean --model svm --kernel rbf --C 1
"""
import argparse
import math
from pathlib import Path

import pandas as pd

from models import kNNAlgorithm, rfAlgorithm, svmAlgorithm
from parser import load_fold
from preprocessing import preprocess, preprocess_hypothyroid

HERE = Path(__file__).resolve().parent
DATASETS = ("soybean", "hypothyroid")


def positive_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer")

    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")

    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=("knn", "svm", "rf"), required=True)
    parser.add_argument("--folds", type=positive_int, choices=range(1, 11), default=10, metavar="1..10")
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV path")
    # kNN
    parser.add_argument("--k", type=positive_int, default=3)
    parser.add_argument("--distance-metric", choices=("euclidean", "manhattan", "canberra", "heom"), default="euclidean")
    parser.add_argument("--voting-scheme", choices=("majority", "inverse_distance"), default="majority")
    parser.add_argument("--retention-policy", choices=("never_retain", "always_retain", "different_class", "oblivion_by_goodness"), default="never_retain")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--p", type=float, default=1.0)
    # SVM
    parser.add_argument("--kernel", choices=("linear", "rbf", "poly", "sigmoid"), default="rbf")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", default="scale", help="scale, auto, or a positive float")
    parser.add_argument("--degree", type=positive_int, default=3)
    # RF
    parser.add_argument("--n-estimators", type=positive_int, default=100)
    parser.add_argument("--criterion", choices=("gini", "entropy", "log_loss"), default="gini")
    parser.add_argument("--max-depth", type=positive_int, default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not math.isfinite(args.C) or args.C <= 0:
        parser.error("--C must be a positive finite number")

    if not math.isfinite(args.p) or args.p <= 0:
        parser.error("--p must be a positive finite number")

    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")

    if args.gamma not in ("scale", "auto"):
        try:
            gamma = float(args.gamma)
            if not math.isfinite(gamma) or gamma <= 0:
                raise ValueError()
        except ValueError:
            parser.error("--gamma must be scale, auto, or a positive finite float")
    else:
        gamma = args.gamma

    params = {
        "knn": dict(k=args.k, distance_metric=args.distance_metric,
                    voting_scheme=args.voting_scheme, retention_policy=args.retention_policy,
                    alpha=args.alpha, p=args.p),
        "svm": dict(kernel=args.kernel, C=args.C, gamma=gamma, degree=args.degree),
        "rf": dict(n_estimators=args.n_estimators, criterion=args.criterion,
                   max_depth=args.max_depth),
    }[args.model]
    algorithms = {"knn": kNNAlgorithm, "svm": svmAlgorithm, "rf": rfAlgorithm}
    algorithm = algorithms[args.model]
    preprocess_fn = preprocess_hypothyroid if args.dataset == "hypothyroid" else preprocess
    print(f"Dataset: {args.dataset} | Model: {args.model} | Parameters: {params}")
    rows = []
    for fold in range(args.folds):
        X_train, y_train, X_test, y_test = load_fold(str(HERE.parent / "data" / args.dataset), fold)
        X_train, X_test = preprocess_fn(X_train, X_test)
        result = algorithm(X_train, y_train, X_test, y_test, **params)
        rows.append({"dataset": args.dataset, "fold": fold, "model": args.model,
                     **params, "n_test": len(y_test),
                     **{key: value for key, value in result.items() if key != "predictions"}})
        print(f"Fold {fold:02d}: accuracy={result['accuracy']:.4f}, "
              f"f1_macro={result['f1_macro']:.4f}, "
              f"prediction_time={result['prediction_time']:.4f}s")
    df = pd.DataFrame(rows)
    print("\nMean over folds:")
    print(df[["accuracy", "f1_macro", "training_time", "prediction_time", "peak_memory_mb"]].mean().to_string())
    if args.output is not None:
        destination = args.output if args.output.is_absolute() else HERE / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination, index=False)
        print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
