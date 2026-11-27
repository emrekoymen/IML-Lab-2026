"""
Single-config CLI runner for verification and report reproduction.

For full parameter sweeps run experiments/run_<algo>.py directly.

Examples
--------
python main.py --algorithm dbscan --dataset glass --eps 0.5 --min-samples 5 --metric euclidean
python main.py --algorithm birch  --dataset hypothyroid --threshold 0.5 --n-clusters 4
python main.py --algorithm kmeans --dataset soybean --n-clusters 6 --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from core.preprocessing import load_preprocessed
from core.metrics import evaluate_clustering, build_confusion_matrix


# ---------------------------------------------------------------------------
# Model factories — one per algorithm
# ---------------------------------------------------------------------------

def _make_dbscan(args):
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=args.eps, min_samples=args.min_samples,
                  metric=args.metric, algorithm=args.algorithm)


def _make_birch(args):
    from sklearn.cluster import Birch
    n = None if args.n_clusters == 0 else args.n_clusters
    return Birch(threshold=args.threshold, branching_factor=args.branching_factor,
                 n_clusters=n)




FACTORIES = {
    "dbscan": _make_dbscan,
    "birch": _make_birch,

}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a single clustering configuration.")
    p.add_argument("--algorithm", required=True, choices=list(FACTORIES),
                   help="Algorithm to run")
    p.add_argument("--dataset", required=True, choices=["glass", "hypothyroid", "soybean"],
                   help="Dataset name")

    # DBSCAN
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument("--metric", type=str, default="euclidean",
                   help="Distance metric (dbscan)")
    p.add_argument("--algorithm-param", dest="algorithm", default="auto",
                   help="DBSCAN internal algorithm (auto/ball_tree/kd_tree/brute)")

    # BIRCH
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--branching-factor", type=int, default=50)
    p.add_argument("--n-clusters", type=int, default=0,
                   help="Number of clusters (0 = None for BIRCH)")

    # Stochastic
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m", type=float, default=2.0, help="FCM fuzzifier")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(f"\nLoading {args.dataset}...")
    X, y_true, meta = load_preprocessed(args.dataset)
    print(f"  {meta['n_samples']} samples × {meta['n_features']} features  "
          f"({meta['n_classes']} classes)")

    model = FACTORIES[args.algorithm](args)
    print(f"\nRunning {args.algorithm.upper()}...")
    labels = np.asarray(model.fit_predict(X), dtype=int)

    results = evaluate_clustering(X, labels, y_true)
    cm = build_confusion_matrix(y_true, labels)

    print("\n--- Results ---")
    for k, v in results.items():
        print(f"  {k:<22}: {v}")

    print("\n--- Confusion matrix (true rows × cluster cols) ---")
    print(cm.to_string())


if __name__ == "__main__":
    main()
