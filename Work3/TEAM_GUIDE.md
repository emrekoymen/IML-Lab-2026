# Work 3 — Team Guide (Structure, Data Flow & Integration)

This guide explains how `Work3/` is organized, how data is loaded and preprocessed, how experiments are logged, and how **Persons 2, 3, and 4** plug their algorithms into the shared pipeline.

> Read this fully before writing any code. The key decisions are already locked — do not change them.

---

## 1. Rules Everyone Must Follow

1. **Python 3.11 & Relative Paths Only.** Anchor every file path with `Path(__file__).resolve().parent`. Never hardcode `C:\Users\...`.
2. **Scaling is locked: `StandardScaler`.** All numeric features are standardised (zero mean, unit variance). OHE binary columns from `hypothyroid`/`soybean` are compatible. **Do not change this — it would invalidate DBSCAN `eps` and BIRCH `threshold` values.**
3. **No Forbidden Libraries:**
   - `sklearn` **allowed** for: DBSCAN, BIRCH, Isomap, all preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`), and all evaluation metrics.
   - `sklearn` **forbidden** for: K-Means (Task 4), Improved K-Means (Task 5), CLARANS (Task 6), Fuzzy C-Means (Task 7), and PCA (Section 1.2). `scipy.stats.anderson` is allowed for G-Means/X-Means; `numpy` eigendecomposition is required for PCA.
4. **Never leak `y_true`.** Do **not** pass the class labels into any clustering algorithm, PCA/Isomap, or hyperparameter selection. `y_true` is used only post-hoc in `core/metrics.py` for external evaluation (ARI, Purity, F-measure, Confusion Matrix).
5. **Multiple random seeds for stochastic algorithms.** K-Means, IKM, CLARANS, and FCM must each run with `SEEDS = [0, 1, 2, 3, 4]` and report mean ± std. DBSCAN and BIRCH are deterministic — one run per config is enough.
6. **Don't import `data_loader` directly.** It is a low-level internal module. Always use `load_preprocessed()` from `core.preprocessing` as your single data entry point (see §3).

---

## 2. Folder Structure

```text
Work3/
├── data/           # Single-file .arff datasets (glass, hypothyroid, soybean)
├── core/           # Shared loader, preprocessing, metrics, runner, plotting
├── algorithms/     # Algorithm implementations — one file per algorithm
├── experiments/    # Parameter-sweep scripts — one script per algorithm
├── results/
│   ├── runs/       # Raw CSV per run (config × seed)
│   ├── summary/    # Aggregated CSV: 1 row per config with mean & std, plus is_best flag
│   ├── predictions/# Best cluster labels (.npy) for Person 4
│   ├── plots/      # PNG figures per algorithm/dataset
│   └── comparison/ # Cross-algorithm tables & plots (Person 4)
├── main.py         # Single-config CLI for reproduction (not full sweeps — see §6)
├── README.md
└── requirements.txt
```

---

## 3. Data & Preprocessing Flow

Three datasets in `Work3/data/` as single `.arff` files:

| Dataset | Instances | Features | Missing | Classes | Post-OHE dims |
|---|---|---|---|---|---|
| `glass.arff` | 214 | 9 numeric | 0% | 6 | 9 |
| `hypothyroid.arff` | 3,772 | 7 num + 22 nom | ~5.5% | 4 | ~50 |
| `soybean.arff` | 683 | 35 nominal | ~9.8% | 19 | ~100+ |

> ⚠️ **Soybean warning:** 35 nominal features expand to ~100+ dimensions after OHE. Euclidean-based algorithms (DBSCAN, BIRCH, K-Means) may struggle in this high-dimensional space. Plan to discuss this as a limitation in the report and prefer `cosine` distance where possible.

**To get the preprocessed data in any script:**

```python
from core.preprocessing import load_preprocessed

X, y_true, meta = load_preprocessed("hypothyroid")
# X      → np.ndarray (float64), shape (n_samples, n_features_encoded)
#           Numeric: median-imputed + StandardScaler
#           Nominal: mode-imputed + OneHotEncoded
# y_true → np.ndarray of str class labels — POST-HOC ONLY, never pass to fit_predict()
# meta   → dict: {"n_samples", "n_features", "n_classes", "class_names",
#                  "feature_names", "dataset"}
```

**Do not call `core.data_loader` directly.** `load_preprocessed()` calls it internally and applies all cleaning and transforms. Calling the loader directly gives you raw, unprocessed data.

---

## 4. How to Implement Your Algorithm (Persons 2 & 3)

You do **not** need to inherit from any base class. Follow the `fit_predict(X)` convention — the same as `scikit-learn` — and the runner handles everything else.

### Step 1 — Write your algorithm in `algorithms/<your_algo>.py`

```python
import numpy as np

class MyCLARANS:
    def __init__(self, n_clusters=3, max_neighbour=5, num_local=2, random_state=None):
        self.n_clusters = n_clusters
        self.max_neighbour = max_neighbour
        self.num_local = num_local
        self.random_state = random_state
        # optional: extra info logged by the runner
        self.extra_info_ = {}

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        # Implement from scratch using numpy only (no sklearn clustering!)
        # Populate self.extra_info_ if useful:
        #   e.g. {"n_iter": iters, "final_cost": cost}
        # Return 1D integer array of cluster IDs, shape (len(X),)
        # Use 0, 1, ..., k-1 — only use -1 for explicit noise (like DBSCAN)
        ...
        return labels  # np.ndarray, dtype=int
```

### Step 2 — Write your experiment script in `experiments/run_<your_algo>.py`

Define a parameter grid and pass a factory function to `run_experiment_suite`. The runner handles data loading, timing, memory, metric computation, CSV logging, best-config selection, label saving, and confusion matrix plotting automatically.

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.clarans import MyCLARANS
from core.runner import run_experiment_suite

PARAM_GRID = [
    {"n_clusters": k, "max_neighbour": mn}
    for k in [2, 3, 4, 6, 8, 10, 15, 19]
    for mn in [5, 10, 20]
]

def make_model(params: dict, seed: int | None) -> MyCLARANS:
    return MyCLARANS(**params, random_state=seed)

for dataset in ["glass", "hypothyroid", "soybean"]:
    run_experiment_suite(
        algo_name="clarans",
        factory_fn=make_model,
        dataset_name=dataset,
        param_grid=PARAM_GRID,
        seeds=[0, 1, 2, 3, 4],   # pass None for deterministic algorithms
    )
```

### Step 3 — What the runner produces automatically

| File | Content |
|---|---|
| `results/runs/<algo>_<dataset>_runs.csv` | One row per `(config, seed)` with all metrics, runtime, memory, `extra_info_` |
| `results/summary/<algo>_<dataset>_summary.csv` | One row per config: mean & std across seeds + `is_best` flag |
| `results/predictions/<algo>_<dataset>_best_labels.npy` | Cluster labels for the best config (used by Person 4) |
| `results/plots/<algo>/<dataset>_cm.png` | Confusion matrix of the best config vs `y_true` |

You only need to add **algorithm-specific plots** (elbow curve, FCM objective vs $C$, CLARANS cost history, G-Means $k$ estimation) inside your own `run_<algo>.py` using helpers from `core/plotting.py`.

### Reading `params_json` from summary CSVs

The `params_json` column stores hyperparameters as a JSON string. To parse them in analysis:

```python
import json
import pandas as pd

df = pd.read_csv("results/summary/clarans_glass_summary.csv")
df["params"] = df["params_json"].map(json.loads)
# Now df["params"] is a column of dicts, e.g. {"n_clusters": 4, "max_neighbour": 10}
```

---

## 5. Metrics & How to Interpret Them

All metrics computed by `core/metrics.py`. Internal metrics use **only `X` and `labels`** — safe for parameter tuning. External metrics use `y_true` — post-hoc evaluation only.

| Metric | Type | Range | Optimal | Interpretation |
|:---|:---|:---|:---|:---|
| **Silhouette** | Internal | `[-1, 1]` | Higher (`+1`) | `+1` = dense, well-separated clusters; `0` = overlapping; `< 0` = likely misassigned. Computed on non-noise points only. |
| **Davies–Bouldin** | Internal | `[0, ∞)` | **Lower (`0`)** | Average cluster-to-nearest-cluster similarity ratio. `0` = perfectly compact & separated. |
| **Calinski–Harabasz** | Internal | `[0, ∞)` | Higher | Between-cluster vs within-cluster variance ratio. Higher = tighter, better-separated. |
| **ARI** | External | `[-0.5, 1]` | `1.0` | `1.0` = exact match with ground truth; `~0` = random; `< 0` = worse than random. |
| **NMI** | External | `[0, 1]` | `1.0` | `1.0` = perfect mutual information with true classes; `0` = no correlation. |
| **Purity** | External | `[0, 1]` | `1.0` | Fraction of samples in their majority true class per cluster. Noise (`-1`) hurts purity. Inflates when `k` is large. |
| **Macro F-measure** | External | `[0, 1]` | `1.0` | Macro-F1 after Hungarian-optimal cluster↔class alignment. Robust to cluster ID arbitrariness. |

---

## 6. How Person 4 Integrates (PCA, Isomap & Cross-Algorithm Comparison)

### PCA from scratch (`algorithms/pca.py` + part of `experiments/run_visualization.py`)
- Load data with `X, y_true, meta = load_preprocessed(dataset)` — same `X` as everyone else.
- Implement the 9 required steps (PDF Section 1.2.1):
  1. Standardise (already done by `load_preprocessed` — document this).
  2. Compute mean vector.
  3. Compute covariance matrix — **print to console** (required by PDF).
  4. Compute eigenvectors & eigenvalues via `numpy.linalg.eigh` — **print to console**.
  5. Sort eigenvectors by eigenvalue descending — **print to console**.
  6. Select top $d$ components (2D and 3D).
  7. Project data onto principal components.
  8. Plot 2D/3D scatter coloured by `y_true`.
  9. Compute reconstruction error for `glass` (all numeric). For `hypothyroid`/`soybean`, note that OHE columns make MSE reconstruction error misleading for categorical features — discuss as a limitation.

### Isomap (`sklearn.manifold.Isomap`)
- Run on the same `X` from `load_preprocessed()`.
- Visualise in 2D and 3D, coloured by `y_true`.

### Clustering on PCA-reduced data (Section 1.2 Tasks 3–4)
- Read `results/summary/` CSVs from K-Means (`kmeans`), IKM (`ikmeans`), and CLARANS (`clarans`).
- Filter `is_best == True` per dataset, compare their `ari_mean` and `silhouette_mean`, and **explicitly state** which algorithm is selected as best and why.
- Import that class (e.g., `from algorithms.ikmeans import GMeans`), call `fit_predict(X_pca)` on the PCA-reduced data, evaluate with `core/metrics.evaluate_clustering()`, and compare against the full-`X` best labels in `results/predictions/<algo>_<dataset>_best_labels.npy`.
- Generate 2D/3D PCA scatter plots with cluster colour overlays. Repeat on Isomap 2D embedding.

### Cross-algorithm comparison (`experiments/compare_all.py`)
```python
import glob, json
import pandas as pd

files = glob.glob("results/summary/*_summary.csv")
df = pd.concat([pd.read_csv(f) for f in files])
best = df[df["is_best"] == True]
# best now has one row per (algorithm, dataset) — build comparison tables & bar charts
```

---

## 7. `main.py` — What It Is (and What It Isn't)

`main.py` runs **one specific configuration** for quick verification or report reproduction. It is **not** a full experiment sweep.

```bash
# From Work3/
python main.py --algorithm dbscan --dataset glass --eps 0.5 --min-samples 5 --metric euclidean
python main.py --algorithm birch --dataset hypothyroid --threshold 0.5 --n-clusters 4
```

**For full sweeps that generate all results:** run `experiments/run_<algo>.py` directly.
