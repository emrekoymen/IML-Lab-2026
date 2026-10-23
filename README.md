# IML Work 2 - Lazy Learning, SVM, Random Forest & Instance Reduction

> **Datasets:** Soybean (683 instances, 35 features) · Hypothyroid (3,772 instances, 29 features)

---

## Team

- **Hasti Sadat Mousavinia**
- **Michelle Miranda**
- **Emre Köymen**
- **Fatima Zahra**


## Environment

- Python: 3.11.9
- Tested operating system: Windows
- Development environment: PyCharm
- Dependencies: See `requirements.txt`

---

## Project Structure

```
IML-Lab-2026/
├── Work2/
│   ├── KNNClassifier/              #   kNN from scratch (distance metrics, voting, retention)
│   │   ├── knn_classifier.py       #   KNNClassifier class + kNNAlgorithm function
│   │   ├── distance_metrics.py     #   Euclidean, Manhattan, Canberra, HEOM
│   │   ├── voting_schemes.py       #   Majority vote, Inverse-distance weighted
│   │   ├── retention_policies.py   #   NR, AR, DC (IB2), Oblivion-by-Goodness
│   │   ├── statistical_tests.py    #   Demšar (2006): Wilcoxon, Friedman, CD diagrams
│   │   └── KNN.md                  #   Example of KNN evaluation output   
│   ├── reduction/                  #   Instance reduction techniques (from scratch)
│   │   ├── gcnn.py                 #   Condensed: Generalised CNN
│   │   ├── rnge.py                 #   Edited:    Relative Neighbourhood Graph Editing
│   │   └── del_reduction.py        #   Hybrid:    DEL
│   │
│   ├── main.py                     # Configurable single-configuration CLI runner
│   ├── models.py                   # kNNAlgorithm · reduction_kNNAlgorithm · svmAlgorithm · rfAlgorithm
│   ├── parser.py                   # load_fold — reads ARFF fold files
│   ├── preprocessing.py            # Imputation, normalisation, one-hot encoding
│   │
│   ├── experiments/                # Experiment runner scripts (long-running, headless)
│   │   ├── run_knn_experiments.py  #   KNN 10-fold grid search + Demšar stats + CD plots
│   │   ├── run_svm_experiments.py  #   SVM 10-fold grid search
│   │   ├── run_rf_experiments.py   #   RF  10-fold grid search
│   │   └── run_reduction_experiments.py  # Instance reduction + stats
│   │
│   ├── analysis/                   # Statistical analysis (Jupyter notebooks)
│   │   ├── statistical_utils.py    #   Shared loaders + wrappers for notebooks
│   │   ├── 00_data_inspection.ipynb
│   │   ├── 01_knn_model_selection.ipynb   # Best KNN config (Demšar tests)
│   │   ├── 02_svm_model_selection.ipynb   # Best SVM config (Demšar tests)
│   │   ├── 03_rf_model_selection.ipynb    # Best RF  config (Demšar tests)
│   │   ├── 04_algorithm_comparison.ipynb  # KNN vs SVM vs RF (accuracy + time)
│   │   └── 05_reduction_analysis.ipynb    # Reduction impact analysis
│   │
│   └── results/                    # All output CSVs and plots (auto-created)
│       ├── knn/                    #   fold_results, summary_results, stats, CD plots
│       ├── svm/
│       ├── rf/
│       ├── reduction/
│       └── comparison/             #   Cross-algorithm comparison outputs
│
├── data/
│   ├── soybean/                    # soybean.fold.00000X.{train,test}.arff
│   └── hypothyroid/                # hypothyroid.fold.00000X.{train,test}.arff
│
└── requirements.txt
```

---

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Required Python:** 3.11+

---

## Running Experiments

Run from the `Work2/` directory. Experiments are independent and can be run in parallel. The instance reduction script uses predefined best configurations stored in `BEST_CONFIGS` inside `run_reduction_experiments.py`; it does not automatically load them from the earlier experiment results.
```bash
cd Work2

# Step 1 — KNN grid search (k × metric × voting × retention, both datasets)
python experiments/run_knn_experiments.py

# Step 2 — SVM grid search (kernel × C × gamma × degree, both datasets)
python experiments/run_svm_experiments.py

# Step 3 — Random Forest grid search (n_estimators × criterion × max_depth, both datasets)
python experiments/run_rf_experiments.py

# Step 4 — Instance reduction experiments (GCNN, RNGE, DEL applied to best KNN/SVM/RF)
python experiments/run_reduction_experiments.py
```

Each script saves results to `results/{knn,svm,rf,reduction}/`.

---

## Running a Single Configuration with `main.py`

The configurable command-line runner `Work2/main.py` evaluates **one selected model configuration** on the supplied dataset folds. It does **not** search for the best configuration; use the grid-search scripts above for that purpose. It supports KNN, SVM, and Random Forest. Instance-reduction experiments use the separate `experiments/run_reduction_experiments.py` script.

Run the following commands **from the repository root** (`IML-Lab-2026/`), after installing the dependencies:

```bash
# Display all available command-line options
python Work2/main.py --help

# Random Forest: 100 trees, all 10 folds (default)
python Work2/main.py --dataset soybean --model rf --n-estimators 100

# SVM: RBF kernel, C=1.0, quick one-fold test
python Work2/main.py --dataset soybean --model svm --kernel rbf --C 1.0 --folds 1

# KNN: 5 neighbors, Manhattan distance, quick one-fold test
python Work2/main.py --dataset soybean --model knn --k 5 --distance-metric manhattan --folds 1

# Hypothyroid dataset with Random Forest
python Work2/main.py --dataset hypothyroid --model rf --n-estimators 100 --folds 1

# Save the per-fold results to a CSV file
python Work2/main.py --dataset soybean --model rf --n-estimators 100 --output results/custom/soybean_rf.csv
```

### CLI options and defaults

| Option | Applies to | Allowed values | Default |
|---|---|---|---|
| `--dataset` | All | `soybean`, `hypothyroid` | **Required** |
| `--model` | All | `knn`, `svm`, `rf` | **Required** |
| `--folds` | All | Integer from 1 to 10 | `10` |
| `--output` | All | Optional CSV file path | None (print only) |
| `--k` | KNN | Positive integer | `3` |
| `--distance-metric` | KNN | `euclidean`, `manhattan`, `canberra`, `heom` | `euclidean` |
| `--voting-scheme` | KNN | `majority`, `inverse_distance` | `majority` |
| `--retention-policy` | KNN | `never_retain`, `always_retain`, `different_class`, `oblivion_by_goodness` | `never_retain` |
| `--alpha` | KNN | Number from 0 to 1 | `0.2` |
| `--p` | KNN | Positive number | `1.0` |
| `--kernel` | SVM | `linear`, `rbf`, `poly`, `sigmoid` | `rbf` |
| `--C` | SVM | Positive number | `1.0` |
| `--gamma` | SVM | `scale`, `auto`, or positive number | `scale` |
| `--degree` | SVM | Positive integer | `3` |
| `--n-estimators` | RF | Positive integer | `100` |
| `--criterion` | RF | `gini`, `entropy`, `log_loss` | `gini` |
| `--max-depth` | RF | Positive integer | None (unlimited) |

Only the parameters relevant to the selected model are passed to that classifier. Unspecified parameters use the defaults shown above. Options for another classifier do not affect the selected model.

### Evaluation and output

- The runner uses the **provided** ARFF train/test folds; it does not create new cross-validation splits.
- With `--folds 10` (default), folds 0 through 9 are evaluated. With `--folds 1`, only fold 0 is evaluated for a quick test.
- The terminal displays each fold's **accuracy**, **macro-F1**, and **prediction time**, followed by mean accuracy, macro-F1, training time, prediction time, and peak memory usage.
- The optional `--output` writes a **per-fold CSV**, including dataset, fold, model, selected parameters, prediction counts, accuracy, macro-F1, and timing/memory metrics. The runner does not write a separate summary CSV.
- Relative `--output` paths are resolved from the **`Work2/` directory**, regardless of the current working directory. For example, `--output results/custom/soybean_rf.csv` creates `Work2/results/custom/soybean_rf.csv` (and creates missing directories). An existing file at the same path is overwritten.
- Input data are read from `data/soybean/` or `data/hypothyroid/` at the repository root. These dataset folders must be included when packaging the submission.

---

## Statistical Analysis

After experiments complete, open the notebooks in order:

```bash
cd Work2
jupyter lab analysis/
```

| Notebook | Purpose | Key Tests |
|----------|---------|-----------|
| `01_knn_model_selection` | Select best KNN config per dataset | Friedman (top-8 configs) → Nemenyi/BD → CD diagram · Wilcoxon #1 vs #2 · Separate accuracy & time |
| `02_svm_model_selection` | Select best SVM config per dataset | Same structure as above |
| `03_rf_model_selection` | Select best RF config per dataset | Same structure as above |
| `04_algorithm_comparison` | Compare best KNN vs SVM vs RF | Friedman → Nemenyi/BD → 3× Wilcoxon (Holm) · Accuracy block + Time block separately |
| `05_reduction_analysis` | Reduction impact on all algorithms | Load existing results from `run_reduction_experiments.py` · Narrative + plots |

All tests follow **Demšar (2006)**:
- **2 classifiers** → Wilcoxon Signed-Rank Test
- **>2 classifiers** → Friedman omnibus → Nemenyi (all-pairs) + Bonferroni-Dunn (vs control) post-hoc

---

## Results Schema

Model-specific hyperparameters are also included as separate columns
in the main CLI output (e.g., `n_estimators`, `criterion`, and
`max_depth` for Random Forest).

The main CLI (`main.py`) uses the following fold-level CSV schema:

| Column | Description |
|--------|-------------|
| `dataset`, `fold`, `model` | Identity |
| `n_test`, `n_correct`, `n_incorrect` | Prediction counts |
| `accuracy`, `f1_macro` | Performance |
| `training_time`, `prediction_time`, `peak_memory_mb` | Efficiency & memory |
| `n_stored`, `storage_pct` | Instance storage (KNN-specific; 100% for SVM/RF) |


---


### Standalone Experiment Results Schema

The standalone KNN, SVM, and Random Forest experiment scripts
generate fold-level CSV files with the following columns:

| Column | Description |
|--------|-------------|
| `fold` | Fold identifier |
| `model` | Model configuration |
| `correct_classification_count` | Number of correct predictions |
| `misclassification_count` | Number of incorrect predictions |
| `accuracy` | Classification accuracy |
| `prediction_time_ms` | Prediction time in milliseconds |

Their summary CSV files contain:

| Column | Description |
|--------|-------------|
| `model` | Model configuration |
| `accuracy` | Reported classification accuracy |
| `accuracy_std` | Standard deviation of accuracy across folds |
| `prediction_time_ms` | Reported prediction time in milliseconds |

### Instance Reduction Results Schema

The instance-reduction experiments generate fold-level CSV files
with the same columns as KNN, SVM, and RF, plus:

| Column | Description |
|--------|-------------|
| `storage_pct` | Percentage of training instances retained |

The instance-reduction summary CSV contains:

| Column | Description |
|--------|-------------|
| `model` | Model configuration |
| `algorithm` | Classification algorithm |
| `reduction_method` | Instance-reduction technique |
| `accuracy` | Reported classification accuracy |
| `accuracy_std` | Standard deviation of accuracy |
| `f1_macro` | Macro-averaged F1-score |
| `storage_pct` | Percentage of training instances retained |
| `reduction_time` | Time spent on instance reduction |
| `prediction_time` | Reported prediction time |
| `prediction_time_ms` | Prediction time in milliseconds |

Experiment results are saved in:

- `Work2/results/knn/`
- `Work2/results/svm/`
- `Work2/results/rf/`
- `Work2/results/reduction/`

These directories contain fold-level results, summary results,
statistical analysis outputs, and visualizations.

## kNN Parameter Space

| Parameter | Values explored |
|-----------|----------------|
| **k** | 3, 5, 7 |
| **Distance metric** | Euclidean, Manhattan, Canberra, HEOM |
| **Voting scheme** | Majority, Inverse-distance weighted |
| **Retention policy** | Never Retain (NR), Always Retain (AR), Different Class (DC), Oblivion by Goodness (OG) |

## Instance Reduction Techniques

| Type | Method |
|------|--------|
| Condensed | GCNN (Generalised Condensed Nearest Neighbour) |
| Edited | RNGE (Relative Neighbourhood Graph Editing) |
| Hybrid | DEL |

---

## Key Implementation Notes

- **No sklearn for kNN** — implemented from scratch in `KNNClassifier/`
- **Pre-supplied folds** — do not regenerate CV splits; use provided `.arff` files
- **All paths are relative** — scripts anchor with `Path(__file__).resolve().parent`; notebooks anchor with `Path().resolve().parent`
- **Statistical tests** — `KNNClassifier/statistical_tests.py` exports `wilcoxon_test`, `friedman_test`, `plot_critical_difference_diagram` (used by both experiment scripts and notebooks)



---

## Reproducibility

- The experiments use the 10 predefined training/test folds for Soybean and Hypothyroid without regenerating the splits.
- SVM and Random Forest use `random_state=42`.
- The custom KNN implementation has no explicit random seed, and no random-number operations were found in the source search.
- GCNN's internal stratified cross-validation uses a default random seed of `0` when selecting `rho`.
- To reproduce the experiments, use the specified Python environment, dependencies, datasets, and experiment commands.