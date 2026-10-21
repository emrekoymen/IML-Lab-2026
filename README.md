# IML Work 2 — Lazy Learning, SVM, Random Forest & Instance Reduction

> **Course:** Introduction to Machine Learning (MAI — UPC, UB, URV)
> **Datasets:** Soybean (683 instances, 35 features) · Hypothyroid (3,772 instances, 29 features)
> **Evaluation:** Pre-supplied 10-fold cross-validation ARFF files (do not regenerate splits)

---

## Project Structure

```
IML-Lab-2026/
├── Work2/
│   ├── KNNClassifier/              # kNN from scratch (distance metrics, voting, retention)
│   │   ├── knn_classifier.py       #   KNNClassifier class + kNNAlgorithm function
│   │   ├── distance_metrics.py     #   Euclidean, Manhattan, Canberra, HEOM, Hamming
│   │   ├── voting_schemes.py       #   Majority vote, Inverse-distance weighted
│   │   ├── retention_policies.py   #   NR, AR, DC (IB2), Oblivion-by-Goodness
│   │   └── statistical_tests.py    #   Demšar (2006): Wilcoxon, Friedman, CD diagrams
│   │
│   ├── reduction/                  # Instance reduction techniques (from scratch)
│   │   ├── gcnn.py                 #   Condensed: Generalised CNN
│   │   ├── rnge.py                 #   Edited:    Relative Neighbourhood Graph Editing
│   │   └── del_reduction.py        #   Hybrid:    DEL
│   │
│   ├── models.py                   # kNNAlgorithm · reduction_kNNAlgorithm · svmAlgorithm · rfAlgorithm
│   ├── benchmarks.py               # benchmark_classifier — timing + memory + metrics
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
├── docs/
│   ├── Work2_Description.pdf
│   └── Demsar_2006.pdf
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

Run from the `Work2/` directory. Experiments are independent and can be run in parallel, but Step 1 (KNN) should complete before Step 4 (reduction) since reduction uses the best KNN config.

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

Every algorithm produces the same fold-level CSV columns:

| Column | Description |
|--------|-------------|
| `dataset`, `fold`, `algorithm`, `model` | Identity |
| `n_test`, `n_correct`, `n_incorrect` | Prediction counts |
| `accuracy`, `f1_macro` | Performance |
| `training_time`, `prediction_time`, `peak_memory_mb` | Efficiency & memory |
| `n_stored`, `storage_pct` | Instance storage (KNN-specific; 100% for SVM/RF) |

Summary CSVs add `{col}_std` columns (standard deviation across folds).

---

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

