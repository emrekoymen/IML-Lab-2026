# k-Nearest Neighbors (kNN) Classifier Guide

This folder provides a modular and configurable implementation of the k-Nearest Neighbors (kNN) classifier built from scratch in Python for IML Work 2.

---

## 1. What is k-Nearest Neighbors (kNN)?

kNN is an intuitive lazy learning classifier:
- Instead of learning an explicit model during training, it stores the training examples in memory as a case base.
- When classifying a new test sample, it finds the $k$ most similar training examples using a chosen distance metric and determines the predicted class through a voting scheme.
- Optionally, it can dynamically update or prune stored examples using online retention and forgetting policies.

---

## 2. Directory Structure

```
Work2/
├── benchmarks.py           # Common benchmark_classifier function (measures time, memory, accuracy, F1, n_correct, n_incorrect, storage_pct)
├── run_knn_experiments.py  # Automated 10-fold CV grid search, statistical tests (Demšar 2006), and plots
├── knn_experiments.ipynb   # Interactive experiments, single-fold test, and 10-fold grid search
└── KNNClassifier/
    ├── __init__.py          # Exports the package modules cleanly
    ├── knn_classifier.py    # Main KNNClassifier class & kNNAlgorithm(...) wrapper
    ├── distance_metrics.py  # Vectorized distance functions (Euclidean, Manhattan, Canberra, HEOM, Hamming)
    ├── voting_schemes.py    # Voting strategies (Majority, Inverse Distance) and tie-breaking
    ├── retention_policies.py # Online case base managers (NR, AR, DC, OG)
    ├── statistical_tests.py # Wilcoxon signed-rank, Friedman, Nemenyi/Bonferroni-Dunn & CD diagrams
    └── KNN.md               # Documentation and teammate guide
```

---

## 3. The 4 Configurable Parameters

### Parameter 1: Distance Metric (`distance_metric`)

1. **`'euclidean'`:**
   - **Explanation:** Measures the direct straight-line distance (Minkowski $r=2$) between two data points across feature dimensions.
   - **Suitable When:** Features are continuous, normalized, and have dense, geometric relationships without heavy outliers.

2. **`'manhattan'`:**
   - **Explanation:** Measures distance along grid-like right-angle paths (Minkowski $r=1$) by summing the absolute differences across each feature.
   - **Suitable When:** Working with high-dimensional spaces or datasets where feature differences along individual axes should not be exaggerated quadratically.

3. **`'canberra'`:**
   - **Explanation:** A weighted fractional metric that divides absolute differences by the sum of feature magnitudes, making it sensitive to relative proportional changes.
   - **Suitable When:** Data contains positive values or scattered points where relative percentage differences near zero matter more than absolute scale.

4. **`'heom'` (Heterogeneous Euclidean Overlap Metric):**
   - **Explanation:** Introduced by Wilson & Martinez (1997) to handle mixed numerical and nominal attributes properly. Uses the overlap metric ($0$ if equal, $1$ if different, scaled by $0.5$ per one-hot binary column so a nominal mismatch contributes $1.0$ to squared distance) for categorical features and range-normalized difference $|q_f - x_f|$ for continuous features in $[0, 1]$.
   - **Suitable When:** Datasets contain mixed numerical and categorical attributes (such as `hypothyroid`). (`'hamming'` is also available for purely categorical comparisons.)

---

### Parameter 2: $K$ Value (`k`)
- Specifies how many nearest neighbors are retrieved to make a decision.
- Configurable to any positive integer $k \ge 1$ (evaluated across $k \in \{3, 5, 7\}$).

---

### Parameter 3: Voting Schemes (`voting_scheme`)

1. **`'majority'` (Majority Class):**
   - Each of the $k$ nearest neighbors casts an equal vote (weight = 1) for its class, and the most common class wins.
   - **Tie-Breaking Rule:** If two or more classes tie for the highest count, the tie is resolved deterministically by picking the class whose closest single neighbor is nearest to the query point.

2. **`'inverse_distance'` (Inverse Distance Weighted):**
   - Closer neighbors receive stronger voting influence than distant neighbors by weighting each vote inversely proportional to its distance ($w_c = 1 / (d(q, x_c) + \epsilon)^p$, default $p=1$).
   - **Tie-Breaking Rule:** If total weighted scores tie, the class containing the closest single neighbor is chosen.

---

### Parameter 4: Retention & Forgetting Policies (`retention_policy`)

1. **`'never_retain'` (NR - Standard Baseline):**
   - The case base remains strictly the original training set throughout testing (Storage = 100%). No test cases are stored.

2. **`'always_retain'` (AR):**
   - Every classified test case is immediately added to the case base, continuously expanding the stored memory (Storage > 100%).

3. **`'different_class'` (DC / IB2 style):**
   - Only test cases that were incorrectly predicted are added to the case base, selectively reinforcing decision boundaries.

4. **`'oblivion_by_goodness'` (OG):**
   - Stored cases maintain a usefulness score (starting at a neutral $G_0 = 0.5$ baseline).
   - Each time a test case is classified, only the $k$ retrieved neighbors update their goodness via $G_t(c) = G_{t-1}(c) + \alpha (r(c) - G_{t-1}(c))$ with $\alpha = 0.2$ and reward $r(c) \in \{0, 1\}$.
   - If a retrieved neighbor's usefulness drops below $G_0 = 0.5$, it is permanently removed from the case base (Storage < 100%).

---

## 4. Key Performance Optimizations

1. **Vectorized Matrix Computation:**
   - Distance calculations across the test and training matrices are performed simultaneously using vectorized NumPy operations.
2. **Precomputed Incremental Distances for Online Policies:**
   - Stateful policies (`AR`, `DC`, `OG`) precompute the test-to-train distance matrix once and incrementally compute distances only to newly added test points while masking forgotten instances.
3. **Linear-Time Neighbor Selection:**
   - Utilizes `np.argpartition` to locate top-$k$ smallest distances in $O(N)$ time without sorting the entire dataset.

---

## 5. Usage Examples

### Using `kNNAlgorithm(...)`

```python
from KNNClassifier import kNNAlgorithm

predictions, train_time, predict_time = kNNAlgorithm(
    X_train,
    y_train,
    X_test,
    y_test=y_test,
    k=5,
    distance_metric="heom",
    voting_scheme="inverse_distance",
    retention_policy="never_retain",
)
```

### Benchmarking with `benchmarks.py`

```python
from KNNClassifier import KNNClassifier
from benchmarks import benchmark_classifier

# Instantiate classifier with your chosen hyperparameters
model = KNNClassifier(
    k=5,
    distance_metric="heom",
    voting_scheme="majority",
    retention_policy="oblivion_by_goodness",
)

# Benchmark time, memory, accuracy, F1, n_correct, n_incorrect, and storage_pct
results = benchmark_classifier(model, X_train, y_train, X_test, y_test)

print(f"Accuracy:        {results['accuracy']:.4f} ({results['n_correct']} correct, {results['n_incorrect']} incorrect)")
print(f"Macro F1:        {results['f1_macro']:.4f}")
print(f"Final Storage:   {results['storage_pct']:.2f}% ({results['n_stored']} instances)")
print(f"Training Time:   {results['training_time']:.4f} s")
print(f"Prediction Time: {results['prediction_time']:.4f} s")
print(f"Peak Memory:     {results['peak_memory_mb']:.2f} MB")
```

