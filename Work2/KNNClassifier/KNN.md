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
├── benchmarks.py           # Common benchmark_classifier function (measures time, memory, accuracy, F1)
├── knn_experiments.ipynb   # Interactive experiments, single-fold test, and 10-fold grid search
└── KNNClassifier/
    ├── __init__.py         # Exports the package modules cleanly
    ├── knn_classifier.py   # Main KNNClassifier class
    ├── distance_metrics.py # Vectorized distance functions (Euclidean, Manhattan, Canberra, Hamming)
    ├── voting_schemes.py   # Voting strategies (Majority, Inverse Distance) and tie-breaking
    ├── retention_policies.py # Online case base managers (NR, AR, DC, OG)
    └── KNN.md              # Documentation and teammate guide
```

---

## 3. The 4 Configurable Parameters

### Parameter 1: Distance Metric (`distance_metric`)

1. **`'euclidean'`:**
   - **Explanation:** Measures the direct straight-line distance between two data points across continuous feature dimensions.
   - **Suitable When:** Features are continuous, normalized, and have dense, geometric relationships without heavy outliers.

2. **`'manhattan'`:**
   - **Explanation:** Measures distance along grid-like right-angle paths by summing the absolute differences across each feature.
   - **Suitable When:** Working with high-dimensional spaces or datasets where feature differences along individual axes should not be exaggerated quadratically.

3. **`'canberra'`:**
   - **Explanation:** A weighted fractional metric that divides absolute differences by the sum of feature magnitudes, making it sensitive to relative proportional changes.
   - **Suitable When:** Data contains positive values or scattered points where relative percentage differences near zero matter more than absolute scale.

4. **`'hamming'`:**
   - **Explanation:** Measures the proportion of feature positions at which two samples have different values.
   - **Suitable When:** Working with categorical, binary, or one-hot encoded attributes where you need to count exact mismatches between features.

---

### Parameter 2: $K$ Value (`k`)
- Specifies how many nearest neighbors are retrieved to make a decision.
- Configurable to any positive integer $k \ge 1$ (typically tested on values like 3, 5, or 7).

---

### Parameter 3: Voting Schemes (`voting_scheme`)

1. **`'majority'` (Majority Class):**
   - Each of the $k$ nearest neighbors casts an equal vote (weight = 1) for its class, and the most common class wins.
   - **Tie-Breaking Rule:** If two or more classes tie for the highest count, the tie is resolved by picking the class whose closest single neighbor is nearest to the query point.

2. **`'inverse_distance'` (Inverse Distance Weighted):**
   - Closer neighbors receive stronger voting influence than distant neighbors by weighting each vote inversely proportional to its distance.
   - **Tie-Breaking Rule:** If total weighted scores tie, the class containing the closest single neighbor is chosen.

---

### Parameter 4: Retention & Forgetting Policies (`retention_policy`)

1. **`'never_retain'` (NR - Standard Baseline):**
   - The case base remains strictly the original training set throughout testing. No test cases are stored.

2. **`'always_retain'` (AR):**
   - Every classified test case is immediately added to the case base, continuously expanding the stored memory.

3. **`'different_class'` (DC / IB2 style):**
   - Only test cases that were incorrectly predicted are added to the case base, selectively reinforcing decision boundaries.

4. **`'oblivion_by_goodness'` (OG):**
   - Stored cases maintain a usefulness score (starting at a neutral 0.5 baseline).
   - Each time a test case is classified, the $k$ retrieved neighbors receive a reward if they match the true class and a penalty if they do not.
   - If a retrieved neighbor's usefulness drops below the 0.5 baseline, it is permanently removed from the case base to discard noisy or misleading samples.

---

## 4. Key Performance Optimizations

1. **Vectorized Matrix Computation:**
   - For standard fixed evaluation (`never_retain`), distance calculations across the entire test matrix are performed simultaneously using vectorized matrix operations rather than slow Python loops.
2. **Linear-Time Neighbor Selection:**
   - Utilizes fast partial partitioning to locate top-$k$ smallest distances in linear time without sorting the entire dataset.
3. **Dynamic Online Management:**
   - Stateful policies (`AR`, `DC`, `OG`) run through a dedicated manager that maintains case indices and removes forgotten instances seamlessly.

---

## 5. Usage Examples

### Benchmarking with `benchmarks.py`

```python
from KNNClassifier import KNNClassifier
from benchmarks import benchmark_classifier

# Instantiate classifier with your chosen hyperparameters
model = KNNClassifier(
    k=5,
    distance_metric="euclidean",
    voting_scheme="majority",
    retention_policy="never_retain",
)

# Benchmark execution time, peak memory, accuracy, and macro F1
results = benchmark_classifier(model, X_train, y_train, X_test, y_test)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Macro F1: {results['f1_macro']:.4f}")
print(f"Training Time: {results['training_time']:.4f} s")
print(f"Prediction Time: {results['prediction_time']:.4f} s")
print(f"Peak Memory: {results['peak_memory_mb']:.2f} MB")
```

### Direct Model Prediction

```python
from KNNClassifier import KNNClassifier

model = KNNClassifier(
    k=3,
    distance_metric="hamming",
    voting_scheme="inverse_distance",
    retention_policy="oblivion_by_goodness",
)

model.fit(X_train, y_train)
predictions = model.predict(X_test, y_test=y_test)
```
