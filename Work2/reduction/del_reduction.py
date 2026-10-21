"""DEL (Description Length) instance reduction.

Implements the DEL algorithm from:
    D. R. Wilson and T. R. Martinez (2000). Reduction Techniques for Instance-Based
    Learning Algorithms. Machine Learning, 38(3), 257-286.
    DOI: https://doi.org/10.1023/A:1007626913721

The MDL (Minimum Description Length) encoding cost is used to decide whether removing
an instance shortens the total description of the training set. A noise-filter pass
(remove misclassified instances first) is followed by a greedy removal pass ordered
furthest-enemy-first, matching the DROP3/DEL ordering described in the paper (Sec. 5).
"""
import math

import numpy as np

from KNNClassifier.voting_schemes import resolve_votes


class _EncodingCost:
    def __init__(self, n, n_classes):
        self.n = n
        self.class_bits = math.log2(n_classes)
        self.exception_bits = math.log2(n_classes - 1) if n_classes > 1 else 0.0
        self.log_factorial = np.array([math.lgamma(i + 1) for i in range(n + 1)])

    def subset_bits(self, m, n):
        """F(m, n): log* of the number of subsets of size <= m of n items."""
        j = np.arange(min(m, n) + 1)
        log2_terms = (self.log_factorial[n] - self.log_factorial[j] - self.log_factorial[n - j]) / math.log(2)
        top = log2_terms.max()
        log2_sum = top + math.log2(np.sum(np.exp2(log2_terms - top)))
        total, term = 0.0, log2_sum
        while term > 0:
            total += term
            term = math.log2(term)
        return total

    def __call__(self, m, x):
        """Cost of keeping m instances with x exceptions."""
        return (self.subset_bits(m, self.n) + m * self.class_bits
                + self.subset_bits(x, self.n - m) + x * self.exception_bits)


def del_reduction(D, y, k=3, voting_scheme="majority", p=1.0):
    """Sorted indices kept by DEL; k, voting_scheme and p define the inner kNN."""
    classes, codes = np.unique(y, return_inverse=True)
    n = len(codes)
    if n <= 1:
        return np.arange(n)

    cost = _EncodingCost(n, len(classes))
    retained = np.ones(n, dtype=bool)

    without_self = D.copy()
    np.fill_diagonal(without_self, np.inf)
    nearest = np.argsort(without_self, axis=1, kind="stable")[:, : min(k + 1, n - 1)]

    neighbours = [list(row) for row in nearest]
    associates = [set() for _ in range(n)]
    for i, row in enumerate(neighbours):
        for j in row:
            associates[j].add(i)

    def is_misclassified(i, neighbour_list):
        if not neighbour_list:
            return True
        voters = np.asarray(neighbour_list[:k])
        return resolve_votes(codes[voters], D[i, voters], scheme=voting_scheme, p=p) != codes[i]

    # Tracked for every instance; only removed ones count as exceptions
    misclassified = np.array([is_misclassified(i, neighbours[i]) for i in range(n)])
    m, x = n, 0

    def exceptions_after_removing(i):
        change = int(misclassified[i])
        for associate in associates[i]:
            if not retained[associate]:
                others = [j for j in neighbours[associate] if j != i]
                change += int(is_misclassified(associate, others)) - int(misclassified[associate])
        return x + change

    def try_remove(i):
        nonlocal m, x
        if m <= 1:
            return False
        if cost(m - 1, exceptions_after_removing(i)) > cost(m, x):
            return False

        retained[i] = False
        for associate in associates[i]:
            neighbours[associate].remove(i)
            candidates = retained.copy()
            candidates[associate] = False
            candidates[neighbours[associate]] = False
            if candidates.any():
                replacement = int(np.where(candidates, D[associate], np.inf).argmin())
                neighbours[associate].append(replacement)
                associates[replacement].add(associate)
            misclassified[associate] = is_misclassified(associate, neighbours[associate])
        # Its own neighbour list stays (DROP2)
        associates[i] = set()
        m -= 1
        x = int(np.count_nonzero(misclassified & ~retained))
        return True

    # Noise filter
    for i in range(n):
        if retained[i] and misclassified[i]:
            try_remove(i)

    # Furthest from the nearest enemy first
    remaining = np.flatnonzero(retained)
    enemy_distances = D[np.ix_(remaining, remaining)].copy()
    enemy_distances[codes[remaining][:, None] == codes[remaining][None, :]] = np.inf
    removal_order = remaining[np.argsort(-enemy_distances.min(axis=1), kind="stable")]

    changed = True
    while changed:
        changed = False
        for i in removal_order:
            if retained[i] and try_remove(i):
                changed = True

    return np.flatnonzero(retained)
