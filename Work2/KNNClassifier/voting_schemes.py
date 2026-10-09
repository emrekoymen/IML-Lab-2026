"""Voting schemes and tie-breaking for kNN."""

from collections import defaultdict
import numpy as np


def majority_voting(neighbor_labels, neighbor_distances):
    """Majority class voting with closest-neighbor tie-breaker."""
    counts = defaultdict(int)
    min_dist_per_class = {}

    for label, dist in zip(neighbor_labels, neighbor_distances):
        counts[label] += 1
        if label not in min_dist_per_class or dist < min_dist_per_class[label]:
            min_dist_per_class[label] = dist

    max_count = max(counts.values())
    candidates = [cls for cls, count in counts.items() if count == max_count]

    if len(candidates) == 1:
        return candidates[0]

    # Break tie by closest individual neighbor
    return min(candidates, key=lambda cls: min_dist_per_class[cls])


def inverse_distance_voting(neighbor_labels, neighbor_distances, p=1.0, eps=1e-9):
    """Inverse distance weighted voting with closest-neighbor tie-breaker."""
    scores = defaultdict(float)
    min_dist_per_class = {}

    weights = 1.0 / ((neighbor_distances + eps) ** p)

    for label, dist, weight in zip(neighbor_labels, neighbor_distances, weights):
        scores[label] += weight
        if label not in min_dist_per_class or dist < min_dist_per_class[label]:
            min_dist_per_class[label] = dist

    max_score = max(scores.values())
    candidates = [cls for cls, score in scores.items() if np.isclose(score, max_score, atol=1e-12)]

    if len(candidates) == 1:
        return candidates[0]

    return min(candidates, key=lambda cls: min_dist_per_class[cls])


VOTING_SCHEMES = {
    "majority": majority_voting,
    "inverse_distance": inverse_distance_voting,
}


def resolve_votes(neighbor_labels, neighbor_distances, scheme="majority", p=1.0, eps=1e-9):
    key = scheme.lower()
    if key == "majority":
        return majority_voting(neighbor_labels, neighbor_distances)
    elif key in ("inverse_distance", "weighted"):
        return inverse_distance_voting(neighbor_labels, neighbor_distances, p=p, eps=eps)
    else:
        raise ValueError(f"Unknown voting scheme '{scheme}'. Choices: {list(VOTING_SCHEMES.keys())}")
