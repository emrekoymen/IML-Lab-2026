from .knn_classifier import KNNClassifier, k_nearest
from .distance_metrics import (
    compute_distance_matrix,
    euclidean_distance,
    manhattan_distance,
    canberra_distance,
    heom_distance,
    DISTANCES,
)
from .voting_schemes import (
    majority_voting,
    inverse_distance_voting,
    resolve_votes,
    VOTING_SCHEMES,
)
from .retention_policies import RetentionManager
from .statistical_tests import (
    wilcoxon_test,
    friedman_test,
    plot_critical_difference_diagram,
)

__all__ = [
    "KNNClassifier",
    "k_nearest",
    "compute_distance_matrix",
    "euclidean_distance",
    "manhattan_distance",
    "canberra_distance",
    "heom_distance",
    "DISTANCES",
    "majority_voting",
    "inverse_distance_voting",
    "resolve_votes",
    "VOTING_SCHEMES",
    "RetentionManager",
    "wilcoxon_test",
    "friedman_test",
    "plot_critical_difference_diagram",
]


