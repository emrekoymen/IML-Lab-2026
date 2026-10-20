import numpy as np

from ._common import knn_predict, stratified_folds

RHO_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 0.99)


def min_enemy_distance(D, codes):
    """delta_n, skipping zeros (identical samples with different labels)."""
    different = codes[:, None] != codes[None, :]
    distances = D[different & (D > 0)]
    return float(distances.min()) if distances.size else 0.0


def _most_voted(D, members):
    """Each member votes for its nearest other member; ties go to the lowest index."""
    block = D[np.ix_(members, members)].copy()
    np.fill_diagonal(block, np.inf)
    votes = np.bincount(block.argmin(axis=1), minlength=len(members))
    return members[int(votes.argmax())]


def gcnn(D, y, rho=0.5):
    """Sorted indices of the GCNN prototypes, rho in [0, 1)."""
    if not 0 <= rho < 1:
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    codes = np.unique(y, return_inverse=True)[1]
    n = len(codes)
    delta = min_enemy_distance(D, codes)
    labels = np.unique(codes)

    is_prototype = np.zeros(n, dtype=bool)
    nearest_same = np.full(n, np.inf)
    nearest_other = np.full(n, np.inf)

    def add_prototypes(new):
        for prototype in new:
            is_prototype[prototype] = True
            column = D[:, prototype]
            same = codes == codes[prototype]
            nearest_same[same] = np.minimum(nearest_same[same], column[same])
            nearest_other[~same] = np.minimum(nearest_other[~same], column[~same])

    # G1: one voted prototype per class
    add_prototypes([_most_voted(D, np.flatnonzero(codes == label)) for label in labels])

    while True:
        # G2
        absorbed = nearest_other - nearest_same > rho * delta
        # A prototype stays unabsorbed only if an identical sample has another label
        candidates = ~absorbed & ~is_prototype
        if not candidates.any():
            break

        # G3
        add_prototypes([
            _most_voted(D, np.flatnonzero(candidates & (codes == label)))
            for label in labels
            if (candidates & (codes == label)).any()
        ])

    return np.flatnonzero(is_prototype)


def select_rho(D, y, rho_grid=RHO_GRID, n_folds=5, k=1, voting_scheme="majority", p=1.0, seed=0):
    """Inner stratified CV on the training set. Returns (best_rho, {rho: accuracy})."""
    codes = np.unique(y, return_inverse=True)[1]
    folds = stratified_folds(codes, n_folds, seed)

    scores = {}
    for rho in rho_grid:
        accuracies = []
        for fold in folds:
            if fold.size == 0:
                continue
            train = np.setdiff1d(np.arange(len(codes)), fold)
            kept = train[gcnn(D[np.ix_(train, train)], codes[train], rho)]
            predictions = knn_predict(D[np.ix_(fold, kept)], codes[kept], k, voting_scheme, p)
            accuracies.append(np.mean(predictions == codes[fold]))
        scores[rho] = float(np.mean(accuracies))

    best_rho = min(rho_grid, key=lambda rho: (-scores[rho], rho))
    return best_rho, scores
