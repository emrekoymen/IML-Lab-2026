import numpy as np

from KNNClassifier.voting_schemes import majority_voting


def relative_neighbourhood_graph(D, block_size=512):
    """Boolean adjacency: i ~ j when d(i, j) <= max(d(i, k), d(j, k)) for every k."""
    n = D.shape[0]
    adjacency = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        d_i = D[i]
        others = np.arange(i + 1, n)
        # Check in blocks to bound memory
        for start in range(i + 1, n, block_size):
            end = min(start + block_size, n)
            # D[start:end] has shape (B, n); np.maximum(d_i, D[start:end]) has shape (B, n)
            lune_radius = np.maximum(d_i, D[start:end]).min(axis=1)
            adjacency[i, start:end] = d_i[start:end] <= lune_radius
    return adjacency | adjacency.T


def rnge(D, y, order=1):
    """Sorted indices kept by RNG editing of order 1 or 2 (one pass)."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    codes = np.unique(y, return_inverse=True)[1]
    if len(codes) < 2:
        return np.arange(len(codes))

    adjacency = relative_neighbourhood_graph(D)
    discard = np.zeros(len(codes), dtype=bool)
    for i in range(len(codes)):
        neighbours = np.flatnonzero(adjacency[i])
        if neighbours.size:
            discard[i] = majority_voting(codes[neighbours], D[i, neighbours]) != codes[i]

    if order == 2:
        for i in np.flatnonzero(discard):
            same_class = np.flatnonzero(adjacency[i] & (codes == codes[i]))
            group = np.append(same_class, i)
            pool = adjacency[group].any(axis=0)
            pool[i] = False  # i cannot vote for itself
            other = np.count_nonzero(codes[pool] != codes[i])
            discard[i] = other > np.count_nonzero(pool) / 2

    return np.flatnonzero(~discard)
