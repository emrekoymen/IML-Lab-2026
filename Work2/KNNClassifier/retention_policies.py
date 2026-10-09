"""Retention and forgetting policies for kNN."""

import numpy as np


class RetentionManager:
    """Manages online instance retention and forgetting during prediction."""

    def __init__(self, X_train, y_train, policy="never_retain", alpha=0.2, initial_goodness=0.5):
        self.policy = policy.lower()
        self.alpha = alpha
        self.initial_goodness = initial_goodness

        self.X_base = np.array(X_train, dtype=float).copy()
        self.y_base = np.array(y_train).copy()

        if self.policy in ("oblivion_by_goodness", "og"):
            self.goodness = np.full(len(self.y_base), self.initial_goodness, dtype=float)
        else:
            self.goodness = None

    @property
    def is_dynamic(self):
        return self.policy not in ("never_retain", "nr")

    @property
    def size(self):
        return len(self.y_base)

    def update(self, query_x, true_label, pred_label, retrieved_indices):
        if self.policy in ("never_retain", "nr"):
            return

        if self.policy in ("always_retain", "ar"):
            self._add_case(query_x, true_label)

        elif self.policy in ("different_class", "dc", "df"):
            # Retain only misclassified test points (IB2 style)
            if true_label is not None and pred_label != true_label:
                self._add_case(query_x, true_label)

        elif self.policy in ("oblivion_by_goodness", "og"):
            self._update_oblivion(true_label, retrieved_indices)

    def _add_case(self, x, y):
        self.X_base = np.vstack([self.X_base, x.reshape(1, -1)])
        self.y_base = np.append(self.y_base, y)
        if self.goodness is not None:
            self.goodness = np.append(self.goodness, self.initial_goodness)

    def _update_oblivion(self, true_label, retrieved_indices):
        if true_label is None:
            return

        to_remove = set()
        for idx in retrieved_indices:
            reward = 1.0 if self.y_base[idx] == true_label else 0.0
            old_g = self.goodness[idx]
            new_g = old_g + self.alpha * (reward - old_g)
            self.goodness[idx] = new_g

            # Remove case if goodness drops below baseline
            if new_g < self.initial_goodness:
                to_remove.add(idx)

        if to_remove:
            keep = np.ones(len(self.y_base), dtype=bool)
            keep[list(to_remove)] = False
            self.X_base = self.X_base[keep]
            self.y_base = self.y_base[keep]
            self.goodness = self.goodness[keep]
