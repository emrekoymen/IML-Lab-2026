"""Retention and forgetting policies for kNN."""

import numpy as np


class RetentionManager:
    """Manages online instance retention and forgetting during prediction.

    For AR and DC policies the case base grows (new test points are appended).
    For the OG (Oblivion by Goodness) policy the case base uses *soft-deletion*:
    X_base and y_base are never physically shrunk, only ``active_mask`` is updated
    (entries are set to False). This keeps all pre-computed distance indices valid.
    """

    def __init__(self, X_train, y_train, policy="never_retain", alpha=0.2, initial_goodness=0.5):
        self.policy = policy.lower()
        self.alpha = alpha
        self.initial_goodness = initial_goodness

        self.X_base = np.array(X_train, dtype=float).copy()
        self.y_base = np.array(y_train).copy()

        if self.policy in ("oblivion_by_goodness", "og"):
            self.goodness = np.full(len(self.y_base), self.initial_goodness, dtype=float)
            # Soft-deletion mask: True = active, False = logically deleted.
            # Never physically shrink X_base/y_base for OG so distance indices stay valid.
            self.active_mask = np.ones(len(self.y_base), dtype=bool)
        else:
            self.goodness = None
            self.active_mask = None

    @property
    def is_dynamic(self):
        return self.policy not in ("never_retain", "nr")

    @property
    def size(self):
        """Number of *active* cases in the base."""
        if self.active_mask is not None:
            return int(self.active_mask.sum())
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
        if self.active_mask is not None:
            self.active_mask = np.append(self.active_mask, True)

    def _update_oblivion(self, true_label, retrieved_indices):
        """Update goodness values and soft-delete cases whose goodness falls below baseline.

        Soft-deletion (active_mask[idx] = False) is used instead of physical removal so
        that all pre-computed distance vectors remain index-aligned with X_base.
        """
        if true_label is None:
            return

        for idx in retrieved_indices:
            # idx refers to a position in the *full* (unmasked) X_base
            if not self.active_mask[idx]:
                continue  # already deleted; skip
            reward = 1.0 if self.y_base[idx] == true_label else 0.0
            new_g = self.goodness[idx] + self.alpha * (reward - self.goodness[idx])
            self.goodness[idx] = new_g
            # Soft-delete if goodness falls below the initial baseline
            if new_g < self.initial_goodness:
                self.active_mask[idx] = False

