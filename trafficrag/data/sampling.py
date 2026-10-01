"""Epoch-seeded motion segment sampling and class weighting."""

import numpy as np
from torch.utils.data import Sampler


class SegmentSampler(Sampler):
    def __init__(self, rows, seed=42, balanced=False):
        self.rows = list(rows)
        if not self.rows:
            raise ValueError("Motion sampling requires annotated segments")
        self.seed = int(seed)
        self.epoch = 0
        self.balanced = bool(balanced)
        self.labels = np.asarray([row["label"] for row in rows])
        if not np.isin(self.labels, (0, 1)).all():
            raise ValueError("Motion sample labels must be binary")
        if balanced and len(np.unique(self.labels)) < 2:
            raise ValueError("Balanced sampling requires both motion classes")

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if not self.balanced:
            return iter(rng.permutation(len(self)).tolist())
        counts = np.bincount(self.labels, minlength=2)
        weights = 1 / counts[self.labels]
        weights /= weights.sum()
        return iter(rng.choice(len(self), size=len(self), replace=True, p=weights).tolist())

    def state_dict(self):
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "balanced": self.balanced,
            "samples": len(self),
        }

    def load_state_dict(self, values):
        if (
            values["seed"] != self.seed
            or values["balanced"] != self.balanced
            or values["samples"] != len(self)
        ):
            raise ValueError("Motion sampler configuration changed during resume")
        self.epoch = int(values["epoch"])


def class_weights(rows):
    labels = np.asarray([row["label"] for row in rows])
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Class weights require binary motion labels")
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both classes must be represented for class-weighted training")
    return {
        "negative": int(counts[0]),
        "positive": int(counts[1]),
        "positive_weight": float(counts[0] / counts[1]),
    }
