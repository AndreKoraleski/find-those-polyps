"""Small feature sets for protocol tests."""

import numpy as np
import torch

from find_those_polyps.evaluation import Features
from find_those_polyps.tasks import TaskKind, TaskSpec


def blobs(n: int = 20, seed: int = 0) -> tuple[Features, Features, Features]:
    """Two well separated Gaussian blobs split into train, validation and test features."""
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(3):
        labels = np.repeat([0, 1], n // 2)
        values = rng.normal(size=(n, 4)) + (labels[:, None] * 12 - 6)
        parts.append(
            Features(
                values=torch.as_tensor(values, dtype=torch.float32),
                targets=tuple(int(label) for label in labels),
            )
        )
    return parts[0], parts[1], parts[2]


def spec_for(n_classes: int) -> TaskSpec:
    """Classification task over ``n_classes`` synthetic classes."""
    return TaskSpec(
        id="cls/blobs",
        description="Gaussian blobs.",
        dataset="blobs",
        kind=TaskKind.CLASSIFICATION,
        classes=tuple(f"c{i}" for i in range(n_classes)),
        primary_metric="balanced_accuracy",
        loader="memory",
        splits="blobs.json",
    )
