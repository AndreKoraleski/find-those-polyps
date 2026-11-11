"""Nested class stratified subsets of the training partition."""

import numpy as np
from numpy.typing import NDArray

from find_those_polyps.tasks.schema import Budget


def _stratified_order(labels: NDArray[np.int64], rng: np.random.Generator) -> NDArray[np.intp]:
    """Permutation whose every prefix is approximately class balanced.

    Each sample receives the key `(rank + u) / n_c` within its class, where `rank` is its position
    in a shuffled order of the class, `u` is uniform on the unit interval and `n_c` is the class
    size. Sorting by key interleaves classes proportionally. The first sample of every class is
    moved ahead, so that any prefix of length at least the number of classes covers all classes.

    Args:
        labels: Class index of each sample, of shape `(n,)`.
        rng: Generator supplying the shuffles and the offsets `u`.

    Returns:
        Indices into `labels` of shape `(n,)` in drawing order.
    """
    keys = np.empty(len(labels))
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        rng.shuffle(members)
        keys[members] = (np.arange(len(members)) + rng.random(len(members))) / len(members)
        keys[members[0]] -= 1.0
    return np.argsort(keys, kind="stable")


def budget_indices(
    budget: Budget,
    n_samples: int,
    labels: NDArray[np.int64] | None,
    seed: int,
) -> NDArray[np.intp]:
    """Indices of the training samples available under `budget`.

    Every budget draws from one permutation determined by the seed, class stratified when labels
    are given and uniform otherwise. Fraction and count budgets take a prefix of the permutation
    and shot budgets the first samples of each class in it, so that with equal seeds the subsets
    of fraction and count budgets are nested in one another, as are those of shot budgets.

    Args:
        budget: Label budget to draw.
        n_samples: Size of the training partition, which must equal the length of `labels` when
            these are given.
        labels: Class index of each sample for single label tasks, of shape `(n_samples,)`, and
            `None` otherwise.
        seed: Seed of the permutation.

    Returns:
        Sorted indices into the training partition. A full budget returns every index, and a
        budget larger than the partition or than a class returns all of it.

    Raises:
        ValueError: If a shot budget is requested without labels.
    """
    if budget.is_full:
        return np.arange(n_samples)
    rng = np.random.default_rng(seed)
    if labels is None:
        if budget.kind == "shots":
            raise ValueError("shot budgets require single-label targets")
        order = rng.permutation(n_samples)
        return np.sort(order[: budget.size(n_samples, 1)])

    order = _stratified_order(labels, rng)
    if budget.kind == "shots":
        shots = int(budget.value)
        counts: dict[int, int] = {}
        chosen = []
        for index in order:
            label = int(labels[index])
            if counts.get(label, 0) < shots:
                counts[label] = counts.get(label, 0) + 1
                chosen.append(index)
        return np.sort(np.asarray(chosen, dtype=np.intp))
    n_classes = len(np.unique(labels))
    return np.sort(order[: budget.size(n_samples, n_classes)])
