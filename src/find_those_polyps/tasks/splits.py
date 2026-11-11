"""Canonical partitions of a dataset into training, validation and test sets.

The test partition is canonical. An official or community-standard split is adopted verbatim
whenever one exists, so that test results remain comparable with the literature. Training and
validation partitions belong to this benchmark and are carved from the remaining samples by group,
so that no patient, video or centre contributes to more than one partition. Samples sharing a group
with the official test partition are excluded from training and validation. Held out partitions are
chosen among candidate folds so that training keeps every class and the held out partition covers as
many classes as possible. Identical images and confirmed near duplicates share a group as well, so
that copies and neighbouring video frames never fall on both sides of a split.
"""

from collections.abc import Callable, Sequence
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from find_those_polyps.logging import get_logger
from find_those_polyps.tasks.schema import Split
from find_those_polyps.utils.perceptual import Signature, near_pairs

log = get_logger(__name__)

BENCHMARK_SOURCE = "find-those-polyps"
"""Test source of the partitions chosen by the benchmark itself."""

LOSSES = ("same group", "identical copy", "near duplicate", "conflicting labels")
"""Reasons for leaving a sample out of every partition, in order of precedence.

A sample is lost when it shares a patient, video or sequence with a test image, when it is a byte
identical copy of a test image, when it is a confirmed near duplicate of a test image, or when it is
a copy outside the test partition carrying a label that contradicts another copy.
"""


class SplitFile(BaseModel):
    """Sample identifiers of each partition and the provenance of the test set.

    Attributes:
        test_source: Provenance of the test partition, `BENCHMARK_SOURCE` when the benchmark chose
            it.
        seed: Seed of the fold selection, `None` for files written before it was recorded.
        train: Identifiers of the training partition.
        val: Identifiers of the validation partition.
        test: Identifiers of the test partition.
        excluded: Identifiers left out of every partition, because they share a group with the test
            partition or are copies carrying conflicting labels.
        losses: Number of excluded samples for each reason, among `LOSSES`.
    """

    model_config = ConfigDict(frozen=True)

    test_source: str = Field(min_length=1)
    seed: int | None = None
    train: tuple[str, ...] = Field(min_length=1)
    val: tuple[str, ...] = Field(min_length=1)
    test: tuple[str, ...] = Field(min_length=1)
    excluded: tuple[str, ...] = ()
    losses: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _disjoint(self) -> Self:
        """Reject partitions that overlap or repeat an identifier."""
        train, val, test = set(self.train), set(self.val), set(self.test)
        excluded = set(self.excluded)
        parts = (train, val, test, excluded)
        if sum(map(len, parts)) != len(train | val | test | excluded):
            raise ValueError("partitions must be disjoint")
        if len(train) != len(self.train) or len(val) != len(self.val):
            raise ValueError("partitions must not repeat identifiers")
        return self

    def ids(self, split: Split) -> tuple[str, ...]:
        """Identifiers of one partition.

        Args:
            split: Partition to return.

        Returns:
            The identifiers of `split` in file order.
        """
        return {Split.TRAIN: self.train, Split.VAL: self.val, Split.TEST: self.test}[split]


ATTEMPTS = 20
"""Number of seeds of stratified group folds tried for a held out partition."""


def _group_fold(labels: np.ndarray, groups: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Select about a fraction of the samples by whole groups.

    Candidate folds of stratified group k fold under successive seeds are scanned in order, and the
    first one keeping every class outside the held out part while covering every class inside it is
    taken. Failing that, the fold covering the most classes inside, with every class outside, wins.

    Args:
        labels: Class of each sample, all equal when the task has no single label.
        groups: Group of each sample.
        fraction: Share of the samples to hold out.
        seed: First seed of the folds.

    Returns:
        A boolean mask of the held out samples.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = max(2, round(1 / fraction))
    classes = set(labels.tolist())
    best: tuple[tuple[bool, int, float], np.ndarray] | None = None
    for attempt in range(ATTEMPTS):
        folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + attempt)
        for _, held_out in folds.split(np.zeros(len(labels)), labels, groups):
            mask = np.zeros(len(labels), dtype=bool)
            mask[held_out] = True
            kept = set(labels[~mask].tolist()) == classes
            covered = len(set(labels[mask].tolist()))
            score = (kept, covered, -abs(mask.mean() - fraction))
            if best is None or score > best[0]:
                best = (score, mask)
            if kept and covered == len(classes):
                return mask
    assert best is not None
    if not best[0][0]:
        log.warning("no held out fold keeps every class in training", fraction=fraction)
    return best[1]


def merge_duplicates(
    ids: Sequence[str],
    groups: Sequence[str],
    fingerprints: Sequence[str],
    labels: Sequence[int] | None = None,
) -> tuple[list[str], set[str]]:
    """Merge the groups of byte identical samples and find duplicates whose labels disagree.

    Identical images in different groups join one group, so that they never fall on both sides of a
    split. Identical images carrying different labels have no defined target and are returned for
    exclusion.

    Args:
        ids: Identifier of each sample.
        groups: Group of each sample.
        fingerprints: CRC32 and size of the image file of each sample.
        labels: Class of each sample, or `None` when the task has no single label.

    Returns:
        The merged group of each sample and the identifiers of duplicates with conflicting labels.
    """
    copies: dict[str, list[int]] = {}
    for index, fingerprint in enumerate(fingerprints):
        copies.setdefault(fingerprint, []).append(index)
    conflicts: set[str] = set()
    for members in copies.values():
        if len(members) > 1 and labels is not None and len({labels[i] for i in members}) > 1:
            conflicts.update(ids[i] for i in members)
    return _join(groups, [m for m in copies.values() if len(m) > 1]), conflicts


def merge_near_duplicates(
    groups: Sequence[str],
    signatures: Sequence[Signature | None],
    confirm: Callable[[int, int], bool] | None = None,
) -> list[str]:
    """Merge the groups of near duplicates.

    Candidates are found by their perceptual signatures, and only pairs from different groups are
    considered, since pairs within a group change nothing. Resized or recompressed copies and
    neighbouring video frames then never fall on both sides of a split. Unlike identical copies,
    near duplicates may rightly carry different labels, as frames before and after a finding do, so
    none is excluded.

    Args:
        groups: Group of each sample.
        signatures: Signature of each image, `None` for images that could not be hashed.
        confirm: Whether the images at two positions are near duplicates, applied to every
            candidate pair. Without it every candidate is merged.

    Returns:
        The merged group of each sample.
    """
    pairs = [(a, b) for a, b in near_pairs(signatures) if groups[a] != groups[b]]
    if confirm is not None:
        pairs = [(a, b) for a, b in pairs if confirm(a, b)]
    return _join(groups, pairs)


def _join(groups: Sequence[str], clusters: Sequence[Sequence[int]]) -> list[str]:
    """Merge the groups of the samples of every cluster with a union find.

    Args:
        groups: Group of each sample.
        clusters: Positions of samples that must share a group.

    Returns:
        The merged group of each sample, named after the smallest group it absorbed.
    """
    parent = {group: group for group in groups}

    def find(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    for members in clusters:
        roots = sorted({find(groups[i]) for i in members})
        for root in roots[1:]:
            parent[root] = roots[0]
    return [find(group) for group in groups]


def make_split(
    ids: Sequence[str],
    groups: Sequence[str],
    labels: Sequence[int] | None = None,
    *,
    test: float | Sequence[str] = 0.2,
    val: float = 0.1,
    test_source: str = BENCHMARK_SOURCE,
    seed: int = 0,
    excluded: Sequence[str] = (),
    kept: bool = False,
) -> SplitFile:
    """Partition samples into a canonical split.

    Samples sharing a group with the test partition are excluded from training and validation.

    Args:
        ids: Identifier of each sample.
        groups: Patient, video or centre of each sample.
        labels: Class of each sample for the stratification of single label tasks, or `None`.
        test: Identifiers of a fixed test partition, or the share of samples to hold out by group.
        val: Share of all samples held out by group for validation.
        test_source: Provenance of the test partition, which a published one must name.
        seed: First seed of the fold selection.
        excluded: Identifiers already removed from `ids`, which the split file records.
        kept: Whether a test partition given as identifiers is one the benchmark chose earlier and
            keeps, which then carries `BENCHMARK_SOURCE`.

    Returns:
        The split.

    Raises:
        ValueError: If identifiers repeat, the test fraction lies outside zero and one, a published
            test partition has no named source, or fixed test identifiers are unknown.
    """
    ids_array = np.asarray(ids)
    groups_array = np.asarray(groups)
    labels_array = np.zeros(len(ids)) if labels is None else np.asarray(labels)
    if len(set(ids)) != len(ids):
        raise ValueError("sample identifiers must be unique")

    if isinstance(test, float):
        if not 0 < test < 1:
            raise ValueError("test fraction must lie in (0, 1)")
        test_mask = _group_fold(labels_array, groups_array, test, seed)
        test_fraction = test
    else:
        if test_source == BENCHMARK_SOURCE and not kept:
            raise ValueError("official test splits must name their source")
        official = set(test)
        if missing := official - set(ids):
            raise ValueError(f"{len(missing)} fixed test ids are unknown")
        test_mask = np.isin(ids_array, list(official))
        test_fraction = float(test_mask.mean())

    shared = np.isin(groups_array, list(set(groups_array[test_mask]))) & ~test_mask
    if shared.any():
        log.warning("excluding samples that share a group with the test partition", n=shared.sum())
    pool = ~test_mask & ~shared

    val_fraction = val / (1 - test_fraction)
    val_in_pool = _group_fold(labels_array[pool], groups_array[pool], val_fraction, seed)
    pool_ids = ids_array[pool]
    return SplitFile(
        test_source=test_source,
        seed=seed,
        train=tuple(pool_ids[~val_in_pool].tolist()),
        val=tuple(pool_ids[val_in_pool].tolist()),
        test=tuple(ids_array[test_mask].tolist()),
        excluded=tuple(sorted(set(excluded) | set(ids_array[shared].tolist()))),
    )
