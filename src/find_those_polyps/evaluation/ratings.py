"""Ratings of how well every task separates encoders.

The ratings rest on paired differences. The per image records of every model are aligned on their
test images, and every model receives the same resamples of those images, so the difference between
two models is resampled as one statistic and its noise excludes the variation that both share.
Metrics are averaged over the seeds of a protocol on every resample.

Encoders pretrained on some data form the pool, and those pretrained on nothing, such as the
handcrafted colour and texture statistics or an untrained network, are references. Variants of one
encoder at several input sizes count once, through the best of them. Every setting, a task under a
protocol on the full label budget, gets independent flags.

Resolvability compares every pair of pooled encoders with Holm's correction. Metrics that are a mean
over images, such as the Dice score or the retrieval metrics, are compared by the exact paired sign
flip permutation test of the mean difference, which stays valid on small test sets where bootstrap
error bars are too narrow (Bowyer et al. 2025). It enumerates every sign pattern up to `EXACT_FLIPS`
images and draws `FLIPS` patterns beyond. Other metrics, computed on the whole test set, are
compared by the paired bootstrap. A setting is resolved when a pair differs at level `ALPHA`,
equivalent when every 90% paired interval lies inside the region of practical equivalence,
underpowered when neither holds and the minimum detectable effect, 2.80 times the median standard
error of the differences, exceeds that region, and inconclusive otherwise. Saturation is the index
of Akhtar et al. (2026), `exp(-R**2)` with R the gap between the first and the k-th pooled encoder
over the paired standard error of that gap, for k up to five. A setting is ceiling bound when the
interval of the best encoder reaches the ideal value of the metric. The floor test is the one sided
paired interval of the best pooled encoder minus the best reference, and a setting is floor level
when its lower bound is at most zero, or for a mean over images when the one sided sign flip test
does not find the best pooled encoder above the best reference. A setting is pool sensitive when
dropping one encoder changes a flag that stays defined. A setting is small when it has fewer than
`SMALL` test images, since bootstrap rankings are untrustworthy around ten cases (Wiesenfarth et al.
2021), and small settings stay out of the core subset however their tests come out.

The region of practical equivalence of a metric is `ROPE_FRACTION` of its range, the 0.01 of
accuracy that Benavoli et al. (2017) use, scaled so that it has the same relative size for every
metric.
"""

import argparse
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from find_those_polyps.evaluation.records import DETECTION_COLUMNS, matches_from_records
from find_those_polyps.logging import get_logger
from find_those_polyps.metrics import MASKED, METRIC_INFO, metric_names, score
from find_those_polyps.metrics.bootstrap import (
    RESAMPLES,
    SLOW_RESAMPLES,
    Statistic,
    draws,
    index_rows,
    matched_statistic,
    mean_statistic,
    prediction_statistic,
)
from find_those_polyps.metrics.detection import DETECTION, IOU_THRESHOLDS
from find_those_polyps.metrics.retrieval import CLUSTERING, RETRIEVAL, clustering
from find_those_polyps.tasks.schema import Target, TaskKind, TaskSpec

log = get_logger(__name__)

UNTRAINED = "none"
"""Pretraining recorded on the leaderboard for references, the models that saw no data."""
MAIN = ("linear", "linear_seg", "linear_det")
"""Protocols that serve as the main protocol of a task, the first present being used."""
ALPHA = 0.05
"""Family wise error rate of the pairwise tests of a setting, and level of the one sided tests."""
POWER_FACTOR = 2.80
"""Minimum detectable effect in standard errors, for a power of 0.8 at a two sided 0.05."""
ROPE_FRACTION = 0.01
"""Half width of the region of practical equivalence, as a fraction of the range of a metric."""
SATURATED = 0.7
"""Saturation index from which a setting is saturated, the high regime of Akhtar et al."""
TOP_K = 5
"""Number of leading encoders compared by the saturation index."""
METRIC_RESAMPLES = 200
"""Resamples of the metrics other than the primary one."""
EXACT_FLIPS = 16
"""Largest number of images whose sign patterns the sign flip test enumerates in full."""
FLIPS = 10000
"""Sign patterns drawn by the sign flip test on more images."""
SMALL = 20
"""Test images below which a setting is small. It is a choice of this benchmark, between the ten
cases at which Wiesenfarth et al. (2021) find bootstrap rankings untrustworthy and the median of
twenty cases of biomedical challenges, since the literature sets no minimum."""
AGREEMENT = 1e-6
"""Largest difference between the score recomputed from records and the leaderboard score."""
SIGNED = frozenset({"mcc", "cohen_kappa", "quadratic_kappa", "linear_kappa", "spearman"})
"""Metrics ranging over [-1, 1]."""

TASK_COLUMNS = [
    "task",
    "protocol",
    "family",
    "primary_metric",
    "n",
    "small",
    "pool",
    "references",
    "rope",
    "best_model",
    "best",
    "pairs",
    "resolved_pairs",
    "mde",
    "status",
    "saturation",
    "saturated",
    "ceiling_bound",
    "floor_model",
    "floor_gap",
    "floor_lower",
    "floor_level",
    "floor_normalised",
    "above_floor",
    "p_rank1",
    "snr",
    "snr_rank",
    "pool_sensitive",
    "sensitive_to",
    "hard_share",
    "main",
    "core",
    "registry_version",
]
"""Columns of the task ratings."""

Records = dict[int, pd.DataFrame]
"""Per image records of one model on one setting, keyed by seed and sorted by image."""


@dataclass
class Measured:
    """One metric of every model on one setting.

    Attributes:
        points: Value on all images, averaged over seeds, of shape `(M,)`.
        draws: Value on every shared resample, averaged over seeds, of shape `(M, B)`.
        seeds: Value on all images of every seed, of shape `(M, S)`, NaN beyond the seeds of a
            model.
        images: Value on every image, averaged over seeds, of shape `(M, N)`, for metrics that
            are a mean over images and None for the others.
    """

    points: NDArray[np.float64]
    draws: NDArray[np.float64]
    seeds: NDArray[np.float64]
    images: NDArray[np.float64] | None = None


@dataclass
class Setting:
    """Measurements of the models of one task under one protocol on the full label budget.

    Attributes:
        task: Task identifier.
        protocol: Protocol name.
        kind: Kind of the task.
        n_classes: Number of classes of the task.
        primary: Primary metric of the setting.
        n: Number of test images shared by every model.
        models: Model names.
        pretrained: Whether each model is pretrained on some data, of shape `(M,)`.
        metrics: Measurement of every metric, the primary one first.
        hard_share: Share of images that every model gets wrong with every seed, NaN when the
            records do not say whether an image is right.
        version: Registry versions of the leaderboard rows.
    """

    task: str
    protocol: str
    kind: TaskKind
    n_classes: int
    primary: str
    n: int
    models: list[str]
    pretrained: NDArray[np.bool_]
    metrics: dict[str, Measured] = field(default_factory=dict)
    hard_share: float = float("nan")
    version: str = ""

    @property
    def family(self) -> str:
        """Group of settings whose metrics are rated together."""
        return family(self.kind, self.protocol, self.n_classes)


def family(kind: TaskKind, protocol: str, n_classes: int) -> str:
    """Group of settings sharing their metrics, by protocol output and task kind.

    Args:
        kind: Kind of the task.
        protocol: Protocol name.
        n_classes: Number of classes of the task.

    Returns:
        `retrieval` or `clustering` for those protocols, the task kind otherwise, with `binary`
        appended for binary single label tasks.
    """
    if protocol == "retrieval":
        return "retrieval"
    if protocol == "kmeans":
        return "clustering"
    binary = kind in (TaskKind.CLASSIFICATION, TaskKind.ORDINAL) and n_classes == 2
    return f"{kind.value} binary" if binary else kind.value


def metric_range(metric: str, kind: TaskKind, n_classes: int) -> float:
    """Width of the range of values of a metric, from ideal to worst or to chance.

    Args:
        metric: Name of the metric.
        kind: Kind of the task, which tells the grade error of ordinal tasks from the pixel error
            of segmentation.
        n_classes: Number of classes of the task.

    Returns:
        2 for metrics over [-1, 1] and the multiclass Brier score, the number of classes minus one
        for the grade error and the coverage error, the log loss of the uniform prediction for the
        log loss, and 1 otherwise.
    """
    if metric in SIGNED or metric == "brier":
        return 2.0
    if metric == "coverage_error" or (metric == "mae" and kind not in MASKED):
        return float(n_classes - 1)
    if metric == "log_loss":
        return math.log(n_classes)
    return 1.0


def ideal(metric: str) -> float:
    """Best attainable value of a metric, 1 for scores and 0 for losses, NaN when it varies."""
    if metric == "coverage_error":
        return float("nan")
    return 1.0 if _higher(metric) else 0.0


def _higher(metric: str) -> bool:
    """Whether larger values of a metric are better, True for unknown metrics."""
    return METRIC_INFO[metric].higher_is_better if metric in METRIC_INFO else True


def candidates(spec: TaskSpec, protocol: str, columns: Iterable[str]) -> list[str]:
    """Metrics recomputable from the records of a protocol, as reported by it.

    Args:
        spec: Task of the records.
        protocol: Protocol of the records.
        columns: Columns of the records.

    Returns:
        The metric names, in reporting order.
    """
    present = set(columns)
    if protocol == "retrieval":
        return [m for m in RETRIEVAL if m in present]
    if protocol == "kmeans":
        return list(CLUSTERING)
    if spec.kind in MASKED:
        return [m for m in metric_names(spec.kind, len(spec.classes)) if m in present]
    if spec.kind is TaskKind.DETECTION:
        return list(DETECTION)
    return list(metric_names(spec.kind, len(spec.classes)))


Group = tuple[list[str], Callable[[NDArray[np.intp]], dict[str, float]]]
"""Metrics computed together on a resample, with the function returning all of them."""


def statistics(
    spec: TaskSpec, protocol: str, frame: pd.DataFrame, metrics: Sequence[str]
) -> tuple[dict[str, Statistic], Group]:
    """Resampled statistics of every metric from the records of one seed.

    Means of per image columns and metrics with a fast resampled version get their own statistic.
    The others recompute a metric suite and are grouped, so that one call per resample serves all
    of them.

    Args:
        spec: Task of the records.
        protocol: Protocol of the records.
        frame: Records of one model and seed, sorted by image.
        metrics: Metrics to compute, from `candidates`.

    Returns:
        The statistics of single metrics, and the grouped metrics with their function.
    """
    n_classes = len(spec.classes)
    if protocol == "kmeans":
        labels = np.array([spec.classes.index(t) for t in frame["target"]], dtype=np.int64)
        clusters = frame["cluster"].to_numpy(dtype=np.int64)
        return {}, (list(metrics), lambda index: clustering(labels[index], clusters[index]))
    if protocol == "retrieval" or spec.kind in MASKED:
        return {m: mean_statistic(frame[m].to_numpy(dtype=float)) for m in metrics}, ([], dict)
    if spec.kind is TaskKind.DETECTION:
        matched = matches_from_records(frame, len(IOU_THRESHOLDS))
        return {m: matched_statistic(m, matched, n_classes) for m in metrics}, ([], dict)
    probabilities = frame[[f"p_{c}" for c in spec.classes]].to_numpy(dtype=np.float64)
    targets = _targets(spec, frame["target"])
    singles: dict[str, Statistic] = {}
    grouped = []
    for metric in metrics:
        statistic = prediction_statistic(spec.kind, metric, probabilities, targets, n_classes)
        if getattr(statistic, "slow", False):
            grouped.append(metric)
        else:
            singles[metric] = statistic

    def suite(index: NDArray[np.intp]) -> dict[str, float]:
        chosen = tuple(targets[i] for i in index)
        return score(spec.kind, probabilities[index], chosen, n_classes, grouped)

    return singles, (grouped, suite)


def _targets(spec: TaskSpec, column: pd.Series) -> tuple[Target, ...]:
    """Class indices, or indicator vectors for multilabel tasks, of the recorded targets."""
    if spec.kind is TaskKind.MULTILABEL:
        return tuple(
            np.array([c in str(v).split("|") for c in spec.classes], dtype=np.int64)
            if isinstance(v, str)
            else np.zeros(len(spec.classes), dtype=np.int64)
            for v in column
        )
    return tuple(spec.classes.index(str(v)) for v in column)


def measure(
    spec: TaskSpec,
    protocol: str,
    primary: str,
    records: Sequence[Records],
    *,
    resamples: int = RESAMPLES,
    metric_resamples: int = METRIC_RESAMPLES,
) -> tuple[dict[str, Measured], float]:
    """Every metric of every model on shared resamples of the test images.

    Args:
        spec: Task of the records.
        protocol: Protocol of the records.
        primary: Primary metric, which receives `resamples` resamples.
        records: Records of every model, aligned on the same images in the same order.
        resamples: Resamples of the primary metric.
        metric_resamples: Resamples of the other metrics.

    Returns:
        The measurement of every metric, the primary one first, and the share of images that
        every model gets wrong with every seed, NaN when the records do not tell.
    """
    first = next(iter(records[0].values()))
    n = len(first)
    metrics = [primary] + [m for m in candidates(spec, protocol, first.columns) if m != primary]
    count = {m: resamples if m == primary else metric_resamples for m in metrics}
    widest = max(len(r) for r in records)
    averaged = protocol == "retrieval" or spec.kind in MASKED
    images = {m: np.full((len(records), n), np.nan) for m in metrics} if averaged else {}
    points = {m: np.full((len(records), widest), np.nan) for m in metrics}
    sampled: dict[str, list[list[NDArray[np.float64]]]] = {m: [] for m in metrics}
    wrong = np.ones(n, dtype=bool)
    judged = False
    for row, model in enumerate(records):
        for m in metrics:
            sampled[m].append([])
        for column, frame in enumerate(model.values()):
            singles, (grouped, suite) = statistics(spec, protocol, frame, metrics)
            everything = np.arange(n)
            for m, statistic in singles.items():
                points[m][row, column] = statistic(everything)
                sampled[m][row].append(draws(statistic, n, resamples=count[m]))
            if grouped:
                whole = suite(everything)
                values = _grouped(
                    suite, grouped, n, min(max(count[m] for m in grouped), SLOW_RESAMPLES)
                )
                for m in grouped:
                    points[m][row, column] = whole[m]
                    sampled[m][row].append(values[m][: count[m]])
            failed = _failed(frame)
            if failed is not None:
                wrong &= failed
                judged = True
        for m in images:
            seeded = np.array([f[m].to_numpy(dtype=float) for f in model.values()])
            finite = np.isfinite(seeded)
            total = np.where(finite, seeded, 0.0).sum(axis=0)
            defined = finite.sum(axis=0)
            images[m][row] = np.where(defined > 0, total / np.maximum(defined, 1), np.nan)
    measured = {}
    for m in metrics:
        length = min(len(d) for model in sampled[m] for d in model)
        stacked = np.array([np.mean([d[:length] for d in model], axis=0) for model in sampled[m]])
        measured[m] = Measured(np.nanmean(points[m], axis=1), stacked, points[m], images.get(m))
    return measured, float(wrong.mean()) if judged else float("nan")


def _grouped(
    suite: Callable[[NDArray[np.intp]], dict[str, float]],
    names: Sequence[str],
    n: int,
    resamples: int,
) -> dict[str, NDArray[np.float64]]:
    """Values of grouped metrics on the shared resamples, one suite call per resample."""
    values = {m: np.empty(resamples) for m in names}
    position = 0
    for rows in index_rows(n, resamples):
        for index in rows:
            result = suite(index)
            for m in names:
                values[m][position] = result[m]
            position += 1
    return values


def _failed(frame: pd.DataFrame) -> NDArray[np.bool_] | None:
    """Whether each image is wrong, from its correctness or its nearest neighbour, if recorded."""
    if "correct" in frame.columns:
        return np.asarray(~frame["correct"].astype(bool).to_numpy(), dtype=bool)
    if "precision_at_1" in frame.columns:
        return np.asarray(frame["precision_at_1"].to_numpy(dtype=float) == 0, dtype=bool)
    return None


def holm(p_values: Sequence[float], alpha: float = ALPHA) -> NDArray[np.bool_]:
    """Rejections of Holm's step down procedure, valid under any dependence.

    Args:
        p_values: P value of every hypothesis.
        alpha: Family wise error rate.

    Returns:
        Whether each hypothesis is rejected.
    """
    p = np.asarray(p_values, dtype=float)
    rejected = np.zeros(len(p), dtype=bool)
    for rank, position in enumerate(np.argsort(p, kind="stable")):
        if not p[position] <= alpha / (len(p) - rank):
            break
        rejected[position] = True
    return rejected


def _difference(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Paired resampled difference, keeping the resamples where both values are finite."""
    d = a - b
    return d[np.isfinite(d)]


def _p_value(d: NDArray[np.float64]) -> float:
    """Two sided bootstrap p value of a zero difference, never below one over the resamples."""
    if len(d) == 0:
        return float("nan")
    tail = min(int((d <= 0).sum()), int((d >= 0).sum()))
    return min(1.0, 2 * (tail + 1) / (len(d) + 1))


def sign_flip(
    differences: NDArray[np.float64], *, one_sided: bool = False, flips: int = FLIPS
) -> float:
    """P value of the paired sign flip permutation test of a zero mean difference.

    Under the null hypothesis each paired difference is as likely to have either sign, so the
    observed sum is compared with the sums under every pattern of signs. Every pattern is
    enumerated up to `EXACT_FLIPS` differences, which makes the test exact, and `flips` patterns
    are drawn beyond, the observed one counting as one of them. Undefined differences are left out.

    Args:
        differences: Difference of two models on every image.
        one_sided: Whether to test only for a positive mean difference.
        flips: Sign patterns drawn when they are too many to enumerate.

    Returns:
        The p value, NaN without any defined difference.
    """
    d = differences[np.isfinite(differences)]
    n = len(d)
    if n == 0:
        return float("nan")
    observed = float(d.sum())
    tolerance = 1e-9 * max(1.0, float(np.abs(d).sum()))

    def extreme(sums: NDArray[np.float64]) -> int:
        if one_sided:
            return int((sums >= observed - tolerance).sum())
        return int((np.abs(sums) >= abs(observed) - tolerance).sum())

    if n <= EXACT_FLIPS:
        patterns = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
        return float(extreme((2.0 * patterns - 1.0) @ d) / 2**n)
    rng = np.random.default_rng(0)
    rows = max(1, 4_000_000 // n)
    count, drawn = 0, 0
    while drawn < flips:
        block = min(rows, flips - drawn)
        signs = 2.0 * rng.integers(0, 2, (block, n), dtype=np.int8) - 1.0
        count += extreme(signs @ d)
        drawn += block
    return (count + 1) / (flips + 1)


def _p_values(
    pairs: Sequence[tuple[int, int]],
    sampled: NDArray[np.float64],
    images: NDArray[np.float64] | None,
) -> list[float]:
    """Two sided p value of every pair, by sign flips over images or by the paired bootstrap."""
    if images is not None:
        return [sign_flip(images[i] - images[j]) for i, j in pairs]
    return [_p_value(_difference(sampled[i], sampled[j])) for i, j in pairs]


def flags(
    points: NDArray[np.float64],
    sampled: NDArray[np.float64],
    pretrained: NDArray[np.bool_],
    rope: float,
    best: float,
    images: NDArray[np.float64] | None = None,
) -> dict[str, object]:
    """Resolvability, saturation and floor of one setting, from oriented values.

    Args:
        points: Value of every model on all images, larger being better.
        sampled: Value of every model on every shared resample, oriented the same way.
        pretrained: Whether each model belongs to the pool.
        rope: Half width of the region of practical equivalence.
        best: Ideal value of the metric in the same orientation, NaN when unknown.
        images: Value of every model on every image, oriented the same way, for a metric that
            is a mean over images, whose tests are then sign flip tests.

    Returns:
        The values of the task ratings that depend on the pool, indices of models included.
    """
    pool = np.flatnonzero(pretrained)
    references = np.flatnonzero(~pretrained)
    out: dict[str, object] = {
        "best_index": -1,
        "pairs": 0,
        "resolved_pairs": 0,
        "mde": float("nan"),
        "status": "",
        "saturation": float("nan"),
        "ceiling_bound": None,
        "floor_index": -1,
        "floor_gap": float("nan"),
        "floor_lower": float("nan"),
        "floor_level": None,
        "floor_normalised": float("nan"),
        "above_floor": 0,
        "p_rank1": float("nan"),
        "snr": float("nan"),
    }
    if len(pool) == 0:
        return out
    top = int(pool[np.argmax(points[pool])])
    out["best_index"] = top
    finite = sampled[top][np.isfinite(sampled[top])]
    if np.isfinite(best) and len(finite):
        high = np.percentile(finite, 100 * (1 - ALPHA / 2))
        out["ceiling_bound"] = bool(high >= best - 1e-12)
    if len(pool) >= 2:
        pairs = [(int(i), int(j)) for i, j in combinations(pool, 2)]
        differences = [_difference(sampled[i], sampled[j]) for i, j in pairs]
        errors = np.array([np.std(d, ddof=1) for d in differences])
        rejected = holm(_p_values(pairs, sampled, images))
        inside = all(
            np.all(np.abs(np.percentile(d, [100 * ALPHA, 100 * (1 - ALPHA)])) <= rope)
            for d in differences
        )
        mde = POWER_FACTOR * float(np.median(errors))
        out["pairs"], out["resolved_pairs"], out["mde"] = len(pairs), int(rejected.sum()), mde
        if rejected.any():
            out["status"] = "resolved"
        elif inside:
            out["status"] = "equivalent"
        elif mde > rope:
            out["status"] = "underpowered"
        else:
            out["status"] = "inconclusive"
        order = pool[np.argsort(-points[pool], kind="stable")]
        last = int(order[min(TOP_K, len(order)) - 1])
        gap = points[top] - points[last]
        error = float(np.std(_difference(sampled[top], sampled[last]), ddof=1))
        out["saturation"] = math.exp(-((gap / error) ** 2)) if error > 0 else float(gap == 0)
        finite = np.isfinite(sampled[pool]).all(axis=0)
        leaders = pool[np.argmax(sampled[pool][:, finite], axis=0)]
        out["p_rank1"] = float(np.mean(leaders == top)) if finite.any() else float("nan")
        spread = float(points[pool].max() - points[pool].min())
        median = float(np.median(errors))
        out["snr"] = spread / median if median > 0 else float("nan")
    if len(references):
        floor = int(references[np.argmax(points[references])])
        lower = [
            np.percentile(d, 100 * ALPHA) if len(d) else np.nan
            for d in (_difference(sampled[i], sampled[floor]) for i in pool)
        ]
        out["floor_index"] = floor
        out["floor_gap"] = float(points[top] - points[floor])
        bound = float(lower[int(np.flatnonzero(pool == top)[0])])
        out["floor_lower"] = bound
        if images is None:
            above = np.array(lower) > 0
        else:
            above = np.array(
                [sign_flip(images[i] - images[floor], one_sided=True) <= ALPHA for i in pool]
            )
        out["floor_level"] = not bool(above[int(np.flatnonzero(pool == top)[0])])
        out["above_floor"] = int(above.sum())
        room = best - points[floor]
        if np.isfinite(room) and room > 0:
            out["floor_normalised"] = float((points[top] - points[floor]) / room)
    return out


VERDICTS = ("status", "saturated", "ceiling_bound", "floor_level")
"""Flags whose change on dropping one encoder makes a setting pool sensitive."""


def _verdicts(values: Mapping[str, object]) -> dict[str, object]:
    """Flags of a setting compared when encoders are dropped, None or empty when undefined."""
    saturation = float(str(values["saturation"]))
    return {
        "status": values["status"],
        "saturated": None if math.isnan(saturation) else saturation >= SATURATED,
        "ceiling_bound": values["ceiling_bound"],
        "floor_level": values["floor_level"],
    }


def rate_setting(setting: Setting, rope_fraction: float = ROPE_FRACTION) -> dict[str, object]:
    """Task rating of one setting from its primary metric.

    Args:
        setting: Measurements of the setting.
        rope_fraction: Half width of the region of practical equivalence over the metric range.

    Returns:
        One row of the task ratings, without the columns set across settings.
    """
    measured = setting.metrics[setting.primary]
    sign = 1.0 if _higher(setting.primary) else -1.0
    rope = rope_fraction * metric_range(setting.primary, setting.kind, setting.n_classes)
    best = sign * ideal(setting.primary)
    points, sampled = sign * measured.points, sign * measured.draws
    images = None if measured.images is None else sign * measured.images
    values = flags(points, sampled, setting.pretrained, rope, best, images)
    reference = _verdicts(values)
    changed = []
    for dropped in range(len(setting.models)):
        keep = np.arange(len(setting.models)) != dropped
        kept = None if images is None else images[keep]
        other = _verdicts(
            flags(points[keep], sampled[keep], setting.pretrained[keep], rope, best, kept)
        )
        if any(
            other[k] not in (None, "")
            and reference[k] not in (None, "")
            and other[k] != reference[k]
            for k in VERDICTS
        ):
            changed.append(setting.models[dropped])
    top, floor = int(str(values["best_index"])), int(str(values["floor_index"]))
    saturation = float(str(values["saturation"]))
    return {
        "task": setting.task,
        "protocol": setting.protocol,
        "family": setting.family,
        "primary_metric": setting.primary,
        "n": setting.n,
        "small": setting.n < SMALL,
        "pool": " ".join(np.array(setting.models)[setting.pretrained]),
        "references": " ".join(np.array(setting.models)[~setting.pretrained]),
        "rope": rope,
        "best_model": setting.models[top] if top >= 0 else "",
        "best": float(measured.points[top]) if top >= 0 else float("nan"),
        "pairs": values["pairs"],
        "resolved_pairs": values["resolved_pairs"],
        "mde": values["mde"],
        "status": values["status"],
        "saturation": saturation,
        "saturated": None if math.isnan(saturation) else saturation >= SATURATED,
        "ceiling_bound": values["ceiling_bound"],
        "floor_model": setting.models[floor] if floor >= 0 else "",
        "floor_gap": values["floor_gap"],
        "floor_lower": values["floor_lower"],
        "floor_level": values["floor_level"],
        "floor_normalised": values["floor_normalised"],
        "above_floor": values["above_floor"],
        "p_rank1": values["p_rank1"],
        "snr": values["snr"],
        "pool_sensitive": bool(changed),
        "sensitive_to": " ".join(changed),
        "hard_share": setting.hard_share,
        "registry_version": setting.version,
    }


def rate_tasks(settings: Sequence[Setting], rope_fraction: float = ROPE_FRACTION) -> pd.DataFrame:
    """Task ratings of every setting, with the main protocol and the core subset marked.

    A setting belongs to the core subset when it is the main protocol of its task, is resolved
    or equivalent, and is neither small, floor level, saturated nor ceiling bound.

    Args:
        settings: Measured settings.
        rope_fraction: Half width of the region of practical equivalence over the metric range.

    Returns:
        One row per setting with the columns of `TASK_COLUMNS`, sorted by task and protocol.
    """
    rows = [rate_setting(s, rope_fraction) for s in settings]
    table = pd.DataFrame(
        rows, columns=[c for c in TASK_COLUMNS if c not in ("snr_rank", "main", "core")]
    )
    table["snr_rank"] = table["snr"].rank(ascending=False, method="min")
    table["main"] = False
    for _, group in table.groupby("task"):
        present = [p for p in MAIN if p in set(group["protocol"])]
        if present:
            table.loc[group.index[group["protocol"] == present[0]], "main"] = True
    table["core"] = (
        table["main"]
        & table["status"].isin(["resolved", "equivalent"])
        & ~table["small"].astype(bool)
        & (table["floor_level"] != True)  # noqa: E712
        & (table["saturated"] != True)  # noqa: E712
        & (table["ceiling_bound"] != True)  # noqa: E712
    )
    return table[TASK_COLUMNS].sort_values(["task", "protocol"], ignore_index=True)


def backbone(model: str, input_size: int) -> str:
    """Name of a model without the suffix of its input size, which variants of it share."""
    return model.removesuffix(f"-{input_size}")


def read_records(
    roots: Sequence[Path],
    task: str,
    model: str,
    protocol: str,
    accept: Callable[[Records], bool] | None = None,
) -> Records | None:
    """Full budget records of a model under a protocol, from the first root that holds them.

    Args:
        roots: Folders of per image results, searched in order.
        task: Task identifier.
        model: Model name, which names the file.
        protocol: Protocol whose rows are kept.
        accept: Test the records must pass, which lets a root holding records of an older run
            be passed over for one further down.

    Returns:
        The records of every seed sorted by image, or None when no root holds accepted ones.
    """
    for root in roots:
        path = root / task.replace("/", "__") / f"{model}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"sample_id": str})
        frame = frame[(frame["protocol"] == protocol) & (frame["budget"].astype(str) == "full")]
        if len(frame):
            records = {
                int(str(seed)): group.sort_values("sample_id", ignore_index=True)
                for seed, group in frame.groupby("seed", sort=True)
            }
            if accept is None or accept(records):
                return records
    return None


def agrees(
    spec: TaskSpec,
    protocol: str,
    primary: str,
    expected: float,
    records: Records,
    tolerance: float = AGREEMENT,
) -> bool:
    """Whether records reproduce the score of their leaderboard row.

    Records left by an earlier run of the same model under the same name hold other predictions,
    and the score they give tells them apart from those of the current row.

    Args:
        spec: Task of the records.
        protocol: Protocol of the records.
        primary: Primary metric of the row.
        expected: Score of the row, the mean of the primary metric over seeds.
        records: Records of every seed.
        tolerance: Largest accepted absolute difference.

    Returns:
        True when the mean over seeds of the primary metric on the records matches the row, and
        False for detection records written before the matches were kept, which cannot tell.
    """
    if spec.kind is TaskKind.DETECTION and any(
        column not in frame.columns for frame in records.values() for column in DETECTION_COLUMNS
    ):
        log.warning("detection records lack the matches", task=spec.id, protocol=protocol)
        return False
    values = []
    for frame in records.values():
        singles, (_, suite) = statistics(spec, protocol, frame, [primary])
        everything = np.arange(len(frame))
        values.append(
            singles[primary](everything) if primary in singles else suite(everything)[primary]
        )
    found = float(np.mean(values))
    if abs(found - expected) > tolerance:
        log.info("records disagree", task=spec.id, protocol=protocol, found=found, score=expected)
        return False
    return True


def _align(records: Mapping[str, Records]) -> tuple[dict[str, Records], int]:
    """Records restricted to the images that every model and seed holds."""
    shared = set.intersection(
        *(set(frame["sample_id"]) for model in records.values() for frame in model.values())
    )
    aligned = {
        name: {
            seed: frame[frame["sample_id"].isin(shared)].reset_index(drop=True)
            for seed, frame in model.items()
        }
        for name, model in records.items()
    }
    return aligned, len(shared)


@dataclass
class Job:
    """Everything needed to measure one setting in a worker process.

    Attributes:
        spec: Task specification.
        protocol: Protocol name.
        primary: Primary metric.
        models: Model names.
        pretrained: Whether each model is pretrained on some data.
        records: Aligned records of every model.
        n: Number of shared images.
        version: Registry versions of the leaderboard rows.
        resamples: Resamples of the primary metric.
        metric_resamples: Resamples of the other metrics.
    """

    spec: TaskSpec
    protocol: str
    primary: str
    models: list[str]
    pretrained: NDArray[np.bool_]
    records: list[Records]
    n: int
    version: str
    resamples: int
    metric_resamples: int


def jobs(
    board: pd.DataFrame,
    specs: Mapping[str, TaskSpec],
    roots: Sequence[Path],
    *,
    resamples: int = RESAMPLES,
    metric_resamples: int = METRIC_RESAMPLES,
) -> list[Job]:
    """Settings of the current full budget leaderboard rows whose records are available.

    Of variants of a model at several input sizes, the best on the leaderboard is kept. Models
    without records are left out with a warning, and so are settings left with fewer than two
    models, which includes detection settings whose records lack the matches.

    Args:
        board: Leaderboard table.
        specs: Task specifications by identifier.
        roots: Folders of per image results, searched in order.
        resamples: Resamples of the primary metric.
        metric_resamples: Resamples of the other metrics.

    Returns:
        One job per setting, sorted by task and protocol.
    """
    if "current" in board.columns:
        board = board[board["current"].astype(bool)]
    rows = board[(board["budget"].astype(str) == "full") & board["score"].notna()]
    found = []
    for (task, protocol), group in rows.groupby(["task", "protocol"], sort=True):
        if task not in specs:
            log.warning("skipping unregistered task", task=task)
            continue
        primary = str(group["primary_metric"].iloc[0])
        sign = 1.0 if _higher(primary) else -1.0
        group = group.assign(
            key=[
                backbone(m, int(s))
                for m, s in zip(group["model"], group["input_size"], strict=True)
            ],
            oriented=sign * group["score"],
        )
        kept = group.sort_values("oriented", ascending=False).drop_duplicates(
            ["pretraining", "key"]
        )
        records = {}
        for model, expected in zip(kept["model"], kept["score"], strict=True):
            check = partial(agrees, specs[str(task)], str(protocol), primary, float(expected))
            read = read_records(roots, str(task), str(model), str(protocol), check)
            if read is None:
                log.warning("no matching records", task=task, protocol=protocol, model=model)
            else:
                records[str(model)] = read
        if len(records) < 2:
            continue
        aligned, n = _align(records)
        if n == 0:
            log.warning("no shared images", task=task, protocol=protocol)
            continue
        models = list(aligned)
        pretraining = kept.set_index("model").loc[models, "pretraining"]
        versions = sorted(set(group.get("registry_version", pd.Series(dtype=str)).astype(str)))
        found.append(
            Job(
                spec=specs[str(task)],
                protocol=str(protocol),
                primary=primary,
                models=models,
                pretrained=(pretraining != UNTRAINED).to_numpy(),
                records=[aligned[m] for m in models],
                n=n,
                version=" ".join(versions),
                resamples=resamples,
                metric_resamples=metric_resamples,
            )
        )
    return found


def run(job: Job) -> Setting:
    """Measure the setting of a job.

    Args:
        job: Setting to measure.

    Returns:
        The measured setting.
    """
    metrics, hard = measure(
        job.spec,
        job.protocol,
        job.primary,
        job.records,
        resamples=job.resamples,
        metric_resamples=job.metric_resamples,
    )
    log.info("measured setting", task=job.spec.id, protocol=job.protocol, images=job.n)
    return Setting(
        task=job.spec.id,
        protocol=job.protocol,
        kind=job.spec.kind,
        n_classes=len(job.spec.classes),
        primary=job.primary,
        n=job.n,
        models=job.models,
        pretrained=job.pretrained,
        metrics=metrics,
        hard_share=hard,
        version=job.version,
    )


def measure_all(work: Sequence[Job], workers: int = 0) -> list[Setting]:
    """Measure every job, in `workers` processes or in this one when zero.

    Args:
        work: Settings to measure.
        workers: Number of worker processes.

    Returns:
        The measured settings in the order of `work`.
    """
    if workers <= 0:
        return [run(job) for job in work]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(run, work))


def render(tasks: pd.DataFrame) -> str:
    """Markdown summary of the ratings.

    Args:
        tasks: Task ratings from `rate_tasks`.

    Returns:
        The summary, with the core subset and the counts of every flag over the main protocols.
    """
    main = tasks[tasks["main"]]
    core = sorted(tasks.loc[tasks["core"], "task"])
    lines = [
        "# Task ratings",
        "",
        f"Core subset, {len(core)} of {len(main)} tasks: "
        + (", ".join(f"`{t}`" for t in core) or "none")
        + ".",
        "",
        "## Main protocols",
        "",
        "| Flag | Settings |",
        "|---|---|",
    ]
    for status in ("resolved", "equivalent", "underpowered", "inconclusive"):
        lines.append(f"| {status} | {int((main['status'] == status).sum())} |")
    for flag in ("small", "saturated", "ceiling_bound", "floor_level", "pool_sensitive"):
        lines.append(f"| {flag.replace('_', ' ')} | {int((main[flag] == True).sum())} |")  # noqa: E712
    return "\n".join(lines) + "\n"


def write(tasks: pd.DataFrame, directory: Path) -> list[Path]:
    """Write the ratings as a CSV file and their Markdown summary.

    Args:
        tasks: Task ratings.
        directory: Destination folder, created when missing.

    Returns:
        The written paths.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tasks.csv"
    tasks.round(4).to_csv(path, index=False, lineterminator="\n")
    summary = directory / "ratings.md"
    summary.write_text(render(tasks), encoding="utf-8", newline="\n")
    return [path, summary]


def main(argv: list[str] | None = None) -> int:
    """Rate the tasks of the leaderboard from the per image records.

    Args:
        argv: Command line arguments without the program name, `sys.argv[1:]` when `None`.

    Returns:
        0 once the ratings are written.

    Raises:
        SystemExit: If the arguments are invalid, no leaderboard exists at the given path or no
            setting has records of two models, with status 2.
    """
    from find_those_polyps.evaluation.runner import EvaluationConfig
    from find_those_polyps.logging import configure_logging
    from find_those_polyps.registry.store import load_registry

    parser = argparse.ArgumentParser(prog="find-those-polyps ratings", description=__doc__)
    default = EvaluationConfig.model_fields["leaderboard"].default
    parser.add_argument("--leaderboard", type=Path, default=default, help=f"default {default}")
    parser.add_argument(
        "--per-image-dir",
        type=Path,
        nargs="+",
        default=[Path("results/per_image")],
        help="folders of per image results, searched in order, default results/per_image",
    )
    parser.add_argument("--output", type=Path, default=Path("results/ratings"))
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--metric-resamples", type=int, default=METRIC_RESAMPLES)
    parser.add_argument("--rope-fraction", type=float, default=ROPE_FRACTION)
    parser.add_argument("--workers", type=int, default=0, help="worker processes, none by default")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    if not args.leaderboard.exists():
        parser.error(f"no leaderboard at {args.leaderboard}")
    specs = {spec.id: spec for spec in load_registry(check=False).tasks}
    work = jobs(
        pd.read_csv(args.leaderboard),
        specs,
        args.per_image_dir,
        resamples=args.resamples,
        metric_resamples=args.metric_resamples,
    )
    if not work:
        parser.error("no setting has the records of two models")
    settings = measure_all(work, args.workers)
    tasks = rate_tasks(settings, args.rope_fraction)
    for path in write(tasks, args.output):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
