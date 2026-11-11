"""Bootstrap confidence intervals of a metric over the test samples.

The percentile interval of Efron and Tibshirani (1993) resamples test samples with replacement and
recomputes the statistic. It quantifies the uncertainty due to the finite test set, which the
standard deviation over seeds does not capture. Resamples on which the statistic is undefined, for
instance AUROC without a positive sample, are ignored. Agreement measures between clusterings are
inflated on every resample, since duplicated samples agree with themselves, so their interval is the
bootstrap spread centred on the estimate.

A statistic maps the indices of a resample to the value of the metric on it. It may carry the
attributes `slow`, for statistics that recompute a whole metric suite and receive fewer resamples,
and `centred`, for agreement measures whose interval is centred on the estimate.
"""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from find_those_polyps.metrics import score
from find_those_polyps.metrics.classification import (
    lrap,
    macro_ap,
    macro_auroc,
    micro_ap,
    ranked_examples,
    varying_labels,
)
from find_those_polyps.metrics.detection import match, summarize
from find_those_polyps.tasks.schema import Target, TaskKind

RESAMPLES = 1000
"""Default number of bootstrap resamples."""
SLOW_RESAMPLES = 200
"""Number of resamples of statistics marked `slow`."""
LEVEL = 0.95
"""Default coverage of the confidence interval."""

Statistic = Callable[[NDArray[np.intp]], float]
"""Function mapping the sample indices of a resample to the value of a metric on it."""


def interval(
    statistic: Statistic,
    n: int,
    *,
    resamples: int = RESAMPLES,
    level: float = LEVEL,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval of `statistic` over `n` samples.

    Statistics that recompute a whole metric suite, marked `slow`, use at most `SLOW_RESAMPLES`
    resamples. Statistics marked `centred` get the percentile interval shifted by the difference
    between the median of the resamples and the estimate on all samples, which is the spread of the
    resamples around that estimate. Resamples with a non finite value are ignored.

    Args:
        statistic: Statistic to resample.
        n: Number of test samples.
        resamples: Number of resamples drawn with replacement.
        level: Coverage of the interval in `(0, 1)`.
        seed: Seed of the resampling.

    Returns:
        Lower and upper bounds of the interval, both NaN when there are no samples or no resample
        has a finite value.
    """
    if getattr(statistic, "slow", False):
        resamples = min(resamples, SLOW_RESAMPLES)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = np.array([statistic(rng.integers(0, n, n)) for _ in range(resamples)], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    if getattr(statistic, "centred", False):
        shift = float(np.median(values)) - statistic(np.arange(n))
        low, high = low - shift, high - shift
    return float(low), float(high)


def detection_statistic(
    metric: str,
    predictions: Sequence[NDArray[np.float64]],
    targets: Sequence[Target],
    n_classes: int,
) -> Statistic:
    """Resampled detection `metric`, with the detections of every image matched once.

    Matching depends on one image only, so a resample merely pools the matches of its images
    before the precision recall curves are computed.

    Args:
        metric: Detection metric, one of `DETECTION`.
        predictions: One table of rows class, score, x0, y0, x1, y1 per image.
        targets: One table of rows class, x0, y0, x1, y1 per image.
        n_classes: Number of classes, including the background.

    Returns:
        Statistic of `metric` over the resampled images, NaN when no target occurs in them.
    """
    matched = [
        match(p, np.asarray(t, dtype=np.float64).reshape(-1, 5), n_classes)
        for p, t in zip(predictions, targets, strict=True)
    ]

    def statistic(index: NDArray[np.intp]) -> float:
        return summarize([matched[i] for i in index], n_classes)[metric]

    return statistic


def averaged(statistics: Sequence[Statistic]) -> Statistic:
    """Mean of several statistics on the same resample, such as one per seed.

    A resample on which any of them is undefined is undefined. The combined statistic is marked
    `slow` or `centred` when any of the statistics is.

    Args:
        statistics: Statistics over the same samples.

    Returns:
        Statistic of the mean of their values.
    """

    def statistic(index: NDArray[np.intp]) -> float:
        return float(np.mean([each(index) for each in statistics]))

    for marker in ("slow", "centred"):
        setattr(statistic, marker, any(getattr(each, marker, False) for each in statistics))
    return statistic


def mean_statistic(values: NDArray[np.float64]) -> Statistic:
    """Mean of per sample values, ignoring undefined ones.

    Args:
        values: Value of the metric on every sample, NaN where it is undefined.

    Returns:
        Statistic of the mean over the finite values of a resample, NaN when it has none.
    """

    def statistic(index: NDArray[np.intp]) -> float:
        chosen = values[index]
        return float(np.nanmean(chosen)) if np.isfinite(chosen).any() else float("nan")

    return statistic


def _single(
    metric: str, labels: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> Callable[[NDArray[np.intp]], float] | None:
    """Fast resampled version of common single label metrics, or None for the others.

    It avoids recomputing the whole suite of `single_label` and agrees with it when the
    probabilities of every sample sum to one.

    Args:
        metric: Name of the metric.
        labels: Class indices of shape `(N,)`.
        probabilities: Class probabilities of shape `(N, C)`.

    Returns:
        Statistic of `metric`, or None when no fast version exists.
    """
    predictions = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]

    def auroc(index: NDArray[np.intp]) -> float:
        targets = labels[index]
        if len(np.unique(targets)) < n_classes:
            return float("nan")
        if n_classes == 2:
            return float(roc_auc_score(targets, probabilities[index, 1]))
        return float(roc_auc_score(targets, probabilities[index], multi_class="ovr"))

    simple: dict[str, Callable[[NDArray[np.int64], NDArray[np.int64]], float]] = {
        "accuracy": lambda t, p: float(accuracy_score(t, p)),
        "balanced_accuracy": lambda t, p: float(balanced_accuracy_score(t, p)),
        "mcc": lambda t, p: float(matthews_corrcoef(t, p)),
        "macro_f1": lambda t, p: float(
            f1_score(t, p, labels=np.arange(n_classes), average="macro", zero_division=0)
        ),
        "quadratic_kappa": lambda t, p: float(
            cohen_kappa_score(t, p, labels=np.arange(n_classes), weights="quadratic")
        ),
    }
    if metric == "auroc":
        return auroc
    if metric in simple:
        function = simple[metric]
        return lambda index: function(labels[index], predictions[index])
    return None


def prediction_statistic(
    kind: TaskKind,
    metric: str,
    probabilities: NDArray[np.float64],
    targets: tuple[Target, ...],
    n_classes: int,
) -> Statistic:
    """Resampled `metric` of class probabilities against their targets.

    Common single label metrics and multilabel ranking metrics have fast versions, and any other
    metric recomputes `score` on every resample and is marked `slow`.

    Args:
        kind: Kind of the image level task.
        metric: Name of the metric.
        probabilities: Class or label probabilities of shape `(N, C)`.
        targets: Class indices or indicator vectors of the samples.
        n_classes: Number of classes.

    Returns:
        Statistic of `metric` over the resampled samples.
    """
    if kind in (TaskKind.CLASSIFICATION, TaskKind.ORDINAL):
        labels = np.asarray(targets, dtype=np.int64)
        fast = _single(metric, labels, probabilities)
        if fast is not None:
            return fast
    if kind is TaskKind.MULTILABEL and metric in _MULTILABEL:
        indicators = np.stack([np.asarray(t) for t in targets])
        return _multilabel(metric, indicators, probabilities)

    def statistic(index: NDArray[np.intp]) -> float:
        chosen = tuple(targets[i] for i in index)
        return score(kind, probabilities[index], chosen, n_classes, [metric])[metric]

    vars(statistic)["slow"] = True
    return statistic


_MULTILABEL: dict[str, Callable[[NDArray[np.int64], NDArray[np.float64]], float]] = {
    "macro_ap": macro_ap,
    "micro_ap": micro_ap,
    "macro_auroc": macro_auroc,
    "lrap": lrap,
}
"""Multilabel ranking metrics with a fast resampled version, as functions of targets and scores."""


def _multilabel(
    metric: str, indicators: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> Statistic:
    """Fast resampled multilabel ranking metric, under the conventions of `multilabel`.

    Label wise averages leave out the labels that are constant on the resample, and label
    ranking average precision leaves out its examples without a positive label, exactly as the
    point estimate does on the test set. Micro averages pool every label decision.

    Args:
        metric: Name of the metric, a key of `_MULTILABEL`.
        indicators: Label indicators of shape `(N, C)`.
        probabilities: Label probabilities of shape `(N, C)`.

    Returns:
        Statistic of `metric`, NaN on resamples where a label wise average has no label that
            varies or label ranking average precision has no example with a positive label.
    """
    function = _MULTILABEL[metric]

    def statistic(index: NDArray[np.intp]) -> float:
        chosen = indicators[index]
        if metric in ("macro_ap", "macro_auroc") and not varying_labels(chosen).any():
            return float("nan")
        if metric == "lrap" and not ranked_examples(chosen).any():
            return float("nan")
        return function(chosen, probabilities[index])

    return statistic
