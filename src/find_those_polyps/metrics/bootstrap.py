"""Bootstrap confidence intervals of a metric over the test samples.

The percentile interval of Efron and Tibshirani (1993) resamples test samples with replacement and
recomputes the statistic. It quantifies the uncertainty due to the finite test set, which the
standard deviation over seeds does not capture. Resamples on which the statistic is undefined, for
instance AUROC without a positive sample, are ignored. Agreement measures between clusterings are
inflated on every resample, since duplicated samples agree with themselves, so their interval is the
bootstrap spread centred on the estimate.

A statistic maps the indices of a resample to the value of the metric on it. It may carry the
attributes `slow`, for statistics that recompute a whole metric suite and receive fewer resamples,
`centred`, for agreement measures whose interval is centred on the estimate, and `batch`, a
function of a matrix of resamples, one per row, returning the value on every row at once. Batched
statistics draw the same resamples and give the same values, computed with array operations
instead of one call per resample.
"""

import warnings
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata
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
from find_those_polyps.metrics.detection import IOU_THRESHOLDS, curve, match
from find_those_polyps.tasks.schema import Target, TaskKind

RESAMPLES = 1000
"""Default number of bootstrap resamples."""
BATCH_CELLS = 1 << 24
"""Largest number of resampled indices a batched statistic receives at once."""
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
    batch = getattr(statistic, "batch", None)
    if batch is None:
        values = np.array([statistic(rng.integers(0, n, n)) for _ in range(resamples)], dtype=float)
    else:
        rows = max(1, BATCH_CELLS // n)
        parts = []
        for start in range(0, resamples, rows):
            count = min(rows, resamples - start)
            parts.append(np.asarray(batch(rng.integers(0, n, (count, n))), dtype=float))
        values = np.concatenate(parts)
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
    """Resampled detection `metric`, with the detections of every image matched and ranked once.

    Matching depends on one image only, and the detections of each class are ranked by score once
    over all images. A resample then weights every detection by the number of times its image was
    drawn, which gives the curves of `summarize` on the pooled resampled images without pooling
    or sorting them again. Only the overlap thresholds that `metric` reads are computed. Detections
    of different images with exactly equal scores keep the order of their images instead of the
    order of the draw.

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
    columns = {"box_ap50": [0], "box_ap75": [5]}.get(metric, list(range(len(IOU_THRESHOLDS))))
    counts = np.array([m.targets for m in matched], dtype=np.float64).reshape(-1, n_classes)
    image = np.repeat(np.arange(len(matched)), [len(m.scores) for m in matched])
    classes = np.concatenate([m.classes for m in matched]) if matched else np.zeros(0, np.int64)
    scores = np.concatenate([m.scores for m in matched]) if matched else np.zeros(0)
    hits = (
        np.concatenate([m.hits for m in matched])[:, columns]
        if matched
        else np.zeros((0, len(columns)), dtype=bool)
    )
    ranked = {}
    for cls in range(1, n_classes):
        chosen = np.flatnonzero(classes == cls)
        order = chosen[np.argsort(-scores[chosen], kind="stable")]
        ranked[cls] = (image[order], hits[order])

    def statistic(index: NDArray[np.intp]) -> float:
        drawn = np.bincount(index, minlength=len(matched)).astype(np.float64)
        totals = drawn @ counts
        present = [cls for cls in range(1, n_classes) if totals[cls] > 0]
        if not present:
            return float("nan")
        values = np.zeros((len(present), len(columns), 2))
        for row, cls in enumerate(present):
            images, outcomes = ranked[cls]
            weights = drawn[images]
            kept = weights > 0
            for column in range(len(columns)):
                values[row, column] = curve(outcomes[kept, column], totals[cls], weights[kept])
        return float(values[..., 1 if metric == "box_ar100" else 0].mean())

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
    batches = [getattr(each, "batch", None) for each in statistics]
    if statistics and all(batch is not None for batch in batches):

        def batch(index: NDArray[np.intp]) -> NDArray[np.float64]:
            return np.mean(np.stack([each(index) for each in batches if each is not None]), axis=0)

        vars(statistic)["batch"] = batch
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

    def batch(index: NDArray[np.intp]) -> NDArray[np.float64]:
        chosen = values[index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(chosen, axis=1)
        means[~np.isfinite(chosen).any(axis=1)] = np.nan
        return means

    vars(statistic)["batch"] = batch
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
        if n_classes == 2:
            scores = probabilities[:, 1]
            vars(auroc)["batch"] = lambda index: _auroc_batch(labels[index], scores[index])
        return auroc
    if metric in simple:
        function = simple[metric]

        def statistic(index: NDArray[np.intp]) -> float:
            return function(labels[index], predictions[index])

        batched = _BATCHED.get(metric)
        if batched is not None:
            vars(statistic)["batch"] = lambda index: batched(
                _confusions(labels[index], predictions[index], n_classes)
            )
        return statistic
    return None


def _confusions(
    targets: NDArray[np.int64], predictions: NDArray[np.int64], n_classes: int
) -> NDArray[np.float64]:
    """Confusion matrix of every resample, targets along rows and predictions along columns.

    Args:
        targets: Class indices of shape `(B, n)`, one resample per row.
        predictions: Predicted class indices of the same shape.
        n_classes: Number of classes `C`.

    Returns:
        Counts of shape `(B, C, C)` as floats, exactly representable.
    """
    rows = len(targets)
    cells = (np.arange(rows)[:, None] * n_classes + targets) * n_classes + predictions
    counts = np.bincount(cells.ravel(), minlength=rows * n_classes * n_classes)
    return counts.reshape(rows, n_classes, n_classes).astype(np.float64)


def _mcc_batch(confusion: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matthews correlation of every confusion matrix, as `matthews_corrcoef` computes it.

    Every intermediate value is a sum of products of counts, exact in double precision, so the
    results equal those of scikit-learn. A zero denominator gives zero, as there.

    Args:
        confusion: Counts of shape `(B, C, C)`.

    Returns:
        The coefficient of every resample.
    """
    t_sum = confusion.sum(axis=2)
    p_sum = confusion.sum(axis=1)
    n_correct = np.einsum("bii->b", confusion)
    n_samples = p_sum.sum(axis=1)
    cov_ytyp = n_correct * n_samples - (t_sum * p_sum).sum(axis=1)
    cov_ypyp = n_samples**2 - (p_sum * p_sum).sum(axis=1)
    cov_ytyt = n_samples**2 - (t_sum * t_sum).sum(axis=1)
    product = cov_ypyp * cov_ytyt
    with np.errstate(divide="ignore", invalid="ignore"):
        values = cov_ytyp / np.sqrt(product)
    return np.where(product == 0, 0.0, values)


def _quadratic_kappa_batch(confusion: NDArray[np.float64]) -> NDArray[np.float64]:
    """Quadratically weighted kappa of every confusion matrix, as `cohen_kappa_score` computes it.

    Undefined values, without samples or with a single label in common, are NaN as there.

    Args:
        confusion: Counts of shape `(B, C, C)`.

    Returns:
        The kappa of every resample.
    """
    n_classes = confusion.shape[1]
    sum0 = confusion.sum(axis=1)
    sum1 = confusion.sum(axis=2)
    total = sum0.sum(axis=1)
    grid = np.arange(n_classes, dtype=np.float64)
    weights = (grid[None, :] - grid[:, None]) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.einsum("bi,bj->bij", sum0, sum1) / total[:, None, None]
        numerator = (weights * confusion).reshape(len(confusion), -1).sum(axis=1)
        denominator = (weights * expected).reshape(len(confusion), -1).sum(axis=1)
        values = 1 - numerator / denominator
    return np.where((total == 0) | (denominator == 0), np.nan, values)


def _accuracy_batch(confusion: NDArray[np.float64]) -> NDArray[np.float64]:
    """Accuracy of every confusion matrix.

    Args:
        confusion: Counts of shape `(B, C, C)`.

    Returns:
        The share of correct predictions of every resample.
    """
    correct: NDArray[np.float64] = np.einsum("bii->b", confusion)
    return correct / confusion.sum(axis=(1, 2))


_BATCHED: dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
    "mcc": _mcc_batch,
    "quadratic_kappa": _quadratic_kappa_batch,
    "accuracy": _accuracy_batch,
}
"""Single label metrics with a batched version, as functions of the confusion matrices."""


def _auroc_batch(targets: NDArray[np.int64], scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Binary AUROC of every resample from the ranks of its scores.

    The area under the ROC curve equals the Mann Whitney statistic with tied scores counted as
    one half, so it agrees with `roc_auc_score` up to floating point rounding. A resample with a
    single class gives NaN, as the point statistic does.

    Args:
        targets: Class indices of shape `(B, n)` in `{0, 1}`.
        scores: Scores of the positive class of the same shape.

    Returns:
        The AUROC of every resample.
    """
    ranks = rankdata(scores, axis=1, method="average")
    positives = targets.sum(axis=1).astype(np.float64)
    negatives = targets.shape[1] - positives
    rank_sum = np.where(targets == 1, ranks, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return np.where((positives == 0) | (negatives == 0), np.nan, values)


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
