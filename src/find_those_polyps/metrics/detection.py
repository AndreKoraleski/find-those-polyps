"""Object detection metrics following the COCO evaluation protocol of Lin et al. (2014).

Box corners x0, y0, x1, y1 may be in any unit shared by predictions and targets, and the
benchmark normalizes them by the image size, which leaves their overlaps unchanged. A target holds
rows class, x0, y0, x1, y1 and a prediction rows class, score, x0, y0, x1, y1, with object classes
counted from one since class zero is the background. In every image the hundred highest scoring
detections of each class are matched greedily in order of score to the unmatched target of that
class with the highest overlap. Average precision interpolates the precision recall curve of each
class at 101 recall levels over the whole test set, and classes without any target are left out of
every mean. Detections of such classes are ignored.
"""

from collections.abc import Iterable, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)
"""Overlap thresholds of COCO from 0.5 to 0.95 in steps of 0.05."""
MAX_DETECTIONS = 100
"""Number of highest scoring detections per image and class that enter the evaluation."""
RECALL_LEVELS = np.linspace(0, 1, 101)
"""Recall levels at which the precision envelope is sampled."""
DETECTION = ("box_map", "box_ap50", "box_ap75", "box_ar100")
"""Detection metrics reported for every task."""


def box_iou(first: NDArray[np.float64], second: NDArray[np.float64]) -> NDArray[np.float64]:
    """Intersection over union of every pair of boxes from two sets.

    Args:
        first: Corners x0, y0, x1, y1 of shape `(N, 4)`.
        second: Corners x0, y0, x1, y1 of shape `(M, 4)`.

    Returns:
        Overlaps of shape `(N, M)` in `[0, 1]`, zero where the union is empty.
    """
    low = np.maximum(first[:, None, :2], second[None, :, :2])
    high = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.prod(np.clip(high - low, 0, None), axis=2)

    def area(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.prod(np.clip(boxes[:, 2:] - boxes[:, :2], 0, None), axis=1)

    union = area(first)[:, None] + area(second)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


class Matches(NamedTuple):
    """Detections of one image with their matching outcome at every overlap threshold.

    Attributes:
        classes: Class of every kept detection, of shape `(D,)`, in decreasing order of score.
        scores: Score of every kept detection, of shape `(D,)`.
        hits: Whether each detection matched a target at each threshold of `IOU_THRESHOLDS`, of
            shape `(D, T)`.
        targets: Number of targets of every class of the image, of shape `(C,)`, with the
            background first.
    """

    classes: NDArray[np.int64]
    scores: NDArray[np.float64]
    hits: NDArray[np.bool_]
    targets: NDArray[np.int64]


def match(prediction: NDArray[np.float64], target: NDArray[np.float64], n_classes: int) -> Matches:
    """Greedy matching of the detections of one image to its targets, class by class.

    Detections are sorted by decreasing score, with ties in their original order, and only the
    `MAX_DETECTIONS` highest scoring ones of each class are kept. At every threshold each detection
    in turn takes the unmatched target of its class with the highest overlap, provided that overlap
    reaches the threshold.

    Args:
        prediction: Rows class, score, x0, y0, x1, y1, reshaped to `(D, 6)`.
        target: Rows class, x0, y0, x1, y1, reshaped to `(M, 5)`.
        n_classes: Number of classes, including the background.

    Returns:
        Kept detections with their outcomes and the target counts of the image.
    """
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, 6)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 5)
    prediction = prediction[np.argsort(-prediction[:, 1], kind="stable")]
    classes = prediction[:, 0].astype(np.int64)
    rank = np.zeros(len(classes), dtype=np.int64)
    for cls in np.unique(classes):
        chosen = np.flatnonzero(classes == cls)
        rank[chosen] = np.arange(len(chosen))
    prediction, classes = prediction[rank < MAX_DETECTIONS], classes[rank < MAX_DETECTIONS]
    hits = np.zeros((len(prediction), len(IOU_THRESHOLDS)), dtype=bool)
    target_classes = target[:, 0].astype(np.int64)
    for cls in np.unique(classes):
        rows = np.flatnonzero(classes == cls)
        truth = target[target_classes == cls, 1:]
        if not len(truth):
            continue
        overlaps = box_iou(prediction[rows, 2:], truth)
        taken = np.zeros((len(IOU_THRESHOLDS), len(truth)), dtype=bool)
        thresholds = np.arange(len(IOU_THRESHOLDS))
        for row, overlap in zip(rows, overlaps, strict=True):
            candidates = np.where(taken | (overlap < IOU_THRESHOLDS[:, None]), -1.0, overlap)
            best = candidates.argmax(axis=1)
            found = candidates[thresholds, best] >= 0
            taken[thresholds[found], best[found]] = True
            hits[row] = found
    counts = np.bincount(target_classes, minlength=n_classes)[:n_classes]
    return Matches(classes, prediction[:, 1], hits, counts)


def _curve(hits: NDArray[np.bool_], n_targets: int) -> tuple[float, float]:
    """Interpolated average precision and maximal recall of ranked detections of one class.

    The precision envelope, the running maximum of the precision from the lowest score up, is
    sampled at the first rank reaching each of the `RECALL_LEVELS`, and counts zero at the levels
    that no rank reaches.

    Args:
        hits: Whether each detection, in decreasing order of score, is a true positive.
        n_targets: Number of targets of the class, positive.

    Returns:
        Average precision and maximal recall, both in `[0, 1]` and both zero without detections.
    """
    if not len(hits):
        return 0.0, 0.0
    true = np.cumsum(hits)
    precision = true / np.arange(1, len(hits) + 1)
    recall = true / n_targets
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, RECALL_LEVELS, side="left")
    sampled = np.where(index < len(envelope), envelope[np.minimum(index, len(envelope) - 1)], 0)
    return float(sampled.mean()), float(recall[-1])


def summarize(matches: Sequence[Matches], n_classes: int) -> dict[str, float]:
    """COCO average precision and recall over images given by their matches.

    The detections of every class are pooled over images and ranked by score. Metrics average
    over the classes with at least one target, and `box_map` and `box_ar100` also average over the
    overlap thresholds.

    Args:
        matches: Matches of every image.
        n_classes: Number of classes, including the background.

    Returns:
        Every metric of `DETECTION` in `[0, 1]`, all NaN when there are no images or no targets.
    """
    if not matches:
        return dict.fromkeys(DETECTION, float("nan"))
    classes = np.concatenate([m.classes for m in matches])
    scores = np.concatenate([m.scores for m in matches])
    hits = np.concatenate([m.hits for m in matches])
    targets = np.sum([m.targets for m in matches], axis=0)
    present = [c for c in range(1, n_classes) if targets[c] > 0]
    if not present:
        return dict.fromkeys(DETECTION, float("nan"))
    precision = np.zeros((len(present), len(IOU_THRESHOLDS)))
    recall = np.zeros_like(precision)
    for row, cls in enumerate(present):
        chosen = np.flatnonzero(classes == cls)
        ranked = chosen[np.argsort(-scores[chosen], kind="stable")]
        for column in range(len(IOU_THRESHOLDS)):
            precision[row, column], recall[row, column] = _curve(
                hits[ranked, column], int(targets[cls])
            )
    return {
        "box_map": float(precision.mean()),
        "box_ap50": float(precision[:, 0].mean()),
        "box_ap75": float(precision[:, 5].mean()),
        "box_ar100": float(recall.mean()),
    }


def detection(
    predictions: Iterable[NDArray[np.float64]],
    targets: Sequence[NDArray[np.float64]],
    n_classes: int,
    names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Requested detection metrics of per image predictions against their targets.

    Args:
        predictions: One table of rows class, score, x0, y0, x1, y1 per image.
        targets: One table of rows class, x0, y0, x1, y1 per image.
        n_classes: Number of classes, including the background.
        names: Metrics to return, all of `DETECTION` when None.

    Returns:
        Value of every requested metric, as `summarize` gives it.
    """
    matched = [match(p, t, n_classes) for p, t in zip(predictions, targets, strict=True)]
    metrics = summarize(matched, n_classes)
    return metrics if names is None else {name: metrics[name] for name in names}
