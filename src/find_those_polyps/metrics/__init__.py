"""Task metrics, their optimization direction and their literature sources.

`score` dispatches on the task kind to the image level, segmentation and detection metrics, and
`METRIC_INFO` records for every metric whether higher values are better and which work defines it.
"""

from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from find_those_polyps.metrics.classification import TOP_K, multilabel, single_label
from find_those_polyps.metrics.detection import DETECTION, detection
from find_those_polyps.metrics.segmentation import OVERLAP, SALIENCY, segmentation
from find_those_polyps.tasks.schema import Target, TaskKind

Predictions = NDArray[np.float64] | Iterable[NDArray[np.float32]] | Sequence[NDArray[np.float64]]
"""Class probabilities `(N, C)`, an iterable of `(C, H, W)` probability maps or one table of
scored boxes `(M, 6)` per image."""


class MetricInfo(BaseModel):
    """Description of a metric and the work that defines it.

    Attributes:
        name: Key of the metric in result tables.
        higher_is_better: Whether larger values indicate better predictions. Defaults to True.
        reference: Short definition or literature source of the metric.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    higher_is_better: bool = True
    reference: str


def _info(name: str, reference: str, *, lower: bool = False) -> MetricInfo:
    """Metric description, with `lower` marking metrics for which lower values are better."""
    return MetricInfo(name=name, higher_is_better=not lower, reference=reference)


METRIC_INFO: dict[str, MetricInfo] = {
    info.name: info
    for info in (
        _info("accuracy", "Fraction of correct top one predictions"),
        _info("balanced_accuracy", "Brodersen et al. 2010, mean per class recall"),
        _info("macro_f1", "Unweighted mean of per class F1"),
        _info("weighted_f1", "Support weighted mean of per class F1"),
        _info("macro_precision", "Unweighted mean of per class precision"),
        _info("macro_recall", "Unweighted mean of per class recall"),
        _info("mcc", "Matthews 1975, Gorodkin 2004 multiclass extension"),
        _info("cohen_kappa", "Cohen 1960"),
        _info("auroc", "Hand and Till 2001, one versus rest macro average"),
        _info("auprc", "Macro average precision, one versus rest"),
        _info("log_loss", "Negative log likelihood", lower=True),
        _info("brier", "Brier 1950, multiclass form", lower=True),
        _info("ece", "Guo et al. 2017, fifteen equal width bins", lower=True),
        _info("sensitivity", "Recall of the positive class"),
        _info("specificity", "Recall of the negative class"),
        _info("ppv", "Positive predictive value"),
        _info("npv", "Negative predictive value, ASGE PIVI threshold 0.9"),
        _info("f1", "F1 of the positive class"),
        _info("quadratic_kappa", "Cohen 1968, quadratic weights"),
        _info("linear_kappa", "Cohen 1968, linear weights"),
        _info("mae", "Mean absolute error", lower=True),
        _info("within_one_accuracy", "Fraction within one ordinal level"),
        _info("spearman", "Spearman rank correlation"),
        _info("macro_ap", "Mean average precision over labels"),
        _info("micro_ap", "Average precision over all label decisions"),
        _info("macro_auroc", "Mean AUROC over labels"),
        _info("micro_auroc", "AUROC over all label decisions"),
        _info("micro_f1", "F1 over all label decisions"),
        _info("samples_f1", "Mean F1 over examples"),
        _info("hamming_loss", "Fraction of wrong label decisions", lower=True),
        _info("subset_accuracy", "Fraction of exactly predicted label sets"),
        _info("lrap", "Label ranking average precision, Tsoumakas et al. 2010"),
        _info("ranking_loss", "Tsoumakas et al. 2010", lower=True),
        _info("coverage_error", "Tsoumakas et al. 2010", lower=True),
        _info("dice", "Dice 1945, mean over images"),
        _info("iou", "Jaccard 1912, mean over images"),
        _info("precision", "Pixel precision, mean over images"),
        _info("recall", "Pixel recall, mean over images"),
        _info("f2", "Pixel F2, EndoCV2021 and PolypDB"),
        _info("nsd", "Normalized surface Dice, Nikolov et al. 2021, tolerance 2% of the diagonal"),
        _info("nsd_strict", "Normalized surface Dice at a tolerance of 1% of the diagonal"),
        _info("nhd95", "95th percentile Hausdorff distance over the image diagonal", lower=True),
        _info("nassd", "Average symmetric surface distance over the image diagonal", lower=True),
        _info("hd95", "95th percentile Hausdorff distance in pixels, per image only", lower=True),
        _info("assd", "Average symmetric surface distance in pixels, per image only", lower=True),
        _info("top3_accuracy", "Fraction of targets among the three most probable classes"),
        _info("top5_accuracy", "Fraction of targets among the five most probable classes"),
        _info("precision_at_1", "Retrieval precision of the nearest neighbour"),
        _info("precision_at_5", "Retrieval precision of the five nearest neighbours"),
        _info("r_precision", "Precision among the R nearest neighbours, Musgrave et al. 2020"),
        _info("map_at_r", "Mean average precision at R, Musgrave et al. 2020"),
        _info("map", "Mean average precision over the full ranking"),
        _info("nmi", "Normalized mutual information of k-means clusters, arithmetic mean"),
        _info("ami", "Adjusted mutual information of k-means clusters, Vinh et al. 2010"),
        _info("ari", "Adjusted Rand index of k-means clusters, Hubert and Arabie 1985"),
        _info("cluster_accuracy", "Accuracy after Hungarian matching of clusters to classes"),
        _info("s_measure", "Fan et al. 2017, alpha 0.5"),
        _info("mean_e_measure", "Fan et al. 2018, mean over 256 thresholds"),
        _info("max_e_measure", "Fan et al. 2018, maximum over 256 thresholds"),
        _info("weighted_f_measure", "Margolin et al. 2014, beta squared 1"),
        _info("box_map", "COCO box average precision over overlaps 0.5 to 0.95, Lin et al. 2014"),
        _info("box_ap50", "COCO box average precision at overlap 0.5, PASCAL VOC criterion"),
        _info("box_ap75", "COCO box average precision at overlap 0.75"),
        _info("box_ar100", "COCO box recall with 100 detections per class and image"),
    )
}
"""Description of every metric the benchmark reports, keyed by metric name."""

_SINGLE = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "mcc",
    "cohen_kappa",
    "auroc",
    "auprc",
    "log_loss",
    "brier",
    "ece",
)
_BINARY = ("sensitivity", "specificity", "ppv", "npv", "f1")
_ORDINAL = (
    "quadratic_kappa",
    "linear_kappa",
    "mae",
    "within_one_accuracy",
    "spearman",
)
_MULTILABEL = (
    "macro_ap",
    "micro_ap",
    "macro_auroc",
    "micro_auroc",
    "macro_f1",
    "micro_f1",
    "samples_f1",
    "hamming_loss",
    "subset_accuracy",
    "lrap",
    "ranking_loss",
    "coverage_error",
)


MASKED = frozenset({TaskKind.SEGMENTATION, TaskKind.MULTILABEL_SEGMENTATION})
"""Task kinds whose targets are masks at the resolution of the image."""


def metric_names(kind: TaskKind, n_classes: int) -> tuple[str, ...]:
    """Metrics reported for a task of the given kind and number of classes.

    Binary segmentation adds the salient object measures, single label tasks add the top k
    accuracies for every k below the number of classes, binary tasks add the operating point
    metrics of the positive class and ordinal tasks add the ordinal agreement metrics.

    Args:
        kind: Kind of the task.
        n_classes: Number of classes, including the background for dense tasks.

    Returns:
        Names of the reported metrics, all of them keys of `METRIC_INFO`.
    """
    if kind is TaskKind.SEGMENTATION:
        return OVERLAP + (SALIENCY if n_classes == 2 else ())
    if kind is TaskKind.MULTILABEL_SEGMENTATION:
        return OVERLAP
    if kind is TaskKind.DETECTION:
        return DETECTION
    if kind is TaskKind.MULTILABEL:
        return _MULTILABEL
    names = _SINGLE + tuple(f"top{k}_accuracy" for k in TOP_K if n_classes > k)
    names += _BINARY if n_classes == 2 else ()
    return names + (_ORDINAL if kind is TaskKind.ORDINAL else ())


def higher_is_better(name: str) -> bool:
    """Whether larger values of a metric indicate better predictions.

    Args:
        name: Name of the metric, a key of `METRIC_INFO`.

    Returns:
        True for metrics to maximize and False for losses and errors to minimize.
    """
    return METRIC_INFO[name].higher_is_better


def score(
    kind: TaskKind,
    predictions: Predictions,
    targets: Sequence[Target],
    n_classes: int,
    names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Metrics of `kind` for the predictions, optionally restricted to `names`.

    Args:
        kind: Kind of the task, which selects the metric family.
        predictions: Class probabilities of shape `(N, C)` for image level tasks, an iterable of
            probability maps of shape `(C, H, W)` for segmentation, or one table of rows class,
            score, x0, y0, x1, y1 per image for detection.
        targets: Class indices, indicator vectors, masks of shape `(H, W)`, stacks of class masks
            of shape `(C, H, W)` or tables of rows class, x0, y0, x1, y1, matching `predictions`.
        n_classes: Number of classes, including the background for dense tasks.
        names: Metrics to return, all those of the family when None.

    Returns:
        Value of every requested metric.

    Raises:
        TypeError: If `predictions` is a probability matrix for a segmentation or detection task,
            or anything else for an image level task.
    """
    if kind in MASKED:
        if isinstance(predictions, np.ndarray):
            raise TypeError("segmentation expects an iterable of probability maps")
        masks = [np.asarray(t, dtype=np.int64) for t in targets]
        maps = cast(Iterable[NDArray[np.float32]], predictions)
        overlapping = kind is TaskKind.MULTILABEL_SEGMENTATION
        return segmentation(maps, masks, n_classes, names, multilabel=overlapping)
    if kind is TaskKind.DETECTION:
        if isinstance(predictions, np.ndarray):
            raise TypeError("detection expects one table of scored boxes per image")
        boxes = [np.asarray(t, dtype=np.float64).reshape(-1, 5) for t in targets]
        return detection(cast(Iterable[NDArray[np.float64]], predictions), boxes, n_classes, names)
    if not isinstance(predictions, np.ndarray):
        raise TypeError(f"{kind} expects a probability matrix")
    if kind is TaskKind.MULTILABEL:
        metrics = multilabel(np.stack([np.asarray(t) for t in targets]), predictions)
    else:
        labels = np.asarray(targets, dtype=np.int64)
        metrics = single_label(labels, predictions, ordinal=kind is TaskKind.ORDINAL)
    return metrics if names is None else {k: metrics[k] for k in names}


__all__ = [
    "METRIC_INFO",
    "MetricInfo",
    "Predictions",
    "higher_is_better",
    "metric_names",
    "score",
]
