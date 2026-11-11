"""Metrics for image level single label, ordinal and multilabel targets.

Single label predictions are the argmax of the class probabilities, and multilabel predictions
threshold every label probability at one half. Undefined values on test subsets that lack classes,
as in bootstrap resamples and smoke tests, follow the conventions of scikit-learn and SciPy unless
stated otherwise.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    coverage_error,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    label_ranking_loss,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

ECE_BINS = 15
"""Number of equal width confidence bins of the expected calibration error."""
TOP_K = (3, 5)
"""Values of k of the top k accuracies, reported when the task has more than k classes."""


def expected_calibration_error(
    targets: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> float:
    """Top label calibration error over equal width confidence bins (Guo et al. 2017).

    The confidence of a sample is its largest class probability and it is correct when that class
    is the target. The error is the mean over the `ECE_BINS` bins of the gap between accuracy and
    mean confidence, weighted by the fraction of samples in the bin. Bins are closed on the left,
    the last one on both sides, and empty bins contribute nothing.

    Args:
        targets: Class indices of shape `(N,)`.
        probabilities: Class probabilities of shape `(N, C)`.

    Returns:
        Expected calibration error in `[0, 1]`, zero when there are no samples.
    """
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    edges = np.linspace(0, 1, ECE_BINS + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1]), 0, ECE_BINS - 1)
    error = 0.0
    for b in range(ECE_BINS):
        members = bins == b
        if members.any():
            gap = abs(correct[members].mean() - confidence[members].mean())
            error += members.mean() * gap
    return float(error)


def _ratio(numerator: float, denominator: float) -> float:
    """Quotient of the two counts, NaN when the denominator is zero."""
    return numerator / denominator if denominator else float("nan")


def _binary(targets: NDArray[np.int64], predictions: NDArray[np.int64]) -> dict[str, float]:
    """Operating point metrics with class 1 as the positive class.

    Args:
        targets: Binary class indices of shape `(N,)`.
        predictions: Predicted class indices of shape `(N,)`.

    Returns:
        Sensitivity, specificity, positive and negative predictive values and F1 of the positive
        class, each in `[0, 1]` and NaN when its denominator is zero.
    """
    tp = float(np.sum((predictions == 1) & (targets == 1)))
    tn = float(np.sum((predictions == 0) & (targets == 0)))
    fp = float(np.sum((predictions == 1) & (targets == 0)))
    fn = float(np.sum((predictions == 0) & (targets == 1)))
    return {
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def single_label(
    targets: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    *,
    ordinal: bool = False,
) -> dict[str, float]:
    """Discrimination, agreement, calibration and, if applicable, ordinal metrics.

    Probabilities are renormalized to sum to one per sample. Macro averages run over all classes
    and count an undefined per class score as zero. The AUROC is NaN unless every class occurs in
    the targets, the AUPRC averages over the classes that occur, and the Spearman correlation is NaN
    when only one class occurs. Binary tasks add the metrics of class 1 as the positive class, and
    tasks with more than k classes add the top k accuracy for every k in `TOP_K`. Test subsets
    lacking classes, as in bootstrap resamples and smoke tests, follow the conventions of
    scikit-learn and SciPy without their warnings.

    Args:
        targets: Class indices of shape `(N,)`.
        probabilities: Class probabilities of shape `(N, C)`.
        ordinal: Whether the classes are ordered levels, which adds the weighted kappas, the mean
            absolute error in levels, the accuracy within one level and the Spearman correlation.

    Returns:
        Value of every metric. Accuracies, F1, precision, recall, AUROC, AUPRC and ECE lie in
        `[0, 1]`, MCC, the kappas and the Spearman correlation in `[-1, 1]`, the Brier score in
        `[0, 2]`, and the log loss and mean absolute error are nonnegative.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "y_pred contains classes not in y_true")
        warnings.filterwarnings("ignore", "A single label was found")
        warnings.simplefilter("ignore", ConstantInputWarning)
        return _single_label(targets, probabilities, ordinal=ordinal)


def _single_label(
    targets: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    *,
    ordinal: bool,
) -> dict[str, float]:
    """Metrics of `single_label`, computed without silencing warnings."""
    n_classes = probabilities.shape[1]
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    labels = np.arange(n_classes)
    predictions = probabilities.argmax(axis=1)
    present = np.unique(targets)
    indicators = np.eye(n_classes)[targets]
    full_support = len(present) == n_classes

    def macro(metric: Callable[..., Any]) -> float:
        return float(metric(targets, predictions, labels=labels, average="macro", zero_division=0))

    if not full_support:
        auroc = float("nan")
    elif n_classes == 2:
        auroc = float(roc_auc_score(targets, probabilities[:, 1]))
    else:
        auroc = float(roc_auc_score(targets, probabilities, multi_class="ovr", average="macro"))
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": macro(f1_score),
        "weighted_f1": float(
            f1_score(targets, predictions, labels=labels, average="weighted", zero_division=0)
        ),
        "macro_precision": macro(precision_score),
        "macro_recall": macro(recall_score),
        "mcc": float(matthews_corrcoef(targets, predictions)),
        "cohen_kappa": float(cohen_kappa_score(targets, predictions, labels=labels)),
        "auroc": auroc,
        "auprc": float(
            average_precision_score(
                indicators[:, present], probabilities[:, present], average="macro"
            )
        ),
        "log_loss": float(log_loss(targets, probabilities, labels=labels)),
        "brier": float(np.mean(np.sum((probabilities - indicators) ** 2, axis=1))),
        "ece": expected_calibration_error(targets, probabilities),
    }
    for k in TOP_K:
        if n_classes > k:
            metrics[f"top{k}_accuracy"] = float(
                top_k_accuracy_score(targets, probabilities, k=k, labels=labels)
            )
    if n_classes == 2:
        metrics |= _binary(targets, predictions)
    if ordinal:
        distance = np.abs(predictions - targets)
        rho = spearmanr(targets, predictions).statistic if len(present) > 1 else np.nan
        metrics |= {
            "quadratic_kappa": float(
                cohen_kappa_score(targets, predictions, labels=labels, weights="quadratic")
            ),
            "linear_kappa": float(
                cohen_kappa_score(targets, predictions, labels=labels, weights="linear")
            ),
            "mae": float(distance.mean()),
            "within_one_accuracy": float((distance <= 1).mean()),
            "spearman": float(rho),
        }
    return metrics


def varying_labels(targets: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Labels that are present on some examples and absent on others.

    Args:
        targets: Label indicators of shape `(N, C)`.

    Returns:
        Mask of shape `(C,)` of the labels whose ranking metrics are defined.
    """
    totals = targets.sum(axis=0)
    return np.asarray((totals > 0) & (totals < len(targets)), dtype=np.bool_)


def ranked_examples(targets: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Examples with at least one positive label, which example wise rankings consider.

    Args:
        targets: Label indicators of shape `(N, C)`.

    Returns:
        Mask of shape `(N,)` of the examples with a positive label.
    """
    return np.asarray(targets.sum(axis=1) > 0, dtype=np.bool_)


def macro_ap(targets: NDArray[np.int64], probabilities: NDArray[np.float64]) -> float:
    """Mean average precision over the labels that are not constant.

    Args:
        targets: Label indicators of shape `(N, C)`.
        probabilities: Label probabilities of shape `(N, C)`.

    Returns:
        The label wise average precision averaged over the labels of `varying_labels`.
    """
    varying = varying_labels(targets)
    return float(
        average_precision_score(targets[:, varying], probabilities[:, varying], average="macro")
    )


def micro_ap(targets: NDArray[np.int64], probabilities: NDArray[np.float64]) -> float:
    """Average precision of every label decision pooled together.

    Args:
        targets: Label indicators of shape `(N, C)`.
        probabilities: Label probabilities of shape `(N, C)`.

    Returns:
        The average precision of the flattened indicators and probabilities.
    """
    return float(average_precision_score(targets, probabilities, average="micro"))


def macro_auroc(targets: NDArray[np.int64], probabilities: NDArray[np.float64]) -> float:
    """Mean AUROC over the labels that are not constant.

    Args:
        targets: Label indicators of shape `(N, C)`.
        probabilities: Label probabilities of shape `(N, C)`.

    Returns:
        The label wise AUROC averaged over the labels of `varying_labels`.
    """
    varying = varying_labels(targets)
    return float(roc_auc_score(targets[:, varying], probabilities[:, varying], average="macro"))


def lrap(targets: NDArray[np.int64], probabilities: NDArray[np.float64]) -> float:
    """Label ranking average precision over the examples with a positive label.

    Args:
        targets: Label indicators of shape `(N, C)`.
        probabilities: Label probabilities of shape `(N, C)`.

    Returns:
        The label ranking average precision of the examples of `ranked_examples`.
    """
    ranked = ranked_examples(targets)
    return float(label_ranking_average_precision_score(targets[ranked], probabilities[ranked]))


def multilabel(targets: NDArray[np.int64], probabilities: NDArray[np.float64]) -> dict[str, float]:
    """Ranking, threshold and example based metrics for label sets.

    Labels are predicted where their probability is at least one half. Label wise averages of
    average precision and AUROC exclude labels that are constant on the test set, whose ranking
    metrics are undefined, while micro averages pool every label decision. Label ranking average
    precision and coverage error leave out examples without any positive label. F1 scores count
    an undefined value as zero.

    Args:
        targets: Label indicators of shape `(N, C)`.
        probabilities: Label probabilities of shape `(N, C)`.

    Returns:
        Value of every metric. The coverage error lies in `[1, C]` and every other metric in
        `[0, 1]`.
    """
    predictions = (probabilities >= 0.5).astype(np.int64)
    ranked = ranked_examples(targets)
    return {
        "macro_ap": macro_ap(targets, probabilities),
        "micro_ap": micro_ap(targets, probabilities),
        "macro_auroc": macro_auroc(targets, probabilities),
        "micro_auroc": float(roc_auc_score(targets, probabilities, average="micro")),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(targets, predictions, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(targets, predictions)),
        "subset_accuracy": float(accuracy_score(targets, predictions)),
        "lrap": lrap(targets, probabilities),
        "ranking_loss": float(label_ranking_loss(targets, probabilities)),
        "coverage_error": float(coverage_error(targets[ranked], probabilities[ranked])),
    }
