import warnings

import numpy as np
import pytest

from find_those_polyps.metrics.classification import (
    expected_calibration_error,
    multilabel,
    single_label,
)


def test_perfect_multiclass_prediction() -> None:
    targets = np.array([0, 1, 2, 2, 1, 0])
    metrics = single_label(targets, np.eye(3)[targets])
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "mcc", "cohen_kappa", "auroc"):
        assert metrics[name] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.0)
    assert metrics["ece"] == pytest.approx(0.0)


def test_binary_operating_point() -> None:
    targets = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    predictions = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    probabilities = np.eye(2)[predictions] * 0.8 + 0.1
    metrics = single_label(targets, probabilities)
    assert metrics["sensitivity"] == pytest.approx(2 / 3)
    assert metrics["specificity"] == pytest.approx(4 / 5)
    assert metrics["ppv"] == pytest.approx(2 / 3)
    assert metrics["npv"] == pytest.approx(4 / 5)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_ordinal_errors_are_weighted_by_distance() -> None:
    targets = np.array([0, 1, 2, 3])
    near = single_label(targets, np.eye(4)[[1, 1, 2, 3]], ordinal=True)
    far = single_label(targets, np.eye(4)[[3, 1, 2, 3]], ordinal=True)
    assert near["accuracy"] == far["accuracy"]
    assert near["quadratic_kappa"] > far["quadratic_kappa"]
    assert near["mae"] == pytest.approx(0.25)
    assert near["within_one_accuracy"] == 1.0
    assert far["within_one_accuracy"] == 0.75


def test_missing_class_leaves_auroc_undefined() -> None:
    metrics = single_label(np.array([0, 1]), np.eye(3)[[0, 1]])
    assert np.isnan(metrics["auroc"])
    assert not np.isnan(metrics["auprc"])


def test_expected_calibration_error_of_overconfident_model() -> None:
    targets = np.array([0, 1, 0, 1])
    probabilities = np.array([[0.9, 0.1]] * 4)
    assert expected_calibration_error(targets, probabilities) == pytest.approx(0.4)


def test_multilabel_ranking_and_set_metrics() -> None:
    targets = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    metrics = multilabel(targets, targets.astype(np.float64))
    assert metrics["macro_ap"] == pytest.approx(1.0)
    assert metrics["subset_accuracy"] == 1.0
    assert metrics["hamming_loss"] == 0.0
    assert metrics["ranking_loss"] == 0.0
    worst = multilabel(targets, 1.0 - targets)
    assert worst["subset_accuracy"] == 0.0
    assert worst["macro_auroc"] == pytest.approx(0.0)


def test_degenerate_subsets_score_without_warnings() -> None:
    targets = np.array([1, 1, 1])
    probabilities = np.array([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.1, 0.1, 0.8]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = single_label(targets, probabilities, ordinal=True)
    assert np.isnan(metrics["spearman"])
