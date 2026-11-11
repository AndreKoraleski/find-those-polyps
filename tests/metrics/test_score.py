import numpy as np
import pytest

from find_those_polyps.metrics import METRIC_INFO, higher_is_better, metric_names, score
from find_those_polyps.tasks import TaskKind


def _maps(masks: list[np.ndarray]) -> list[np.ndarray]:
    return [np.stack([1 - m, m]).astype(np.float32) for m in masks]


@pytest.mark.parametrize("kind", list(TaskKind))
@pytest.mark.parametrize("n_classes", [2, 4])
def test_every_reported_metric_is_documented(kind: TaskKind, n_classes: int) -> None:
    names = metric_names(kind, n_classes)
    assert len(set(names)) == len(names)
    assert set(names) <= set(METRIC_INFO)


def test_binary_and_ordinal_extras() -> None:
    assert "npv" in metric_names(TaskKind.CLASSIFICATION, 2)
    assert "npv" not in metric_names(TaskKind.CLASSIFICATION, 3)
    assert "quadratic_kappa" in metric_names(TaskKind.ORDINAL, 4)
    assert "s_measure" in metric_names(TaskKind.SEGMENTATION, 2)
    assert "s_measure" not in metric_names(TaskKind.SEGMENTATION, 3)
    assert not higher_is_better("mae")
    assert higher_is_better("dice")


def test_score_returns_exactly_the_reported_metrics() -> None:
    targets = [0, 1, 1, 0, 1]
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.3, 0.7], [0.1, 0.9]])
    metrics = score(TaskKind.CLASSIFICATION, probabilities, targets, 2)
    assert set(metrics) == set(metric_names(TaskKind.CLASSIFICATION, 2))
    assert score(TaskKind.CLASSIFICATION, probabilities, targets, 2, ["mcc"]).keys() == {"mcc"}


def test_segmentation_is_averaged_per_image() -> None:
    target = np.zeros((4, 4), dtype=np.int64)
    target[:2] = 1
    half = np.zeros((4, 4), dtype=np.int64)
    half[0] = 1
    empty = np.zeros((4, 4), dtype=np.int64)
    metrics = score(TaskKind.SEGMENTATION, _maps([target, half, empty]), [target, target, empty], 2)
    assert set(metrics) == set(metric_names(TaskKind.SEGMENTATION, 2))
    assert metrics["dice"] == pytest.approx((1 + 2 * 4 / 12 + 1) / 3)
    assert metrics["iou"] == pytest.approx((1 + 0.5 + 1) / 3)


def test_prediction_type_is_checked() -> None:
    with pytest.raises(TypeError):
        score(TaskKind.SEGMENTATION, np.zeros((2, 2)), [0, 1], 2)
    with pytest.raises(TypeError):
        score(TaskKind.CLASSIFICATION, [np.zeros((2, 2, 2), dtype=np.float32)], [0], 2)
