import numpy as np
import pytest

from find_those_polyps.metrics.detection import box_iou, detection, match

TARGET = np.array([[1, 0.1, 0.1, 0.4, 0.4], [2, 0.5, 0.5, 0.9, 0.8]])


def _scored(target: np.ndarray, scores: list[float]) -> np.ndarray:
    return np.column_stack([target[:, 0], scores, target[:, 1:]])


def test_box_iou_of_overlapping_and_disjoint_boxes() -> None:
    first = np.array([[0.0, 0.0, 2.0, 2.0]])
    second = np.array([[1.0, 1.0, 3.0, 3.0], [5.0, 5.0, 6.0, 6.0]])
    assert box_iou(first, second) == pytest.approx(np.array([[1 / 7, 0.0]]))


def test_perfect_detections_score_one_and_missing_ones_zero() -> None:
    perfect = detection([_scored(TARGET, [0.9, 0.8])], [TARGET], 3)
    assert perfect == {"box_map": 1.0, "box_ap50": 1.0, "box_ap75": 1.0, "box_ar100": 1.0}
    empty = detection([np.zeros((0, 6))], [TARGET], 3)
    assert empty["box_map"] == 0.0 and empty["box_ar100"] == 0.0
    assert np.isnan(detection([np.zeros((0, 6))], [np.zeros((0, 5))], 3)["box_map"])


def test_wrong_class_and_duplicates_are_false_positives() -> None:
    wrong = TARGET.copy()
    wrong[:, 0] = [2, 1]
    assert detection([_scored(wrong, [0.9, 0.8])], [TARGET], 3)["box_ap50"] == 0.0
    doubled = np.vstack([_scored(TARGET[:1], [0.9]), _scored(TARGET[:1], [0.8])])
    matches = match(doubled, TARGET, 3)
    assert matches.hits[:, 0].tolist() == [True, False]


def test_scaling_an_axis_leaves_scores_unchanged() -> None:
    rng = np.random.default_rng(0)
    noisy = _scored(TARGET, [0.9, 0.7])
    noisy[:, 2:] += rng.normal(0, 0.03, (2, 4))
    scale = np.array([1, 640, 480, 640, 480])
    normalized = detection([noisy], [TARGET], 3)
    pixels = detection([noisy * np.r_[1, scale]], [TARGET * scale], 3)
    assert normalized == pytest.approx(pixels)
