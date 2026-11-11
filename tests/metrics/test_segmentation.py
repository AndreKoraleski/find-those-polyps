import numpy as np
import pytest

from find_those_polyps.metrics.segmentation import (
    _surface_distances,
    boundary,
    e_measure,
    hd95,
    overlap,
    per_image,
    s_measure,
    saliency,
    segmentation,
    weighted_f_measure,
)


@pytest.mark.parametrize(
    ("n_classes", "multilabel", "names"),
    [
        (2, False, None),
        (3, False, None),
        (2, True, None),
        (2, False, ["dice", "mae"]),
        (2, False, ["dice", "hd95", "unknown"]),
        (2, False, ["hd95"]),
    ],
)
def test_no_images_give_nan_for_every_metric(
    n_classes: int, multilabel: bool, names: list[str] | None
) -> None:
    rng = np.random.default_rng(0)
    maps = rng.dirichlet(np.ones(n_classes), (6, 7)).transpose(2, 0, 1)
    probability = maps.astype(np.float32)
    target = rng.integers(0, 2, (n_classes, 6, 7)) if multilabel else rng.integers(0, 2, (6, 7))
    one = segmentation([probability], [target], n_classes, names, multilabel=multilabel)
    empty = segmentation([], [], n_classes, names, multilabel=multilabel)
    assert list(empty) == list(one)
    assert all(np.isnan(value) for value in empty.values())


@pytest.fixture
def disc() -> np.ndarray:
    y, x = np.ogrid[:40, :50]
    return np.asarray((y - 20) ** 2 + (x - 25) ** 2 < 100, dtype=bool)


def test_perfect_map_scores_one(disc: np.ndarray) -> None:
    probability = disc.astype(np.float64)
    assert s_measure(probability, disc) == pytest.approx(1.0, abs=1e-6)
    assert e_measure(probability, disc).max() == pytest.approx(1.0, abs=1e-3)
    assert weighted_f_measure(probability, disc) == pytest.approx(1.0, abs=1e-6)
    metrics = saliency(probability, disc)
    assert metrics["mae"] == 0.0


def test_inverted_map_scores_low(disc: np.ndarray) -> None:
    probability = 1.0 - disc
    assert s_measure(probability, disc) < 0.2
    assert weighted_f_measure(probability, disc) < 0.2
    assert saliency(probability, disc)["mae"] == 1.0


def test_empty_target_conventions() -> None:
    empty = np.zeros((10, 10), dtype=bool)
    assert s_measure(np.zeros((10, 10)), empty) == 1.0
    assert weighted_f_measure(np.zeros((10, 10)), empty) == 0.0
    assert overlap(empty, empty)["dice"] == 1.0
    assert hd95(empty, empty) == 0.0
    full = np.ones((10, 10), dtype=bool)
    assert s_measure(np.ones((10, 10)), full) == 1.0
    assert hd95(full, empty) == pytest.approx(np.hypot(10, 10))


def test_overlap_counts() -> None:
    target = np.zeros((4, 4), dtype=bool)
    target[:2] = True
    prediction = np.zeros((4, 4), dtype=bool)
    prediction[:1] = True
    metrics = overlap(prediction, target)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["dice"] == pytest.approx(2 / 3)
    assert metrics["f2"] == pytest.approx(5 * 0.5 / (4 + 0.5))
    assert metrics["specificity"] == 1.0


def test_hd95_of_shifted_square() -> None:
    a = np.zeros((30, 30), dtype=bool)
    a[5:15, 5:15] = True
    b = np.roll(a, 3, axis=1)
    assert hd95(a, b) == pytest.approx(3.0)


def test_multiclass_per_image_skips_saliency() -> None:
    target = np.array([[0, 1], [2, 2]])
    maps = np.eye(3)[target].transpose(2, 0, 1).astype(np.float32)
    records = per_image([maps], [target], 3)
    assert records[0]["dice"] == 1.0
    assert "s_measure" not in records[0]
    assert per_image([maps], [target], 3, ["iou"]) == [{"iou": 1.0}]


def test_boundary_tolerances_and_distances_scale_with_the_diagonal() -> None:
    a = np.zeros((100, 100), dtype=bool)
    a[20:60, 20:60] = True
    diagonal = np.hypot(100, 100)
    same = boundary(a, a)
    assert same["nsd"] == same["nsd_strict"] == 1.0
    assert same["nhd95"] == same["hd95"] == 0.0
    shifted = boundary(a, np.roll(a, 2, axis=1))
    assert shifted["nsd"] == 1.0
    assert 0.0 < shifted["nsd_strict"] < 1.0
    far = boundary(a, np.roll(a, 8, axis=1))
    assert 0.0 < far["nsd"] < shifted["nsd"]
    assert far["nhd95"] == pytest.approx(far["hd95"] / diagonal)
    assert far["nassd"] == pytest.approx(far["assd"] / diagonal)
    empty = np.zeros_like(a)
    assert boundary(a, empty) | {"hd95": 0, "assd": 0} == {
        "nsd": 0.0,
        "nsd_strict": 0.0,
        "nhd95": 1.0,
        "nassd": 1.0,
        "hd95": 0,
        "assd": 0,
    }
    assert boundary(empty, empty)["nsd"] == 1.0


def test_relative_distances_do_not_depend_on_resolution() -> None:
    small = np.zeros((50, 80), dtype=bool)
    small[10:30, 20:50] = True
    large = small.repeat(4, axis=0).repeat(4, axis=1)
    shift_small = boundary(small, np.roll(small, 3, axis=1))
    shift_large = boundary(large, np.roll(large, 12, axis=1))
    assert shift_large["nhd95"] == pytest.approx(shift_small["nhd95"], rel=0.1)
    assert shift_large["hd95"] == pytest.approx(4 * shift_small["hd95"], rel=0.1)


def test_multiclass_ignores_classes_absent_from_both_masks() -> None:
    target = np.zeros((6, 6), dtype=np.int64)
    target[:3] = 1
    maps = np.zeros((3, 6, 6), dtype=np.float32)
    maps[0, 3:] = 1
    maps[1, :3] = 1
    (record,) = per_image([maps], [target], 3)
    assert record["dice"] == 1.0
    maps[1, :3], maps[2, :3] = 0, 1
    (record,) = per_image([maps], [target], 3)
    assert record["dice"] == 0.0


def test_boundary_crop_matches_full_image_distances() -> None:
    rng = np.random.default_rng(0)
    prediction = np.zeros((60, 80), dtype=bool)
    target = np.zeros((60, 80), dtype=bool)
    prediction[0:12, 5:30] = rng.random((12, 25)) > 0.3
    target[8:25, 20:41] = True
    to_t, to_p = _surface_distances(prediction, target)
    expected = max(np.percentile(to_t, 95), np.percentile(to_p, 95))
    assert boundary(prediction, target)["hd95"] == pytest.approx(expected)


def test_boundaries_are_skipped_unless_requested(disc: np.ndarray) -> None:
    assert "hd95" not in overlap(disc, disc, boundaries=False)
    probabilities = np.stack([~disc, disc]).astype(np.float32)
    (record,) = per_image([probabilities], [disc.astype(np.int64)], 2, ["dice"])
    assert record == {"dice": 1.0}


def test_multilabel_masks_are_scored_per_overlapping_class(disc: np.ndarray) -> None:
    target = np.stack([disc, disc, np.zeros_like(disc)]).astype(np.int64)
    probability = np.stack([disc, ~disc, np.zeros_like(disc)]).astype(np.float32)
    (record,) = per_image([probability], [target], 3, ["dice"], multilabel=True)
    assert record["dice"] == pytest.approx(0.5)
