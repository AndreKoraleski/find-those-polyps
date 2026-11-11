import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score, matthews_corrcoef, roc_auc_score

from find_those_polyps.metrics.bootstrap import (
    averaged,
    detection_statistic,
    interval,
    mean_statistic,
    prediction_statistic,
)
from find_those_polyps.metrics.classification import multilabel, ranked_examples, varying_labels
from find_those_polyps.metrics.detection import match, summarize
from find_those_polyps.tasks import TaskKind


def test_interval_brackets_the_mean_and_shrinks_with_samples() -> None:
    rng = np.random.default_rng(0)
    small, large = rng.normal(size=50), rng.normal(size=5000)
    low_s, high_s = interval(mean_statistic(small), len(small))
    low_l, high_l = interval(mean_statistic(large), len(large))
    assert low_s < small.mean() < high_s
    assert high_l - low_l < high_s - low_s
    assert np.isnan(interval(mean_statistic(np.array([np.nan])), 1)[0])
    assert np.isnan(interval(mean_statistic(np.zeros(0)), 0)[0])


@pytest.mark.parametrize("metric", ["mcc", "auroc", "quadratic_kappa", "brier"])
def test_resampled_metrics_match_the_full_metrics(metric: str) -> None:
    rng = np.random.default_rng(1)
    targets = rng.integers(0, 3, 300)
    probabilities = rng.dirichlet(np.ones(3), 300)
    probabilities[np.arange(300), targets] += 0.5
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    statistic = prediction_statistic(
        TaskKind.ORDINAL, metric, probabilities, tuple(int(t) for t in targets), 3
    )
    value = statistic(np.arange(300))
    if metric == "mcc":
        assert value == pytest.approx(matthews_corrcoef(targets, probabilities.argmax(1)))
    if metric == "auroc":
        assert value == pytest.approx(roc_auc_score(targets, probabilities, multi_class="ovr"))
    assert np.isfinite(value)


def _label_sets() -> tuple[np.ndarray, np.ndarray]:
    """Label sets with a label that never occurs and examples without any positive label."""
    rng = np.random.default_rng(3)
    targets = (rng.random((120, 4)) < [0.5, 0.3, 0.2, 0.0]).astype(np.int64)
    targets[:10] = 0
    probabilities = np.clip(0.6 * targets + rng.random((120, 4)) * 0.5, 0, 1)
    return targets, probabilities


@pytest.mark.parametrize("metric", sorted(multilabel(*_label_sets())))
def test_resampled_multilabel_metrics_match_the_point_estimate(metric: str) -> None:
    targets, probabilities = _label_sets()
    assert not varying_labels(targets).all()
    assert not ranked_examples(targets).all()
    statistic = prediction_statistic(
        TaskKind.MULTILABEL, metric, probabilities, tuple(targets), targets.shape[1]
    )
    assert statistic(np.arange(len(targets))) == pytest.approx(
        multilabel(targets, probabilities)[metric]
    )


def test_resampled_multilabel_metrics_are_undefined_without_varying_labels() -> None:
    targets, probabilities = _label_sets()
    empty = np.arange(10)
    for metric in ("macro_ap", "macro_auroc", "lrap"):
        statistic = prediction_statistic(
            TaskKind.MULTILABEL, metric, probabilities, tuple(targets), 4
        )
        assert np.isnan(statistic(empty))


def test_centred_interval_contains_clustering_agreement() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 20, 600)
    clusters = np.where(rng.random(600) < 0.4, labels, rng.integers(0, 20, 600))

    def agreement(index: np.ndarray) -> float:
        return float(adjusted_mutual_info_score(labels[index], clusters[index]))

    estimate = agreement(np.arange(600))
    low, high = interval(agreement, 600, resamples=100)
    assert low > estimate
    agreement.centred = True  # type: ignore[attr-defined]
    low, high = interval(agreement, 600, resamples=100)
    assert low < estimate < high


def test_averaged_statistics_share_resamples_and_markers() -> None:
    first, second = mean_statistic(np.arange(4.0)), mean_statistic(np.arange(4.0) * 3)
    second.centred = True  # type: ignore[attr-defined]
    both = averaged([first, second])
    assert both(np.array([0, 3])) == pytest.approx((1.5 + 4.5) / 2)
    assert both.__dict__ == {"slow": False, "centred": True}


def _detections(seed: int, n: int, n_classes: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    predictions, targets = [], []
    for _ in range(n):
        boxes = []
        for _ in range(rng.integers(0, 4)):
            x, y = rng.uniform(0, 0.6, size=2)
            boxes.append([rng.integers(1, n_classes), x, y, x + 0.3, y + 0.3])
        target = np.array(boxes).reshape(-1, 5)
        rows = []
        for box in target:
            jitter = rng.normal(0, 0.05, size=4)
            rows.append([box[0], rng.uniform(), *(box[1:] + jitter)])
        for _ in range(rng.integers(0, 5)):
            x, y = rng.uniform(0, 0.7, size=2)
            rows.append([rng.integers(1, n_classes), rng.uniform(), x, y, x + 0.2, y + 0.2])
        predictions.append(np.array(rows).reshape(-1, 6))
        targets.append(target)
    return predictions, targets


@pytest.mark.parametrize("metric", ["box_ap50", "box_ap75", "box_map", "box_ar100"])
def test_weighted_detection_resamples_match_pooled_images(metric: str) -> None:
    predictions, targets = _detections(0, 60, 4)
    statistic = detection_statistic(metric, predictions, targets, 4)
    matched = [match(p, t, 4) for p, t in zip(predictions, targets, strict=True)]
    rng = np.random.default_rng(1)
    for _ in range(30):
        index = rng.integers(0, 60, size=60)
        expected = summarize([matched[i] for i in index], 4)[metric]
        assert statistic(index) == pytest.approx(expected, abs=1e-12, nan_ok=True)
    assert statistic(np.arange(60)) == pytest.approx(summarize(matched, 4)[metric])
    assert np.isnan(
        detection_statistic(metric, [np.zeros((0, 6))], [np.zeros((0, 5))], 4)(
            np.zeros(1, dtype=np.intp)
        )
    )
