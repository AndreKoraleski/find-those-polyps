import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from conftest import make_spec
from scipy.stats import permutation_test
from sklearn.metrics import matthews_corrcoef

from find_those_polyps.evaluation.ratings import (
    FLIPS,
    Measured,
    Setting,
    agrees,
    flags,
    holm,
    jobs,
    main,
    measure_all,
    metric_range,
    rate_tasks,
    sign_flip,
)
from find_those_polyps.evaluation.records import (
    classification_records,
    detection_records,
    matches_from_records,
)
from find_those_polyps.metrics.bootstrap import (
    detection_statistic,
    draws,
    index_rows,
    interval,
    matched_statistic,
    mean_statistic,
)
from find_those_polyps.metrics.detection import IOU_THRESHOLDS, match
from find_those_polyps.tasks.schema import TaskKind, TaskSpec

RNG = np.random.default_rng(0)


def _boxes() -> tuple[list[np.ndarray], list[np.ndarray]]:
    predictions, targets = [], []
    for image in range(12):
        shift = image % 4
        targets.append(np.array([[1 + image % 2, 10, 10, 30 + shift, 30]], dtype=np.float64))
        predictions.append(
            np.array(
                [
                    [1 + image % 2, 0.9 - image / 100, 11, 10, 31 + shift, 30],
                    [1, 0.3 + image / 100, 50, 50, 60, 60],
                ]
            )
            if image % 3
            else np.zeros((0, 6))
        )
    return predictions, targets


def test_detection_records_rebuild_the_matches(tmp_path: Path) -> None:
    predictions, targets = _boxes()
    matched = [match(p, t, 3) for p, t in zip(predictions, targets, strict=True)]
    path = tmp_path / "records.csv"
    detection_records(matched).to_csv(path, index=False)
    rebuilt = matches_from_records(pd.read_csv(path), len(IOU_THRESHOLDS))
    for original, copy in zip(matched, rebuilt, strict=True):
        assert np.array_equal(original.classes, copy.classes)
        assert np.array_equal(original.scores, copy.scores)
        assert np.array_equal(original.hits, copy.hits)
        assert np.array_equal(original.targets, copy.targets)
    for metric in ("box_ap50", "box_map"):
        direct = detection_statistic(metric, predictions, targets, 3)
        stored = matched_statistic(metric, rebuilt, 3)
        assert interval(direct, 12) == interval(stored, 12)


def test_old_detection_records_are_refused() -> None:
    old = pd.DataFrame({"sample_id": ["a"], "targets": [1]})
    with pytest.raises(ValueError, match="det_hits"):
        matches_from_records(old, len(IOU_THRESHOLDS))
    spec = make_spec(
        id="det/toy",
        dataset="toy",
        kind=TaskKind.DETECTION,
        classes=("background", "polyp"),
        primary_metric="map50",
    )
    assert not agrees(spec, "linear_det", "map50", 0.5, {0: old})


def test_draws_share_their_first_resamples() -> None:
    values = RNG.random(50)
    statistic = mean_statistic(values)
    assert np.array_equal(
        draws(statistic, 50, resamples=40), draws(statistic, 50, resamples=80)[:40]
    )
    rows = np.concatenate(list(index_rows(50, 30)))
    assert np.allclose(draws(statistic, 50, resamples=30), values[rows].mean(axis=1))


def test_holm_steps_down_until_a_hypothesis_is_kept() -> None:
    assert holm([0.01, 0.04, 0.03, 0.2]).tolist() == [True, False, False, False]
    assert holm([0.001, 0.012, 0.02]).tolist() == [True, True, True]
    assert holm([float("nan"), 0.001]).tolist() == [False, True]


def test_sign_flip_test_is_exact_on_few_images_and_sampled_on_many() -> None:
    wins = np.array([0.1, 0.2, 0.05, 0.3, 0.15, 0.1, 0.2])
    assert sign_flip(wins) == pytest.approx(2 / 2**7)
    assert sign_flip(wins, one_sided=True) == pytest.approx(1 / 2**7)
    assert sign_flip(-wins, one_sided=True) == 1
    assert sign_flip(np.array([0.1, -0.1, 0.2, -0.2])) == 1
    assert np.isnan(sign_flip(np.array([np.nan])))
    differences = RNG.normal(0, 1, 12)
    exhaustive = permutation_test(
        (differences,),
        lambda x: float(np.mean(x)),
        permutation_type="samples",
        vectorized=False,
        n_resamples=2**12,
        alternative="two-sided",
    )
    assert sign_flip(differences) == pytest.approx(exhaustive.pvalue)
    assert sign_flip(RNG.normal(1, 1, 200)) == pytest.approx(1 / (FLIPS + 1))
    assert 0.2 < sign_flip(RNG.normal(0, 1, 200)) <= 1


def test_flags_use_sign_flips_over_images_when_given() -> None:
    images = np.array([[0.9, 0.8, 0.85, 0.95, 0.9, 0.7, 0.8], [0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.5]])
    points = images.mean(axis=1)
    sampled = points[:, None] + RNG.normal(0, 0.001, (2, 500))
    both = np.array([True, True])
    assert flags(points, sampled, both, 0.01, 1.0, images)["status"] == "resolved"
    noisy = images[[0, 0]] + np.array([[0.0], [-0.01]])
    noisy[1, ::2] += 0.03
    spread = noisy.mean(axis=1)[:, None] + RNG.normal(0, 0.0001, (2, 500))
    assert flags(noisy.mean(axis=1), spread, both, 0.001, 1.0)["status"] == "resolved"
    assert flags(noisy.mean(axis=1), spread, both, 0.001, 1.0, noisy)["status"] != "resolved"
    reference = np.array([True, False])
    assert flags(points, sampled, reference, 0.01, 1.0, images)["floor_level"] is False
    close = images[[0, 0]] - np.array([[0.0], [0.001]])
    close[1, 0] += 0.01
    assert flags(close.mean(axis=1), sampled, reference, 0.01, 1.0, close)["floor_level"] is True


def _flags(
    means: list[float], noise: float, pretrained: list[bool], rope: float = 0.01
) -> dict[str, object]:
    shared = RNG.normal(0, noise, 1000)
    own = RNG.normal(0, noise / 4, (len(means), 1000))
    sampled = np.array(means)[:, None] + shared + own
    return flags(np.array(means), sampled, np.array(pretrained), rope, 1.0)


def test_flags_separate_resolved_equivalent_and_underpowered_settings() -> None:
    resolved = _flags([0.80, 0.70, 0.75, 0.20], 0.02, [True, True, True, False])
    assert resolved["status"] == "resolved"
    assert resolved["resolved_pairs"] == 3
    assert resolved["best_index"] == 0 and resolved["floor_index"] == 3
    assert resolved["floor_level"] is False and resolved["above_floor"] == 3
    assert float(str(resolved["saturation"])) < 0.01
    assert resolved["ceiling_bound"] is False
    assert _flags([0.700, 0.701], 0.004, [True, True], rope=0.05)["status"] == "equivalent"
    weak = _flags([0.70, 0.69], 0.4, [True, True])
    assert weak["status"] == "underpowered"
    assert float(str(weak["mde"])) > 0.01
    shortcut = _flags([0.60, 0.59, 0.62], 0.02, [True, True, False])
    assert shortcut["floor_level"] is True
    assert shortcut["above_floor"] == 0


def test_flags_mark_a_best_encoder_at_the_ideal_value() -> None:
    sampled = np.vstack([np.ones(200), np.full(200, 0.5)])
    values = flags(np.array([1.0, 0.5]), sampled, np.array([True, True]), 0.01, 1.0)
    assert values["ceiling_bound"] is True


def test_metric_ranges_give_every_metric_the_same_relative_rope() -> None:
    assert metric_range("dice", TaskKind.SEGMENTATION, 2) == 1
    assert metric_range("mcc", TaskKind.CLASSIFICATION, 4) == 2
    assert metric_range("mae", TaskKind.ORDINAL, 4) == 3
    assert metric_range("mae", TaskKind.SEGMENTATION, 2) == 1
    assert metric_range("log_loss", TaskKind.CLASSIFICATION, 4) == pytest.approx(np.log(4))


def _measured(points: list[float], noise: float) -> Measured:
    shared = RNG.normal(0, noise, 400)
    sampled = np.array(points)[:, None] + shared + RNG.normal(0, noise / 4, (len(points), 400))
    seeds = np.array(points)[:, None] + RNG.normal(0, noise / 10, (len(points), 3))
    return Measured(np.array(points), sampled, seeds)


def _setting(task: str, points: list[float], twin: list[float]) -> Setting:
    return Setting(
        task=task,
        protocol="linear",
        kind=TaskKind.CLASSIFICATION,
        n_classes=3,
        primary="mcc",
        n=400,
        models=["a", "b", "c", "colour-texture"],
        pretrained=np.array([True, True, True, False]),
        metrics={
            "mcc": _measured(points, 0.01),
            "accuracy": _measured([p / 2 + 0.4 for p in points], 0.01),
            "macro_f1": _measured(twin, 0.2),
        },
    )


def test_tasks_are_rated_from_settings() -> None:
    settings = [
        _setting("cls/one", [0.8, 0.6, 0.7, 0.2], [0.1, 0.3, 0.2, 0.0]),
        _setting("cls/two", [0.5, 0.7, 0.6, 0.1], [0.3, 0.1, 0.2, 0.0]),
    ]
    tasks = rate_tasks(settings).set_index("task")
    assert tasks.loc["cls/one", "status"] == "resolved"
    assert tasks.loc["cls/one", "best_model"] == "a"
    assert tasks.loc["cls/two", "best_model"] == "b"
    assert bool(tasks.loc["cls/one", "core"])
    assert tasks.loc["cls/one", "pool"] == "a b c"
    assert tasks.loc["cls/one", "rope"] == pytest.approx(0.02)
    assert not tasks.loc["cls/one", "small"]
    tiny = rate_tasks([replace(settings[0], n=12)]).iloc[0]
    assert tiny["small"] and tiny["status"] == "resolved" and not tiny["core"]


def _spec() -> TaskSpec:
    return make_spec(
        id="cls/toy",
        dataset="toy",
        kind=TaskKind.CLASSIFICATION,
        classes=("x", "y", "z"),
        primary_metric="mcc",
    )


def _write(root: Path, spec: TaskSpec, model: str, accuracy: float, seeds: int) -> float:
    labels = np.arange(90) % 3
    frames = []
    scores = []
    for seed in range(seeds):
        rng = np.random.default_rng([zlib.crc32(model.encode()), seed])
        right = rng.random(90) < accuracy
        predicted = np.where(right, labels, (labels + 1) % 3)
        probabilities = np.full((90, 3), 0.1)
        probabilities[np.arange(90), predicted] = 0.8
        scores.append(matthews_corrcoef(labels, predicted))
        frame = classification_records(spec, probabilities, tuple(int(v) for v in labels))
        frame.insert(0, "sample_id", [f"s{i:03d}" for i in range(90)][::-1])
        frame.insert(1, "path", "")
        frame.insert(0, "protocol", "linear")
        frame.insert(1, "budget", "full")
        frame.insert(2, "seed", seed)
        frames.append(frame)
    folder = root / "cls__toy"
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(folder / f"{model}.csv", index=False)
    return float(np.mean(scores))


def _board(scores: dict[str, float]) -> pd.DataFrame:
    rows = [
        ("good", "imagenet-1k", 224),
        ("fair", "lvd-142m", 224),
        ("fair-448", "lvd-142m", 448),
        ("random", "none", 224),
        ("missing", "imagenet-21k", 224),
    ]
    return pd.DataFrame(
        [
            {
                "task": "cls/toy",
                "protocol": "linear",
                "model": model,
                "pretraining": pretraining,
                "input_size": size,
                "budget": "full",
                "primary_metric": "mcc",
                "score": scores.get(model, 0.3),
                "current": True,
                "registry_version": "1",
            }
            for model, pretraining, size in rows
        ]
    )


def test_settings_are_built_from_the_records_of_the_leaderboard(tmp_path: Path) -> None:
    spec = _spec()
    scores = {
        model: _write(tmp_path / "current", spec, model, accuracy, seeds=2)
        for model, accuracy in (("good", 0.95), ("fair", 0.6), ("fair-448", 0.5), ("random", 0.34))
    }
    _write(tmp_path / "stale", spec, "good", 0.7, seeds=2)
    roots = [tmp_path / "stale", tmp_path / "current"]
    work = jobs(_board(scores), {spec.id: spec}, roots, resamples=100, metric_resamples=20)
    assert len(work) == 1
    assert work[0].models == ["good", "fair", "random"]
    assert work[0].pretrained.tolist() == [True, True, False]
    assert work[0].n == 90
    assert list(work[0].records[0][0]["sample_id"][:2]) == ["s000", "s001"]
    (setting,) = measure_all(work)
    assert next(iter(setting.metrics)) == "mcc"
    assert setting.metrics["mcc"].draws.shape == (3, 100)
    assert setting.metrics["auroc"].draws.shape == (3, 20)
    assert setting.metrics["log_loss"].draws.shape == (3, 20)
    assert setting.metrics["mcc"].seeds.shape == (3, 2)
    assert setting.metrics["mcc"].points[0] == pytest.approx(scores["good"])
    assert 0 <= setting.hard_share < 0.5
    row = rate_tasks([setting]).iloc[0]
    assert row["status"] == "resolved"
    assert row["best_model"] == "good"
    assert row["floor_model"] == "random"
    assert not row["floor_level"]


def test_command_writes_the_ratings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec()
    scores = {
        model: _write(tmp_path / "records", spec, model, accuracy, seeds=1)
        for model, accuracy in (("good", 0.85), ("fair", 0.6), ("random", 0.34))
    }
    board = tmp_path / "leaderboard.csv"
    _board(scores).to_csv(board, index=False)

    registry = SimpleNamespace(tasks=[spec])
    monkeypatch.setattr("find_those_polyps.registry.store.load_registry", lambda check: registry)
    status = main(
        [
            "--leaderboard",
            str(board),
            "--per-image-dir",
            str(tmp_path / "records"),
            "--output",
            str(tmp_path / "ratings"),
            "--resamples",
            "50",
            "--metric-resamples",
            "10",
        ]
    )
    assert status == 0
    assert len(pd.read_csv(tmp_path / "ratings" / "tasks.csv")) == 1
    assert not (tmp_path / "ratings" / "metrics.csv").exists()
    summary = (tmp_path / "ratings" / "ratings.md").read_text(encoding="utf-8")
    assert "Core subset, 1 of 1 tasks: `cls/toy`." in summary
