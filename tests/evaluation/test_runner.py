from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import MeanColour, Noise, SquareLoader, make_spec, make_task
from PIL import Image

from find_those_polyps.data import LocalFiles
from find_those_polyps.evaluation import (
    EvaluationConfig,
    Leaderboard,
    LeaderboardRow,
    LinearSegmentation,
    evaluate,
    evaluate_task,
    runner,
)
from find_those_polyps.evaluation.__main__ import summary
from find_those_polyps.evaluation.features import extract
from find_those_polyps.evaluation.protocols import LinearDetection
from find_those_polyps.evaluation.runner import budget_validation, resolve_protocols, shrink
from find_those_polyps.evaluation.store import FeatureStore
from find_those_polyps.models import Encoder
from find_those_polyps.registry import Registry
from find_those_polyps.tasks import FULL, Budget, Split, Task, TaskKind, TaskSpec, make_split
from find_those_polyps.tasks.loaders.boxes import mask_boxes
from find_those_polyps.tasks.loaders.folders import FolderClassification
from find_those_polyps.tasks.schema import Sample


def _config(tmp_path: Path, **fields: object) -> EvaluationConfig:
    base: dict[str, object] = {
        "seeds": 2,
        "device": "cpu",
        "bootstrap": 50,
        "feature_cache": False,
        "leaderboard": tmp_path / "leaderboard.csv",
        "per_image_dir": tmp_path / "per_image",
    }
    return EvaluationConfig.model_validate(base | fields)


def test_classification_leaderboard_ranks_models(colour_task: Task, tmp_path: Path) -> None:
    config = _config(tmp_path, protocols=("knn", "linear"))
    for encoder in (Noise(), MeanColour()):
        evaluate(encoder, config, tasks=[colour_task], registry=Registry(version=3, sha256=""))
    rows = Leaderboard(config.leaderboard).read()
    assert {(r.protocol, r.budget) for r in rows} == {
        ("knn", "full"),
        ("knn", "shots:2"),
        ("linear", "full"),
        ("linear", "shots:2"),
    }
    full = [r for r in rows if r.budget == "full"]
    assert all(r.rank == 1 and r.score > 0.9 for r in full if r.model == "mean-colour")
    assert all(r.runs == 2 for r in rows if r.budget == "shots:2")
    assert all(r.runs == 1 for r in full)
    assert {r.registry_version for r in rows} == {3}


def test_per_image_records_trace_test_images(colour_task: Task, tmp_path: Path) -> None:
    config = _config(tmp_path, protocols=("knn",))
    evaluate(MeanColour(), config, tasks=[colour_task], registry=Registry(version=0, sha256=""))
    records = pd.read_csv(tmp_path / "per_image" / "cls__colour" / "mean-colour.csv")
    test_ids = set(colour_task.split.test)
    assert set(records["sample_id"]) == test_ids
    assert records["path"].str.startswith("colour/").all()
    assert {"p_red", "p_green", "p_blue", "correct", "confidence"} <= set(records.columns)
    assert len(records) == len(test_ids) * (1 + 2)


def test_rerun_replaces_rows(colour_task: Task, tmp_path: Path) -> None:
    config = _config(tmp_path, protocols=("knn",), seeds=1)
    for _ in range(2):
        evaluate(MeanColour(), config, tasks=[colour_task], registry=Registry(version=0, sha256=""))
    frame = Leaderboard(config.leaderboard).frame()
    assert not frame.duplicated(["model", "task", "protocol", "budget"]).any()
    records = pd.read_csv(tmp_path / "per_image" / "cls__colour" / "mean-colour.csv")
    assert not records.duplicated(["protocol", "budget", "seed", "sample_id"]).any()


def test_overrides_and_unknown_protocols(colour_task: Task, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown protocols"):
        resolve_protocols(["svm"])
    rows = evaluate(
        MeanColour(),
        _config(tmp_path),
        tasks=[colour_task],
        registry=Registry(version=0, sha256=""),
        protocols=("knn",),
        seeds=1,
    )
    assert {row.protocol for row in rows} == {"knn"}


@pytest.mark.parametrize("fixture", ["ordinal_task", "tag_task"])
def test_other_task_kinds(fixture: str, request: pytest.FixtureRequest, tmp_path: Path) -> None:
    task: Task = request.getfixturevalue(fixture)
    rows = evaluate_task(
        MeanColour(), task, resolve_protocols(["knn", "linear"]), _config(tmp_path), 0
    )
    full = [row for row in rows if row.budget == "full"]
    assert len(full) == 2
    assert all(row.score > 0.8 for row in full)


def test_linear_segmentation(square_task: Task, tmp_path: Path) -> None:
    protocol = LinearSegmentation(steps=300, eval_every=100, label_size=32, lr=0.05)
    rows = evaluate_task(MeanColour(), square_task, [protocol], _config(tmp_path, seeds=1), 0)
    (row,) = rows
    assert row.means["dice"] > 0.7
    assert row.means["s_measure"] > 0.7
    records = pd.read_csv(tmp_path / "per_image" / "seg__square" / "mean-colour.csv")
    assert {"dice", "hd95", "path"} <= set(records.columns)


def test_dense_protocol_skipped_without_dense_features(square_task: Task, tmp_path: Path) -> None:
    rows = evaluate_task(Noise(), square_task, [LinearSegmentation()], _config(tmp_path), 0)
    assert rows == []


def test_missing_datasets_are_skipped(tmp_path: Path) -> None:
    config = _config(tmp_path, data_root=tmp_path / "empty", seeds=1, stream=False)
    rows = evaluate(MeanColour(), config)
    assert rows == []


def test_zero_shot_protocols(colour_task: Task, tmp_path: Path) -> None:
    protocols = resolve_protocols(["retrieval", "kmeans"])
    rows = evaluate_task(MeanColour(), colour_task, protocols, _config(tmp_path, seeds=2), 0)
    kmeans = [row for row in rows if row.protocol == "kmeans"]
    assert [row.budget for row in kmeans] == ["full"]
    assert kmeans[0].primary_metric == "ami"
    assert kmeans[0].score > 0.8
    assert kmeans[0].runs == 2
    low, high = kmeans[0].score_ci_low, kmeans[0].score_ci_high
    assert low is not None and high is not None
    assert low <= kmeans[0].score <= high
    retrieval = {row.budget: row for row in rows if row.protocol == "retrieval"}
    assert retrieval["full"].means["map_at_r"] > 0.8
    assert retrieval["shots:2"].runs == 2
    records = pd.read_csv(tmp_path / "per_image" / "cls__colour" / "mean-colour.csv")
    assert {"nearest_path", "cluster", "map_at_r"} <= set(records.columns)


def test_failing_tasks_do_not_stop_the_others(
    colour_task: Task, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = colour_task.spec.model_copy(update={"id": "cls/broken"})
    broken = colour_task.model_copy(update={"spec": spec})
    original = runner.evaluate_task

    def flaky(encoder: Encoder, task: Task, *args: Any, **kwargs: Any) -> list[LeaderboardRow]:
        if task.spec.id == "cls/broken":
            raise RuntimeError("boom")
        return original(encoder, task, *args, **kwargs)

    monkeypatch.setattr(runner, "evaluate_task", flaky)
    config = _config(tmp_path, protocols=("knn",), seeds=1)
    registry = Registry(version=0, sha256="")
    with pytest.raises(runner.EvaluationError, match="cls/broken"):
        evaluate(MeanColour(), config, tasks=[broken, colour_task], registry=registry)
    assert len(Leaderboard(config.leaderboard).read()) == 2


def test_budgets_subsample_validation_and_report_intervals(
    colour_task: Task, tmp_path: Path
) -> None:
    rows = evaluate_task(
        MeanColour(), colour_task, resolve_protocols(["linear"]), _config(tmp_path), 0
    )
    full = next(row for row in rows if row.budget == "full")
    assert full.runs == 1
    assert full.stds["balanced_accuracy"] != full.stds["balanced_accuracy"]
    assert full.score_ci_low is not None and full.score_ci_high is not None
    assert full.score_ci_low <= full.score <= full.score_ci_high
    val = colour_task.samples(Split.VAL)
    features = extract(MeanColour(), val, dense=False, batch_size=8, device=torch.device("cpu"))
    budget = Budget.model_validate("shots:2")
    subset = budget_validation(features, budget, colour_task.spec, 0)
    assert np.bincount(subset.labels()).tolist() == [2, 2, 2]
    assert budget_validation(features, FULL, colour_task.spec, 0) is features


class CountingColour(MeanColour):
    """Mean colour encoder counting the images it preprocesses."""

    def __init__(self) -> None:
        super().__init__()
        self.seen = 0

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        self.seen += 1
        return super().preprocess(image)


def _folder_tasks(root: Path) -> list[Task]:
    rng = np.random.default_rng(0)
    for index in range(36):
        folder = root / "shared" / "set" / "abc"[index % 3]
        folder.mkdir(parents=True, exist_ok=True)
        pixels = rng.integers(0, 80, size=(16, 16, 3))
        pixels[..., index % 3] += 150
        Image.fromarray(pixels.astype(np.uint8)).save(folder / f"i{index}.png")
    tasks = []
    for name, classes in (("cls/three", [["a"], ["b"], ["c"]]), ("cls/two", [["a"], ["b", "c"]])):
        loader = FolderClassification(
            LocalFiles("shared", root / "shared"), root="set", classes=classes
        )
        ids = loader.ids()
        labels = loader.labels()
        split = make_split(ids, ids, [labels[i] for i in ids], test=0.3, val=0.2)
        spec = TaskSpec(
            id=name,
            description="Shared images.",
            dataset="shared",
            kind=TaskKind.CLASSIFICATION,
            classes=tuple(f"c{i}" for i in range(len(classes))),
            primary_metric="balanced_accuracy",
            loader="folders",
            splits=f"{name[4:]}.json",
        )
        tasks.append(Task(spec=spec, loader=loader, split=split))
    return tasks


def test_tasks_sharing_images_encode_them_once(tmp_path: Path) -> None:
    encoder = CountingColour()
    config = _config(tmp_path, protocols=("knn",), seeds=1)
    rows = evaluate(
        encoder, config, tasks=_folder_tasks(tmp_path), registry=Registry(version=0, sha256="")
    )
    assert encoder.seen == 36
    assert {row.task for row in rows} == {"cls/three", "cls/two"}
    assert all(row.score > 0.8 for row in rows if row.budget == "full")


def test_resume_skips_settings_already_recorded(colour_task: Task, tmp_path: Path) -> None:
    config = _config(tmp_path, protocols=("knn",), seeds=1)
    registry = Registry(version=0, sha256="")
    first = evaluate(MeanColour(), config, tasks=[colour_task], registry=registry)
    again = evaluate(MeanColour(), config, tasks=[colour_task], registry=registry, resume=True)
    assert first and again == []
    other = evaluate(Noise(), config, tasks=[colour_task], registry=registry, resume=True)
    assert len(other) == len(first)
    assert "cls/colour" in summary(first)
    assert "knn" in summary(first).splitlines()[1]


def test_smoke_runs_use_few_samples_and_separate_outputs(colour_task: Task, tmp_path: Path) -> None:
    config = _config(tmp_path, protocols=("knn",), seeds=1, limit=6)
    encoder = CountingColour()
    rows = evaluate(encoder, config, tasks=[colour_task], registry=Registry(version=0, sha256=""))
    assert rows
    assert encoder.seen <= 18
    assert (tmp_path / "smoke" / "leaderboard.csv").exists()
    assert not (tmp_path / "leaderboard.csv").exists()
    shrunk = shrink(colour_task, 4)
    assert [len(shrunk.split.ids(s)) for s in Split] == [4, 4, 4]


def test_feature_cache_skips_encoding_in_later_runs(square_task: Task, tmp_path: Path) -> None:
    protocols = [LinearSegmentation(steps=20, eval_every=10, label_size=32)]
    config = _config(tmp_path, seeds=1, feature_cache=True, data_root=tmp_path / "data")
    first = CountingColour()
    store = FeatureStore(runner.feature_directory(first, config))
    evaluate_task(first, square_task, protocols, config, 0, store=store)
    store.flush()
    second = CountingColour()
    store = FeatureStore(runner.feature_directory(second, config))
    rows = evaluate_task(second, square_task, protocols, config, 0, store=store)
    assert first.seen > 0 and second.seen == 0 and rows
    (directory,) = (tmp_path / "data" / ".features").iterdir()
    assert directory.name.startswith("mean-colour-")


def test_cached_global_features_reach_every_task(tmp_path: Path) -> None:
    config = _config(tmp_path, protocols=("knn",), seeds=1, feature_cache=True)
    config = config.model_copy(update={"data_root": tmp_path / "data"})
    tasks = _folder_tasks(tmp_path)
    for encoder in (CountingColour(), CountingColour()):
        evaluate(encoder, config, tasks=tasks, registry=Registry(version=0, sha256=""))
    assert encoder.seen == 0


def test_detection_tasks_report_box_metrics_and_records(tmp_path: Path) -> None:
    loader = SquareLoader(LocalFiles("synthetic", tmp_path))
    loader.samples = {
        key: sample.model_copy(update={"target": mask_boxes(np.asarray(sample.target))})
        for key, sample in loader.samples.items()
    }
    spec = make_spec(
        id="det/square",
        dataset="square",
        kind=TaskKind.DETECTION,
        classes=("background", "square"),
        primary_metric="box_map",
    )
    task = make_task(loader, spec)
    protocol = LinearDetection(steps=60, eval_every=30, lr=0.05)
    (row,) = evaluate_task(MeanColour(), task, [protocol], _config(tmp_path, seeds=1), 0)
    assert set(row.means) == {"box_map", "box_ap50", "box_ap75", "box_ar100"}
    assert row.primary_metric == "box_map" and 0 <= row.score <= 1
    assert row.score_ci_low is not None
    records = pd.read_csv(tmp_path / "per_image" / "det__square" / "mean-colour.csv")
    assert {"targets", "detections", "matched_50", "path"} <= set(records.columns)
    assert (records["targets"] == 1).all()


def test_linear_segmentation_of_overlapping_masks(tmp_path: Path) -> None:
    loader = SquareLoader(LocalFiles("synthetic", tmp_path))
    loader.samples = {
        key: sample.model_copy(update={"target": np.stack([sample.target, 1 - sample.target])})
        for key, sample in loader.samples.items()
    }
    spec = make_spec(
        id="seg/layers",
        dataset="square",
        kind=TaskKind.MULTILABEL_SEGMENTATION,
        classes=("square", "surround"),
        primary_metric="dice",
    )
    protocol = LinearSegmentation(steps=300, eval_every=100, label_size=32, lr=0.05)
    config = _config(tmp_path, seeds=1)
    (row,) = evaluate_task(MeanColour(), make_task(loader, spec), [protocol], config, 0)
    assert row.means["dice"] > 0.7
    assert "s_measure" not in row.means


class DeclaredColour(MeanColour):
    """Mean colour encoder declaring a pretraining corpus."""

    name = "declared-colour"
    pretrained_on = ("pool",)


def test_pretraining_declarations_are_recorded(colour_task: Task, tmp_path: Path) -> None:
    config = _config(tmp_path, protocols=("knn",), seeds=1)
    registry = Registry(version=0, sha256="")
    for encoder in (MeanColour(), DeclaredColour()):
        evaluate(encoder, config, tasks=[colour_task], registry=registry)
    rows = Leaderboard(config.leaderboard).read()
    assert {row.model: row.pretraining for row in rows} == {
        "mean-colour": "undeclared",
        "declared-colour": "pool",
    }
    assert runner.describe(runner.declaration(DeclaredColour(), config)) == "pool"
    extended = config.model_copy(update={"pretrained_on": ("extra",)})
    assert runner.declaration(DeclaredColour(), extended) == ("extra", "pool")
    assert runner.declaration(MeanColour(), extended) == ("extra",)
    assert runner.describe(()) == "none"
    report = config.leaderboard.with_suffix(".md").read_text(encoding="utf-8")
    assert "mean-colour (undeclared)" in report and "| declared-colour | pool | 32 |" in report


def test_rows_record_the_input_size_and_dense_feature_grid(
    colour_task: Task, square_task: Task, tmp_path: Path
) -> None:
    config = _config(tmp_path, seeds=1)
    protocol = LinearSegmentation(steps=10, eval_every=10, label_size=32, lr=0.05)
    (dense,) = evaluate_task(MeanColour(), square_task, [protocol], config, 0)
    assert (dense.input_size, dense.feature_grid) == (32, "8x8")
    knn = resolve_protocols(["knn"])
    (flat, *_) = evaluate_task(Noise(), colour_task, knn, config, 0)
    assert (flat.input_size, flat.feature_grid) == (8, "")


class CountingLoader(SquareLoader):
    """Square loader counting the samples it loads."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loads = 0

    def load(self, sample_id: str) -> Sample:
        self.loads += 1
        return super().load(sample_id)


class OtherColour(CountingColour):
    """Counting mean colour encoder under another name."""

    name = "other-colour"
    image_size = 16


def test_encoders_evaluated_together_read_each_image_once(tmp_path: Path) -> None:
    loader = CountingLoader(LocalFiles("synthetic", tmp_path))
    spec = make_spec(
        id="seg/square",
        dataset="square",
        kind=TaskKind.SEGMENTATION,
        classes=("background", "square"),
        primary_metric="dice",
    )
    task = make_task(loader, spec)
    protocols = ("knn", "linear_seg")
    config = _config(tmp_path, protocols=protocols, seeds=1, bootstrap=0)
    first, second = CountingColour(), OtherColour()
    rows = evaluate([first, second], config, tasks=[task], registry=Registry(version=0, sha256=""))
    assert {row.model for row in rows} == {"mean-colour", "other-colour"}
    assert {(row.model, row.protocol) for row in rows} >= {
        ("mean-colour", "linear_seg"),
        ("other-colour", "linear_seg"),
    }
    assert loader.loads == len(loader.ids())
    assert first.seen == second.seen == len(loader.ids())
    assert "model" in summary(rows).splitlines()[0]


def test_encoders_keep_their_own_task_selection(colour_task: Task, tmp_path: Path) -> None:
    config = _config(tmp_path, protocols=("knn",), seeds=1)
    first, second = CountingColour(), OtherColour()
    registry = Registry(version=0, sha256="")
    evaluate([first, second], config, tasks=[colour_task], registry=registry)
    again = evaluate([first, second], config, tasks=[colour_task], registry=registry, resume=True)
    assert again == []
    with pytest.raises(ValueError, match="distinct names"):
        evaluate([first, CountingColour()], config, tasks=[colour_task], registry=registry)
    with pytest.raises(ValueError, match="no encoder"):
        evaluate([], config, tasks=[colour_task], registry=registry)


def test_task_order_groups_tasks_by_dataset() -> None:
    specs = [
        make_spec(
            id=f"cls/{name}",
            dataset=dataset,
            kind=TaskKind.CLASSIFICATION,
            classes=("a", "b"),
            primary_metric="mcc",
        )
        for name, dataset in (("a1", "a"), ("b1", "b"), ("a2", "a"), ("c1", "c"), ("b2", "b"))
    ]
    order = [spec.id for spec in runner.task_order(specs)]
    assert order == ["cls/a1", "cls/a2", "cls/b1", "cls/b2", "cls/c1"]


def test_datasets_that_fail_to_stream_count_as_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = LocalFiles("shared", tmp_path / "missing")
    files.stream_error = "certificate verify failed"
    loader = FolderClassification(files, root="set", classes=[["a"], ["b"]])
    task = _folder_tasks(tmp_path)[0]
    monkeypatch.setattr(
        runner, "build_task", lambda *args, **kwargs: task.model_copy(update={"loader": loader})
    )
    spec = task.spec.model_copy(update={"splits_sha256": "x"})
    with pytest.raises(runner.EvaluationError, match="could not be streamed"):
        runner._prepare(spec, Registry(version=0, sha256=""), _config(tmp_path))
