import json
from pathlib import Path

import pytest

from find_those_polyps.evaluation import EvaluationConfig, select_tasks
from find_those_polyps.registry import (
    REGISTRY_PATH,
    StaleRegistryError,
    load_registry,
    sync_registry,
)

DATASET = {
    "name": "Synthetic",
    "labelled": True,
    "modality": "white light",
    "licence": "CC BY 4.0",
    "access": "open",
    "url": "https://example.org",
    "citation": "Doe 2026",
    "provenance": "Generated.",
    "family": "synthetic",
}


def _task(task_id: str, dataset: str) -> dict[str, object]:
    return {
        "id": task_id,
        "description": "Synthetic task.",
        "dataset": dataset,
        "kind": "classification",
        "classes": ["a", "b"],
        "primary_metric": "balanced_accuracy",
        "loader": "synthetic",
        "splits": f"{dataset}.json",
    }


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    splits = tmp_path / "splits"
    splits.mkdir()
    for name in ("pool", "a", "b"):
        (splits / f"{name}.json").write_text("{}", encoding="utf-8")
    datasets = [
        {**DATASET, "key": "pool", "labelled": False},
        {
            **DATASET,
            "key": "a",
            "overlaps": [{"dataset": "pool", "kind": "same_source", "evidence": "Doe 2026"}],
        },
        {**DATASET, "key": "b"},
    ]
    content = {
        "version": 0,
        "sha256": "",
        "datasets": datasets,
        "tasks": [_task("cls/a", "a"), _task("cls/b", "b")],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_packaged_registry_is_current() -> None:
    load_registry(REGISTRY_PATH)


def test_sync_bumps_version_only_on_change(registry_path: Path) -> None:
    assert sync_registry(registry_path)
    first = load_registry(registry_path)
    assert first.version == 1
    assert not sync_registry(registry_path)
    (registry_path.parent / "splits" / "a.json").write_text("[]", encoding="utf-8")
    with pytest.raises(StaleRegistryError):
        load_registry(registry_path)
    assert sync_registry(registry_path)
    assert load_registry(registry_path).version == 2


def test_pretraining_overlap_excludes_tasks(registry_path: Path) -> None:
    sync_registry(registry_path)
    registry = load_registry(registry_path)
    config = EvaluationConfig(pretrained_on=("pool",))
    assert [t.id for t in select_tasks(registry, config)] == ["cls/b"]
    config = EvaluationConfig(exclude_datasets=("b",), tasks=("cls/*",))
    assert [t.id for t in select_tasks(registry, config)] == ["cls/a"]
    with pytest.raises(ValueError, match="unknown dataset"):
        select_tasks(registry, EvaluationConfig(exclude_datasets=("typo",)))
