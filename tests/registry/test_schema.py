from typing import Any

import pytest
from pydantic import ValidationError

from find_those_polyps.metrics import metric_names
from find_those_polyps.registry import REGISTRY_PATH, Registry, load_registry


def _card(key: str, **fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "key": key,
        "name": key,
        "labelled": True,
        "modality": "m",
        "licence": "l",
        "access": "open",
        "url": "u",
        "citation": "c",
        "provenance": "p",
        "family": "f",
    }
    return base | fields


def test_unregistered_overlap_targets_need_a_name() -> None:
    overlap = {"dataset": "ghost", "kind": "duplicate", "evidence": "e"}
    with pytest.raises(ValidationError, match="unknown dataset"):
        Registry.model_validate(
            {"version": 0, "sha256": "", "datasets": [_card("a", overlaps=[overlap])]}
        )
    named = Registry.model_validate(
        {
            "version": 0,
            "sha256": "",
            "datasets": [_card("a", overlaps=[overlap | {"name": "Ghost pool"}])],
        }
    )
    assert named.corpora() == {"a", "ghost"}
    assert named.overlapping({"ghost"}) == {"a", "ghost"}


def test_duplicate_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Registry.model_validate({"version": 0, "sha256": "", "datasets": [_card("a"), _card("a")]})


def test_overlapping_is_symmetric() -> None:
    overlap = {"dataset": "pool", "kind": "same_source", "evidence": "e"}
    registry = Registry.model_validate(
        {
            "version": 0,
            "sha256": "",
            "datasets": [_card("pool", labelled=False), _card("a", overlaps=[overlap]), _card("b")],
        }
    )
    assert registry.overlapping({"pool"}) == {"pool", "a"}
    assert registry.overlapping({"a"}) == {"a", "pool"}


def test_packaged_tasks_use_valid_primary_metrics() -> None:
    registry = load_registry(REGISTRY_PATH)
    for task in registry.tasks:
        assert task.primary_metric in metric_names(task.kind, len(task.classes)), task.id
    assert {card.key for card in registry.datasets} >= {task.dataset for task in registry.tasks}


def test_task_names_must_be_unique() -> None:
    task = {
        "description": "d",
        "dataset": "a",
        "kind": "classification",
        "classes": ["x", "y"],
        "primary_metric": "auroc",
        "loader": "l",
        "splits": "s.json",
    }
    tasks = [task | {"id": "cls/same"}, task | {"id": "ordinal/same"}]
    with pytest.raises(ValidationError, match="task names"):
        Registry.model_validate(
            {"version": 0, "sha256": "", "datasets": [_card("a")], "tasks": tasks}
        )
