from pathlib import Path

from find_those_polyps.registry import REGISTRY_PATH, load_registry
from find_those_polyps.registry.docs import DOCS_PATH, render, warnings, write_docs


def test_packaged_registry_documents_known_leakage() -> None:
    registry = load_registry(REGISTRY_PATH)
    lines = warnings(registry)
    assert any("HyperKvasir unlabeled images" in w and "PolypDB" in w for w in lines)
    assert all(w.startswith("**") and "Evidence" in w for w in lines)
    registered = {card.key for card in registry.datasets}
    assert all(task.dataset in registered for task in registry.tasks)
    assert {task.dataset for task in registry.tasks} == registered


def test_committed_docs_match_registry() -> None:
    assert DOCS_PATH.read_text(encoding="utf-8") == render(load_registry(REGISTRY_PATH))


def test_write_docs_reports_changes(tmp_path: Path) -> None:
    registry = load_registry(REGISTRY_PATH)
    target = tmp_path / "datasets.md"
    assert write_docs(registry, target)
    assert not write_docs(registry, target)
    text = target.read_text(encoding="utf-8")
    assert "## Leakage warnings" in text
    assert "| `seg/kvasir-seg` | segmentation | 2 | `dice` | " in text
    assert "| Task | Kind | Classes | Primary metric | Partitions | Lost | Test source |" in text
    assert "not built" in render(registry, None)
