from pathlib import Path

import pytest
from conftest import ColourLoader

from find_those_polyps.data import LocalFiles
from find_those_polyps.registry import __main__ as registry_main
from find_those_polyps.registry import build
from find_those_polyps.registry.store import load_registry


def test_registry_command_reports_a_synchronized_registry() -> None:
    assert registry_main.main([]) == 0


def test_split_command_builds_selected_and_missing_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = load_registry()
    monkeypatch.setattr(build, "SPLITS_DIR", tmp_path)
    monkeypatch.setattr(build, "load_registry", lambda **_: registry)
    monkeypatch.setattr(build, "sync_registry", lambda: False)
    monkeypatch.setattr(build, "write_docs", lambda _: False)
    monkeypatch.setattr(
        build, "open_loader", lambda *a, **k: ColourLoader(LocalFiles("x", tmp_path))
    )
    assert build.main(["cls/limuc-*", "--root", str(tmp_path)]) == 0
    assert (tmp_path / "limuc.json").exists()
    assert "cls/limuc-remission train" in capsys.readouterr().out
    assert build.main(["--missing", "--root", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob("*.json"))) == len({t.splits for t in registry.tasks})
    with pytest.raises(SystemExit):
        build.main(["nothing/*"])


def test_registry_check_reports_stale_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert registry_main.main(["--check"]) == 0
    stale = tmp_path / "datasets.md"
    stale.write_text("old")
    monkeypatch.setattr(registry_main, "DOCS_PATH", stale)
    assert registry_main.main(["--check"]) == 1
    assert "stale" in capsys.readouterr().err


def test_split_command_refuses_to_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = load_registry()
    monkeypatch.setattr(build, "SPLITS_DIR", tmp_path)
    monkeypatch.setattr(build, "load_registry", lambda **_: registry)
    monkeypatch.setattr(build, "sync_registry", lambda: False)
    monkeypatch.setattr(build, "write_docs", lambda _: False)
    monkeypatch.setattr(
        build, "open_loader", lambda *a, **k: ColourLoader(LocalFiles("x", tmp_path))
    )
    (tmp_path / "limuc.json").write_text("{}")
    with pytest.raises(SystemExit):
        build.main(["ordinal/limuc-mayo"])
    assert "--force" in capsys.readouterr().err
    assert build.main(["ordinal/limuc-mayo", "--force", "--redraw"]) == 0
    assert (tmp_path / "limuc.json").read_text() != "{}"
