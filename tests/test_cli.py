from pathlib import Path

import pytest

import find_those_polyps
from find_those_polyps.cli import COMMANDS, main


def test_usage_lists_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in COMMANDS)
    assert main([]) == 2
    assert main(["unknown"]) == 2


def test_dispatches_to_the_named_tool(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data", "tasks"]) == 0
    assert "seg/kvasir-seg" in capsys.readouterr().out


def test_package_exports_are_lazy() -> None:
    assert find_those_polyps.load_registry().version > 0
    assert set(find_those_polyps.__all__) >= {"evaluate", "Encoder", "TorchEncoder"}
    with pytest.raises(AttributeError):
        _ = find_those_polyps.missing


def test_data_root_is_accepted_after_the_subcommand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "datasets", "--root", str(tmp_path)]) == 0
    assert "streamable" in capsys.readouterr().out
