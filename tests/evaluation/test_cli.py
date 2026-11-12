from datetime import UTC, datetime
from pathlib import Path

import pytest

from find_those_polyps.baselines import RandomResNet50
from find_those_polyps.evaluation import LeaderboardRow
from find_those_polyps.evaluation.__main__ import load_encoder, main, summary


def test_load_encoder_from_reference() -> None:
    encoder = load_encoder("find_those_polyps.baselines:RandomResNet50")
    assert isinstance(encoder, RandomResNet50)


@pytest.mark.parametrize(
    ("reference", "error"),
    [("find_those_polyps.baselines", ValueError), ("pathlib:Path", ValueError)],
)
def test_load_encoder_rejects_bad_references(reference: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        load_encoder(reference)


def test_cli_runs_without_data(tmp_path: Path) -> None:
    status = main(
        [
            "find_those_polyps.baselines:RandomResNet50",
            "--data-root",
            str(tmp_path),
            "--leaderboard",
            str(tmp_path / "board.csv"),
            "--device",
            "cpu",
            "--seeds",
            "1",
            "--no-stream",
            "--workers",
            "0",
            "--prefetch",
            "3",
            "--preprocess-on-device",
        ]
    )
    assert status == 0
    assert not (tmp_path / "board.csv").exists()


FACTORY = """
from find_those_polyps.baselines import RandomResNet50


def build():
    return RandomResNet50(seed=3)
"""


def test_load_encoder_from_files_and_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "local_factory.py").write_text(FACTORY)
    assert load_encoder(f"{tmp_path / 'local_factory.py'}:build").name == "random-resnet50-s3"
    monkeypatch.chdir(tmp_path)
    assert load_encoder("local_factory:build").name == "random-resnet50-s3"
    with pytest.raises(ValueError, match="no attribute"):
        load_encoder("local_factory:missing")
    with pytest.raises(ValueError, match="no such file"):
        load_encoder("absent.py:build")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--exclude-datasets", "hyperkvasir"], "did you mean"),
        (["--tasks", "cls/nothing-like-this"], "unknown task patterns"),
        (["--protocols", "svm"], "choose from"),
        (["--seeds", "0"], "seeds: Input should be greater than 0"),
        (["--bootstrap", "-1"], "bootstrap: Input should be greater than or equal to 0"),
    ],
)
def test_user_mistakes_exit_with_a_message(
    arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["x:y", *arguments, "--dry-run"])
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_dry_run_lists_selected_tasks(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["x:y", "--dry-run", "--pretrained-on", "hyperkvasir-unlabeled"]) == 0
    out = capsys.readouterr().out
    assert "ordinal/limuc-mayo" in out
    assert "cls/hyperkvasir " not in out


def _row(task: str, low: float | None, high: float | None) -> LeaderboardRow:
    return LeaderboardRow(
        model="a",
        description="d",
        task=task,
        task_hash="h",
        dataset="x",
        protocol="knn",
        protocol_hash="p",
        budget="full",
        primary_metric="auroc",
        runs=1,
        registry_version=1,
        evaluated_at=datetime(2026, 1, 1, tzinfo=UTC),
        means={"auroc": 0.8},
        stds={"auroc": float("nan")},
        score_ci_low=low,
        score_ci_high=high,
    )


def test_summary_omits_unknown_intervals() -> None:
    nan = float("nan")
    rows = [_row("cls/a", 0.7, 0.9), _row("cls/b", nan, nan), _row("cls/c", None, None)]
    text = summary(rows)
    lines = text.splitlines()[1:]
    assert lines[0].endswith("0.800  [0.700, 0.900]")
    assert [line.endswith("0.800  ") for line in lines[1:]] == [True, True]
    assert "nan" not in text
