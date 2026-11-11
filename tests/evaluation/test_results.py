from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from find_those_polyps.evaluation import Leaderboard, LeaderboardRow
from find_those_polyps.evaluation.results import main


def test_outputs_are_rewritten_from_the_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    row = LeaderboardRow(
        model="a",
        description="d",
        task="cls/x",
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
    )
    board.upsert([row])
    board.workbook.unlink()
    board.report.unlink()
    assert main(["--leaderboard", str(board.path)]) == 0
    assert "rewrote" in capsys.readouterr().out
    assert "**" not in board.report.read_text(encoding="utf-8")
    assert list(pd.read_excel(board.workbook, sheet_name=None)) == [
        "index",
        "comparison",
        "x.knn.full",
    ]
    with pytest.raises(SystemExit):
        main(["--leaderboard", str(tmp_path / "missing.csv")])
