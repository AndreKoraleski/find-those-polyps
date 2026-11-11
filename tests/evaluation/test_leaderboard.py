from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from find_those_polyps.evaluation import Leaderboard, LeaderboardRow


def _row(model: str, score: float, primary: str = "auroc", **fields: object) -> LeaderboardRow:
    base: dict[str, object] = {
        "model": model,
        "description": "d",
        "task": "cls/x",
        "task_hash": "h",
        "dataset": "x",
        "protocol": "knn",
        "protocol_hash": "p",
        "budget": "full",
        "primary_metric": primary,
        "runs": 1,
        "registry_version": 1,
        "evaluated_at": datetime(2026, 1, 1, tzinfo=UTC),
        "means": {primary: score, "mcc": 0.5},
        "stds": {primary: 0.01, "mcc": 0.02},
    }
    return LeaderboardRow.model_validate(base | fields)


def test_one_row_per_run_with_a_column_per_metric(tmp_path: Path) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    board.upsert([_row("a", 0.9), _row("a", 0.8, budget="shots:1")])
    frame = pd.read_csv(board.path)
    assert len(frame) == 2
    columns = list(frame.columns)
    assert columns.index("mcc") < columns.index("auroc") < columns.index("mcc_std")
    assert frame["score"].tolist() == frame["auroc"].tolist()
    assert frame["score_std"].tolist() == [0.01, 0.01]
    assert frame["mcc_std"].tolist() == [0.02, 0.02]


def test_ranking_respects_metric_direction(tmp_path: Path) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    board.upsert([_row("a", 0.9), _row("b", 0.7), _row("c", 0.9)])
    board.upsert([_row(m, s, "log_loss", task="cls/y") for m, s in (("a", 0.5), ("b", 0.2))])
    ranks = {(r.task, r.model): r.rank for r in board.read()}
    assert ranks[("cls/x", "a")] == ranks[("cls/x", "c")] == 1
    assert ranks[("cls/x", "b")] == 3
    assert ranks[("cls/y", "b")] == 1
    assert ranks[("cls/y", "a")] == 2


def test_tasks_with_different_metrics_share_the_table(tmp_path: Path) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    board.upsert([_row("a", 0.9)])
    board.upsert([_row("a", 0.4, "dice", task="seg/z", means={"dice": 0.4}, stds={"dice": 0.0})])
    rows = {r.task: r for r in board.read()}
    assert rows["seg/z"].means == {"dice": 0.4}
    assert rows["cls/x"].means == {"auroc": 0.9, "mcc": 0.5}
    assert rows["seg/z"].score == 0.4


def test_nan_is_unranked_and_hashes_separate_settings(tmp_path: Path) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    board.upsert([_row("a", float("nan")), _row("b", 0.6, task_hash="new")])
    ranks = {r.model: r.rank for r in board.read()}
    assert ranks == {"a": None, "b": 1}


def test_upsert_replaces_earlier_results(tmp_path: Path) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    board.upsert([_row("a", 0.5)])
    board.upsert([_row("a", 0.8)])
    assert [r.score for r in board.read()] == [0.8]
    assert Leaderboard(tmp_path / "absent.csv").read() == []


def test_upsert_writes_the_workbook(tmp_path: Path) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    board.upsert([_row("a", 0.9), _row("b", 0.4)])
    sheets = pd.read_excel(board.workbook, sheet_name=None)
    assert list(sheets) == ["index", "comparison", "x.knn.full"]
    assert sheets["x.knn.full"]["rank"].tolist() == [1, 2]
    assert [r.configuration for r in board.read()] == ["x.knn.full"] * 2


def test_upserting_nothing_writes_nothing(tmp_path: Path) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    board.upsert([])
    assert not board.path.exists()
    board.upsert([_row("a", 0.9)])
    board.upsert([])
    assert len(board.read()) == 1


def test_hashes_and_identifiers_round_trip_as_text(tmp_path: Path) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    board.upsert([_row("007", 0.9, task_hash="012345678901", protocol_hash="1234e5678901")])
    (row,) = board.read()
    assert (row.model, row.task_hash, row.protocol_hash) == ("007", "012345678901", "1234e5678901")


def test_undefined_primary_scores_and_intervals_read_back(tmp_path: Path) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    board.upsert([_row("a", float("nan"), score_ci_low=0.1, score_ci_high=0.3)])
    (row,) = board.read()
    assert row.score != row.score
    assert (row.score_ci_low, row.score_ci_high) == (0.1, 0.3)


def test_rows_under_old_definitions_are_stale_and_left_out_of_the_workbook(
    tmp_path: Path,
) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    board.upsert([_row("a", 0.9, evaluated_at=datetime(2026, 1, 1, tzinfo=UTC))])
    board.upsert([_row("b", 0.5, task_hash="new", evaluated_at=datetime(2026, 2, 1, tzinfo=UTC))])
    frame = pd.read_csv(board.path)
    assert dict(zip(frame["model"], frame["current"], strict=True)) == {"a": False, "b": True}
    sheet = pd.read_excel(board.workbook, sheet_name="x.knn.full")
    assert sheet["model"].tolist() == ["b"]


def test_markdown_report_marks_the_best_model(tmp_path: Path) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    board.upsert([_row("a", 0.9, score_ci_low=0.8, score_ci_high=0.95), _row("b", 0.4)])
    board.upsert(
        [_row("a", 0.5, "log_loss", task="cls/y"), _row("b", 0.2, "log_loss", task="cls/y")]
    )
    text = board.report.read_text(encoding="utf-8")
    assert "| `cls/x` | knn | auroc | **0.900 [0.800, 0.950]** | 0.400 |" in text
    assert "| `cls/y` | knn | log_loss | 0.500 | **0.200** |" in text


def test_input_size_and_feature_grid_follow_the_pretraining(tmp_path: Path) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    dense = _row("a", 0.4, "dice", protocol="linear_seg", input_size=448, feature_grid="28x28")
    board.upsert([_row("a", 0.9, input_size=448), dense, _row("b", 0.5)])
    columns = list(pd.read_csv(board.path).columns)
    assert columns[columns.index("pretraining") + 1 : columns.index("task")] == [
        "input_size",
        "feature_grid",
    ]
    rows = {(r.model, r.protocol): r for r in board.read()}
    assert (rows["a", "linear_seg"].input_size, rows["a", "linear_seg"].feature_grid) == (
        448,
        "28x28",
    )
    assert (rows["a", "knn"].feature_grid, rows["b", "knn"].input_size) == ("", 0)
    text = board.report.read_text(encoding="utf-8")
    assert "| Model | Pretrained on | Input size |" in text
    assert "| a | undeclared | 448 |" in text
    assert "| b | undeclared | unknown |" in text
    sheet = pd.read_excel(board.workbook, sheet_name="x.linear_seg.full")
    assert sheet[["input_size", "feature_grid"]].values.tolist() == [[448, "28x28"]]


def test_old_files_without_the_new_columns_are_backfilled(tmp_path: Path) -> None:
    board = Leaderboard(tmp_path / "board.csv")
    board.upsert([_row("a", 0.9, input_size=224, feature_grid="14x14")])
    old = pd.read_csv(board.path).drop(columns=["pretraining", "input_size", "feature_grid"])
    old.to_csv(board.path, index=False)
    (row,) = board.read()
    assert (row.pretraining, row.input_size, row.feature_grid) == ("undeclared", 0, "")
    board.upsert([_row("b", 0.5, input_size=518, feature_grid="37x37")])
    rows = {r.model: (r.input_size, r.feature_grid) for r in board.read()}
    assert rows == {"a": (0, ""), "b": (518, "37x37")}
    frame = pd.read_csv(board.path)
    frame.loc[frame["model"] == "b", ["input_size", "feature_grid"]] = None
    frame.to_csv(board.path, index=False)
    assert {r.model: (r.input_size, r.feature_grid) for r in board.read()} == {
        "a": (0, ""),
        "b": (0, ""),
    }
