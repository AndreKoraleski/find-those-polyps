from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from find_those_polyps.evaluation import PROTOCOLS
from find_those_polyps.evaluation.workbook import (
    INDEX,
    configuration_name,
    sheet_name,
    write_workbook,
)
from find_those_polyps.registry.store import load_registry


def _frame() -> pd.DataFrame:
    stamp = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    rows = []
    for model, score, budget in (("a", 0.9, "full"), ("b", 0.7, "full"), ("a", 0.6, "shots:5")):
        rows.append(
            {
                "configuration": configuration_name("cls/x", "knn", budget),
                "model": model,
                "description": "d",
                "task": "cls/x",
                "dataset": "x",
                "protocol": "knn",
                "budget": budget,
                "primary_metric": "auroc",
                "score": score,
                "score_std": 0.0,
                "rank": 1 if model == "a" else 2,
                "runs": 1,
                "auroc": score,
                "dice": None,
                "auroc_std": 0.0,
                "dice_std": None,
                "task_hash": "h",
                "protocol_hash": "p",
                "registry_version": 1,
                "evaluated_at": stamp,
            }
        )
    return pd.DataFrame(rows)


def test_configuration_names() -> None:
    assert (
        configuration_name("ordinal/limuc-mayo", "linear", "shots:5") == "limuc-mayo.linear.5shot"
    )
    assert (
        configuration_name("seg/kvasir-seg", "linear_seg", "n:100") == "kvasir-seg.linear_seg.n100"
    )
    assert configuration_name("cls/x", "knn", "frac:0.01") == "x.knn.1pct"
    long = configuration_name("cls/" + "x" * 30, "knn", "full")
    assert len(sheet_name(long)) == 31
    assert sheet_name(long) != sheet_name(long.replace("x", "y"))
    assert sheet_name("x.knn.full") == "x.knn.full"


def test_every_registered_configuration_has_a_valid_name() -> None:
    names = [
        configuration_name(task.id, protocol.name, str(budget))
        for task in load_registry().tasks
        for protocol in PROTOCOLS.values()
        if task.kind in protocol.kinds
        for budget in task.budgets
    ]
    assert len(set(names)) == len(names)
    sheets = [sheet_name(name) for name in names]
    assert len(set(sheets)) == len(sheets)
    assert all(len(sheet) <= 31 and not set(sheet) & set("[]:*?/\\") for sheet in sheets)


def test_workbook_has_an_index_and_a_sheet_per_configuration(tmp_path: Path) -> None:
    path = tmp_path / "board.xlsx"
    write_workbook(_frame(), path)
    sheets = pd.read_excel(path, sheet_name=None)
    assert list(sheets) == [INDEX, "comparison", "x.knn.full", "x.knn.5shot"]
    comparison = sheets["comparison"]
    assert comparison["configuration"].tolist() == ["x.knn.full", "x.knn.5shot"]
    assert comparison["a"].tolist() == [0.9, 0.6]
    assert comparison["b"].isna().tolist() == [False, True]
    index = sheets[INDEX]
    assert index["best_model"].tolist() == ["a", "a"]
    assert index["models"].tolist() == [2, 1]
    full = sheets["x.knn.full"]
    assert full["model"].tolist() == ["a", "b"]
    assert "auroc" in full.columns
    assert "dice" not in full.columns
