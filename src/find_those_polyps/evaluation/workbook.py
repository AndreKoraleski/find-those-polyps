"""Excel rendering of the leaderboard with one sheet per evaluation configuration.

A configuration is a task evaluated under one protocol and label budget, and its standard name joins
the task name, the protocol and the budget label, for instance `limuc-mayo.linear.5shot`. Its
sheet carries the same title, shortened with a hash suffix when it exceeds the thirty one characters
Excel allows. The first sheet indexes every configuration with its best model and links to its
sheet, and the second compares the primary scores of all models across configurations.
"""

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from find_those_polyps.metrics import METRIC_INFO
from find_those_polyps.tasks.schema import Budget

SHEET_LIMIT = 31
INDEX = "index"
COMPARISON = "comparison"
SCORE_FORMAT = "0.0000"
RUN_COLUMNS = [
    "rank",
    "model",
    "description",
    "input_size",
    "feature_grid",
    "score",
    "score_std",
    "score_ci_low",
    "score_ci_high",
    "runs",
]
TRACE_COLUMNS = ["task_hash", "protocol_hash", "registry_version", "evaluated_at"]
INDEX_COLUMNS = [
    "configuration",
    "sheet",
    "task",
    "dataset",
    "protocol",
    "budget",
    "primary_metric",
    "models",
    "best_model",
    "best_score",
]


def configuration_name(task: str, protocol: str, budget: str) -> str:
    """Standard name of a configuration.

    Args:
        task: Task identifier, of which only the part after the last slash is kept.
        protocol: Name of the protocol.
        budget: Label budget in its textual form, for instance `shots:5`.

    Returns:
        The task name, the protocol and the budget slug joined by dots.
    """
    return f"{task.rsplit('/', 1)[-1]}.{protocol}.{Budget.model_validate(budget).slug}"


def sheet_name(configuration: str) -> str:
    """Excel sheet title of a configuration, its name or a stable shortening of it.

    Args:
        configuration: Standard name of the configuration.

    Returns:
        The name itself when it fits in `SHEET_LIMIT` characters, and otherwise its first 26
        characters followed by a tilde and four hexadecimal digits of its SHA-1 digest.
    """
    if len(configuration) <= SHEET_LIMIT:
        return configuration
    digest = hashlib.sha1(configuration.encode(), usedforsecurity=False).hexdigest()[:4]
    return f"{configuration[: SHEET_LIMIT - 5]}~{digest}"


def _ordered(frame: pd.DataFrame) -> list[str]:
    """Configurations sorted by task, protocol and budget, the full and larger budgets first."""
    first = frame.drop_duplicates("configuration")
    keys: dict[str, tuple[str, str, tuple[int, float]]] = {
        str(row["configuration"]): (
            str(row["task"]),
            str(row["protocol"]),
            Budget.model_validate(str(row["budget"])).order,
        )
        for row in first.to_dict("records")
    }
    return sorted(keys, key=keys.__getitem__)


def _write(sheet: Worksheet, table: pd.DataFrame, scores: list[str]) -> None:
    """Fill a sheet with a table, a bold frozen header, filters and fitted widths.

    Missing values become empty cells, and the first two columns stay visible when scrolling.

    Args:
        sheet: Empty worksheet to fill.
        table: Rows to write below a header of its column names.
        scores: Columns shown with four decimals.
    """
    sheet.append(list(table.columns))
    for values in table.itertuples(index=False):
        sheet.append([None if pd.isna(value) else value for value in values])
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    sheet.freeze_panes = "C2"
    sheet.auto_filter.ref = sheet.dimensions
    for position, column in enumerate(table.columns, start=1):
        letter = get_column_letter(position)
        if column in scores:
            for cell in sheet[letter][1:]:
                cell.number_format = SCORE_FORMAT
        width = max([len(str(column)), *(len(str(v)) for v in table[column].head(200))])
        sheet.column_dimensions[letter].width = min(width + 2, 60)


def write_workbook(frame: pd.DataFrame, path: Path) -> None:
    """Write the current rows of a ranked leaderboard table as an Excel workbook.

    Each configuration sheet lists its models by rank with the metrics defined for at least one of
    them. The creation time of the workbook is set to the time of the latest evaluation, while
    openpyxl sets its modification time to the moment of saving. The file is replaced atomically
    through a temporary file.

    Args:
        frame: Ranked leaderboard table, filtered to current rows when it has a `current` column.
        path: Destination workbook, whose parent directories are created.
    """
    if "current" in frame.columns:
        frame = frame[frame["current"].astype(bool)]
    book = Workbook()
    index = book.active
    assert index is not None
    index.title = INDEX
    rows = []
    for name in _ordered(frame):
        table = frame[frame["configuration"] == name].sort_values(["rank", "model"])
        best = table.iloc[0]
        rows.append([name, sheet_name(name), best.task, best.dataset, best.protocol, best.budget])
        rows[-1] += [best.primary_metric, len(table), best.model, best.score]
        metrics = [m for m in METRIC_INFO if m in table.columns and table[m].notna().any()]
        stds = [m + "_std" for m in metrics]
        columns = RUN_COLUMNS + metrics + stds + TRACE_COLUMNS
        sheet = book.create_sheet(sheet_name(name))
        scores = ["score", "score_std", "score_ci_low", "score_ci_high", *metrics, *stds]
        _write(sheet, table[[c for c in columns if c in table.columns]], scores)
    _write(index, pd.DataFrame(rows, columns=INDEX_COLUMNS), ["best_score"])
    order = [row[0] for row in rows]
    if order:
        models = sorted(frame["model"].unique())
        pivot = frame.pivot_table(
            index="configuration", columns="model", values="score", aggfunc="first", dropna=False
        )
        comparison = pivot.reindex(index=order, columns=models).reset_index()
        _write(book.create_sheet(COMPARISON, 1), comparison, models)
    for row, title in enumerate((r[1] for r in rows), start=2):
        cell = index.cell(row=row, column=1)
        cell.hyperlink = f"#'{title}'!A1"
        cell.font = Font(color="0563C1", underline="single")
    stamp = pd.to_datetime(frame["evaluated_at"]).max() if len(frame) else None
    if isinstance(stamp, datetime):
        book.properties.created = book.properties.modified = stamp.to_pydatetime()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.xlsx")
    book.save(temporary)
    temporary.replace(path)
