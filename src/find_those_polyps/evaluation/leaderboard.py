"""Global results table with one row per evaluation and one column per metric.

Each row holds the test scores of one model on one task under one protocol and label budget, the
configuration named by `configuration_name`. The mean of every metric over seeds has its own
column, followed by the standard deviations with the suffix `_std`. Rows are ranked by the primary
metric of their task within comparable settings. The CSV file is the record of all results, and an
Excel workbook and a Markdown summary beside it are derived from it. `pretraining` records what the
model declared to have been pretrained on, `undeclared` when it declared nothing and `none` when it
was not pretrained. `input_size` is the side in pixels of the images given to the encoder, zero for
rows recorded before it was kept, and `feature_grid` the height and width of the dense feature
maps, for instance `14x14`, empty under global protocols.
"""

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from find_those_polyps.evaluation.report import write_report
from find_those_polyps.evaluation.workbook import configuration_name, write_workbook
from find_those_polyps.metrics import METRIC_INFO

SETTING = ["task", "task_hash", "protocol", "protocol_hash", "budget"]
KEY = ["model", "task", "protocol", "budget"]
STD = "_std"


class LeaderboardRow(BaseModel):
    """Aggregated test scores of one model under one evaluation setting.

    Attributes:
        model: Name of the encoder.
        description: One sentence summary of the encoder.
        pretraining: Declared pretraining datasets joined by `+`, `none` when the encoder was not
            pretrained and `undeclared` by default.
        input_size: Side in pixels of the encoder input, 0 when unknown.
        feature_grid: Height and width of the dense feature maps written as `hxw`, empty by
            default and under global protocols.
        task: Identifier of the task.
        task_hash: Hash of the task definition the scores were obtained under.
        dataset: Key of the dataset of the task.
        protocol: Name of the protocol.
        protocol_hash: Hash of the protocol definition the scores were obtained under.
        budget: Label budget, for instance `full` or `shots:5`.
        primary_metric: Metric that ranks the row.
        runs: Number of seeds averaged.
        rank: Competition rank among comparable rows, `None` until ranked or when the score is
            undefined.
        registry_version: Version of the registry the task was taken from.
        evaluated_at: Time at which the scores were computed.
        means: Mean of every metric over the runs.
        stds: Sample standard deviation of every metric over the runs, NaN for a single run.
        score_ci_low: Lower bound of the bootstrap interval of the primary score, `None` when
            unknown and NaN when the bootstrap was disabled.
        score_ci_high: Upper bound of the bootstrap interval of the primary score, `None` when
            unknown and NaN when the bootstrap was disabled.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    description: str
    pretraining: str = "undeclared"
    input_size: int = 0
    feature_grid: str = ""
    task: str
    task_hash: str
    dataset: str
    protocol: str
    protocol_hash: str
    budget: str
    primary_metric: str
    runs: int
    rank: int | None = None
    registry_version: int
    evaluated_at: datetime
    means: dict[str, float]
    stds: dict[str, float]
    score_ci_low: float | None = None
    score_ci_high: float | None = None

    @property
    def configuration(self) -> str:
        """Standard name of the task, protocol and budget combination."""
        return configuration_name(self.task, self.protocol, self.budget)

    @property
    def score(self) -> float:
        """Mean of the primary metric, NaN when it is undefined."""
        return self.means.get(self.primary_metric, float("nan"))


HEAD = [
    "configuration",
    "model",
    "description",
    "pretraining",
    "input_size",
    "feature_grid",
    "task",
    "dataset",
    "protocol",
    "budget",
    "primary_metric",
    "score",
    "score_std",
    "score_ci_low",
    "score_ci_high",
    "rank",
    "current",
    "runs",
]
TAIL = ["task_hash", "protocol_hash", "registry_version", "evaluated_at"]
DERIVED = ("configuration", "score", "score_std", "current")
TEXT = [*KEY, "configuration", "description", "pretraining", "feature_grid", "dataset"]
TEXT += ["primary_metric", *SETTING]
BACKFILL: dict[str, str | int] = {"pretraining": "undeclared", "input_size": 0, "feature_grid": ""}


def _record(row: LeaderboardRow) -> dict[str, object]:
    """Flat CSV record of a row with its derived columns and one column per metric statistic."""
    fields = row.model_dump(mode="json", exclude={"means", "stds"})
    fields["configuration"] = row.configuration
    fields["score"] = row.score
    fields["score_std"] = row.stds.get(row.primary_metric, float("nan"))
    return fields | row.means | {name + STD: value for name, value in row.stds.items()}


def _columns(frame: pd.DataFrame) -> list[str]:
    """Fixed columns, metric means in registry order, then their standard deviations."""
    metrics = [name for name in METRIC_INFO if name in frame.columns]
    return [*HEAD, *metrics, *(name + STD for name in metrics), *TAIL]


def rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Competition ranking on the primary score within comparable settings.

    Rows are comparable when they share the task, the protocol, the budget and the hashes of the
    task and protocol definitions. Ties share the best rank, lower scores rank first for metrics
    where lower is better, and undefined scores stay unranked.

    Args:
        frame: Leaderboard table with at least the setting columns, `model`, `primary_metric` and
            `score`.

    Returns:
        A copy with a nullable integer `rank` column, sorted by task, protocol, budget, rank and
        model, with unranked rows last and a fresh index.
    """
    higher = frame["primary_metric"].map(
        lambda m: METRIC_INFO[m].higher_is_better if m in METRIC_INFO else True
    )
    signed = frame["score"].where(higher, -frame["score"])
    ranks = signed.groupby([frame[c] for c in SETTING]).rank(method="min", ascending=False)
    ranked = frame.assign(rank=ranks.astype("Int64"))
    ranked = ranked.sort_values(["task", "protocol", "budget", "rank", "model"], na_position="last")
    return ranked.reset_index(drop=True)


def _current(frame: pd.DataFrame) -> pd.Series:
    """Whether each row was obtained under the latest task and protocol definitions.

    The latest definitions of a task and protocol pair are the hashes of its most recently
    evaluated row. Rows evaluated before a task or protocol changed keep their scores but are
    marked stale, and the workbook and the summary show current rows only.

    Args:
        frame: Leaderboard table with the task, protocol, hash and `evaluated_at` columns.

    Returns:
        A boolean series aligned with the index of `frame`.
    """
    hashes = frame["task_hash"] + "/" + frame["protocol_hash"]
    newest = frame.assign(_hashes=hashes).sort_values("evaluated_at")
    latest = newest.groupby(["task", "protocol"])["_hashes"].last()
    keys = pd.MultiIndex.from_frame(frame[["task", "protocol"]])
    return pd.Series(hashes.to_numpy() == latest.reindex(keys).to_numpy(), index=frame.index)


class Leaderboard:
    """CSV file holding every result, with the workbook and the summary derived from it.

    The whole table is rewritten after each update. The CSV file and the workbook are replaced
    atomically through a temporary file, while the Markdown summary is written in place.

    Attributes:
        path: CSV file holding every result.
        workbook: Excel workbook beside it, with the suffix `.xlsx`.
        report: Markdown summary beside it, with the suffix `.md`.
    """

    def __init__(self, path: Path) -> None:
        """Bind the leaderboard to its CSV file, which need not exist yet.

        Args:
            path: CSV file holding every result.
        """
        self.path = path
        self.workbook = path.with_suffix(".xlsx")
        self.report = path.with_suffix(".md")

    def frame(self) -> pd.DataFrame:
        """The table as a data frame, empty if the file does not exist.

        Columns added after a file was written are filled with their defaults, and so are the empty
        cells of rows merged from such files.

        Returns:
            The table with text columns read as strings and integer ranks and input sizes.
        """
        if not self.path.exists():
            return pd.DataFrame(columns=[*HEAD, *TAIL])
        types = dict.fromkeys(TEXT, str) | {"rank": "Int64", "input_size": "Int64"}
        frame = pd.read_csv(self.path, dtype=types, keep_default_na=False, na_values=[""])
        for column, value in BACKFILL.items():
            frame[column] = frame[column].fillna(value) if column in frame.columns else value
        return frame.astype({"input_size": int})

    def read(self) -> list[LeaderboardRow]:
        """Rows of the table, omitting metrics whose mean is undefined.

        Undefined ranks and interval bounds become `None`, and derived columns are dropped.

        Returns:
            One validated row per line of the CSV file.
        """
        frame = self.frame()
        metrics = [name for name in METRIC_INFO if name in frame.columns]
        rows = []
        for record in frame.to_dict("records"):
            fields = {key: record[key] for key in HEAD + TAIL if key not in DERIVED}
            for key in ("rank", "score_ci_low", "score_ci_high"):
                fields[key] = None if pd.isna(fields[key]) else fields[key]
            present = [name for name in metrics if pd.notna(record[name])]
            fields["means"] = {name: record[name] for name in present}
            fields["stds"] = {name: record[name + STD] for name in present}
            rows.append(LeaderboardRow.model_validate(fields))
        return rows

    def refresh(self) -> None:
        """Rewrite the ranking, the stale flags, the workbook and the summary from the CSV file.

        Nothing is written when the table is empty or missing.
        """
        frame = self.frame()
        if len(frame):
            self._write(frame)

    def upsert(self, rows: Iterable[LeaderboardRow]) -> None:
        """Insert rows, replacing earlier results of the same model, task, protocol and budget.

        A replaced result is dropped even when it was obtained under other task or protocol
        definitions.

        Args:
            rows: New results, written nowhere when empty.
        """
        records = [_record(row) for row in rows]
        if not records:
            return
        new = pd.DataFrame(records)
        existing = self.frame()
        merged = pd.concat([existing, new]) if len(existing) else new
        self._write(merged.drop_duplicates(subset=KEY, keep="last"))

    def _write(self, merged: pd.DataFrame) -> None:
        """Rank `merged`, flag stale rows and write the CSV file, the workbook and the summary."""
        merged = merged.assign(current=_current(merged))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        ranked = rank(merged)[_columns(merged)]
        ranked.to_csv(temporary, index=False, lineterminator="\n")
        temporary.replace(self.path)
        write_workbook(ranked, self.workbook)
        write_report(ranked, self.report)
