"""Markdown summary of the leaderboard, readable without a spreadsheet.

For every task and protocol the primary score of each model on the full label budget is listed with
its bootstrap interval, and the best model of each row is set in bold. Only results obtained under
the current task and protocol definitions are shown. Models that did not declare their pretraining
data are marked, since their scores may stem from having seen the test images. The input size of
every model is listed beside its pretraining data, since scores depend on the image resolution.
"""

from pathlib import Path

import pandas as pd

from find_those_polyps.metrics import METRIC_INFO


def _cell(row: pd.Series, best: bool) -> str:
    """Table cell of one score with its interval when known, `n/a` when the score is undefined.

    Args:
        row: Leaderboard row with `score` and optionally the interval bounds.
        best: Whether to set the cell in bold.

    Returns:
        The score and its interval with three decimals.
    """
    if pd.isna(row["score"]):
        return "n/a"
    text = f"{row['score']:.3f}"
    if pd.notna(row.get("score_ci_low")) and pd.notna(row.get("score_ci_high")):
        text += f" [{row['score_ci_low']:.3f}, {row['score_ci_high']:.3f}]"
    return f"**{text}**" if best else text


def _sizes(values: pd.Series) -> str:
    """Known input sizes of one model joined by commas, `unknown` when none was recorded."""
    known = sorted({int(value) for value in values if pd.notna(value)} - {0})
    return ", ".join(str(size) for size in known) or "unknown"


def render(frame: pd.DataFrame) -> str:
    """Markdown tables of full budget primary scores, one row per task and protocol.

    A first table lists the pretraining data and input sizes of every model, and a second one the
    scores with one column per model. The best score of a row is set in bold when at least two
    models were evaluated on it, following the direction of its metric.

    Args:
        frame: Ranked leaderboard table, filtered to current rows when it has a `current` column.

    Returns:
        The Markdown text ending with a newline.
    """
    if "current" in frame.columns:
        frame = frame[frame["current"].astype(bool)]
    full = frame[frame["budget"] == "full"]
    models = sorted(full["model"].unique())
    declared = full.groupby("model")["pretraining"].agg(lambda values: sorted(set(values)))
    sizes = full.get("input_size", pd.Series(0, index=full.index)).groupby(full["model"])
    size = sizes.agg(_sizes)
    undeclared = {model for model, values in declared.items() if "undeclared" in values}
    heads = [f"{model} (undeclared)" if model in undeclared else model for model in models]
    lines = [
        "# Leaderboard summary",
        "",
        "Primary score on the full label budget with its 95% bootstrap interval over test images.",
        "The best model of each row is in bold. Every budget and metric is in `leaderboard.csv`",
        "and `leaderboard.xlsx`.",
        "",
        "| Model | Pretrained on | Input size |",
        "|---|---|---|",
        *(f"| {model} | {', '.join(declared[model])} | {size[model]} |" for model in models),
        "",
        "Models marked undeclared did not state their pretraining data, so they may have seen the",
        "test images. The probes never do.",
        "",
        "| Task | Protocol | Metric | " + " | ".join(heads) + " |",
        "|---|---|---|" + "---|" * len(models),
    ]
    for (task, protocol), group in full.groupby(["task", "protocol"], sort=True):
        metric = str(group["primary_metric"].iloc[0])
        higher = METRIC_INFO[metric].higher_is_better if metric in METRIC_INFO else True
        scores = group["score"] if higher else -group["score"]
        best = scores.idxmax() if scores.notna().any() and len(group) > 1 else None
        by_model = {row["model"]: _cell(row, index == best) for index, row in group.iterrows()}
        cells = [by_model.get(model, "") for model in models]
        lines.append(f"| `{task}` | {protocol} | {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(frame: pd.DataFrame, path: Path) -> None:
    """Write the Markdown summary of a leaderboard table.

    Args:
        frame: Ranked leaderboard table.
        path: Destination file, whose parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(frame), encoding="utf-8", newline="\n")
