"""Per image test results for error analysis.

Each task holds one CSV per model under `per_image_dir / task`, where the slashes of the task
identifier become double underscores. Every row identifies a test image by its id and its
canonical path relative to the data root, together with the protocol, budget and seed that
produced the prediction.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from find_those_polyps.metrics.detection import Matches
from find_those_polyps.tasks.schema import Target, TaskKind, TaskSpec

RUN_COLUMNS = ["protocol", "budget", "seed"]


def detection_records(matches: Sequence[Matches]) -> pd.DataFrame:
    """Counts and matching outcome of the detections of each image.

    The last four columns hold the matches of the image as space separated fields, from which
    `matches_from_records` rebuilds them, so that any detection metric can be resampled later.

    Args:
        matches: Matching outcome of every test image.

    Returns:
        One row per image with the number of targets, the number of kept detections, the number
        of detections matched at the first overlap threshold, 0.5, and the highest detection
        score, NaN when there is no detection. Then the number of targets of every class with the
        background first, and the class, score and hits of every kept detection, the hits as one
        string of zeros and ones per detection with one character per overlap threshold.
    """
    return pd.DataFrame(
        {
            "targets": [int(m.targets.sum()) for m in matches],
            "detections": [len(m.scores) for m in matches],
            "matched_50": [int(m.hits[:, 0].sum()) for m in matches],
            "top_score": [
                float(m.scores.max()) if len(m.scores) else float("nan") for m in matches
            ],
            "target_counts": [" ".join(str(int(c)) for c in m.targets) for m in matches],
            "det_classes": [" ".join(str(int(c)) for c in m.classes) for m in matches],
            "det_scores": [" ".join(repr(float(v)) for v in m.scores) for m in matches],
            "det_hits": [
                " ".join("".join("1" if h else "0" for h in row) for row in m.hits) for m in matches
            ],
        }
    )


DETECTION_COLUMNS = ("target_counts", "det_classes", "det_scores", "det_hits")
"""Columns of detection records from which the matches are rebuilt."""


def matches_from_records(frame: pd.DataFrame, n_thresholds: int) -> list[Matches]:
    """Matches of every image rebuilt from detection records.

    Args:
        frame: Detection records with the columns of `DETECTION_COLUMNS`.
        n_thresholds: Number of overlap thresholds, the length of every hit string.

    Returns:
        The matches of every row in order.

    Raises:
        ValueError: If a column of `DETECTION_COLUMNS` is missing, as in records written before
            they were stored.
    """
    missing = [c for c in DETECTION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"detection records lack {', '.join(missing)}")
    rebuilt = []
    for row in frame[list(DETECTION_COLUMNS)].itertuples(index=False):
        counts, classes, scores, hits = (_fields(value) for value in row)
        rebuilt.append(
            Matches(
                classes=np.array([int(c) for c in classes], dtype=np.int64),
                scores=np.array([float(v) for v in scores], dtype=np.float64),
                hits=np.array([[h == "1" for h in word] for word in hits], dtype=bool).reshape(
                    len(hits), n_thresholds
                ),
                targets=np.array([int(c) for c in counts], dtype=np.int64),
            )
        )
    return rebuilt


def _fields(value: object) -> list[str]:
    """Space separated fields of a cell, none for an empty cell read back as NaN."""
    return str(value).split() if isinstance(value, str) else []


def classification_records(
    spec: TaskSpec, probabilities: NDArray[np.float64], targets: Sequence[Target]
) -> pd.DataFrame:
    """Target, prediction and class probabilities of each image.

    Multilabel tasks predict every class of probability at least one half and list classes joined
    by `|`, counting an image as correct when all its classes are right. Other tasks predict the
    most probable class and also report its probability as the confidence.

    Args:
        spec: Task whose class names label the columns.
        probabilities: Class probabilities of shape (N, C).
        targets: Class index or indicator vector of each image.

    Returns:
        One row per image with the target, the prediction, whether it is correct, the confidence
        for single label tasks and one `p_` column per class.
    """
    names = [f"p_{name}" for name in spec.classes]
    frame = pd.DataFrame(probabilities, columns=names)
    if spec.kind is TaskKind.MULTILABEL:
        labels = np.stack([np.asarray(t) for t in targets])
        decided = probabilities >= 0.5
        frame.insert(0, "target", ["|".join(np.array(spec.classes)[row == 1]) for row in labels])
        frame.insert(1, "prediction", ["|".join(np.array(spec.classes)[row]) for row in decided])
        frame.insert(2, "correct", (decided == (labels == 1)).all(axis=1))
        return frame
    labels = np.asarray(targets, dtype=np.int64)
    predicted = probabilities.argmax(axis=1)
    frame.insert(0, "target", np.array(spec.classes)[labels])
    frame.insert(1, "prediction", np.array(spec.classes)[predicted])
    frame.insert(2, "correct", predicted == labels)
    frame.insert(3, "confidence", probabilities.max(axis=1))
    return frame


def write_records(
    directory: Path, spec: TaskSpec, model: str, records: Sequence[pd.DataFrame]
) -> Path:
    """Merge records into the file of the model, replacing earlier rows of the same settings.

    Earlier rows sharing a protocol and budget with the new records are dropped whatever their
    seed, so that a rerun with fewer seeds leaves no rows of the old run behind. The file is
    replaced atomically through a temporary file.

    Args:
        directory: Root of the per image results.
        spec: Task the records belong to.
        model: Name of the encoder, which names the file.
        records: Nonempty frames with the `protocol`, `budget` and `seed` columns.

    Returns:
        The path of the written CSV file.
    """
    path = directory / spec.id.replace("/", "__") / f"{model}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.concat(records, ignore_index=True)
    if path.exists():
        old = pd.read_csv(path)
        replaced = old.set_index(RUN_COLUMNS[:2]).index.isin(
            new.set_index(RUN_COLUMNS[:2]).index.unique()
        )
        new = pd.concat([old[~replaced], new], ignore_index=True)
    temporary = path.with_suffix(".tmp")
    new.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)
    return path
