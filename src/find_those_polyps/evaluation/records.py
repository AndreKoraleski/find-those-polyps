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
    """Target and detection counts of each image with its matches at an overlap of one half.

    Args:
        matches: Matching outcome of every test image.

    Returns:
        One row per image with the number of targets, the number of kept detections, the number
        of detections matched at the first overlap threshold, 0.5, and the highest detection
        score, NaN when there is no detection.
    """
    return pd.DataFrame(
        {
            "targets": [int(m.targets.sum()) for m in matches],
            "detections": [len(m.scores) for m in matches],
            "matched_50": [int(m.hits[:, 0].sum()) for m in matches],
            "top_score": [
                float(m.scores.max()) if len(m.scores) else float("nan") for m in matches
            ],
        }
    )


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
