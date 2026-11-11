import numpy as np
import pytest
import torch
from pydantic import ValidationError

from find_those_polyps.evaluation import Features
from find_those_polyps.evaluation.protocols.detection import LinearDetection, assign
from find_those_polyps.metrics import score
from find_those_polyps.tasks import TaskKind, TaskSpec

CPU = torch.device("cpu")
SPEC = TaskSpec(
    id="det/squares",
    description="Bright squares.",
    dataset="squares",
    kind=TaskKind.DETECTION,
    classes=("background", "square"),
    primary_metric="box_map",
    loader="memory",
    splits="squares.json",
)


def _squares(n: int, seed: int, size: int = 8) -> Features:
    """Feature maps of one square each, encoding at every cell inside it the log distances to its
    sides, so that a pointwise linear head can recover the box exactly.
    """
    rng = np.random.default_rng(seed)
    values = torch.zeros(n, 6, size, size)
    targets = []
    for index in range(n):
        side = int(rng.integers(2, 5))
        top, left = (int(v) for v in rng.integers(0, size - side + 1, 2))
        for row in range(top, top + side):
            for column in range(left, left + side):
                x, y = column + 0.5, row + 0.5
                sides = [x - left, y - top, left + side - x, top + side - y]
                values[index, 0, row, column] = 1.0
                values[index, 1:5, row, column] = torch.tensor(sides).log()
        values[index, 5] = 1 - values[index, 0]
        box = np.array([[1, left, top, left + side, top + side]], dtype=np.float64)
        box[:, 1:] /= size
        targets.append(box)
    return Features(values=values.half(), targets=tuple(targets))


def test_every_box_gets_a_cell_and_the_smallest_box_wins() -> None:
    boxes = np.array([[1, 0.0, 0.0, 1.0, 1.0], [2, 0.3, 0.3, 0.7, 0.7], [1, 0.9, 0.9, 0.95, 0.95]])
    classes, assigned, positive = assign(boxes, 2, 4, 4)
    assert positive.all()
    assert classes[1, 1, 1] == 1 and classes[0, 1, 1] == 0
    assert torch.allclose(assigned[:, 1, 1], torch.tensor([0.3, 0.3, 0.7, 0.7]))
    assert torch.allclose(assigned[:, 3, 3], torch.tensor([0.9, 0.9, 0.95, 0.95]))


def test_linear_detection_localizes_squares() -> None:
    train, val, test = _squares(64, 0), _squares(16, 1), _squares(32, 2)
    protocol = LinearDetection(steps=400, eval_every=100, lr=0.05)
    predictions = protocol.predict(train, val, test, SPEC, seed=0, device=CPU)
    assert len(predictions) == len(test.targets)
    assert all(p.shape[1] == 6 for p in predictions)
    metrics = score(TaskKind.DETECTION, predictions, test.targets, 2)
    assert metrics["box_map"] > 0.9


def test_even_kernels_are_rejected() -> None:
    assert LinearDetection(kernel=3).kernel == 3
    with pytest.raises(ValidationError, match="kernel must be odd"):
        LinearDetection(kernel=2)
