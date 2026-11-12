import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn

from find_those_polyps.evaluation import Features
from find_those_polyps.evaluation.protocols.base import Draw, Protocol
from find_those_polyps.evaluation.protocols.detection import LinearDetection
from find_those_polyps.evaluation.protocols.heads import StackedHeads
from find_those_polyps.evaluation.protocols.segmentation import LinearSegmentation
from find_those_polyps.tasks import TaskKind, TaskSpec

CPU = torch.device("cpu")


def _spec(kind: TaskKind, primary: str) -> TaskSpec:
    return TaskSpec(
        id="dense/squares",
        description="Squares.",
        dataset="squares",
        kind=kind,
        classes=("background", "square"),
        primary_metric=primary,
        loader="memory",
        splits="squares.json",
    )


def _squares(n: int, seed: int, detect: bool) -> Features:
    """Feature maps marking one square each, with its mask or its box as the target."""
    rng = np.random.default_rng(seed)
    values = torch.as_tensor(rng.normal(size=(n, 6, 8, 8)), dtype=torch.float32)
    targets: list[NDArray[np.uint8] | NDArray[np.float64]] = []
    for index in range(n):
        side = int(rng.integers(2, 5))
        top, left = (int(v) for v in rng.integers(0, 8 - side + 1, 2))
        values[index, :2, top : top + side, left : left + side] += 3.0
        if detect:
            targets.append(np.array([[1, left, top, left + side, top + side]]) / [1, 8, 8, 8, 8])
        else:
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[2 * top : 2 * (top + side), 2 * left : 2 * (left + side)] = 1
            targets.append(mask)
    ids = tuple(f"{seed}-{index}" for index in range(n))
    return Features(values=values.half(), targets=tuple(targets), ids=ids, paths=ids)


@pytest.mark.parametrize("kernel", [1, 3])
def test_stacked_heads_compute_every_head_apart(kernel: int) -> None:
    torch.manual_seed(0)
    heads = [
        nn.Sequential(nn.BatchNorm2d(5), nn.Conv2d(5, 3, kernel, padding=kernel // 2))
        for _ in range(4)
    ]
    stacked = StackedHeads(heads)
    values = torch.randn(6, 4, 5, 7, 7)
    outputs = stacked(values)
    for index, head in enumerate(heads):
        assert torch.allclose(outputs[:, index], head(values[:, index]), atol=1e-5)
        member = stacked.member(index)
        for (key, value), expected in zip(
            member.state_dict().items(), head.state_dict().values(), strict=True
        ):
            assert torch.allclose(value, expected), key


def test_members_draw_nothing_from_the_generator() -> None:
    stacked = StackedHeads([nn.Sequential(nn.BatchNorm2d(5), nn.Conv2d(5, 3, 1))])
    state = torch.random.get_rng_state()
    stacked.member(0)
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize("detect", [False, True])
def test_repeats_trained_together_match_repeats_trained_apart(detect: bool) -> None:
    train, val, test = (_squares(n, seed, detect) for n, seed in ((24, 0), (8, 1), (6, 2)))
    spec = _spec(TaskKind.DETECTION, "box_map") if detect else _spec(TaskKind.SEGMENTATION, "dice")
    probe: Protocol = (
        LinearDetection(steps=60, eval_every=20, min_steps=20, patience=1, lr=0.05)
        if detect
        else LinearSegmentation(steps=60, eval_every=20, min_steps=20, patience=1, lr=0.05)
    )
    rng = np.random.default_rng(3)
    draws = [
        Draw(0, np.arange(24)),
        Draw(1, np.sort(rng.choice(24, 10, replace=False)), np.array([0, 2, 5, 7])),
        Draw(2, np.sort(rng.choice(24, 16, replace=False))),
    ]
    together = [list(p) for p in probe.predict_runs(train, val, test, spec, draws, device=CPU)]
    for draw, predictions in zip(draws, together, strict=True):
        subset, held_out = draw.partitions(train, val)
        apart = list(probe.predict(subset, held_out, test, spec, seed=draw.seed, device=CPU))
        for got, expected in zip(predictions, apart, strict=True):
            assert isinstance(got, np.ndarray) and isinstance(expected, np.ndarray)
            assert got.shape == expected.shape
            assert np.allclose(got, expected, atol=2e-3 if detect else 1e-4)
