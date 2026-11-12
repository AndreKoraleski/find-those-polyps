from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from support import blobs, spec_for

from find_those_polyps.evaluation import KNN, PROTOCOLS, Features, LinearProbe
from find_those_polyps.evaluation.protocols import detection, segmentation
from find_those_polyps.evaluation.protocols.base import Protocol, better
from find_those_polyps.evaluation.protocols.zero_shot import KMeansClustering
from find_those_polyps.metrics import Predictions, score
from find_those_polyps.tasks import TaskKind, TaskSpec

CPU = torch.device("cpu")


def test_protocol_hash_tracks_hyperparameters() -> None:
    assert KNN().hash == KNN().hash
    assert KNN(k=5).hash != KNN().hash
    assert set(PROTOCOLS) == {"knn", "linear", "linear_seg", "linear_det", "retrieval", "kmeans"}


def test_better_handles_direction_and_nan() -> None:
    assert better(0.9, 0.8, "auroc")
    assert better(0.1, 0.2, "mae")
    assert better(0.1, float("nan"), "auroc")
    assert not better(float("nan"), 0.1, "auroc")


def test_knn_probabilities_are_normalized() -> None:
    train, val, test = blobs()
    probabilities = KNN(k=3).predict(train, val, test, spec_for(2), seed=0, device=CPU)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert (probabilities.argmax(axis=1) == test.labels()).mean() == 1.0


def test_linear_probe_selects_on_validation() -> None:
    train, val, test = blobs()
    probe = LinearProbe(c_grid=(1e-6, 1.0))
    probabilities = probe.predict(train, val, test, spec_for(2), seed=0, device=CPU)
    assert (probabilities.argmax(axis=1) == test.labels()).mean() == 1.0


def test_linear_probe_with_absent_class() -> None:
    train, val, test = blobs()
    single = Features(values=train.values[train.labels() == 0], targets=(0,) * 10)
    probabilities = LinearProbe().predict(single, val, test, spec_for(3), seed=0, device=CPU)
    assert probabilities.shape == (len(test.targets), 3)


def test_kmeans_handles_fewer_images_than_classes() -> None:
    train, val, test = blobs(n=4)
    metrics, records = KMeansClustering().measure(train, val, test, spec_for(6), seed=0, device=CPU)
    assert len(records) == 4
    assert 0 <= metrics["nmi"] <= 1


def test_direct_protocol_hash_tracks_the_primary_metric() -> None:
    class ByNmi(KMeansClustering):
        primary: ClassVar[str] = "nmi"

    assert ByNmi().hash != KMeansClustering().hash


def _dense(n: int, detect: bool) -> Features:
    """Random dense features with a square mask, or its box, as the target of every image."""
    mask = np.zeros((8, 8), dtype=np.int64)
    mask[2:6, 2:6] = 1
    target: NDArray[np.int64] | NDArray[np.float64] = mask
    if detect:
        target = np.array([[1, 0.25, 0.25, 0.75, 0.75]])
    return Features(values=torch.randn(n, 3, 4, 4), targets=(target,) * n)


@pytest.mark.parametrize("detect", [False, True])
def test_dense_probes_stop_once_validation_stalls(
    detect: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = "box_map" if detect else "dice"
    evaluated: list[int] = []

    def constant(
        kind: TaskKind,
        predictions: Predictions,
        targets: Sequence[object],
        n_classes: int,
        names: Sequence[str] | None = None,
    ) -> dict[str, float]:
        evaluated.append(len(targets))
        return {primary: 0.5}

    def device_dice(
        head: torch.nn.Module,
        values: torch.Tensor,
        masks: Sequence[torch.Tensor],
        device: torch.device,
        **options: object,
    ) -> float:
        evaluated.append(len(masks))
        return 0.5

    if detect:
        monkeypatch.setattr(detection, "score", constant)
    else:
        monkeypatch.setattr(segmentation, "_device_dice", device_dice)
    spec = TaskSpec(
        id="dense/squares",
        description="Squares.",
        dataset="squares",
        kind=TaskKind.DETECTION if detect else TaskKind.SEGMENTATION,
        classes=("background", "square"),
        primary_metric=primary,
        loader="memory",
        splits="squares.json",
    )
    train, val, test = _dense(8, detect), _dense(4, detect), _dense(4, detect)
    probe: Protocol = (
        detection.LinearDetection(steps=1000, eval_every=10, patience=3, min_steps=50)
        if detect
        else segmentation.LinearSegmentation(steps=1000, eval_every=10, patience=3, min_steps=50)
    )
    predictions = probe.predict(train, val, test, spec, seed=0, device=CPU)
    assert len(list(predictions)) == 4
    assert evaluated == [4] * 5


@pytest.mark.parametrize(
    ("kind", "n_classes"),
    [
        (TaskKind.SEGMENTATION, 2),
        (TaskKind.SEGMENTATION, 4),
        (TaskKind.MULTILABEL_SEGMENTATION, 3),
    ],
)
def test_validation_dice_on_the_device_equals_the_host_score(
    kind: TaskKind, n_classes: int
) -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    multilabel = kind is TaskKind.MULTILABEL_SEGMENTATION
    head = torch.nn.Conv2d(6, n_classes, 1)
    values = torch.randn(11, 6, 4, 5)
    targets = []
    for index in range(11):
        shape = (n_classes, 23, 31) if multilabel else (23, 31)
        high = 2 if multilabel else n_classes
        mask = rng.integers(0, high, shape).astype(np.uint8)
        if index == 3:
            mask[:] = 0
        targets.append(mask)
    maps = segmentation._probability_maps(head, values, targets, CPU, multilabel=multilabel)
    expected = score(kind, maps, tuple(targets), n_classes, ["dice"])["dice"]
    masks = [torch.from_numpy(target) for target in targets]
    dice = segmentation._device_dice(
        head, values, masks, CPU, n_classes=n_classes, multilabel=multilabel
    )
    assert dice == expected
