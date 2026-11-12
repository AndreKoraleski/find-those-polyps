import numpy as np
import torch
from conftest import MeanColour

from find_those_polyps.evaluation import extract
from find_those_polyps.evaluation.features import extract_many
from find_those_polyps.evaluation.store import FeatureStore
from find_those_polyps.tasks import Split, Task


def test_extract_global_and_dense(colour_task: Task) -> None:
    samples = colour_task.samples(Split.TEST)
    device = torch.device("cpu")
    global_features = extract(MeanColour(), samples, dense=False, batch_size=7, device=device)
    dense_features = extract(MeanColour(), samples, dense=True, batch_size=7, device=device)
    assert global_features.values.shape == (len(samples), 3)
    assert global_features.values.dtype == torch.float32
    assert dense_features.values.shape == (len(samples), 3, 8, 8)
    assert dense_features.values.dtype == torch.float16
    assert global_features.ids == colour_task.split.test
    assert all(path.startswith("colour/") for path in global_features.paths)


def test_subset_and_target_views(colour_task: Task) -> None:
    samples = colour_task.samples(Split.TRAIN)
    features = extract(
        MeanColour(), samples, dense=False, batch_size=64, device=torch.device("cpu")
    )
    subset = features.subset(np.array([0, 2]))
    assert subset.ids == (features.ids[0], features.ids[2])
    assert subset.values.shape == (2, 3)
    assert features.indicators(3).sum(axis=1).tolist() == [1.0] * len(samples)
    assert features.labels().tolist() == [s.target for s in samples]


def test_preprocessing_ahead_of_the_device_gives_identical_features(colour_task: Task) -> None:
    class Smaller(MeanColour):
        name = "smaller-colour"
        image_size = 16

    samples = colour_task.samples(Split.TEST)
    device = torch.device("cpu")
    encoders = [MeanColour(), Smaller()]
    serial = extract_many(encoders, samples, dense=True, batch_size=5, device=device)
    partial = FeatureStore()
    first = extract(
        encoders[0], samples[:7], dense=True, batch_size=5, device=device, store=partial
    )
    assert len(first.values) == 7
    parallel = extract_many(
        encoders,
        samples,
        dense=True,
        batch_size=5,
        device=device,
        stores=[partial, None],
        workers=3,
        prefetch=1,
    )
    for alone, together in zip(serial, parallel, strict=True):
        assert torch.equal(alone.values, together.values)
        assert alone.ids == together.ids and alone.paths == together.paths
