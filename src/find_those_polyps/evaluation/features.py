"""Feature extraction with a frozen encoder."""

from collections.abc import Sequence
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from tqdm.auto import tqdm

from find_those_polyps.evaluation.store import FeatureStore
from find_those_polyps.models.encoder import Encoder, run
from find_those_polyps.tasks.schema import Sample, Target


class Features(BaseModel):
    """Frozen representations of one partition with the corresponding targets.

    Attributes:
        values: Features on the CPU, of shape (N, D) for global features in float32 or
            (N, D, h, w) for dense features in float16.
        targets: Target of each sample, a class index, an indicator vector, a mask or a table of
            boxes depending on the task kind.
        ids: Sample identifiers in the order of `values`, empty when unknown.
        paths: Canonical image paths relative to the data root in the order of `values`, empty
            when unknown.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    values: torch.Tensor
    targets: tuple[Target, ...]
    ids: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()

    def subset(self, indices: NDArray[np.intp]) -> Self:
        """Samples at the given positions, in the given order.

        Identifiers and paths are indexed like the values, so they must be filled for a nonempty
        selection.

        Args:
            indices: Positions into the partition, possibly repeated.

        Returns:
            A copy holding the selected values, targets, identifiers and paths.
        """
        index = torch.as_tensor(indices, dtype=torch.long)
        return self.model_copy(
            update={
                "values": self.values[index],
                "targets": tuple(self.targets[i] for i in indices),
                "ids": tuple(self.ids[i] for i in indices),
                "paths": tuple(self.paths[i] for i in indices),
            }
        )

    def labels(self) -> NDArray[np.int64]:
        """Class indices of single label targets.

        Returns:
            An integer array of shape (N,).
        """
        return np.asarray(self.targets, dtype=np.int64)

    def indicators(self, n_classes: int) -> NDArray[np.float64]:
        """Targets as an indicator matrix.

        Indicator vectors are stacked as they are and class indices are one hot encoded.

        Args:
            n_classes: Number of classes C.

        Returns:
            A float array of shape (N, C).
        """
        if self.targets and isinstance(self.targets[0], np.ndarray):
            return np.stack([np.asarray(t, dtype=np.float64) for t in self.targets])
        return np.eye(n_classes)[self.labels()]


def extract(
    encoder: Encoder,
    samples: Sequence[Sample],
    *,
    dense: bool,
    batch_size: int,
    device: torch.device,
    description: str = "",
    store: FeatureStore | None = None,
) -> Features:
    """Encode every sample, keeping global features in float32 and dense maps in float16.

    Samples may arrive in any order, and the result follows `samples.ids` when the sequence
    provides it, so that partitions keep their canonical order. Images whose features are in
    `store` are not encoded again, and fresh features are added to it. The encoder is put in
    evaluation mode and moved to `device`.

    Args:
        encoder: Frozen encoder applied to the images.
        samples: Samples of one partition, read once in iteration order.
        dense: Whether to extract dense feature maps instead of global embeddings.
        batch_size: Number of images encoded together.
        device: Device on which the encoder runs.
        description: Label of the progress bar.
        store: Features already encoded, read and extended in place when given.

    Returns:
        The features on the CPU with the targets, identifiers and paths of the samples.

    Raises:
        ValueError: If the encoder produces features of an unexpected shape.
    """
    (features,) = extract_many(
        [encoder],
        samples,
        dense=dense,
        batch_size=batch_size,
        device=device,
        description=description,
        stores=[store],
    )
    return features


@torch.inference_mode()
def extract_many(
    encoders: Sequence[Encoder],
    samples: Sequence[Sample],
    *,
    dense: bool,
    batch_size: int,
    device: torch.device,
    description: str = "",
    stores: Sequence[FeatureStore | None] | None = None,
) -> list[Features]:
    """Encode every sample with several encoders, reading and decoding each image once.

    Every image is handed to each encoder whose store lacks its features, and each encoder runs
    on its own batches, so that a streamed dataset is downloaded once for all of them. All
    encoders are put in evaluation mode and moved to `device` together. Ordering and precision
    follow `extract`.

    Args:
        encoders: Frozen encoders applied to the images.
        samples: Samples of one partition, read once in iteration order.
        dense: Whether to extract dense feature maps instead of global embeddings.
        batch_size: Number of images encoded together by each encoder.
        device: Device on which the encoders run.
        description: Label of the progress bar.
        stores: Features already encoded by each encoder, read and extended in place, or `None`
            to store nothing.

    Returns:
        The features of each encoder in the order of `encoders`, on the CPU, with the targets,
        identifiers and paths of the samples.

    Raises:
        ValueError: If an encoder produces features of an unexpected shape, or if the number of
            stores differs from the number of encoders.
    """
    stores = [None] * len(encoders) if stores is None else list(stores)
    if len(stores) != len(encoders):
        raise ValueError(f"{len(encoders)} encoders but {len(stores)} stores")
    for encoder in encoders:
        encoder.eval().to(device)
    rows: list[list[torch.Tensor | None]] = [[] for _ in encoders]
    images: list[list[NDArray[np.uint8]]] = [[] for _ in encoders]
    waiting: list[list[int]] = [[] for _ in encoders]
    targets: list[Target] = []
    ids: list[str] = []
    paths: list[str] = []

    def flush(index: int) -> None:
        """Encode the images waiting for one encoder, place their rows and store them."""
        output = run(encoders[index], images[index], device, dense=dense).detach().float().cpu()
        output = output.half() if dense else output
        for position, row in zip(waiting[index], output, strict=True):
            rows[index][position] = row
        store = stores[index]
        if store is not None:
            store.put([paths[position] for position in waiting[index]], output, dense)
        images[index].clear()
        waiting[index].clear()

    for sample in tqdm(samples, desc=description, leave=False):
        targets.append(sample.target)
        ids.append(sample.id)
        paths.append(sample.path)
        for index, store in enumerate(stores):
            cached = store.get(sample.path, dense) if store is not None else None
            rows[index].append(cached)
            if cached is None:
                images[index].append(sample.image)
                waiting[index].append(len(ids) - 1)
                if len(images[index]) == batch_size:
                    flush(index)
    for index in range(len(encoders)):
        if images[index]:
            flush(index)
    canonical = getattr(samples, "ids", None)
    order = None
    if canonical is not None and list(canonical) != ids:
        position = {sample_id: i for i, sample_id in enumerate(ids)}
        order = [position[sample_id] for sample_id in canonical]
        targets = [targets[i] for i in order]
        ids = [ids[i] for i in order]
        paths = [paths[i] for i in order]
    results = []
    for encoded in rows:
        values = torch.stack([row for row in encoded if row is not None])
        if order is not None:
            values = values[torch.as_tensor(order, dtype=torch.long)]
        results.append(
            Features(values=values, targets=tuple(targets), ids=tuple(ids), paths=tuple(paths))
        )
    return results
