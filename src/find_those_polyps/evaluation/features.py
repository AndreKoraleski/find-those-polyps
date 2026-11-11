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


@torch.inference_mode()
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
    encoder.eval().to(device)
    rows: list[torch.Tensor | None] = []
    targets: list[Target] = []
    ids: list[str] = []
    paths: list[str] = []
    images: list[NDArray[np.uint8]] = []
    waiting: list[int] = []

    def flush() -> None:
        """Encode the waiting images, place their rows and add them to the store."""
        output = run(encoder, images, device, dense=dense).detach().float().cpu()
        output = output.half() if dense else output
        for position, row in zip(waiting, output, strict=True):
            rows[position] = row
        if store is not None:
            store.put([paths[position] for position in waiting], output, dense)
        images.clear()
        waiting.clear()

    for sample in tqdm(samples, desc=description, leave=False):
        cached = store.get(sample.path, dense) if store is not None else None
        rows.append(cached)
        targets.append(sample.target)
        ids.append(sample.id)
        paths.append(sample.path)
        if cached is None:
            images.append(sample.image)
            waiting.append(len(rows) - 1)
            if len(images) == batch_size:
                flush()
    if images:
        flush()
    values = torch.stack([row for row in rows if row is not None])
    canonical = getattr(samples, "ids", None)
    if canonical is not None and list(canonical) != ids:
        position = {sample_id: i for i, sample_id in enumerate(ids)}
        order = [position[sample_id] for sample_id in canonical]
        values = values[torch.as_tensor(order, dtype=torch.long)]
        targets = [targets[i] for i in order]
        ids = [ids[i] for i in order]
        paths = [paths[i] for i in order]
    return Features(values=values, targets=tuple(targets), ids=tuple(ids), paths=tuple(paths))
