"""A task instantiated on its dataset files and canonical split."""

from collections.abc import Iterator, Sequence
from typing import overload

from pydantic import BaseModel, ConfigDict

from find_those_polyps.tasks.loaders.base import Loader
from find_those_polyps.tasks.schema import Sample, Split, TaskSpec
from find_those_polyps.tasks.splits import SplitFile


class LazySamples(Sequence[Sample]):
    """Sequence of samples read on access.

    Indexing returns samples in the order of `ids`. Iteration streams them in the storage order of
    the loader, which is faster for remote archives, so consumers that keep results must order them
    by `ids`.

    Args:
        loader: Reader of the dataset.
        ids: Identifiers of the samples in the sequence.
    """

    def __init__(self, loader: Loader, ids: Sequence[str]) -> None:
        self._loader = loader
        self._ids = ids

    @property
    def ids(self) -> Sequence[str]:
        """Identifiers of the samples in index order."""
        return self._ids

    def __len__(self) -> int:
        """Number of samples."""
        return len(self._ids)

    @overload
    def __getitem__(self, index: int) -> Sample: ...
    @overload
    def __getitem__(self, index: slice) -> "LazySamples": ...
    def __getitem__(self, index: int | slice) -> "Sample | LazySamples":
        """Read one sample, or select several without reading them.

        Args:
            index: Position of a sample, or a slice of positions.

        Returns:
            The sample at `index`, or a lazy sequence over the sliced identifiers.
        """
        if isinstance(index, slice):
            return LazySamples(self._loader, self._ids[index])
        return self._loader.load(self._ids[index])

    def __iter__(self) -> Iterator[Sample]:
        """Stream every sample once, in the storage order of the loader."""
        return self._loader.stream(self._ids)


class Task(BaseModel):
    """Task definition bound to a loader and its canonical split.

    Attributes:
        spec: Declarative definition of the task.
        loader: Reader of the dataset the task is defined on.
        split: Canonical partition of the sample identifiers.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    spec: TaskSpec
    loader: Loader
    split: SplitFile

    def samples(self, split: Split) -> LazySamples:
        """Samples of one partition.

        Args:
            split: Partition to read.

        Returns:
            Lazy sequence over the identifiers of the partition, in the order of the split file.
        """
        return LazySamples(self.loader, self.split.ids(split))
