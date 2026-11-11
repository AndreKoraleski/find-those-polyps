"""Dataset readers and their registry."""

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from find_those_polyps.tasks.schema import Sample

if TYPE_CHECKING:
    from find_those_polyps.data.files import DatasetFiles

DECODE_WORKERS = 8
DECODE_AHEAD = 64


class Loader(ABC):
    """Reader for one dataset whose files are served by `files`.

    Loaders read data the user obtains under the dataset's own licence, either from a local copy or
    streamed from the official host, and the benchmark never redistributes images.

    Args:
        files: Files of the dataset.
        **options: Options that no subclass consumed, kept in `options`.

    Attributes:
        files: Files of the dataset.
        options: Remaining keyword options.
    """

    def __init__(self, files: "DatasetFiles", **options: Any) -> None:
        self.files = files
        self.options = options

    def available(self) -> bool:
        """Whether the dataset can be read.

        Returns:
            Whether the files of the dataset are present or reachable.
        """
        return self.files.available()

    @abstractmethod
    def ids(self) -> list[str]:
        """Identifiers of all samples, used to construct split files.

        Returns:
            Sorted sample identifiers.
        """

    @abstractmethod
    def groups(self) -> dict[str, str]:
        """Patient, video or centre of each sample.

        Returns:
            Group of every sample keyed by identifier.
        """

    @abstractmethod
    def load(self, sample_id: str) -> Sample:
        """Read one sample.

        Args:
            sample_id: Identifier of the sample.

        Returns:
            The decoded sample.
        """

    def stream(self, ids: Sequence[str]) -> Iterator[Sample]:
        """Samples of `ids` in an order convenient for reading, each exactly once.

        The base implementation loads them one by one in the given order.

        Args:
            ids: Identifiers of the samples to read.

        Returns:
            Iterator over the samples.
        """
        return (self.load(sample_id) for sample_id in ids)

    def labels(self) -> dict[str, int] | None:
        """Class of each sample for stratified splitting.

        Returns:
            Class index of every sample keyed by identifier, or `None` when the task does not have
            one label per image.
        """
        return None

    def official_test(self) -> tuple[list[str], str] | None:
        """Identifiers of the canonical test partition and its source, if one is published.

        Returns:
            Sorted test identifiers with a description of their source, or `None` when the
            dataset publishes no test partition.
        """
        return None


class FileLoader(Loader):
    """Loader whose samples are decoded from named files of the dataset.

    Subclasses state the files each sample needs and decode their contents. Streaming reads samples
    in storage order so that remote archives are fetched in large sequential windows, and decodes
    them in `DECODE_WORKERS` parallel threads up to `DECODE_AHEAD` samples ahead of consumption.
    """

    @abstractmethod
    def members(self, sample_id: str) -> tuple[str, ...]:
        """Names of the files holding one sample, its image first.

        Args:
            sample_id: Identifier of the sample.

        Returns:
            File names relative to the dataset root.
        """

    @abstractmethod
    def decode(self, sample_id: str, contents: Mapping[str, bytes]) -> Sample:
        """Build a sample from the contents of its files.

        Args:
            sample_id: Identifier of the sample.
            contents: Bytes of every file named by `members`, keyed by file name.

        Returns:
            The decoded sample.
        """

    def path(self, name: str) -> str:
        """Canonical path of a file relative to the data root.

        Args:
            name: File name relative to the dataset root.

        Returns:
            The name prefixed by the dataset key.
        """
        return f"{self.files.key}/{name}"

    def load(self, sample_id: str) -> Sample:
        """Read the files of one sample and decode them.

        Args:
            sample_id: Identifier of the sample.

        Returns:
            The decoded sample.
        """
        names = self.members(sample_id)
        return self.decode(sample_id, {name: self.files.read(name) for name in names})

    def stream(self, ids: Sequence[str]) -> Iterator[Sample]:
        """Samples of `ids` in the storage order of their images, decoded in parallel.

        Args:
            ids: Identifiers of the samples to read.

        Yields:
            Each sample once, ordered by the storage position of its first file.
        """
        ordered = sorted(ids, key=lambda i: self.files.position(self.members(i)[0]))
        contents = self.files.stream([self.members(i) for i in ordered])
        with ThreadPoolExecutor(DECODE_WORKERS) as pool:
            pending: deque[Future[Sample]] = deque()
            for sample_id, content in zip(ordered, contents, strict=True):
                pending.append(pool.submit(self.decode, sample_id, content))
                if len(pending) >= DECODE_AHEAD:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()


LOADERS: dict[str, type[Loader]] = {}


def register_loader(key: str) -> Callable[[type[Loader]], type[Loader]]:
    """Class decorator registering a loader under `key`.

    Args:
        key: Name by which task definitions select the loader.

    Returns:
        Decorator that records the class in `LOADERS` and returns it unchanged. It raises
        `ValueError` when `key` is already registered.
    """

    def decorator(cls: type[Loader]) -> type[Loader]:
        """Record `cls` in `LOADERS` under `key`.

        Args:
            cls: Loader class to register.

        Returns:
            The class unchanged.

        Raises:
            ValueError: If `key` is already registered.
        """
        if key in LOADERS:
            raise ValueError(f"loader {key!r} is already registered")
        LOADERS[key] = cls
        return cls

    return decorator
