"""Torch dataset that reads curated corpora while training.

`CorpusDataset` is an iterable dataset over the frames of one or several corpora. While some unit
is not cached yet, an epoch visits each unit once, in an order drawn from the seed and the epoch,
and the workers of a data loader share the units among themselves, so that several archives
download at once. A unit that is not cached is streamed from its host, or downloaded whole, as its
reader is set up, and its frames leave through a shuffle buffer, since the frames of a unit arrive
together. Once every unit is cached, an epoch draws the frames of all units in one random order.
"""

import io
import random
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from PIL import Image, UnidentifiedImageError
from torch.utils.data import IterableDataset, get_worker_info

from find_those_polyps.data.remote import HostUnreachableError
from find_those_polyps.logging import get_logger
from find_those_polyps.pretraining.corpora import Archive
from find_those_polyps.pretraining.reader import CorpusReader, Frame

log = get_logger(__name__)


def shard[T](items: Sequence[T], worker: int, workers: int) -> list[T]:
    """Items that one worker of a data loader reads.

    Args:
        items: Every item of the epoch, in the same order for all workers.
        worker: Index of the worker, from zero.
        workers: Number of workers.

    Returns:
        Every item whose rank is `worker` modulo `workers`.
    """
    return list(items[worker::workers])


class CorpusDataset(IterableDataset[Any]):
    """Frames of curated corpora, decoded and transformed, for self supervised training.

    Call `set_epoch` before every epoch so that the order of the units changes. A streamed unit
    whose host cannot be read after every retry is skipped for the epoch, which is logged, and
    training goes on. An archive that no longer matches its pin raises.

    Args:
        readers: Readers of the corpora to draw from.
        transform: Function applied to each decoded RGB image, the identity when `None`.
        shuffle_buffer: Number of frames held to shuffle the stream, at most one to disable.
            Defaults to 256.
        seed: Seed of the unit order and of the shuffle buffer. Defaults to zero.

    Attributes:
        readers: Readers of the corpora.
        transform: Function applied to each decoded image, or `None`.
        shuffle_buffer: Number of frames held to shuffle the stream.
        seed: Seed of the unit order and of the shuffle buffer.
        epoch: Epoch the next iteration belongs to.
    """

    def __init__(
        self,
        readers: Sequence[CorpusReader],
        transform: Callable[[Image.Image], Any] | None = None,
        *,
        shuffle_buffer: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.readers = tuple(readers)
        self.transform = transform
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Choose the epoch of the next iteration, which decides the order of the units.

        Args:
            epoch: Index of the epoch, from zero.
        """
        self.epoch = epoch

    def units(self) -> list[tuple[CorpusReader, Archive]]:
        """Every unit that may be read with its reader.

        Returns:
            The units of each reader in reader and card order.
        """
        return [(reader, archive) for reader in self.readers for archive in reader.units()]

    def expected(self) -> int | None:
        """Number of frames of one epoch.

        Returns:
            The sum over the units, or `None` when a card does not give the length of a unit.
        """
        counts = [reader.expected(archive) for reader, archive in self.units()]
        return None if any(count is None for count in counts) else sum(filter(None, counts))

    def _frames(self, units: Sequence[tuple[CorpusReader, Archive]]) -> Iterator[Frame]:
        """Frames of the given units in order, skipping a streamed unit whose host fails."""
        for reader, archive in units:
            try:
                yield from reader.frames(archive)
            except HostUnreachableError as error:
                log.error(
                    "unit skipped this epoch",
                    corpus=reader.card.key,
                    unit=archive.unit,
                    error=str(error),
                )

    def _decode(self, frame: Frame) -> Any:
        """Decode a frame to RGB and transform it, or `None` when it is not an image."""
        try:
            image = Image.open(io.BytesIO(frame.content)).convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            log.warning("frame skipped", corpus=frame.corpus, frame=frame.name, error=str(error))
            return None
        return image if self.transform is None else self.transform(image)

    def __iter__(self) -> Iterator[Any]:
        """Transformed frames of one epoch, for the worker that iterates.

        Yields:
            The output of `transform` for each decodable frame of the units of this worker.
        """
        units = self.units()
        order = random.Random(f"{self.seed}-{self.epoch}")
        info = get_worker_info()
        worker, workers = (0, 1) if info is None else (info.id, info.num_workers)
        if all(reader.cached(archive) for reader, archive in units):
            files = [
                (reader.card.key, archive.unit, index, path)
                for reader, archive in units
                for index, path in reader.paths(archive)
            ]
            order.shuffle(files)
            for key, unit, index, path in shard(files, worker, workers):
                decoded = self._decode(Frame(key, unit, index, path.name, path.read_bytes()))
                if decoded is not None:
                    yield decoded
            return
        order.shuffle(units)
        draw = random.Random(f"{self.seed}-{self.epoch}-{worker}")
        buffer: list[Frame] = []
        for frame in self._frames(shard(units, worker, workers)):
            if len(buffer) < self.shuffle_buffer:
                buffer.append(frame)
                continue
            if buffer:
                slot = draw.randrange(len(buffer))
                frame, buffer[slot] = buffer[slot], frame
            decoded = self._decode(frame)
            if decoded is not None:
                yield decoded
        draw.shuffle(buffer)
        for frame in buffer:
            decoded = self._decode(frame)
            if decoded is not None:
                yield decoded
