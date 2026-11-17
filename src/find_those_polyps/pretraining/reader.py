"""Frames of a curated corpus, read from a local cache, a local archive or the official host.

Each archive of a corpus holds the frames of one unit, a video or a patient, in no particular
order. A unit is read from the first source that can serve it.

1. The cache under `data_root / "corpora" / key / unit`, when an earlier pass kept its frames.
2. A whole archive under `data_root / ".archives" / key`, downloaded first when `download` is set.
3. The official host, decompressed while it downloads, when `stream` is set.

Frames are subsampled by their index within the unit, to about `fps` frames per second of
recording, and the kept frames are written to the cache when `cache` is set, so that a corpus
becomes a local copy as training reads it. An archive read from the host or from a kept file is
hashed as it passes and compared with the checksum pinned on the card once it has been read whole.
A mismatch discards the frames cached from it and raises. Held out units are never read.
"""

import hashlib
import io
import json
import shutil
import tarfile
from collections.abc import Generator, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from tqdm.auto import tqdm

from find_those_polyps.data.download import ARCHIVES
from find_those_polyps.data.paths import default_data_root
from find_those_polyps.data.remote import CHUNK, ResumableReader, fetch_file
from find_those_polyps.logging import get_logger
from find_those_polyps.pretraining.corpora import Archive, CorpusCard
from find_those_polyps.registry.schema import Download

log = get_logger(__name__)

CORPORA_DIR = "corpora"
"""Folder of the corpus caches below the data root."""

MARKERS = ".complete"
"""Folder, below the cache of a corpus, of the markers of the units read whole."""

PARTIAL = ".part"


class CorpusUnavailableError(RuntimeError):
    """Raised when a unit is neither cached nor allowed to be downloaded or streamed."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One frame of a corpus.

    Attributes:
        corpus: Key of the corpus.
        unit: Video or patient the frame belongs to.
        index: Index of the frame within its unit.
        name: Base name of the frame file.
        content: Encoded bytes of the image.
    """

    corpus: str
    unit: str
    index: int
    name: str
    content: bytes


class _Hashing:
    """Binary stream that hashes every byte read through it.

    Args:
        handle: Stream to read from.
        algorithm: Name of a `hashlib` algorithm.
    """

    def __init__(self, handle: IO[bytes], algorithm: str) -> None:
        self.handle = handle
        self.digest = hashlib.new(algorithm)
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        """Read and hash up to `size` bytes.

        Args:
            size: Largest number of bytes to read, everything left when negative.

        Returns:
            The bytes read, empty at the end of the stream.
        """
        data = self.handle.read(size) or b""
        self.digest.update(data)
        self.count += len(data)
        return data

    def drain(self) -> None:
        """Read the stream to its end, so that the hash covers the whole file."""
        while self.read(CHUNK):
            pass


class CorpusReader:
    """Reads the units of one corpus, keeping about `fps` frames per second of recording.

    Args:
        card: Corpus to read.
        data_root: Directory holding the cache and the archives, `FIND_THOSE_POLYPS_DATA` or
            `data` when `None`.
        fps: Frames kept per second of recording, or `None` to keep every frame. Defaults to one.
        stream: Whether a unit that is not local may be read from its host while it downloads.
            Defaults to `True`.
        cache: Whether the kept frames are written below the data root. Defaults to `True`.
        download: Whether the archive of a unit that is not cached is downloaded whole and
            verified before it is read, instead of streamed. Defaults to `False`.
        keep_archives: Whether an archive downloaded whole stays on disk once read. Defaults to
            `False`.

    Attributes:
        card: Corpus being read.
        fps: Frames kept per second of recording, or `None`.
        stream: Whether units may be streamed from their host.
        cache: Whether kept frames are written to disk.
        download: Whether archives are downloaded whole first.
        keep_archives: Whether downloaded archives stay on disk.
        root: Cache directory of the corpus.
        archives_dir: Directory of the archives downloaded whole.
    """

    def __init__(
        self,
        card: CorpusCard,
        data_root: Path | None = None,
        *,
        fps: float | None = 1.0,
        stream: bool = True,
        cache: bool = True,
        download: bool = False,
        keep_archives: bool = False,
    ) -> None:
        if fps is not None and fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        root = default_data_root() if data_root is None else data_root
        self.card = card
        self.fps = fps
        self.stream = stream
        self.cache = cache
        self.download = download
        self.keep_archives = keep_archives
        self.root = root / CORPORA_DIR / card.key
        self.archives_dir = root / ARCHIVES / card.key

    def units(self) -> tuple[Archive, ...]:
        """Archives that may be read, one per unit.

        Returns:
            The archives of the card whose unit is not held out.
        """
        return self.card.selected()

    def expected(self, archive: Archive) -> int | None:
        """Number of frames the subsampling keeps from a unit.

        Args:
            archive: Archive of the unit.

        Returns:
            The count of frame indexes from zero that are multiples of the stride, or `None` when
            the card does not give the length of the unit.
        """
        if archive.frames is None:
            return None
        return -(-archive.frames // archive.stride(self.fps))

    def _marker(self, archive: Archive) -> Path:
        """Path of the marker recording that a unit was read whole."""
        return self.root / MARKERS / f"{archive.unit}.json"

    def cached(self, archive: Archive) -> bool:
        """Whether the cache holds every frame the subsampling keeps from a unit.

        A unit cached at a stride that divides the requested one holds a superset of the
        requested frames, so it counts as cached.

        Args:
            archive: Archive of the unit.

        Returns:
            `True` when a marker written for the pinned archive covers the requested stride.
        """
        marker = self._marker(archive)
        if not marker.exists():
            return False
        record = json.loads(marker.read_text(encoding="utf-8"))
        stride = archive.stride(self.fps)
        return record.get("checksum") == archive.checksum and any(
            stride % done == 0 for done in record.get("strides", [])
        )

    def state(self, archive: Archive) -> str:
        """Where the next read of a unit comes from.

        Args:
            archive: Archive of the unit.

        Returns:
            `cached`, `archive` when a whole archive is on disk, or `remote`.
        """
        if self.cached(archive):
            return "cached"
        return "archive" if (self.archives_dir / archive.filename).exists() else "remote"

    def _mark(self, archive: Archive, stride: int) -> None:
        """Record that every frame of a unit at a stride is in the cache."""
        marker = self._marker(archive)
        strides = {stride}
        if marker.exists():
            record = json.loads(marker.read_text(encoding="utf-8"))
            if record.get("checksum") == archive.checksum:
                strides |= set(record.get("strides", []))
        marker.parent.mkdir(parents=True, exist_ok=True)
        record = {"checksum": archive.checksum, "strides": sorted(strides)}
        partial = marker.with_name(marker.name + PARTIAL)
        partial.write_text(json.dumps(record) + "\n", encoding="utf-8", newline="\n")
        partial.replace(marker)

    def _store(self, archive: Archive, name: str, content: bytes) -> None:
        """Write one frame to the cache through a `.part` file."""
        path = self.root / archive.unit / name
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + PARTIAL)
        partial.write_bytes(content)
        partial.replace(path)

    def _discard(self, archive: Archive) -> None:
        """Remove the cached frames and the marker of a unit."""
        shutil.rmtree(self.root / archive.unit, ignore_errors=True)
        self._marker(archive).unlink(missing_ok=True)

    def paths(self, archive: Archive) -> list[tuple[int, Path]]:
        """Cached frame files of a unit that the subsampling keeps.

        Args:
            archive: Archive of the unit, which must be cached.

        Returns:
            The index and path of every kept frame, by increasing index.
        """
        stride = archive.stride(self.fps)
        kept = []
        for path in (self.root / archive.unit).iterdir():
            index = self.card.frame_index(path.name)
            if index is not None and index % stride == 0:
                kept.append((index, path))
        return sorted(kept)

    def _local(self, archive: Archive) -> Iterator[Frame]:
        """Frames of a cached unit, by increasing index."""
        for index, path in self.paths(archive):
            yield Frame(self.card.key, archive.unit, index, path.name, path.read_bytes())

    def _unpack(
        self, archive: Archive, handle: IO[bytes], stride: int, *, verify: bool
    ) -> Iterator[Frame]:
        """Frames of a tarball read as a stream that never seeks.

        Args:
            archive: Archive being read.
            handle: Binary stream of the tarball.
            stride: Gap between kept frame indexes.
            verify: Whether to compare the bytes read with the pinned checksum at the end.

        Yields:
            The kept frames in archive order.

        Raises:
            ValueError: If the archive read does not match its pinned checksum or size, in which
                case the frames cached from it are discarded.
        """
        algorithm, _, expected = archive.checksum.partition(":")
        hashing = _Hashing(handle, algorithm)
        with tarfile.open(fileobj=hashing, mode="r|*") as bundle:  # type: ignore[call-overload]
            for info in bundle:
                index = self.card.frame_index(info.name) if info.isfile() else None
                if index is None or index % stride:
                    continue
                extracted = bundle.extractfile(info)
                if extracted is None:
                    continue
                name = info.name.rsplit("/", 1)[-1]
                content = extracted.read()
                if self.cache:
                    self._store(archive, name, content)
                yield Frame(self.card.key, archive.unit, index, name, content)
        if verify:
            hashing.drain()
            if hashing.count != archive.size or hashing.digest.hexdigest() != expected:
                self._discard(archive)
                raise ValueError(
                    f"{archive.filename} does not match its pin, {hashing.count} bytes read"
                )
        if self.cache:
            self._mark(archive, stride)

    def frames(self, archive: Archive) -> Generator[Frame]:
        """Kept frames of one unit, from the cache, a whole archive or the host.

        Args:
            archive: Archive of the unit, which must not be held out.

        Yields:
            The frames whose index is a multiple of the stride of the unit, by increasing index
            from the cache and in archive order otherwise.

        Raises:
            ValueError: If the unit is held out, or the archive does not match its pin.
            CorpusUnavailableError: If the unit is not local and may be neither downloaded nor
                streamed.
            HostUnreachableError: If a streamed archive could not be read after every retry.
        """
        if archive.unit in self.card.excluded:
            raise ValueError(f"unit {archive.unit} of {self.card.key} is held out")
        stride = archive.stride(self.fps)
        if self.cached(archive):
            yield from self._local(archive)
            return
        path = self.archives_dir / archive.filename
        if self.download:
            item = Download(
                url=archive.url,
                filename=archive.filename,
                checksum=archive.checksum,
                size=archive.size,
                extract=False,
            )
            log.info("downloading archive", corpus=self.card.key, archive=archive.filename)
            path = fetch_file(item, self.archives_dir)
            with path.open("rb") as handle:
                yield from self._unpack(archive, handle, stride, verify=False)
            if not self.keep_archives:
                path.unlink()
        elif path.exists():
            with path.open("rb") as handle:
                yield from self._unpack(archive, handle, stride, verify=True)
        elif self.stream:
            log.info("streaming archive", corpus=self.card.key, archive=archive.filename)
            with io.BufferedReader(ResumableReader(archive.url, archive.size), CHUNK) as raw:
                yield from self._unpack(archive, raw, stride, verify=True)
        else:
            raise CorpusUnavailableError(
                f"unit {archive.unit} of {self.card.key} is not under {self.root} and neither "
                "streaming nor downloading is allowed"
            )

    def __iter__(self) -> Iterator[Frame]:
        """Kept frames of every unit that may be read, unit after unit.

        Yields:
            The frames of each unit in card order.
        """
        for archive in self.units():
            yield from self.frames(archive)


def fetch(reader: CorpusReader, units: Sequence[str] | None = None, workers: int = 4) -> int:
    """Read units of a corpus once, so that the cache holds their kept frames.

    Units already cached are left alone. The others are read as the reader is set up, streamed
    or downloaded whole, several at a time.

    Args:
        reader: Reader of the corpus, which must write to the cache.
        units: Units to fetch, every unit that may be read when `None`.
        workers: Number of units read at once. Defaults to four.

    Returns:
        The number of units read.

    Raises:
        ValueError: If the reader does not cache, or a unit is unknown or held out.
    """
    if not reader.cache:
        raise ValueError("fetching needs a reader that writes to the cache")
    known = {archive.unit: archive for archive in reader.units()}
    missing = sorted(set(units or ()) - set(known))
    if missing:
        raise ValueError(f"{reader.card.key} has no readable units {missing}")
    wanted = [known[unit] for unit in dict.fromkeys(units)] if units else list(known.values())
    pending = [archive for archive in wanted if not reader.cached(archive)]

    def read(archive: Archive) -> int:
        return sum(1 for _ in reader.frames(archive))

    with ThreadPoolExecutor(max(1, workers)) as pool:
        counts = tqdm(pool.map(read, pending), total=len(pending), desc=reader.card.key)
        for archive, count in zip(pending, counts, strict=True):
            log.info("unit cached", corpus=reader.card.key, unit=archive.unit, frames=count)
    return len(pending)
