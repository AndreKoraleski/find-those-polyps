"""Uniform access to the files of a dataset, local or streamed from its official host.

Names are POSIX paths relative to the dataset directory, laid out as after extracting every archive
of the dataset card. `LocalFiles` reads a directory, and `RemoteFiles` reads the official
archives with HTTP range requests, optionally caching each fetched file at its local path so that
later runs read from disk. `open_files` chooses between them.

Some archives nest tarballs whose members cannot be addressed by range. `expand` reads such a
tarball once, sequentially as it downloads, and places its files beside it as `tar` would.
"""

import io
import json
import tarfile
import tempfile
import zlib
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import IO

from find_those_polyps.data.remote import (
    TRANSIENT,
    ArchiveIndex,
    Member,
    MemberReader,
    RangeUnsupportedError,
    fetch_file,
    fetch_members,
    read_index,
    stream,
)
from find_those_polyps.logging import get_logger
from find_those_polyps.registry.schema import Access, DatasetCard, Download

log = get_logger(__name__)

COMPLETE = ".complete"
INDEX_DIR = ".index"
EXPANDED_DIR = ".expanded"
PARTIAL = ".part"
READ_AHEAD = 64
TARBALLS = (".tar", ".tar.gz", ".tgz")


def _listing_name(tarball: str) -> str:
    """File name of the JSON listing of an expanded tarball, with slashes as double underscores."""
    return tarball.replace("/", "__") + ".json"


def _unpack(tarball: str, handle: IO[bytes]) -> Iterator[tuple[str, bytes]]:
    """Read the regular files of a tarball sequentially, as a stream that never seeks.

    Args:
        tarball: Name of the tarball relative to the dataset directory, whose folder becomes the
            folder of its members.
        handle: Binary stream of the tarball, plain or compressed.

    Yields:
        The name of each file relative to the dataset directory with its content.

    Raises:
        ValueError: If a member path is absolute, climbs with `..` or holds a colon.
    """
    base = PurePosixPath(tarball).parent
    with tarfile.open(fileobj=handle, mode="r|*") as bundle:
        for info in bundle:
            extracted = bundle.extractfile(info) if info.isfile() else None
            if extracted is None:
                continue
            member = PurePosixPath(info.name.replace("\\", "/"))
            name = (base / member).as_posix().removeprefix("./")
            unsafe = ".." in member.parts or member.is_absolute() or ":" in info.name
            if unsafe:
                raise ValueError(f"unsafe member {info.name!r} in {tarball}")
            yield name, extracted.read()


class DatasetFiles(ABC):
    """Read only view of the files of one dataset.

    Args:
        key: Key of the dataset card.

    Attributes:
        key: Key of the dataset card.
    """

    def __init__(self, key: str) -> None:
        self.key = key

    @abstractmethod
    def available(self) -> bool:
        """Whether the files can be read.

        Returns:
            `True` when the view can serve the files of the dataset.
        """

    @abstractmethod
    def names(self) -> list[str]:
        """List every file name in storage order.

        Returns:
            POSIX paths relative to the dataset directory.
        """

    @abstractmethod
    def read(self, name: str) -> bytes:
        """Read the content of one file.

        Args:
            name: POSIX path relative to the dataset directory.

        Returns:
            The bytes of the file.
        """

    @abstractmethod
    def expand(self, tarball: str) -> list[str]:
        """Make the files of a nested tarball readable.

        Args:
            tarball: Name of the tarball.

        Returns:
            The names of its regular files, placed beside the tarball.
        """

    def _refresh(self) -> None:
        """Drop the cached name set and positions after the names have changed."""
        for attribute in ("name_set", "positions"):
            self.__dict__.pop(attribute, None)

    def fingerprint(self, name: str) -> str:
        """CRC32 and size of a file, which identify byte identical files.

        Args:
            name: Name of the file.

        Returns:
            The CRC32 as eight lower case hexadecimal digits, a colon and the size in bytes.
        """
        content = self.read(name)
        return f"{zlib.crc32(content):08x}:{len(content)}"

    def read_text(self, name: str) -> str:
        """Read a file as UTF-8 text, dropping a leading byte order mark.

        Args:
            name: Name of the file.

        Returns:
            The decoded text.
        """
        return self.read(name).decode("utf-8-sig")

    def exists(self, name: str) -> bool:
        """Whether a file of that name belongs to the dataset.

        Args:
            name: Name of the file.

        Returns:
            `True` when `name` is among `names`.
        """
        return name in self.name_set

    @cached_property
    def name_set(self) -> frozenset[str]:
        """Every file name, cached for membership tests until the names change."""
        return frozenset(self.names())

    def under(self, prefix: str, suffixes: frozenset[str] | None = None) -> list[str]:
        """List the names below a directory, optionally filtered by suffix.

        Args:
            prefix: Directory relative to the dataset directory, with or without a trailing slash.
                An empty prefix selects every name.
            suffixes: Lower case suffixes with their dot, for instance `.png`, matched against
                the lower cased suffix of each name. Every suffix is kept when `None`.

        Returns:
            The matching names in storage order.
        """
        start = prefix.rstrip("/") + "/" if prefix else ""
        return [
            name
            for name in self.names()
            if name.startswith(start)
            and (suffixes is None or Path(name).suffix.lower() in suffixes)
        ]

    def position(self, name: str) -> int:
        """Rank of a file in storage order, used to read samples sequentially.

        Args:
            name: Name of the file.

        Returns:
            The zero based rank of the file, zero for an unknown name.
        """
        return self.positions.get(name, 0)

    @cached_property
    def positions(self) -> dict[str, int]:
        """Rank of every file name in storage order, cached until the names change."""
        return {name: rank for rank, name in enumerate(self.names())}

    def stream(self, groups: Sequence[Sequence[str]]) -> Iterator[Mapping[str, bytes]]:
        """Read each group of names in order, ahead of consumption.

        Eight threads read the files of up to `READ_AHEAD` groups beyond the one being consumed.

        Args:
            groups: Names needed by each sample, in consumption order.

        Yields:
            The content of every name of each group by name, one mapping per group.
        """
        with ThreadPoolExecutor(8) as pool:
            pending = deque(pool.submit(self._read_group, g) for g in groups[:READ_AHEAD])
            for index in range(len(groups)):
                if index + READ_AHEAD < len(groups):
                    pending.append(pool.submit(self._read_group, groups[index + READ_AHEAD]))
                yield pending.popleft().result()

    def _read_group(self, names: Sequence[str]) -> dict[str, bytes]:
        """Content of every name of one group by name."""
        return {name: self.read(name) for name in names}


class LocalFiles(DatasetFiles):
    """Files of a dataset stored under a local directory.

    Every path with a component that starts with a dot is ignored, which hides the bookkeeping
    folders, the completion marker and hidden files. Files ending in `.part`, left by interrupted
    downloads, are ignored as well.

    Args:
        key: Key of the dataset card.
        root: Dataset directory.

    Attributes:
        key: Key of the dataset card.
        root: Dataset directory.
        stream_error: Why streaming the dataset failed when this view replaces a streamed one,
            `None` otherwise.
    """

    def __init__(self, key: str, root: Path) -> None:
        super().__init__(key)
        self.root = root
        self.stream_error: str | None = None

    def available(self) -> bool:
        """Whether the dataset directory exists.

        Returns:
            `True` when `root` is a directory.
        """
        return self.root.is_dir()

    def names(self) -> list[str]:
        """List every file below the dataset directory, except hidden and partial downloads.

        Returns:
            POSIX paths relative to `root` in sorted order, empty when the directory is missing.
        """
        if not self.root.is_dir():
            return []
        names = []
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root)
            hidden = any(part.startswith(".") for part in relative.parts)
            if path.is_file() and not hidden and path.suffix != PARTIAL:
                names.append(relative.as_posix())
        return sorted(names)

    def read(self, name: str) -> bytes:
        """Read one file from disk.

        Args:
            name: POSIX path relative to `root`.

        Returns:
            The bytes of the file.
        """
        return (self.root / name).read_bytes()

    def expand(self, tarball: str) -> list[str]:
        """Extract a nested tarball beside itself, once.

        The names are recorded in a JSON listing under the `.expanded` folder of `root`, which
        later calls return without reading the tarball again.

        Args:
            tarball: Name of the tarball relative to `root`.

        Returns:
            The names of its regular files.

        Raises:
            ValueError: If a member path escapes the dataset directory.
        """
        listing = self.root / EXPANDED_DIR / _listing_name(tarball)
        if listing.exists():
            names: list[str] = json.loads(listing.read_text(encoding="utf-8"))
            return names
        names = []
        with (self.root / tarball).open("rb") as handle:
            for name, content in _unpack(tarball, handle):
                path = self.root / name
                if not path.resolve().is_relative_to(self.root.resolve()):
                    raise ValueError(f"unsafe member {name!r} in {tarball}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                names.append(name)
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text(json.dumps(names), encoding="utf-8", newline="\n")
        self._refresh()
        return names


class RemoteFiles(DatasetFiles):
    """Files read from the official archives of an open dataset with range requests.

    Fetched files are written below `cache` at their local path when a cache is given, so that a
    dataset becomes a local copy as it is used. Archive indexes and the listings of expanded
    tarballs are kept under `cache / ".index"`. Without a cache, the files of expanded tarballs
    are held in memory and plain files are downloaded to a temporary directory, read and deleted.

    Args:
        card: Open dataset whose published files are read.
        cache: Dataset directory that receives the fetched files, or `None` to keep nothing on
            disk.
        window_bytes: Largest window of a streamed read in compressed bytes. Defaults to 64
            mebibytes.

    Attributes:
        key: Key of the dataset card.
        card: Dataset card.
        cache: Dataset directory that receives the fetched files, or `None`.
        window_bytes: Largest window of a streamed read in compressed bytes.
        index_dir: Folder of the cached archive indexes, `None` without a cache.
        memory: Files of expanded tarballs held in memory by name, used without a cache.
        expanded: Names of the files of every expanded tarball, by tarball name.
    """

    def __init__(self, card: DatasetCard, cache: Path | None, *, window_bytes: int = 64 << 20):
        super().__init__(card.key)
        self.card = card
        self.cache = cache
        self.window_bytes = window_bytes
        self.index_dir = cache / INDEX_DIR if cache is not None else None
        self.memory: dict[str, bytes] = {}
        self.expanded: dict[str, list[str]] = {}

    def available(self) -> bool:
        """Whether the dataset is open and publishes at least one file.

        Returns:
            `True` when the files can be read from the official host.
        """
        return self.card.access is Access.OPEN and bool(self.card.downloads)

    def _index(self, item: Download) -> ArchiveIndex:
        """Load the index of one archive, from the cache when it still matches the card pins.

        Args:
            item: Published archive of the dataset card.

        Returns:
            The index of the archive, saved to the cache when it was read from the host.

        Raises:
            ValueError: If the index read from the host differs from the pinned index or size.
        """
        path = None if self.index_dir is None else self.index_dir / f"{item.filename}.json"
        if path is not None and path.exists():
            cached = ArchiveIndex.load(path)
            pinned = (not item.index or cached.digest == item.index) and (
                item.size is None or cached.size == item.size
            )
            if pinned:
                return cached
            log.warning("cached archive index differs from the card", archive=item.filename)
        log.info("reading archive index", dataset=self.key, archive=item.filename)
        index = read_index(item.url)
        if item.index and index.digest != item.index:
            raise ValueError(f"{item.filename} changed on the server, index {index.digest}")
        if item.size is not None and index.size != item.size:
            raise ValueError(f"{item.filename} has {index.size} bytes, expected {item.size}")
        if path is not None:
            index.save(path)
        return index

    @cached_property
    def members(self) -> dict[str, tuple[str, Member | None]]:
        """Archive URL and member of every published name, with `None` as member for plain files.

        Plain files map to their own URL. The table is built once, reading every archive index.
        """
        table: dict[str, tuple[str, Member | None]] = {}
        for item in self.card.downloads:
            if not item.extract:
                table[item.filename] = (item.url, None)
                continue
            for member in self._index(item).members:
                table[member.name] = (item.url, member)
        return table

    def _listing(self, tarball: str) -> Path | None:
        """Path of the cached listing of an expanded tarball, `None` without a cache."""
        if self.cache is None:
            return None
        return self.cache / INDEX_DIR / EXPANDED_DIR / _listing_name(tarball)

    def names(self) -> list[str]:
        """List every published name, then the files of every expanded tarball.

        Tarballs whose listing is cached count as expanded, so their files are listed without
        reading them again.

        Returns:
            The published names in card and storage order, followed by the files of the
                expanded tarballs.
        """
        names = list(self.members)
        for tarball in names:
            listing = self._listing(tarball) if tarball.endswith(TARBALLS) else None
            if listing is not None and listing.exists() and tarball not in self.expanded:
                self.expanded[tarball] = json.loads(listing.read_text(encoding="utf-8"))
        return names + [name for files in self.expanded.values() for name in files]

    def _cached(self, name: str) -> Path | None:
        """Local path of a cached file, `None` when it is not on disk or there is no cache."""
        if self.cache is None:
            return None
        path = self.cache / name
        return path if path.exists() else None

    def _store(self, name: str, content: bytes) -> None:
        """Write a fetched file to the cache through a `.part` file, or do nothing without one.

        Args:
            name: Name of the file relative to the dataset directory.
            content: Bytes of the file.

        Raises:
            ValueError: If the name escapes the cache directory.
        """
        if self.cache is None:
            return
        path = self.cache / name
        if not path.resolve().is_relative_to(self.cache.resolve()):
            raise ValueError(f"unsafe member name {name!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + PARTIAL)
        partial.write_bytes(content)
        partial.replace(path)

    def _plain(self, name: str) -> bytes:
        """Download a plain published file whole, keeping it only when there is a cache.

        Without a cache the file is downloaded to a temporary directory, which is removed once
        the file has been read, so that the working directory is never touched.

        Args:
            name: Name of the file, which must be the file name of a published download.

        Returns:
            The bytes of the file.

        Raises:
            KeyError: If no published download has that file name.
        """
        item = next((item for item in self.card.downloads if item.filename == name), None)
        if item is None:
            raise KeyError(f"{name} is not a file of {self.key}")
        if self.cache is not None:
            return fetch_file(item, self.cache).read_bytes()
        with tempfile.TemporaryDirectory(prefix="polyps-") as directory:
            return fetch_file(item, Path(directory)).read_bytes()

    def read(self, name: str) -> bytes:
        """Read one file from memory, the cache or the official host.

        A cached archive member is checked against the size and CRC32 of the index and fetched
        again when it differs. A fetched member is written to the cache.

        Args:
            name: Name of the file relative to the dataset directory.

        Returns:
            The bytes of the file.

        Raises:
            KeyError: If the file is neither held, cached nor published.
        """
        if name in self.memory:
            return self.memory[name]
        cached = self._cached(name)
        url, member = self.members.get(name, ("", None))
        if cached is not None:
            content = cached.read_bytes()
            if member is None or (zlib.crc32(content), len(content)) == (member.crc, member.size):
                return content
            log.warning("cached file is corrupt, fetching it again", file=name)
        if member is None:
            return self._plain(name)
        content = fetch_members(url, [member])[name]
        self._store(name, content)
        return content

    def expand(self, tarball: str) -> list[str]:
        """Stream a nested tarball once and make its files readable.

        The tarball is decompressed while it downloads, without being stored, and its files go to
        the cache or, without one, to memory. With a cache their names are also recorded in a
        listing, so that later runs skip the download.

        Args:
            tarball: Name of the tarball, which must be a member of a published archive.

        Returns:
            The names of its regular files.

        Raises:
            KeyError: If no published archive holds `tarball`.
            ValueError: If `tarball` is a plain published file rather than an archive member, or
                one of its member paths is unsafe.
        """
        self.names()
        if tarball in self.expanded:
            return self.expanded[tarball]
        url, member = self.members[tarball]
        if member is None:
            raise ValueError(f"{tarball} is not an archive member")
        log.info("expanding nested archive", dataset=self.key, archive=tarball, size=member.size)
        names = []
        with MemberReader(url, member) as raw:
            for name, content in _unpack(tarball, io.BufferedReader(raw, 1 << 20)):
                if self.cache is None:
                    self.memory[name] = content
                else:
                    self._store(name, content)
                names.append(name)
        listing = self._listing(tarball)
        if listing is not None:
            listing.parent.mkdir(parents=True, exist_ok=True)
            listing.write_text(json.dumps(names), encoding="utf-8", newline="\n")
        self.expanded[tarball] = names
        self._refresh()
        return names

    def fingerprint(self, name: str) -> str:
        """CRC32 and size from the archive index when known, without downloading the file.

        Plain published files and files of expanded tarballs are read and hashed instead.

        Args:
            name: Name of the file.

        Returns:
            The CRC32 as eight lower case hexadecimal digits, a colon and the size in bytes.
        """
        self.names()
        _, member = self.members.get(name, ("", None))
        if member is None:
            return super().fingerprint(name)
        return f"{member.crc:08x}:{member.size}"

    def _pending(self, name: str) -> tuple[str, Member] | None:
        """Archive URL and member of a file still to download, `None` when held or not a member."""
        if name in self.memory or self._cached(name) is not None:
            return None
        url, member = self.members.get(name, ("", None))
        return None if member is None else (url, member)

    def stream(self, groups: Sequence[Sequence[str]]) -> Iterator[Mapping[str, bytes]]:
        """Read each group in order, downloading later windows while earlier ones are consumed.

        Archive members not yet held are fetched in windows of at most `window_bytes` compressed
        bytes and written to the cache. Every other name is read with `read`.

        Args:
            groups: Names needed by each sample, in consumption order.

        Yields:
            The content of every name of each group by name, one mapping per group.
        """
        missing = [[p for n in g if (p := self._pending(n)) is not None] for g in groups]
        fetched = stream(missing, window_bytes=self.window_bytes)
        for group in groups:
            contents = next(fetched)
            for name, content in contents.items():
                self._store(name, content)
            yield {name: contents[name] if name in contents else self.read(name) for name in group}


def open_files(
    card: DatasetCard, data_root: Path, *, stream: bool = True, cache: bool = True
) -> DatasetFiles:
    """Open a local copy of a dataset if complete or manually provided, else a streamed view.

    A directory holding a `.complete` marker was fetched in full, and a directory without the
    `.index` folder of a stream cache was provided by the user, so both are read locally.
    Otherwise open datasets are streamed and read through the cache directory. A dataset whose
    host ignores range requests or cannot be reached falls back to the local view, which may then
    be unavailable, and which records the error in `stream_error`.

    Args:
        card: Dataset to open.
        data_root: Directory that holds one directory per dataset key.
        stream: Whether an open dataset may be streamed from its host. Defaults to `True`.
        cache: Whether streamed files are written to the dataset directory. Defaults to `True`.

    Returns:
        A `LocalFiles` view of `data_root / card.key` or a `RemoteFiles` view of the official
            archives whose indexes are already read.

    Raises:
        ValueError: If an archive index read from the host differs from its pin on the card.
    """
    local = LocalFiles(card.key, data_root / card.key)
    remote_possible = stream and card.access is Access.OPEN and bool(card.downloads)
    complete = (local.root / COMPLETE).exists()
    manual = local.available() and not (local.root / INDEX_DIR).is_dir()
    if complete or manual or not remote_possible:
        return local
    try:
        files = RemoteFiles(card, local.root if cache else None)
        files.names()
    except (RangeUnsupportedError, *TRANSIENT) as error:
        log.warning("cannot stream dataset", dataset=card.key, error=str(error))
        local.stream_error = str(error)
        return local
    return files
