"""Range based access to the members of remote zip archives.

The central directory of an archive is read with a few HTTP range requests and kept as an index of
members with their offsets, sizes and CRC32 values. Members are then fetched in windows whose
neighbouring files are coalesced into shared spans, so that a window can be processed while the
next ones download. Long spans are split into pieces requested in parallel, since many hosts limit
the throughput of each connection. Every member is verified against the CRC32 recorded in the
index.
"""

import hashlib
import http.client
import io
import json
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Buffer, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from tqdm.auto import tqdm

from find_those_polyps.registry.schema import Download
from find_those_polyps.utils.hashing import canonical_sha256

USER_AGENT = "find-those-polyps"
RETRIES = 5
TIMEOUT = 60
BLOCK = 1 << 16
LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")
LOCAL_SIGNATURE = b"PK\x03\x04"
MAX_GAP = 1 << 20
MAX_RUN = 32 << 20
TAIL = 1 << 16
END_SIGNATURE = b"PK" + bytes([5, 6])
ZIP64_LOCATOR = b"PK" + bytes([6, 7])
CHUNK = 1 << 20
FIRST_WINDOW = 4 << 20
PIECE = 4 << 20
TRANSIENT = (OSError, urllib.error.URLError, http.client.HTTPException)


class RangeUnsupportedError(RuntimeError):
    """Raised when a server answers a range request without partial content."""


def _request(url: str, start: int, end: int) -> bytes:
    """Read a byte range of a remote file.

    Truncated responses, which some hosts send for large ranges, are kept and completed by a request
    for the remaining bytes. Requests that fail transiently or yield nothing are retried with waits
    of 2, 4, 8 and 16 seconds, and at most `RETRIES` attempts may fail over the whole range.

    Args:
        url: Address of the file.
        start: Offset of the first byte.
        end: Offset of the last byte, inclusive.

    Returns:
        The `end - start + 1` bytes of the range.

    Raises:
        RangeUnsupportedError: If the server does not answer with partial content.
        OSError: If no data arrives after every retry, or the last attempt fails with an
            operating system or URL error.
        http.client.HTTPException: If the last attempt fails with a malformed HTTP exchange.
    """
    parts: list[bytes] = []
    position = start
    failures = 0
    while position <= end:
        headers = {"User-Agent": USER_AGENT, "Range": f"bytes={position}-{end}"}
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                if response.status != 206:
                    raise RangeUnsupportedError(url)
                data: bytes = response.read()
        except http.client.IncompleteRead as error:
            data = error.partial
        except TRANSIENT:
            if failures == RETRIES - 1:
                raise
            data = b""
        if not data:
            failures += 1
            if failures == RETRIES:
                raise OSError(f"no data received from {url}")
            time.sleep(2**failures)
            continue
        parts.append(data[: end - position + 1])
        position += len(parts[-1])
    return b"".join(parts)


def checksum(path: Path, algorithm: str) -> str:
    """Hash a local file in chunks of one mebibyte.

    Args:
        path: File to hash.
        algorithm: Name of a `hashlib` algorithm, for instance `sha256`.

    Returns:
        The hexadecimal digest, without the algorithm prefix used on dataset cards.
    """
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, item: Download) -> bool:
    """Whether a local file exists and matches the checksum pin, when the card gives one."""
    algorithm, _, expected = item.checksum.partition(":")
    return path.exists() and (not expected or checksum(path, algorithm) == expected)


def _download(url: str, partial: Path, label: str) -> None:
    """Write a whole remote file to a local file with one request, truncating it first.

    Args:
        url: Address of the file.
        partial: Local file that receives the content.
        label: Description shown by the progress bar.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=label, leave=False) as bar:
            while chunk := response.read(CHUNK):
                out.write(chunk)
                bar.update(len(chunk))


def fetch_file(item: Download, directory: Path) -> Path:
    """Download a whole published file unless a verified copy exists, then verify it.

    The download is written to a `.part` file beside the target and renamed once complete. An
    attempt that fails transiently is repeated from the start, rewriting the `.part` file, after
    waits of 2, 4, 8 and 16 seconds, for at most `RETRIES` attempts. A file whose card gives no
    checksum is accepted as soon as it exists.

    Args:
        item: Published file of a dataset card.
        directory: Directory that receives the file under `item.filename`.

    Returns:
        The path of the verified file.

    Raises:
        ValueError: If the downloaded file does not match its checksum, in which case it is
            deleted.
        OSError: If the last attempt fails with an operating system or URL error.
        http.client.HTTPException: If the last attempt fails with a malformed HTTP exchange.
    """
    path = directory / item.filename
    if _verified(path, item):
        return path
    directory.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            _download(item.url, partial, item.filename)
            break
        except TRANSIENT:
            if attempt == RETRIES:
                raise
            time.sleep(2**attempt)
    partial.replace(path)
    if not _verified(path, item):
        path.unlink()
        raise ValueError(f"checksum mismatch for {item.filename}")
    return path


def remote_size(url: str) -> int:
    """Read the size of a remote file from the answer to a one byte range request.

    Args:
        url: Address of the file.

    Returns:
        The size of the file in bytes.

    Raises:
        RangeUnsupportedError: If the server does not answer with partial content.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        if response.status != 206:
            raise RangeUnsupportedError(url)
        content_range = response.headers.get("Content-Range", "")
    return int(content_range.rpartition("/")[2])


class RangeFile(io.RawIOBase):
    """Seekable read only view of a remote file backed by cached range requests.

    The file is read in aligned blocks of `BLOCK` bytes, each fetched once and kept in memory.

    Args:
        url: Address of the file.
        size: Size of the file in bytes, read from the server when `None`.

    Attributes:
        url: Address of the file.
        size: Size of the file in bytes.
        position: Offset of the next byte to read.
        blocks: Fetched blocks by block number.
    """

    def __init__(self, url: str, size: int | None = None) -> None:
        super().__init__()
        self.url = url
        self.size = remote_size(url) if size is None else size
        self.position = 0
        self.blocks: dict[int, bytes] = {}

    def readable(self) -> bool:
        """Whether the view can be read, which it always can.

        Returns:
            `True`.
        """
        return True

    def seekable(self) -> bool:
        """Whether the view supports random access, which it always does.

        Returns:
            `True`.
        """
        return True

    def tell(self) -> int:
        """Current read offset.

        Returns:
            The offset of the next byte to read.
        """
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the read offset without any request.

        Args:
            offset: Displacement in bytes.
            whence: Reference point, one of `io.SEEK_SET`, `io.SEEK_CUR` and `io.SEEK_END`.

        Returns:
            The new read offset.
        """
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def preload(self, start: int, end: int) -> None:
        """Fill the block cache over a byte range with one request.

        Nothing is requested when every block of the range is already cached.

        Args:
            start: Offset of the first byte.
            end: Offset just past the last byte, exclusive.
        """
        first, last = start // BLOCK, (end - 1) // BLOCK
        if all(index in self.blocks for index in range(first, last + 1)):
            return
        base = first * BLOCK
        data = _request(self.url, base, min((last + 1) * BLOCK, self.size) - 1)
        for index in range(first, last + 1):
            offset = index * BLOCK - base
            self.blocks[index] = data[offset : offset + BLOCK]

    def _block(self, index: int) -> bytes:
        """Content of block number `index`, fetched on first use."""
        if index not in self.blocks:
            start = index * BLOCK
            self.blocks[index] = _request(self.url, start, min(start + BLOCK, self.size) - 1)
        return self.blocks[index]

    def readinto(self, buffer: Buffer) -> int:
        """Read bytes from the current offset into a buffer, stopping at the end of the file.

        Args:
            buffer: Writable buffer filled from its start.

        Returns:
            The number of bytes written, zero at the end of the file.
        """
        view = memoryview(buffer).cast("B")
        end = min(self.position + len(view), self.size)
        written = 0
        while self.position < end:
            block = self._block(self.position // BLOCK)
            start = self.position % BLOCK
            piece = block[start : start + end - self.position]
            view[written : written + len(piece)] = piece
            written += len(piece)
            self.position += len(piece)
        return written


class Member(BaseModel):
    """Location and integrity data of one file inside an archive.

    Attributes:
        name: Path of the file inside the archive.
        offset: Archive offset of its local header.
        end: Archive offset just past its stored data, which is the local header of the next
            member or the start of the central directory.
        compressed: Size of its stored data in bytes.
        size: Size of its content in bytes.
        crc: CRC32 of its content.
        method: Zip compression method, stored or deflated for members that can be read.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    offset: int
    end: int
    compressed: int
    size: int
    crc: int
    method: int


class ArchiveIndex(BaseModel):
    """Members of a remote zip archive in storage order.

    Attributes:
        url: Address of the archive.
        size: Size of the archive in bytes.
        members: Files of the archive, without directory entries, by increasing offset.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    size: int
    members: tuple[Member, ...]

    @property
    def digest(self) -> str:
        """Hash of member names, sizes and CRC32 values, independent of the host.

        The members are sorted by name, so neither the address nor the storage order nor the
        compression of the archive affects the hash.

        Returns:
            The SHA256 of the canonical listing, prefixed by `sha256` and a colon as on dataset
                cards.
        """
        listing = [[m.name, m.size, m.crc] for m in sorted(self.members, key=lambda m: m.name)]
        return "sha256:" + canonical_sha256(listing)

    def save(self, path: Path) -> None:
        """Write the index as compact JSON, creating the parent directories.

        Args:
            path: Destination file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.model_dump(mode="json"), separators=(",", ":"))
        path.write_text(text + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: Path) -> "ArchiveIndex":
        """Read an index written by `save`.

        Args:
            path: JSON file of the index.

        Returns:
            The validated index.
        """
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def _directory_span(handle: RangeFile) -> tuple[int, int]:
    """Locate the central directory from the end of central directory records.

    The last `TAIL` bytes of the archive are fetched in one request, and the ZIP64 record is read
    with one more when the classic record overflows.

    Args:
        handle: Range view of the archive.

    Returns:
        The archive offsets of the start of the central directory and just past its end.

    Raises:
        ValueError: If no end of central directory record is found.
    """
    tail_start = max(0, handle.size - TAIL)
    handle.preload(tail_start, handle.size)
    handle.seek(tail_start)
    tail = handle.read(handle.size - tail_start)
    record = tail.rfind(END_SIGNATURE)
    if record < 0:
        raise ValueError(f"{handle.url} is not a zip archive")
    size, start = struct.unpack_from("<II", tail, record + 12)
    locator = tail.rfind(ZIP64_LOCATOR, 0, record)
    if 0xFFFFFFFF in (size, start) and locator >= 0:
        (record64,) = struct.unpack_from("<Q", tail, locator + 8)
        handle.preload(record64, record64 + 56)
        handle.seek(record64 + 40)
        size, start = struct.unpack("<QQ", handle.read(16))
    return start, start + size


def read_index(url: str) -> ArchiveIndex:
    """Read the index of a remote zip archive from its central directory.

    One request reads the size of the archive, one or two read the end of central directory
    records, and at most one more reads the whole central directory, so no member is downloaded.

    Args:
        url: Address of the archive, which must support range requests.

    Returns:
        The index of every file member, directory entries excluded.

    Raises:
        RangeUnsupportedError: If the server ignores range requests.
        ValueError: If the file is not a zip archive.
    """
    handle = RangeFile(url)
    handle.preload(*_directory_span(handle))
    with zipfile.ZipFile(io.BufferedReader(handle, BLOCK)) as bundle:
        infos = sorted(bundle.infolist(), key=lambda info: info.header_offset)
        directory_start = bundle.start_dir
    ends = [info.header_offset for info in infos[1:]] + [directory_start]
    members = tuple(
        Member(
            name=info.filename,
            offset=info.header_offset,
            end=end,
            compressed=info.compress_size,
            size=info.file_size,
            crc=info.CRC,
            method=info.compress_type,
        )
        for info, end in zip(infos, ends, strict=True)
        if not info.is_dir()
    )
    return ArchiveIndex(url=url, size=handle.size, members=members)


def decode(member: Member, buffer: bytes, base: int) -> bytes:
    """Extract and verify the content of a member from a buffer of archive bytes.

    Args:
        member: Member to decode.
        buffer: Archive bytes that cover the local header and the stored data of the member.
        base: Archive offset of the first byte of `buffer`.

    Returns:
        The decompressed content of the member.

    Raises:
        ValueError: If the local header is malformed or the content does not match the size and
            CRC32 of the index.
        NotImplementedError: If the member is neither stored nor deflated.
    """
    start = member.offset - base
    header = LOCAL_HEADER.unpack_from(buffer, start)
    if header[0] != LOCAL_SIGNATURE:
        raise ValueError(f"bad local header for {member.name}")
    data_start = start + LOCAL_HEADER.size + header[9] + header[10]
    raw = buffer[data_start : data_start + member.compressed]
    if member.method == zipfile.ZIP_STORED:
        content = raw
    elif member.method == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(raw, -zlib.MAX_WBITS)
    else:
        raise NotImplementedError(f"compression method {member.method} of {member.name}")
    if len(content) != member.size or zlib.crc32(content) != member.crc:
        raise ValueError(f"CRC mismatch for {member.name}")
    return content


class MemberReader(io.RawIOBase):
    """Decompressed content of one member read progressively from a single range request.

    Large members such as nested tarballs are thereby consumed while they download, and an
    interrupted transfer resumes with a new range request at the first missing byte. The CRC32 of
    the content is verified once the member has been read to its end. The local header is read on
    construction, which opens the request.

    Args:
        url: Address of the archive.
        member: Member to read.

    Attributes:
        url: Address of the archive.
        member: Member being read.
        position: Archive offset of the next stored byte to receive.
        response: Open HTTP response, or `None` between requests.
        remaining: Stored bytes of the member not yet received.
        inflater: Decompressor of a deflated member, `None` for a stored one.
        pending: Decompressed bytes not yet returned.
        crc: Running CRC32 of the decompressed content.
        length: Number of decompressed bytes produced so far.

    Raises:
        NotImplementedError: If the member is neither stored nor deflated.
        ValueError: If the local header of the member is malformed.
        RangeUnsupportedError: If the server ignores range requests.
    """

    def __init__(self, url: str, member: Member) -> None:
        super().__init__()
        self.url = url
        self.member = member
        if member.method not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise NotImplementedError(f"compression method {member.method} of {member.name}")
        self.position = member.offset
        self.response: Any = None
        header = LOCAL_HEADER.unpack(self._exactly(LOCAL_HEADER.size))
        if header[0] != LOCAL_SIGNATURE:
            raise ValueError(f"bad local header for {member.name}")
        self._exactly(header[9] + header[10])
        self.remaining = member.compressed
        self.inflater = zlib.decompressobj(-zlib.MAX_WBITS) if member.method else None
        self.pending = b""
        self.crc = 0
        self.length = 0

    def _open(self) -> None:
        """Open a range request from the current position to the end of the member storage."""
        headers = {
            "User-Agent": USER_AGENT,
            "Range": f"bytes={self.position}-{self.member.end - 1}",
        }
        request = urllib.request.Request(self.url, headers=headers)
        self.response = urllib.request.urlopen(request, timeout=TIMEOUT)
        if self.response.status != 206:
            raise RangeUnsupportedError(self.url)

    def _exactly(self, size: int) -> bytes:
        """Receive the next stored bytes of the member, resuming after interruptions.

        Each interruption closes the response and reopens the request at the first missing byte
        after waits of 2, 4, 8 and 16 seconds, for at most `RETRIES` failures per call.

        Args:
            size: Number of bytes to receive.

        Returns:
            Exactly `size` bytes.
        """
        parts: list[bytes] = []
        missing = size
        failures = 0
        while missing:
            try:
                if self.response is None:
                    self._open()
                part: bytes = self.response.read(missing)
                if not part:
                    raise OSError(f"connection closed while reading {self.member.name}")
            except TRANSIENT:
                failures += 1
                if failures == RETRIES:
                    raise
                if self.response is not None:
                    self.response.close()
                self.response = None
                time.sleep(2**failures)
                continue
            parts.append(part)
            missing -= len(part)
            self.position += len(part)
        return b"".join(parts)

    def readable(self) -> bool:
        """Whether the reader can be read, which it always can.

        Returns:
            `True`.
        """
        return True

    def readinto(self, buffer: Buffer) -> int:
        """Decompress the next bytes of the member into a buffer.

        Stored data is received in chunks of `CHUNK` bytes as the decompressed bytes run out.

        Args:
            buffer: Writable buffer filled from its start.

        Returns:
            The number of bytes written, zero once the whole member has been returned.

        Raises:
            ValueError: If the complete content does not match the size and CRC32 of the index.
        """
        while not self.pending and self.remaining:
            raw = self._exactly(min(CHUNK, self.remaining))
            self.remaining -= len(raw)
            self.pending = self.inflater.decompress(raw) if self.inflater else raw
            if not self.remaining and self.inflater is not None:
                self.pending += self.inflater.flush()
            self.crc = zlib.crc32(self.pending, self.crc)
            self.length += len(self.pending)
            if not self.remaining and (self.crc, self.length) != (
                self.member.crc,
                self.member.size,
            ):
                raise ValueError(f"CRC mismatch for {self.member.name}")
        view = memoryview(buffer).cast("B")
        size = min(len(view), len(self.pending))
        view[:size] = self.pending[:size]
        self.pending = self.pending[size:]
        return size

    def close(self) -> None:
        """Close the open HTTP response, if any, then the reader."""
        if self.response is not None:
            self.response.close()
        super().close()


def runs(members: Iterable[Member]) -> list[list[Member]]:
    """Group members into runs of nearby storage that one request can cover.

    Members are sorted by offset, and a member joins the current run when the gap before it is at
    most `MAX_GAP` bytes and the run then spans at most `MAX_RUN` bytes.

    Args:
        members: Members of one archive.

    Returns:
        The runs in storage order, each sorted by offset.
    """
    grouped: list[list[Member]] = []
    for member in sorted(members, key=lambda m: m.offset):
        if grouped:
            run = grouped[-1]
            gap = member.offset - run[-1].end
            if gap <= MAX_GAP and member.end - run[0].offset <= MAX_RUN:
                run.append(member)
                continue
        grouped.append([member])
    return grouped


def pieces(start: int, stop: int, size: int | None = None) -> list[tuple[int, int]]:
    """Split a byte span into consecutive pieces of at most `size` bytes.

    Args:
        start: Offset of the first byte.
        stop: Offset just past the last byte.
        size: Largest piece in bytes, `PIECE` when `None`.

    Returns:
        The first and last offset, inclusive, of every piece in order.
    """
    size = PIECE if size is None else size
    return [(offset, min(offset + size, stop) - 1) for offset in range(start, stop, size)]


def fetch_run(url: str, run: Sequence[Member]) -> dict[str, bytes]:
    """Read a run of members with a single range request.

    Args:
        url: Address of the archive.
        run: Members sorted by offset, as produced by `runs`.

    Returns:
        The verified content of every member by name.
    """
    base = run[0].offset
    buffer = _request(url, base, run[-1].end - 1)
    return {member.name: decode(member, buffer, base) for member in run}


def fetch_members(url: str, members: Iterable[Member]) -> dict[str, bytes]:
    """Read arbitrary members of one archive, coalescing neighbours into shared requests.

    Args:
        url: Address of the archive.
        members: Members to read, in any order.

    Returns:
        The verified content of every member by name.
    """
    contents: dict[str, bytes] = {}
    for run in runs(members):
        contents |= fetch_run(url, run)
    return contents


def windows(
    groups: Sequence[Sequence[tuple[str, Member]]], window_bytes: int
) -> list[list[Sequence[tuple[str, Member]]]]:
    """Batch consecutive groups by compressed size, starting small so that processing starts early.

    A window closes once its compressed size reaches its limit, so every window but the last holds
    at least that many bytes. The first limit is `FIRST_WINDOW` bytes and each following one
    doubles, both capped at `window_bytes`.

    Args:
        groups: Pairs of archive URL and member needed by each sample, in consumption order.
        window_bytes: Largest limit of a window in compressed bytes.

    Returns:
        The windows in order, each a list of consecutive groups.
    """
    batches: list[list[Sequence[tuple[str, Member]]]] = []
    limit = total = min(FIRST_WINDOW, window_bytes)
    for group in groups:
        if total >= limit:
            if batches:
                limit = min(2 * limit, window_bytes)
            batches.append([])
            total = 0
        batches[-1].append(group)
        total += sum(member.compressed for _, member in group)
    return batches


def stream(
    groups: Sequence[Sequence[tuple[str, Member]]],
    *,
    window_bytes: int = 64 << 20,
    lookahead: int = 2,
    workers: int = 8,
) -> Iterator[dict[str, bytes]]:
    """Read the members of each group in order, with later windows downloading in the background.

    Each group pairs archive URLs with the members one sample needs. Groups are batched into windows
    by `windows`, at most `lookahead` windows beyond the one being consumed are in flight, and each
    window is fetched as coalesced runs whose pieces of at most `PIECE` bytes are requested by
    `workers` threads at once.

    Args:
        groups: Pairs of archive URL and member needed by each sample, in consumption order.
        window_bytes: Largest window in compressed bytes. Defaults to 64 mebibytes.
        lookahead: Number of windows fetched ahead of the one being consumed. Defaults to 2.
        workers: Number of threads, and so of parallel connections, that run the range
            requests. Defaults to 8.

    Yields:
        The verified content of the members of each group by member name, one mapping per group.
    """
    batches = windows(groups, window_bytes)

    def fetch_window(batch: list[Sequence[tuple[str, Member]]]) -> dict[str, bytes]:
        """Contents of every member of one window, its pieces fetched on the worker threads."""
        by_url: dict[str, list[Member]] = {}
        for group in batch:
            for url, member in group:
                by_url.setdefault(url, []).append(member)
        jobs = [(url, run) for url, members in by_url.items() for run in runs(members)]
        spans = [(url, span) for url, run in jobs for span in pieces(run[0].offset, run[-1].end)]
        fetched = iter(pool.map(lambda job: _request(job[0], *job[1]), spans))
        contents: dict[str, bytes] = {}
        for _, run in jobs:
            count = len(pieces(run[0].offset, run[-1].end))
            buffer = b"".join(next(fetched) for _ in range(count))
            base = run[0].offset
            contents |= {member.name: decode(member, buffer, base) for member in run}
        return contents

    with ThreadPoolExecutor(workers) as pool, ThreadPoolExecutor(lookahead + 1) as planner:
        pending: list[Future[dict[str, bytes]]] = []
        position = 0
        for index, batch in enumerate(batches):
            while position < len(batches) and position <= index + lookahead:
                pending.append(planner.submit(fetch_window, batches[position]))
                position += 1
            contents = pending[index].result()
            pending[index] = Future()
            for group in batch:
                yield {member.name: contents[member.name] for _, member in group}
