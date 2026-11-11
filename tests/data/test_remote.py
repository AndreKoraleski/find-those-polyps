import hashlib
import http.client
import http.server
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

from find_those_polyps.data import remote
from find_those_polyps.data.remote import (
    Member,
    RangeFile,
    RangeUnsupportedError,
    fetch_file,
    fetch_members,
    read_index,
    remote_size,
    runs,
    stream,
    windows,
)
from find_those_polyps.registry import Download

Server = tuple[str, Path, http.server.ThreadingHTTPServer]


def _publish(directory: Path, name: str = "set.zip", n: int = 12) -> dict[str, bytes]:
    contents = {f"images/{i:02d}.bin": bytes([i]) * (100 + 37 * i) for i in range(n)}
    contents["labels.csv"] = b"name,label\n" + b"".join(
        f"{i:02d},{i % 3}\n".encode() for i in range(n)
    )
    with zipfile.ZipFile(directory / name, "w") as bundle:
        bundle.mkdir("images")
        for index, (member, data) in enumerate(contents.items()):
            method = zipfile.ZIP_DEFLATED if index % 2 else zipfile.ZIP_STORED
            bundle.writestr(member, data, compress_type=method)
    return contents


def test_range_file_reads_like_a_local_file(range_server: Server) -> None:
    url, directory, _ = range_server
    (directory / "blob").write_bytes(bytes(range(256)) * 600)
    assert remote_size(f"{url}/blob") == 153600
    handle = RangeFile(f"{url}/blob")
    handle.seek(70000)
    assert handle.read(4) == bytes([112, 113, 114, 115])
    handle.seek(-2, 2)
    assert handle.read(10) == bytes([254, 255])


def test_index_lists_members_and_pins_content(range_server: Server) -> None:
    url, directory, _ = range_server
    contents = _publish(directory)
    index = read_index(f"{url}/set.zip")
    assert {m.name for m in index.members} == set(contents)
    assert index.size == (directory / "set.zip").stat().st_size
    assert index.digest == read_index(f"{url}/set.zip").digest
    _publish(directory, n=13)
    assert read_index(f"{url}/set.zip").digest != index.digest


def test_members_are_fetched_in_coalesced_runs(range_server: Server) -> None:
    url, directory, _ = range_server
    contents = _publish(directory)
    index = read_index(f"{url}/set.zip")
    assert len(runs(index.members)) == 1
    fetched = fetch_members(f"{url}/set.zip", index.members[3:7])
    assert fetched == {m.name: contents[m.name] for m in index.members[3:7]}


def test_corrupted_members_are_rejected(range_server: Server) -> None:
    url, directory, _ = range_server
    _publish(directory)
    index = read_index(f"{url}/set.zip")
    member = next(m for m in index.members if m.method == zipfile.ZIP_STORED)
    raw = bytearray((directory / "set.zip").read_bytes())
    raw[member.end - 1] ^= 0xFF
    (directory / "set.zip").write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="CRC"):
        fetch_members(f"{url}/set.zip", [member])


def test_stream_yields_groups_in_order_across_windows(range_server: Server) -> None:
    url, directory, _ = range_server
    contents = _publish(directory)
    members = {m.name: m for m in read_index(f"{url}/set.zip").members}
    order = [f"images/{i:02d}.bin" for i in (5, 1, 9, 0, 11)]
    archive = f"{url}/set.zip"
    groups = [[(archive, members[order[0]]), (archive, members["labels.csv"])]]
    groups += [[(archive, members[name])] for name in order]
    results = list(stream(groups, window_bytes=300, lookahead=1, workers=2))
    assert set(results[0]) == {"images/05.bin", "labels.csv"}
    assert [next(iter(r)) for r in results[1:]] == order
    assert all(content == contents[name] for r in results for name, content in r.items())


def test_long_runs_are_fetched_as_parallel_pieces(
    range_server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, directory, _ = range_server
    contents = _publish(directory)
    archive = f"{url}/set.zip"
    members = read_index(archive).members
    spans: list[tuple[int, int]] = []
    request = remote._request

    def recording(address: str, start: int, end: int) -> bytes:
        spans.append((start, end))
        return request(address, start, end)

    monkeypatch.setattr(remote, "_request", recording)
    monkeypatch.setattr(remote, "PIECE", 256)
    results = list(stream([[(archive, m) for m in members]], workers=4))
    assert results[0] == {m.name: contents[m.name] for m in members}
    assert len(spans) > 1 and all(end - start < 256 for start, end in spans)
    assert remote.pieces(0, 600, 256) == [(0, 255), (256, 511), (512, 599)]


def test_servers_without_ranges_are_detected(range_server: Server) -> None:
    url, directory, server = range_server
    _publish(directory)
    server.ignore_ranges = True  # type: ignore[attr-defined]
    with pytest.raises(RangeUnsupportedError):
        read_index(f"{url}/set.zip")


def test_windows_start_small_and_double() -> None:
    member = Member(name="m", offset=0, end=1, compressed=1 << 20, size=1, crc=0, method=0)
    batches = windows([[("u", member)]] * 40, window_bytes=16 << 20)
    assert [len(batch) for batch in batches] == [4, 8, 16, 12]


class _Truncated:
    """Response whose body loses its last bytes, as some hosts send for large ranges."""

    status = 206

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self) -> "_Truncated":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def read(self) -> bytes:
        raise http.client.IncompleteRead(self.data[:4], len(self.data) - 4)


def test_failed_requests_are_retried(range_server: Server, monkeypatch: pytest.MonkeyPatch) -> None:
    url, directory, _ = range_server
    _publish(directory)
    opened = urllib.request.urlopen
    calls = []

    def flaky(request: urllib.request.Request, timeout: float) -> object:
        calls.append(request)
        if len(calls) == 1:
            raise http.client.IncompleteRead(b"", 10)
        return opened(request, timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr("time.sleep", lambda _: None)
    assert len(remote._request(f"{url}/set.zip", 0, 9)) == 10
    assert len(calls) == 2


def test_truncated_responses_are_completed(
    range_server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, directory, _ = range_server
    _publish(directory)
    expected = (directory / "set.zip").read_bytes()[100:200]
    opened = urllib.request.urlopen
    ranges = []

    def truncating(request: urllib.request.Request, timeout: float) -> object:
        ranges.append(request.get_header("Range"))
        response = opened(request, timeout=timeout)
        if len(ranges) > 1:
            return response
        with response:
            return _Truncated(response.read())

    monkeypatch.setattr(urllib.request, "urlopen", truncating)
    assert remote._request(f"{url}/set.zip", 100, 199) == expected
    assert ranges == ["bytes=100-199", "bytes=104-199"]


class _Interrupted:
    """Response that sends its first bytes and then loses the connection."""

    status = 200
    headers: dict[str, str] = {}  # noqa: RUF012

    def __init__(self) -> None:
        self.sent = False

    def __enter__(self) -> "_Interrupted":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def read(self, _: int) -> bytes:
        if self.sent:
            raise http.client.IncompleteRead(b"", 10)
        self.sent = True
        return b"garbage"


def _plain(url: str, content: bytes) -> Download:
    digest = "md5:" + hashlib.md5(content).hexdigest()
    return Download(url=f"{url}/blob.bin", filename="blob.bin", checksum=digest, extract=False)


def test_whole_downloads_are_retried_from_the_start(
    range_server: Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, directory, _ = range_server
    content = bytes(range(256)) * 40
    (directory / "blob.bin").write_bytes(content)
    opened = urllib.request.urlopen
    calls: list[urllib.request.Request] = []

    def flaky(request: urllib.request.Request, timeout: float) -> object:
        calls.append(request)
        if len(calls) == 1:
            raise urllib.error.URLError("refused")
        if len(calls) == 2:
            return _Interrupted()
        return opened(request, timeout=timeout)

    delays: list[float] = []
    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr("time.sleep", delays.append)
    path = fetch_file(_plain(url, content), tmp_path / "out")
    assert path.read_bytes() == content
    assert delays == [2, 4]
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["blob.bin"]


def test_whole_downloads_give_up_after_every_retry(
    range_server: Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, _, _ = range_server

    def refused(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.URLError("refused")

    delays: list[float] = []
    monkeypatch.setattr(urllib.request, "urlopen", refused)
    monkeypatch.setattr("time.sleep", delays.append)
    with pytest.raises(urllib.error.URLError):
        fetch_file(_plain(url, b"x"), tmp_path)
    assert delays == [2, 4, 8, 16]
