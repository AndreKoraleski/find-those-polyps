import hashlib
import http.server
import io
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from find_those_polyps.pretraining import (
    CorpusCard,
    CorpusDataset,
    CorpusReader,
    CorpusUnavailableError,
    fetch,
)
from find_those_polyps.pretraining.__main__ import main
from find_those_polyps.pretraining.dataset import shard

Server = tuple[str, Path, http.server.ThreadingHTTPServer]
FRAMES = 12
UNITS = ("v0", "v1", "v2")


def _image(unit: int, index: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (unit * 40, index * 10, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _publish(directory: Path, unit: str) -> dict[str, object]:
    """Tarball of one unit, its frames out of order beside a file that is not a frame."""
    number = int(unit[1:])
    path = directory / f"{unit}_frames.tar.gz"
    with tarfile.open(path, "w:gz") as bundle:
        bundle.addfile(tarfile.TarInfo(f"{unit}_frames/"))
        for index in [5, 0, 11, 3, 8, 1, 10, 2, 7, 4, 9, 6]:
            content = _image(number, index)
            info = tarfile.TarInfo(f"{unit}_frames/{unit}_{index}.jpg")
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
        note = tarfile.TarInfo(f"{unit}_frames/notes.txt")
        note.size = 4
        bundle.addfile(note, io.BytesIO(b"note"))
    data = path.read_bytes()
    return {
        "unit": unit,
        "url": "",
        "filename": path.name,
        "checksum": "md5:" + hashlib.md5(data).hexdigest(),
        "size": len(data),
        "frames": FRAMES,
        "fps": 4.0,
    }


@pytest.fixture
def card(range_server: Server) -> CorpusCard:
    url, directory, _ = range_server
    archives = [_publish(directory, unit) for unit in UNITS]
    for archive in archives:
        archive["url"] = f"{url}/{archive['filename']}"
    return CorpusCard.model_validate(
        {
            "key": "toy",
            "name": "Toy",
            "modality": "colonoscopy",
            "licence": "CC BY 4.0",
            "url": url,
            "citation": "c",
            "provenance": "p",
            "family": "toy",
            "frame_pattern": r"v\d_(\d+)\.jpg",
            "archives": archives,
            "held_out": [{"units": ["v2"], "reason": "test", "evidence": "e"}],
        }
    )


def _offline(server: http.server.ThreadingHTTPServer, directory: Path) -> None:
    """Remove every published file, so that any request to the host fails."""
    for path in directory.iterdir():
        path.unlink()


def test_streamed_units_are_subsampled_verified_and_cached(
    card: CorpusCard, range_server: Server, tmp_path: Path
) -> None:
    _, directory, server = range_server
    reader = CorpusReader(card, tmp_path / "data", fps=2.0)
    first = card.selected()[0]
    assert reader.state(first) == "remote" and reader.expected(first) == 6
    frames = list(reader.frames(first))
    assert [frame.index for frame in frames] == [0, 8, 10, 2, 4, 6]
    assert frames[0].content == _image(0, 0) and frames[0].name == "v0_0.jpg"
    assert {frame.unit for frame in frames} == {"v0"}
    folder = tmp_path / "data" / "corpora" / "toy" / "v0"
    assert sorted(path.name for path in folder.iterdir()) == [
        f"v0_{index}.jpg" for index in (0, 10, 2, 4, 6, 8)
    ]
    assert reader.state(first) == "cached"
    _offline(server, directory)
    assert [frame.index for frame in reader.frames(first)] == [0, 2, 4, 6, 8, 10]


def test_a_finer_cache_serves_coarser_requests_only(
    card: CorpusCard, range_server: Server, tmp_path: Path
) -> None:
    root = tmp_path / "data"
    first = card.selected()[0]
    assert len(list(CorpusReader(card, root, fps=2.0).frames(first))) == 6
    assert CorpusReader(card, root, fps=1.0).cached(first)
    assert [f.index for f in CorpusReader(card, root, fps=1.0, stream=False).frames(first)] == [
        0,
        4,
        8,
    ]
    every = CorpusReader(card, root, fps=None)
    assert not every.cached(first)
    assert len(list(every.frames(first))) == FRAMES
    assert CorpusReader(card, root, fps=2.0).cached(first)


def test_held_out_units_are_never_read(card: CorpusCard, tmp_path: Path) -> None:
    reader = CorpusReader(card, tmp_path / "data")
    assert [archive.unit for archive in reader.units()] == ["v0", "v1"]
    with pytest.raises(ValueError, match="held out"):
        next(reader.frames(card.archives[2]))
    assert {frame.unit for frame in reader} == {"v0", "v1"}
    with pytest.raises(ValueError, match="no readable units"):
        fetch(reader, ["v2"])


def test_an_archive_that_changed_is_refused_and_leaves_no_cache(
    card: CorpusCard, range_server: Server, tmp_path: Path
) -> None:
    _, directory, _ = range_server
    path = directory / "v0_frames.tar.gz"
    path.write_bytes(path.read_bytes() + b"\0")
    reader = CorpusReader(card, tmp_path / "data", fps=None)
    first = card.selected()[0]
    with pytest.raises(ValueError, match="does not match its pin"):
        list(reader.frames(first))
    assert not reader.cached(first)
    assert not (tmp_path / "data" / "corpora" / "toy" / "v0").exists()


def test_an_interrupted_pass_is_not_taken_for_a_complete_one(
    card: CorpusCard, tmp_path: Path
) -> None:
    reader = CorpusReader(card, tmp_path / "data", fps=None)
    first = card.selected()[0]
    frames = reader.frames(first)
    next(frames)
    frames.close()
    assert not reader.cached(first)
    assert len(list(reader.frames(first))) == FRAMES
    assert reader.cached(first)


@pytest.mark.parametrize("keep", [False, True])
def test_whole_downloads_are_verified_then_read(
    card: CorpusCard, range_server: Server, tmp_path: Path, keep: bool
) -> None:
    _, directory, server = range_server
    root = tmp_path / "data"
    reader = CorpusReader(card, root, fps=2.0, download=True, keep_archives=keep, cache=False)
    first = card.selected()[0]
    assert len(list(reader.frames(first))) == 6
    archive = root / ".archives" / "toy" / "v0_frames.tar.gz"
    assert archive.exists() == keep
    assert not (root / "corpora" / "toy").exists()
    _offline(server, directory)
    local = CorpusReader(card, root, fps=2.0, stream=False)
    if keep:
        assert local.state(first) == "archive"
        assert len(list(local.frames(first))) == 6
        assert local.cached(first)
    else:
        with pytest.raises(CorpusUnavailableError, match="neither"):
            list(local.frames(first))


def test_fetch_caches_units_once(card: CorpusCard, range_server: Server, tmp_path: Path) -> None:
    _, directory, server = range_server
    reader = CorpusReader(card, tmp_path / "data", fps=2.0)
    assert fetch(reader, ["v1", "v1"], workers=2) == 1
    assert fetch(reader, workers=2) == 1
    _offline(server, directory)
    assert fetch(reader) == 0
    assert len(list(reader)) == 12
    with pytest.raises(ValueError, match="writes to the cache"):
        fetch(CorpusReader(card, tmp_path / "data", cache=False))
    with pytest.raises(ValueError, match="positive"):
        CorpusReader(card, tmp_path / "data", fps=0)


def test_dataset_reads_every_unit_once_per_epoch(card: CorpusCard, tmp_path: Path) -> None:
    reader = CorpusReader(card, tmp_path / "data", fps=None)
    dataset = CorpusDataset([reader], lambda image: image.getpixel((0, 0)), shuffle_buffer=5)
    assert dataset.expected() == 2 * FRAMES
    first = list(dataset)
    expected = sorted((unit * 40, index * 10, 0) for unit in (0, 1) for index in range(FRAMES))
    assert sorted(first) == expected
    assert first != expected
    cached = list(dataset)
    assert sorted(cached) == expected and cached != expected
    assert list(dataset) == cached
    dataset.set_epoch(1)
    second = list(dataset)
    assert sorted(second) == expected and second != cached
    units = [colour[0] for colour in second]
    assert units != sorted(units)
    plain = CorpusDataset([reader], shuffle_buffer=0)
    assert [image.size for image in plain] == [(8, 6)] * (2 * FRAMES)


def test_workers_share_the_units() -> None:
    units = list(range(7))
    parts = [shard(units, worker, 3) for worker in range(3)]
    assert sorted(unit for part in parts for unit in part) == units
    assert parts[0] == [0, 3, 6]


def test_dataset_skips_units_whose_host_fails_and_frames_that_are_not_images(
    card: CorpusCard, range_server: Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, directory, _ = range_server
    monkeypatch.setattr("time.sleep", lambda _: None)
    reader = CorpusReader(card, tmp_path / "data", fps=None)
    list(reader.frames(card.selected()[0]))
    (tmp_path / "data" / "corpora" / "toy" / "v0" / "v0_3.jpg").write_bytes(b"not an image")
    (directory / "v1_frames.tar.gz").unlink()
    dataset = CorpusDataset([reader], shuffle_buffer=0)
    assert len(list(dataset)) == FRAMES - 1


def test_cli_lists_fetches_and_verifies(
    card: CorpusCard,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("find_those_polyps.pretraining.__main__.load_corpora", lambda: (card,))
    monkeypatch.setattr("find_those_polyps.pretraining.__main__.corpus", lambda key: card)
    root = str(tmp_path / "data")
    assert main(["list", "--root", root, "--fps", "all"]) == 0
    listed = capsys.readouterr().out
    assert "units 2 of 3 (1 held out)" in listed and "24 frames kept, 0 units cached" in listed
    assert main(["fetch", "toy", "--root", root, "--fps", "2", "--whole", "--units", "v1"]) == 0
    assert "1 units read" in capsys.readouterr().out
    assert main(["list", "--root", root, "--fps", "2"]) == 0
    assert "12 frames kept, 1 units cached" in capsys.readouterr().out
    assert main(["fetch", "toy", "--root", root, "--units", "v2"]) == 1
    assert "no readable units" in capsys.readouterr().err
    assert main(["verify", "toy"]) == 0
    assert capsys.readouterr().out.count("ok ") == 3


def test_cli_rejects_unknown_corpora(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fetch", "absent"]) == 1
    assert "unknown corpus" in capsys.readouterr().err
    assert main(["list"]) == 0
    assert "real-colon-train" in capsys.readouterr().out
