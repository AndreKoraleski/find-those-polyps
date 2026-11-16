import hashlib
import http.server
import io
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import MeanColour
from PIL import Image

from find_those_polyps.data import LocalFiles, RemoteFiles, fetch, open_files, read_index
from find_those_polyps.data.__main__ import main, state
from find_those_polyps.data.files import COMPLETE, INDEX_DIR
from find_those_polyps.evaluation.features import extract
from find_those_polyps.registry import DatasetCard, Download, load_registry
from find_those_polyps.tasks import Split, SplitFile, Task
from find_those_polyps.tasks.loaders.kvasir_seg import KvasirSEG

Server = tuple[str, Path, http.server.ThreadingHTTPServer]
NOTES = b"id\nk0\n"


def _png(value: int, mask: bool = False) -> bytes:
    array = np.full((8, 10) if mask else (8, 10, 3), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _publish(directory: Path, n: int = 9) -> None:
    with zipfile.ZipFile(directory / "kvasir.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for i in reversed(range(n)):
            bundle.writestr(f"Kvasir-SEG/images/k{i}.jpg", _png(20 * i))
        for i in range(n):
            bundle.writestr(f"Kvasir-SEG/masks/k{i}.jpg", _png(255 * (i % 2), mask=True))
    (directory / "notes.csv").write_bytes(NOTES)


def _card(url: str, index: str = "", downloads: bool = True) -> DatasetCard:
    pin = index or "sha256:" + "0" * 64
    notes = "md5:" + hashlib.md5(NOTES).hexdigest()
    items = (
        Download(url=f"{url}/kvasir.zip", filename="kvasir.zip", index=pin),
        Download(url=f"{url}/notes.csv", filename="notes.csv", checksum=notes, extract=False),
    )
    return DatasetCard.model_validate(
        {
            "key": "kvasir-seg",
            "name": "Toy",
            "labelled": True,
            "modality": "m",
            "licence": "CC BY 4.0",
            "access": "open",
            "url": url,
            "citation": "c",
            "provenance": "p",
            "family": "f",
            "downloads": items if downloads else (),
        }
    )


@pytest.fixture
def card(range_server: Server) -> DatasetCard:
    url, directory, _ = range_server
    _publish(directory)
    return _card(url, read_index(f"{url}/kvasir.zip").digest)


def test_plain_files_need_a_checksum() -> None:
    with pytest.raises(ValueError, match="checksum"):
        Download(url="u", filename="f.csv", extract=False)


def test_remote_files_list_read_and_cache(card: DatasetCard, tmp_path: Path) -> None:
    cache = tmp_path / "data" / "kvasir-seg"
    files = RemoteFiles(card, cache)
    assert files.available()
    assert "Kvasir-SEG/masks/k3.jpg" in files.names()
    assert files.read_text("notes.csv") == "id\nk0\n"
    assert files.read("Kvasir-SEG/images/k2.jpg") == _png(40)
    assert (cache / "Kvasir-SEG/images/k2.jpg").read_bytes() == _png(40)
    assert (cache / INDEX_DIR / "kvasir.zip.json").exists()


@pytest.mark.parametrize("cached", [True, False])
def test_unknown_remote_files_raise_key_error(
    card: DatasetCard, tmp_path: Path, cached: bool
) -> None:
    files = RemoteFiles(card, tmp_path / "cache" if cached else None)
    with pytest.raises(KeyError, match=r"absent\.png"):
        files.read("absent.png")


def test_uncached_plain_files_leave_the_working_directory_alone(
    card: DatasetCard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    files = RemoteFiles(card, None)
    assert files.read("notes.csv") == NOTES
    assert list(work.iterdir()) == []
    (work / "notes.csv").write_bytes(NOTES)
    assert files.read("notes.csv") == NOTES
    assert [path.name for path in work.iterdir()] == ["notes.csv"]


def test_local_files_skip_partial_downloads(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(b"a")
    (tmp_path / "images" / "b.png.part").write_bytes(b"b")
    (tmp_path / "notes.csv.part").write_bytes(b"n")
    assert LocalFiles("toy", tmp_path).names() == ["images/a.png"]


def test_changed_archives_are_refused(range_server: Server, tmp_path: Path) -> None:
    url, directory, _ = range_server
    _publish(directory)
    with pytest.raises(ValueError, match="changed"):
        RemoteFiles(_card(url), tmp_path).names()


def test_open_files_prefers_complete_and_manual_copies(card: DatasetCard, tmp_path: Path) -> None:
    root = tmp_path / "data"
    assert isinstance(open_files(card, root), RemoteFiles)
    assert isinstance(open_files(card, root, stream=False), LocalFiles)
    manual = tmp_path / "manual"
    (manual / "kvasir-seg").mkdir(parents=True)
    (manual / "kvasir-seg" / "own.png").write_bytes(b"x")
    assert isinstance(open_files(card, manual), LocalFiles)
    restricted = card.model_copy(update={"downloads": ()})
    assert isinstance(open_files(restricted, tmp_path / "empty"), LocalFiles)


def test_streamed_task_matches_local_order(card: DatasetCard, tmp_path: Path) -> None:
    loader = KvasirSEG(RemoteFiles(card, None))
    ids = loader.ids()
    assert ids == sorted(f"k{i}" for i in range(9))
    streamed = [sample.id for sample in loader.stream(ids)]
    assert sorted(streamed) == ids
    assert streamed != ids
    split = SplitFile(
        test_source="t", train=tuple(ids[:5]), val=tuple(ids[5:7]), test=tuple(ids[7:])
    )
    task = Task(spec=load_registry().task("seg/kvasir-seg"), loader=loader, split=split)
    samples = task.samples(Split.TRAIN)
    features = extract(MeanColour(), samples, dense=False, batch_size=2, device=torch.device("cpu"))
    assert features.ids == split.train
    assert features.paths[0] == "kvasir-seg/Kvasir-SEG/images/k0.jpg"


def test_fetch_streams_every_file_then_marks_complete(card: DatasetCard, tmp_path: Path) -> None:
    root = tmp_path / "data"
    destination = fetch(card, root)
    assert (destination / COMPLETE).exists()
    assert (destination / "Kvasir-SEG/masks/k8.jpg").exists()
    assert (destination / "notes.csv").exists()
    assert isinstance(open_files(card, root), LocalFiles)


def test_cli_lists_states_and_indexes(
    card: DatasetCard, range_server: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    url, _, _ = range_server
    assert main(["index", f"{url}/kvasir.zip"]) == 0
    assert card.downloads[0].index in capsys.readouterr().out
    assert main(["tasks"]) == 0
    assert "seg/kvasir-seg" in capsys.readouterr().out


def test_dataset_states(card: DatasetCard, tmp_path: Path) -> None:
    root = tmp_path / "data"
    assert state(card, root) == "streamable"
    assert state(card.model_copy(update={"downloads": ()}), root) == "open access"
    (root / "kvasir-seg").mkdir(parents=True)
    assert state(card, root) == "manual copy"
    (root / "kvasir-seg" / ".index").mkdir()
    assert state(card, root) == "cached"
    (root / "kvasir-seg" / ".complete").write_text("x")
    assert state(card, root) == "local"


def test_cached_files_and_indexes_are_verified(card: DatasetCard, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    RemoteFiles(card, cache).read("Kvasir-SEG/images/k2.jpg")
    (cache / "Kvasir-SEG/images/k2.jpg").write_bytes(b"corrupt")
    assert RemoteFiles(card, cache).read("Kvasir-SEG/images/k2.jpg") == _png(40)
    index = cache / INDEX_DIR / "kvasir.zip.json"
    index.write_text(index.read_text().replace('"size":', '"size":1', 1))
    files = RemoteFiles(card, cache)
    assert "Kvasir-SEG/masks/k3.jpg" in files.names()
    expected = _png(255, mask=True)
    assert (
        files.fingerprint("Kvasir-SEG/masks/k3.jpg")
        == f"{zlib.crc32(expected):08x}:{len(expected)}"
    )


@pytest.mark.parametrize("keep", [False, True])
def test_whole_fetch_downloads_each_file_once_and_verifies_its_index(
    card: DatasetCard, tmp_path: Path, keep: bool
) -> None:
    root = tmp_path / "data"
    destination = fetch(card, root, whole=True, keep_archives=keep)
    assert (destination / COMPLETE).exists()
    assert (destination / "Kvasir-SEG/images/k3.jpg").read_bytes() == _png(60)
    assert (destination / "notes.csv").read_bytes() == NOTES
    assert not (destination / INDEX_DIR).exists()
    assert (root / ".archives" / "kvasir-seg" / "kvasir.zip").exists() == keep
    assert isinstance(open_files(card, root), LocalFiles)


def test_whole_fetch_refuses_an_archive_that_changed(card: DatasetCard, tmp_path: Path) -> None:
    stale = card.model_copy(
        update={
            "downloads": (
                card.downloads[0].model_copy(update={"index": "sha256:" + "0" * 64}),
                card.downloads[1],
            )
        }
    )
    with pytest.raises(ValueError, match="no longer matches its index"):
        fetch(stale, tmp_path / "data", whole=True)
    assert not (tmp_path / "data" / "kvasir-seg" / COMPLETE).exists()
    assert not (tmp_path / "data" / ".archives" / "kvasir-seg" / "kvasir.zip").exists()


def test_open_files_can_download_whole_instead_of_streaming(
    card: DatasetCard, tmp_path: Path
) -> None:
    root = tmp_path / "data"
    files = open_files(card, root, download=True)
    assert isinstance(files, LocalFiles)
    assert files.read("Kvasir-SEG/images/k2.jpg") == _png(40)
    assert (root / "kvasir-seg" / COMPLETE).exists()
    assert not (root / "kvasir-seg" / INDEX_DIR).exists()
    assert isinstance(open_files(card, root, download=True, stream=False), LocalFiles)


def test_a_failed_download_is_reported_like_a_failed_stream(
    card: DatasetCard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("find_those_polyps.data.remote.time.sleep", lambda _: None)
    unreachable = card.model_copy(
        update={
            "downloads": tuple(
                item.model_copy(update={"url": "http://127.0.0.1:9/" + item.filename})
                for item in card.downloads
            )
        }
    )
    files = open_files(unreachable, tmp_path / "data", download=True)
    assert isinstance(files, LocalFiles)
    assert files.stream_error
    assert not files.available()
    assert not (tmp_path / "data" / "kvasir-seg").exists()
