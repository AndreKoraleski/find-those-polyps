import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from find_those_polyps.registry import load_registry
from find_those_polyps.registry.manifest import (
    COLUMNS,
    Reader,
    _Print,
    check_corpus,
    read_manifest,
    stale_manifests,
    write_manifest,
)
from find_those_polyps.utils.perceptual import dhash


def _scene(seed: int) -> Image.Image:
    """Smooth random texture, whose gradients resemble those of endoscopy frames."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, (6, 8, 3)).astype(np.uint8)
    return Image.fromarray(coarse).resize((160, 120), Image.Resampling.BICUBIC)


def test_hash_survives_resizing_and_recompression() -> None:
    image = _scene(0)
    buffer = io.BytesIO()
    image.resize((80, 60)).save(buffer, "JPEG", quality=70)
    copy = Image.open(io.BytesIO(buffer.getvalue()))
    assert (dhash(image) ^ dhash(copy)).bit_count() <= 4
    assert (dhash(image) ^ dhash(_scene(1))).bit_count() > 10


def test_corpus_check_finds_identical_and_near_copies(tmp_path: Path) -> None:
    tests, corpus = tmp_path / "test", tmp_path / "corpus"
    tests.mkdir()
    (corpus / "nested").mkdir(parents=True)
    rows = []
    for index in range(3):
        path = tests / f"t{index}.png"
        _scene(index).save(path)
        item = _Print(path)
        assert item.signature is not None
        coarse, fine = f"{item.signature.coarse:016x}", f"{item.signature.fine:064x}"
        rows.append(["cls/x", f"t{index}", f"x/t{index}.png", item.crc, item.size, coarse, fine])
    manifests = pd.DataFrame(rows, columns=COLUMNS)
    (corpus / "same.png").write_bytes((tests / "t0.png").read_bytes())
    _scene(1).resize((80, 60)).save(corpus / "nested" / "small.jpg", quality=80)
    _scene(9).save(corpus / "other.png")
    (corpus / "notes.txt").write_text("not an image")

    def read(path: str) -> bytes | None:
        return (tests / Path(path).name).read_bytes()

    def kinds(reader: Reader | None) -> set[tuple[str, str, str]]:
        matches = check_corpus([corpus], manifests=manifests, read=reader)
        columns = matches[["file", "sample_id", "match"]].to_numpy()
        return {(Path(f).name, s, m) for f, s, m in columns}

    assert kinds(read) == {("same.png", "t0", "identical"), ("small.jpg", "t1", "near duplicate")}
    assert kinds(None) == {("same.png", "t0", "identical"), ("small.jpg", "t1", "candidate")}
    with pytest.raises(ValueError, match="distance"):
        check_corpus([corpus], distance=300, manifests=manifests)


def test_manifests_are_written_reproducibly(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            ["seg/b", "2", "d/2.png", "0000abcd", 7, "00ff", "0f"],
            ["cls/a", "1", "d/1.png", "1", 5, "1", "1"],
        ],
        columns=COLUMNS,
    )
    first, second = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    write_manifest(frame, first)
    write_manifest(frame.iloc[::-1], second)
    assert first.read_bytes() == second.read_bytes()
    assert read_manifest(first)["crc32"].tolist() == ["1", "0000abcd"]


def test_committed_manifests_match_the_test_partitions() -> None:
    assert stale_manifests(load_registry()) == []
