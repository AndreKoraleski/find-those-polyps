from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from find_those_polyps.data import LocalFiles
from find_those_polyps.registry.build import build_split
from find_those_polyps.registry.store import load_registry
from find_those_polyps.tasks.loaders.kvasir_seg import KvasirSEG
from find_those_polyps.tasks.splits import LOSSES


@pytest.fixture
def kvasir_root(tmp_path: Path) -> Path:
    root = tmp_path / "kvasir-seg"
    for folder in ("images", "masks"):
        (root / "Kvasir-SEG" / folder).mkdir(parents=True)
    for index in range(30):
        image = np.full((12, 16, 3), index, dtype=np.uint8)
        mask = np.zeros((12, 16), dtype=np.uint8)
        mask[2:6, 3:9] = 255
        Image.fromarray(image).save(root / "Kvasir-SEG" / "images" / f"img{index}.jpg")
        Image.fromarray(mask).save(root / "Kvasir-SEG" / "masks" / f"img{index}.jpg")
    return root


def test_loader_reads_binary_masks(kvasir_root: Path) -> None:
    loader = KvasirSEG(LocalFiles("kvasir-seg", kvasir_root))
    assert loader.available()
    assert len(loader.ids()) == 30
    sample = loader.load("img3")
    assert sample.image.shape == (12, 16, 3)
    assert isinstance(sample.target, np.ndarray)
    assert set(np.unique(sample.target)) == {0, 1}
    assert sample.target.sum() == 24
    assert sample.path == "kvasir-seg/Kvasir-SEG/images/img3.jpg"
    assert loader.groups()["img3"] == "img3"


def test_benchmark_split_without_official_test(kvasir_root: Path, tmp_path: Path) -> None:
    split = build_split(load_registry().task("seg/kvasir-seg"), tmp_path)
    assert split.test_source == "find-those-polyps"
    assert len(split.test) == 6
    assert len(split.train) + len(split.val) + len(split.test) == 30


def test_rebuilds_keep_the_test_partition_the_benchmark_chose(
    kvasir_root: Path, tmp_path: Path
) -> None:
    spec = load_registry().task("seg/kvasir-seg")
    first = build_split(spec, tmp_path)
    chosen = first.model_copy(update={"test": tuple(sorted(first.test + first.val[:2]))})
    kept = build_split(spec, tmp_path, previous=chosen)
    assert kept.test == chosen.test
    assert kept.test_source == "find-those-polyps"
    assert set(kept.train).isdisjoint(kept.test)
    redrawn = build_split(spec, tmp_path, previous=None)
    assert redrawn.test == first.test


def test_unavailable_without_masks(tmp_path: Path) -> None:
    (tmp_path / "Kvasir-SEG" / "images").mkdir(parents=True)
    assert not KvasirSEG(LocalFiles("kvasir-seg", tmp_path)).available()


def test_losses_count_every_excluded_sample_once(kvasir_root: Path, tmp_path: Path) -> None:
    split = build_split(load_registry().task("seg/kvasir-seg"), tmp_path)
    assert set(split.losses) == set(LOSSES)
    assert sum(split.losses.values()) == len(split.excluded)
