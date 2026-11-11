from pathlib import Path

import numpy as np
from PIL import Image

from find_those_polyps.data import LocalFiles
from find_those_polyps.tasks.loaders.layers import LayeredMasks


def _pages(path: Path, pages: list[np.ndarray]) -> None:
    first, *rest = (Image.fromarray(page) for page in pages)
    first.save(path, save_all=True, append_images=rest)


def test_pages_become_overlapping_class_masks(tmp_path: Path) -> None:
    for folder in ("a/images", "a/masks", "b/images", "b/masks"):
        (tmp_path / folder).mkdir(parents=True)
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "a/images/x1.png")
    Image.fromarray(image).save(tmp_path / "b/images/y1.png")
    first = np.zeros((4, 6), dtype=np.uint8)
    first[:2] = 255
    second = np.zeros((4, 6), dtype=np.uint8)
    second[1:3] = 200
    _pages(tmp_path / "a/masks/x1.tif", [first, second])
    _pages(tmp_path / "b/masks/y1_mask.tif", [second])
    loader = LayeredMasks(
        LocalFiles("set", tmp_path),
        sources=[("a/images", "a/masks"), ("b/images", "b/masks")],
        layers=3,
        mask_name=("_mask$", ""),
    )
    assert loader.ids() == ["x1", "y1"]
    target = np.asarray(loader.load("x1").target)
    assert target.shape == (3, 4, 6)
    assert target[:, 1, 0].tolist() == [1, 1, 0]
    assert np.asarray(loader.load("y1").target)[0].sum() == 12
