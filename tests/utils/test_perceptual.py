import io

import numpy as np
from PIL import Image

from find_those_polyps.utils.perceptual import confirmer, signature, similar, thumbnail


def _texture(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, (6, 8, 3)).astype(np.uint8)
    return Image.fromarray(coarse).resize((160, 120), Image.Resampling.BICUBIC)


def test_thumbnails_confirm_copies_and_reject_other_images() -> None:
    original = _texture(0)
    buffer = io.BytesIO()
    original.resize((80, 60)).save(buffer, "JPEG", quality=60)
    copy = Image.open(io.BytesIO(buffer.getvalue()))
    assert similar(thumbnail(original), thumbnail(copy))
    assert not similar(thumbnail(original), thumbnail(_texture(1)))
    assert (signature(original).fine ^ signature(copy).fine).bit_count() <= 16


def test_confirmer_caches_thumbnails_and_rejects_unreadable_images() -> None:
    loads: list[int] = []
    images = {0: _texture(0), 1: _texture(0)}

    def load(position: int) -> Image.Image | None:
        loads.append(position)
        return images.get(position)

    confirm = confirmer(load)
    assert confirm(0, 1)
    assert not confirm(0, 2)
    assert loads == [0, 1, 2]
