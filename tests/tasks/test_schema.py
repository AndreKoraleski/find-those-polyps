import numpy as np
import pytest
from pydantic import ValidationError

from find_those_polyps.tasks import Sample, Task


def test_sample_rejects_mismatched_mask() -> None:
    with pytest.raises(ValidationError, match="mask shape"):
        Sample(
            id="x",
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            target=np.zeros((3, 4), dtype=np.int64),
            group="g",
        )


def test_sample_rejects_non_rgb_image() -> None:
    with pytest.raises(ValidationError, match="image"):
        Sample(id="x", image=np.zeros((4, 4), dtype=np.uint8), target=0, group="g")


def test_task_hash_tracks_definition(colour_task: Task) -> None:
    spec = colour_task.spec
    assert spec.hash == spec.model_copy().hash
    assert spec.hash != spec.model_copy(update={"splits_sha256": "x"}).hash


def test_masks_keep_one_byte_per_pixel() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    def target(value: np.ndarray) -> np.ndarray:
        return np.asarray(Sample(id="x", image=image, target=value, group="g").target)

    assert target(np.ones((4, 4), dtype=np.int64)).dtype == np.uint8
    assert target(np.ones((2, 4, 4), dtype=bool).astype(np.int32)).dtype == np.uint8
    assert target(np.full((4, 4), 300, dtype=np.int64)).dtype == np.int64
    assert target(np.array([0, 1, 1])).dtype == np.int64
