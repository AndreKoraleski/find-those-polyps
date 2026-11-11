from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from find_those_polyps.data import LocalFiles
from find_those_polyps.registry.build import build_split, write_split
from find_those_polyps.registry.store import load_registry
from find_those_polyps.tasks import SplitFile
from find_those_polyps.tasks.loaders.limuc import LIMUC


@pytest.fixture
def limuc_root(tmp_path: Path) -> Path:
    root = tmp_path / "limuc"
    rng = np.random.default_rng(0)
    for patient in range(1, 21):
        for index in range(4):
            mayo = (patient + index) % 4
            name = f"UC_patient_{patient}_{index}.bmp"
            pixels = rng.integers(0, 255, size=(6, 8, 3), dtype=np.uint8)
            for parent in [
                root / "patient_based_classified_images" / str(patient) / f"Mayo {mayo}"
            ]:
                parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(parent / name)
            if patient > 16:
                test_dir = root / "test_set" / f"Mayo {mayo}"
                test_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(test_dir / name)
    return root


def test_loader_reads_scores_patients_and_paths(limuc_root: Path) -> None:
    loader = LIMUC(LocalFiles("limuc", limuc_root))
    assert loader.available()
    assert len(loader.ids()) == 80
    sample = loader.load("UC_patient_3_1")
    assert sample.target == 0
    assert sample.group == "3"
    assert sample.image.shape == (6, 8, 3)
    assert sample.path == "limuc/patient_based_classified_images/3/Mayo 0/UC_patient_3_1.bmp"


def test_contradictory_duplicates_are_excluded(limuc_root: Path) -> None:
    pixels = np.zeros((6, 8, 3), dtype=np.uint8)
    for mayo in (1, 2):
        folder = limuc_root / "patient_based_classified_images" / "5" / f"Mayo {mayo}"
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(folder / "UC_patient_5_99.bmp")
    loader = LIMUC(LocalFiles("limuc", limuc_root))
    assert "UC_patient_5_99" not in loader.ids()
    assert loader.conflicts == ["UC_patient_5_99"]


def test_mapping_relabels_scores(limuc_root: Path) -> None:
    loader = LIMUC(LocalFiles("limuc", limuc_root), mapping=[0, 0, 1, 1])
    assert set(loader.labels().values()) == {0, 1}


def test_split_uses_official_test_and_patient_groups(limuc_root: Path, tmp_path: Path) -> None:
    spec = load_registry().task("ordinal/limuc-mayo")
    split = build_split(spec, tmp_path)
    assert len(split.test) == 16
    assert all(int(i.split("_")[2]) > 16 for i in split.test)
    patients = {
        phase: {i.split("_")[2] for i in ids}
        for phase, ids in (("train", split.train), ("val", split.val))
    }
    assert not patients["train"] & patients["val"]
    write_split(split, tmp_path / "limuc.json")
    reloaded = SplitFile.model_validate_json((tmp_path / "limuc.json").read_text())
    assert reloaded == split


def test_missing_data_is_reported(tmp_path: Path) -> None:
    spec = load_registry().task("ordinal/limuc-mayo")
    with pytest.raises(FileNotFoundError):
        build_split(spec, tmp_path / "nowhere", stream=False)
