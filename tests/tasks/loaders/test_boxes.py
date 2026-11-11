from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from find_those_polyps.data import LocalFiles
from find_those_polyps.tasks.loaders.boxes import TableBoxes, mask_boxes, voc_boxes, yolo_boxes


def test_regions_become_normalized_boxes_and_fragments_are_dropped() -> None:
    mask = np.zeros((100, 200), dtype=np.int64)
    mask[10:30, 20:60] = 1
    mask[60:90, 150:190] = 2
    mask[50:52, 100:101] = 1
    boxes = mask_boxes(mask)
    assert boxes.tolist() == [[1, 0.1, 0.1, 0.3, 0.3], [2, 0.75, 0.6, 0.95, 0.9]]


def test_diagonal_neighbours_join_and_empty_masks_have_no_boxes() -> None:
    mask = np.zeros((10, 10), dtype=np.int64)
    mask[2, 2] = mask[3, 3] = 1
    assert mask_boxes(mask, min_area=0).tolist() == [[1, 0.2, 0.2, 0.4, 0.4]]
    assert mask_boxes(np.zeros((4, 4), dtype=np.int64)).shape == (0, 5)


def test_pascal_voc_files_become_normalized_boxes() -> None:
    data = (
        b"<annotation><object><name> bleeding </name><bndbox><xmin>20</xmin><ymin>10</ymin>"
        b"<xmax>60</xmax><ymax>30</ymax></bndbox></object><object><name>bleeding</name>"
        b"</object></annotation>"
    )
    boxes = voc_boxes(data, {"bleeding": 1}, height=100, width=200)
    assert boxes.tolist() == [[1, 0.1, 0.1, 0.3, 0.3]]


def test_table_boxes_keep_listed_classes_and_bound_corners(tmp_path: Path) -> None:
    root = tmp_path / "set"
    (root / "frames").mkdir(parents=True)
    for name in ("v1_1", "v1_2", "v2_1"):
        Image.fromarray(np.zeros((40, 80, 3), dtype=np.uint8)).save(root / "frames" / f"{name}.jpg")
    (root / "boxes.csv").write_text(
        "file;video;kind;x1;y1;x2;y2\n"
        "v1_1.jpg;v1;Ulcer;8;4;40;20\n"
        "v1_1.jpg;v1;Polyp;0;0;8;8\n"
        "v1_2.jpg;v1;Normal;;;;\n"
        "v2_1.jpg;v2;Ulcer;100;30;60;10\n"
    )
    loader = TableBoxes(
        LocalFiles("set", root),
        table="boxes.csv",
        key="file",
        label="kind",
        box_classes={"Ulcer": 1},
        x=["x1", "x2"],
        y=["y1", "y2"],
        root="frames",
        group_column="video",
        delimiter=";",
    )
    assert loader.ids() == ["v1_1", "v2_1"]
    assert loader.groups() == {"v1_1": "v1", "v2_1": "v2"}
    assert np.asarray(loader.load("v1_1").target).tolist() == [[1, 0.1, 0.1, 0.5, 0.5]]
    assert np.asarray(loader.load("v2_1").target).tolist() == [[1, 0.75, 0.25, 1.0, 0.75]]


def test_yolo_files_shift_classes_and_clip_boxes() -> None:
    boxes = yolo_boxes("0 0.5 0.5 0.2 0.4\n\n7 0.95 0.1 0.2 0.4\n")
    assert boxes == pytest.approx(np.array([[1, 0.4, 0.3, 0.6, 0.7], [8, 0.85, 0.0, 1.0, 0.3]]))
