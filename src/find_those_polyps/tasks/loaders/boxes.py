"""Detection targets from segmentation masks or from published box files.

Every connected region of a mask under eight connectivity becomes the box enclosing it, labelled
with the class of the region, as the boxes published with Kvasir-SEG were derived. Regions covering
less than `min_area` of the image are dropped as annotation fragments, and touching objects form
one region and hence one box. On Kvasir-SEG and Kvasir-Instrument these boxes match 99 percent of
the published ones at a median overlap above 0.99. Where published boxes divide regions further, as
on WCEBleedGen, they are read from their Pascal VOC files instead. Tables of box corners such as the
metadata of Kvasir-Capsule and the YOLO files of EAD2020 are read as well.

Every loader returns a box table of shape `(N, 5)` and dtype float64, whose rows hold class, x0,
y0, x1, y1. Classes count from one since zero is the background, and corners are normalized by the
image width and height.
"""

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from find_those_polyps.tasks.loaders.base import register_loader
from find_those_polyps.tasks.loaders.folders import FolderClassification, MaskSegmentation, _Indexed
from find_those_polyps.tasks.loaders.images import IMAGE_SUFFIXES, read_rgb, read_rows, stem
from find_those_polyps.tasks.loaders.kvasir_seg import KvasirSEG
from find_those_polyps.tasks.schema import Sample

if TYPE_CHECKING:
    from find_those_polyps.data.files import DatasetFiles

MIN_AREA = 5e-4


def mask_boxes(mask: NDArray[np.int64], min_area: float = MIN_AREA) -> NDArray[np.float64]:
    """Boxes enclosing the connected regions of a mask.

    Args:
        mask: Class index of each pixel, of shape `(H, W)`, with zero as the background.
        min_area: Smallest region kept, as a fraction of the image area.

    Returns:
        Box table of shape `(N, 5)` with rows class, x0, y0, x1, y1, where x1 and y1 are the
        exclusive pixel bounds of the region and all corners are normalized by the mask size.
    """
    height, width = mask.shape
    rows = []
    for cls in np.unique(mask[mask > 0]):
        regions, _ = ndimage.label(mask == cls, structure=np.ones((3, 3)))
        for index, window in enumerate(ndimage.find_objects(regions), start=1):
            if window is None or np.sum(regions[window] == index) < min_area * height * width:
                continue
            rows_slice, columns_slice = window
            rows.append(
                [
                    float(cls),
                    columns_slice.start / width,
                    rows_slice.start / height,
                    columns_slice.stop / width,
                    rows_slice.stop / height,
                ]
            )
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def _boxed(sample: Sample, min_area: float) -> Sample:
    """Copy of a segmentation sample whose mask target is replaced by the boxes of its regions.

    Args:
        sample: Sample whose target is a mask of shape `(H, W)`.
        min_area: Smallest region kept, as a fraction of the image area.

    Returns:
        The sample with a box table target.
    """
    return sample.model_copy(update={"target": mask_boxes(np.asarray(sample.target), min_area)})


@register_loader("mask-boxes")
class MaskBoxes(MaskSegmentation):
    """Boxes of the mask regions of a `masks` dataset, with the same options.

    Args:
        files: Files of the dataset.
        min_area: Smallest region kept, as a fraction of the image area.
        **options: Options of `MaskSegmentation`.

    Attributes:
        min_area: Smallest region kept, as a fraction of the image area.
    """

    def __init__(self, files: "DatasetFiles", min_area: float = MIN_AREA, **options: Any) -> None:
        super().__init__(files, **options)
        self.min_area = min_area

    def decode(self, sample_id: str, contents: Mapping[str, bytes]) -> Sample:
        """Build a detection sample from its image and mask.

        Args:
            sample_id: Identifier of the sample.
            contents: Bytes of the image and mask keyed by file name.

        Returns:
            Sample whose target is the box table of the mask regions.
        """
        return _boxed(super().decode(sample_id, contents), self.min_area)


@register_loader("kvasir-seg-boxes")
class KvasirSEGBoxes(KvasirSEG):
    """Boxes of the Kvasir-SEG polyp masks.

    Args:
        files: Files of the dataset.
        min_area: Smallest region kept, as a fraction of the image area.
        **options: Options passed on to `Loader`.

    Attributes:
        min_area: Smallest region kept, as a fraction of the image area.
    """

    def __init__(self, files: "DatasetFiles", min_area: float = MIN_AREA, **options: Any) -> None:
        super().__init__(files, **options)
        self.min_area = min_area

    def decode(self, sample_id: str, contents: Mapping[str, bytes]) -> Sample:
        """Build a detection sample from its image and mask.

        Args:
            sample_id: Identifier of the sample.
            contents: Bytes of the image and mask keyed by file name.

        Returns:
            Sample whose target is the box table of the polyp regions.
        """
        return _boxed(super().decode(sample_id, contents), self.min_area)


def voc_boxes(
    data: bytes, classes: Mapping[str, int], height: int, width: int
) -> NDArray[np.float64]:
    """Boxes of a Pascal VOC annotation.

    Objects without a `bndbox` element are skipped, a missing corner becomes NaN and an object name
    absent from `classes` raises `KeyError`.

    Args:
        data: XML annotation file.
        classes: Class index of every object name.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Box table of shape `(N, 5)` with rows class, x0, y0, x1, y1 normalized by the image size.
    """
    rows = []
    for item in ET.fromstring(data).iter("object"):
        box = item.find("bndbox")
        if box is None:
            continue
        x0, y0, x1, y1 = (
            float(box.findtext(side) or "nan") for side in ("xmin", "ymin", "xmax", "ymax")
        )
        name = (item.findtext("name") or "").strip()
        rows.append([float(classes[name]), x0 / width, y0 / height, x1 / width, y1 / height])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


@register_loader("voc-boxes")
class VOCBoxes(FolderClassification):
    """Images of a `folders` dataset with Pascal VOC box files, under the same identifiers.

    Images without a box file contain no object, which keeps the negatives of a classification
    task.

    Args:
        files: Files of the dataset.
        annotation: Regular expression and its replacement mapping the name of an image to the
            name of its box file.
        box_classes: Class index of every object name in the box files.
        **options: Options of `FolderClassification`.

    Attributes:
        annotation: Compiled expression and replacement of `annotation`.
        box_classes: Class index of every object name in the box files.
    """

    def __init__(
        self,
        files: "DatasetFiles",
        annotation: tuple[str, str],
        box_classes: dict[str, int],
        **options: Any,
    ) -> None:
        super().__init__(files, **options)
        self.annotation = (re.compile(annotation[0]), annotation[1])
        self.box_classes = box_classes

    def index(self) -> dict[str, tuple[str, ...]]:
        """Image file of every sample, followed by its box file when one exists.

        Returns:
            Names of the image and of its optional box file keyed by identifier.
        """
        names = set(self.files.names())
        found: dict[str, tuple[str, ...]] = {}
        for key, (image,) in super().index().items():
            box = self.annotation[0].sub(self.annotation[1], image)
            found[key] = (image, box) if box != image and box in names else (image,)
        return found

    def decode(self, sample_id: str, contents: Mapping[str, bytes]) -> Sample:
        """Build a detection sample from its image and optional box file.

        Args:
            sample_id: Identifier of the sample.
            contents: Bytes of the image and box file keyed by file name.

        Returns:
            Sample whose target is the box table of the annotation, empty without a box file.
        """
        image_name, *box_file = self.entries[sample_id]
        image = read_rgb(contents[image_name])
        height, width = image.shape[:2]
        boxes = (
            voc_boxes(contents[box_file[0]], self.box_classes, height, width)
            if box_file
            else np.zeros((0, 5), dtype=np.float64)
        )
        return Sample(
            id=sample_id,
            image=image,
            target=boxes,
            group=self.group_of(sample_id),
            path=self.path(image_name),
        )


@register_loader("table-boxes")
class TableBoxes(_Indexed):
    """Images below `root` with boxes listed in a delimited text table.

    Each row names an image, matched by stem, its class and the pixel coordinates of corners of
    its box, whose extremes bound the box. Rows of other classes, without an image name or with
    missing or non numeric corners are skipped, so that only images holding at least one box of a
    listed class are kept.

    Args:
        files: Files of the dataset.
        table: Table to read, relative to the dataset root.
        key: Column holding the image file name.
        label: Column holding the object class.
        box_classes: Class index of every value of `label` that is kept.
        x: Columns holding horizontal corner coordinates in pixels.
        y: Columns holding vertical corner coordinates in pixels.
        root: Directory below which images are searched, empty for the dataset root.
        group_column: Column holding the patient or video, or `None` to use the `group`
            expression of `_Indexed`.
        delimiter: Column separator of the table.
        **options: Options of `_Indexed`, namely `group`, `official_test` and `expand`.
    """

    def __init__(
        self,
        files: "DatasetFiles",
        table: str,
        key: str,
        label: str,
        box_classes: dict[str, int],
        x: list[str],
        y: list[str],
        root: str = "",
        group_column: str | None = None,
        delimiter: str = ",",
        **options: Any,
    ) -> None:
        super().__init__(files, **options)
        self.table = table
        self.key = key
        self.label = label
        self.box_classes = box_classes
        self.x, self.y = x, y
        self.root = root
        self.group_column = group_column
        self.delimiter = delimiter

    @cached_property
    def rows(self) -> dict[str, tuple[list[tuple[int, float, float, float, float]], str]]:
        """Pixel boxes as class, x0, y0, x1, y1 and the group of every listed stem.

        When a stem is listed several times its last row provides the group.
        """
        boxes: dict[str, list[tuple[int, float, float, float, float]]] = defaultdict(list)
        groups: dict[str, str] = {}
        for row in read_rows(self.files.read_text(self.table), self.delimiter):
            cls = self.box_classes.get(row.get(self.label, "").strip())
            name = row.get(self.key, "").strip()
            try:
                xs = [float(row[column]) for column in self.x]
                ys = [float(row[column]) for column in self.y]
            except (KeyError, ValueError):
                continue
            if cls is None or not name:
                continue
            key = stem(name)
            boxes[key].append((cls, min(xs), min(ys), max(xs), max(ys)))
            groups[key] = row.get(self.group_column, "") if self.group_column else ""
        return {key: (value, groups[key]) for key, value in boxes.items()}

    def index(self) -> dict[str, tuple[str, ...]]:
        """Image file of every listed stem found below `root`.

        Returns:
            Name of the image of every sample, as a tuple of one, keyed by stem.
        """
        images = {stem(n): n for n in self.files.under(self.root, IMAGE_SUFFIXES)}
        return {key: (images[key],) for key in self.rows if key in images}

    def group_of(self, sample_id: str) -> str:
        """Patient or video of one sample.

        Args:
            sample_id: Identifier of the sample.

        Returns:
            The value of `group_column` when that column is set, and the group of
            `_Indexed.group_of` otherwise.
        """
        if self.group_column is None:
            return super().group_of(sample_id)
        return self.rows[sample_id][1]

    def labels(self) -> dict[str, int]:
        """Class of the first box of every image, which stratifies the split.

        Returns:
            Class index keyed by identifier.
        """
        return {key: self.rows[key][0][0][0] for key in self.entries}

    def decode(self, sample_id: str, contents: Mapping[str, bytes]) -> Sample:
        """Build a detection sample from its image and listed boxes.

        Args:
            sample_id: Identifier of the sample.
            contents: Bytes of the image keyed by file name.

        Returns:
            Sample whose target is the box table of its rows, with corners clipped to the image.
        """
        (name,) = self.entries[sample_id]
        image = read_rgb(contents[name])
        height, width = image.shape[:2]
        scale = np.array([1.0, width, height, width, height])
        boxes = np.asarray(self.rows[sample_id][0], dtype=np.float64).reshape(-1, 5) / scale
        boxes[:, 1:] = boxes[:, 1:].clip(0, 1)
        return Sample(
            id=sample_id,
            image=image,
            target=boxes,
            group=self.group_of(sample_id),
            path=self.path(name),
        )


def yolo_boxes(text: str, offset: int = 1) -> NDArray[np.float64]:
    """Boxes of a YOLO file of normalized centres and sizes.

    Each line holds a class followed by the centre and size of a box as fractions of the image,
    and lines with fewer than five fields are skipped.

    Args:
        text: Content of the YOLO file.
        offset: Shift added to every class, one by default since YOLO counts classes from zero
            while the benchmark reserves zero for the background.

    Returns:
        Box table of shape `(N, 5)` with rows class, x0, y0, x1, y1 normalized by the image size
        and clipped to the unit interval.
    """
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        cls, x, y, width, height = (float(value) for value in fields[:5])
        rows.append([cls + offset, x - width / 2, y - height / 2, x + width / 2, y + height / 2])
    boxes = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
    boxes[:, 1:] = boxes[:, 1:].clip(0, 1)
    return boxes


@register_loader("yolo-boxes")
class YOLOBoxes(_Indexed):
    """Images in `images` with YOLO box files of the same stem in `labels`.

    Images without a box file are skipped, and identifiers are image stems.

    Args:
        files: Files of the dataset.
        images: Directory below which images are searched.
        labels: Directory below which the text box files are searched, or `None` for `images`.
        **options: Options of `_Indexed`, namely `group`, `official_test` and `expand`.
    """

    def __init__(
        self, files: "DatasetFiles", images: str, labels: str | None = None, **options: Any
    ) -> None:
        super().__init__(files, **options)
        self.images = images
        self.labels_dir = labels or images

    def index(self) -> dict[str, tuple[str, ...]]:
        """Image and box file of every image with a box file.

        Returns:
            Names of the image and of its box file keyed by image stem.
        """
        boxes = {stem(n): n for n in self.files.under(self.labels_dir, frozenset({".txt"}))}
        found: dict[str, tuple[str, ...]] = {}
        for name in self.files.under(self.images, IMAGE_SUFFIXES):
            label = boxes.get(stem(name))
            if label is not None:
                found[stem(name)] = (name, label)
        return found

    def decode(self, sample_id: str, contents: Mapping[str, bytes]) -> Sample:
        """Build a detection sample from its image and box file.

        Args:
            sample_id: Identifier of the sample.
            contents: Bytes of the image and box file keyed by file name.

        Returns:
            Sample whose target is the box table of the file, with classes shifted by one.
        """
        image, label = self.entries[sample_id]
        return Sample(
            id=sample_id,
            image=read_rgb(contents[image]),
            target=yolo_boxes(contents[label].decode("utf-8", "replace")),
            group=self.group_of(sample_id),
            path=self.path(image),
        )
