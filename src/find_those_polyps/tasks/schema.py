"""Data models describing downstream tasks and their samples."""

import math
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PlainValidator,
    model_serializer,
    model_validator,
)

from find_those_polyps.utils.hashing import canonical_sha256


class TaskKind(StrEnum):
    """Structure of the prediction target.

    Attributes:
        CLASSIFICATION: One class index per image.
        ORDINAL: One ordered class index per image, for instance a Mayo or BBPS grade.
        MULTILABEL: A subset of classes per image, given as a multi hot vector.
        SEGMENTATION: One class index per pixel, with class 0 as the background.
        DETECTION: Boxes with one class each, counted from 1 since 0 is the background.
        MULTILABEL_SEGMENTATION: One binary mask per class, where masks may overlap.
    """

    CLASSIFICATION = "classification"
    ORDINAL = "ordinal"
    MULTILABEL = "multilabel"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    MULTILABEL_SEGMENTATION = "multilabel_segmentation"


class Split(StrEnum):
    """Partition of a task's samples.

    Attributes:
        TRAIN: Training partition, from which label budgets are drawn.
        VAL: Validation partition used for model selection.
        TEST: Canonical test partition.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def _validate_image(value: object) -> NDArray[np.uint8]:
    """Accept an RGB image as a uint8 array of shape `(H, W, 3)`.

    Args:
        value: Candidate image.

    Returns:
        The array unchanged.

    Raises:
        ValueError: If `value` is not a uint8 array of shape `(H, W, 3)`.
    """
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[2] != 3
    ):
        raise ValueError("image must be a uint8 array of shape (H, W, 3)")
    return value


def _validate_target(
    value: object,
) -> int | NDArray[np.int64] | NDArray[np.uint8] | NDArray[np.float64]:
    """Normalize a target to a Python integer, an integer array or a float64 box table.

    Args:
        value: Candidate target. Integers other than booleans are class indices, integer arrays
            of one to three dimensions are multi hot vectors, masks or stacks of masks, and
            floating arrays of shape `(N, 5)` are box tables.

    Returns:
        The class index as `int`, a multi hot vector in int64, a mask or stack of masks in uint8
        when its values lie in `[0, 255]` and in int64 otherwise, or a box table in float64. Arrays
        are cast without copying when possible. Masks keep one byte per pixel, since a task holds
        the masks of all its images at their native resolution.

    Raises:
        ValueError: If `value` matches none of the accepted forms.
    """
    if isinstance(value, int | np.integer) and not isinstance(value, bool):
        return int(value)
    if (
        isinstance(value, np.ndarray)
        and value.ndim in (1, 2, 3)
        and np.issubdtype(value.dtype, np.integer)
    ):
        small = value.ndim > 1 and (value.size == 0 or (value.min() >= 0 and value.max() <= 255))
        if small:
            return value.astype(np.uint8, copy=False)
        return value.astype(np.int64, copy=False)
    if (
        isinstance(value, np.ndarray)
        and value.ndim == 2
        and value.shape[1] == 5
        and np.issubdtype(value.dtype, np.floating)
    ):
        return value.astype(np.float64, copy=False)
    raise ValueError(
        "target must be a class index, a multi-hot vector, a mask, a stack of masks or a box table"
    )


Image = Annotated[NDArray[np.uint8], PlainValidator(_validate_image)]
Target = Annotated[
    int | NDArray[np.int64] | NDArray[np.uint8] | NDArray[np.float64],
    PlainValidator(_validate_target),
]


class Sample(BaseModel):
    """Image at native resolution with its target and grouping identifier.

    Samples sharing a group are never assigned to different partitions.

    Attributes:
        id: Identifier of the sample, unique within its dataset.
        image: RGB image as a uint8 array of shape `(H, W, 3)`.
        target: Class index for classification and ordinal tasks, int64 multi hot vector of shape
            `(C,)` for multilabel tasks, int64 mask of shape `(H, W)` for segmentation, int64
            stack of binary masks of shape `(C, H, W)` for multilabel segmentation, or float64
            box table of shape `(N, 5)` for detection, whose rows hold class, x0, y0, x1, y1 with
            corners normalized by the image width and height.
        group: Patient, video or centre of origin.
        path: Location of the image file relative to the data root, so that per image results can
            be traced back to the canonical file. Empty when unknown.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    image: Image
    target: Target
    group: str
    path: str = ""

    @model_validator(mode="after")
    def _mask_matches_image(self) -> Self:
        """Require masks and mask stacks to share the height and width of the image.

        Returns:
            The validated sample.

        Raises:
            ValueError: If an integer target of two or three dimensions differs from the image in
                its last two dimensions.
        """
        target = self.target
        if (
            isinstance(target, np.ndarray)
            and target.ndim in (2, 3)
            and np.issubdtype(target.dtype, np.integer)
            and target.shape[-2:] != self.image.shape[:2]
        ):
            raise ValueError(f"mask shape {target.shape} differs from image {self.image.shape[:2]}")
        return self


class Budget(BaseModel):
    """Labelled data budget drawn from the training partition.

    Written as `full`, `frac:<p>` for a fraction `p` of the partition, `shots:<k>` for `k` samples
    per class or `n:<k>` for `k` samples in total, and serialized back to that form. Subsets drawn
    with the same seed are nested within fraction and count budgets and within shot budgets.

    Attributes:
        kind: Type of budget.
        value: Fraction in the open unit interval for `frac`, positive integer for `shots` and
            `n`, and zero for `full`.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["full", "frac", "shots", "n"]
    value: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _parse(cls, data: Any) -> Any:
        """Split a written budget such as `frac:0.1` into its kind and value.

        Args:
            data: Raw input, parsed when it is a string and passed through otherwise.

        Returns:
            Mapping of `kind` and `value`, or `data` unchanged.
        """
        if isinstance(data, str):
            kind, _, value = data.partition(":")
            return {"kind": kind, "value": float(value) if value else 0.0}
        return data

    @model_validator(mode="after")
    def _check_value(self) -> Self:
        """Check that the value suits the kind.

        Returns:
            The validated budget.

        Raises:
            ValueError: If a full budget has a value, a fraction lies outside the open unit
                interval or a count is not a positive integer.
        """
        if self.kind == "full" and self.value != 0:
            raise ValueError("budget 'full' takes no value")
        if self.kind == "frac" and not 0 < self.value < 1:
            raise ValueError("fraction budget must lie in (0, 1)")
        if self.kind in {"shots", "n"} and (self.value < 1 or not self.value.is_integer()):
            raise ValueError("count budget must be a positive integer")
        return self

    @model_serializer
    def _serialize(self) -> str:
        """Serialize the budget in its written form."""
        return str(self)

    def __str__(self) -> str:
        """Written form, such as `full`, `frac:0.1`, `shots:5` or `n:100`."""
        if self.kind == "full":
            return "full"
        value = self.value if self.kind == "frac" else int(self.value)
        return f"{self.kind}:{value}"

    @property
    def is_full(self) -> bool:
        """Whether the budget keeps the whole training partition."""
        return self.kind == "full"

    @property
    def slug(self) -> str:
        """Short label used in configuration names, such as `10pct`, `5shot` or `n100`."""
        if self.kind == "frac":
            return f"{self.value * 100:g}pct"
        if self.kind == "shots":
            return f"{int(self.value)}shot"
        if self.kind == "n":
            return f"n{int(self.value)}"
        return "full"

    @property
    def order(self) -> tuple[int, float]:
        """Sort key grouping budgets by kind and placing larger budgets first within a kind.

        Kinds follow the order full, fraction, count and shots.
        """
        return (["full", "frac", "n", "shots"].index(self.kind), -self.value)

    def size(self, n_samples: int, n_classes: int) -> int:
        """Number of samples a budget selects.

        Args:
            n_samples: Size of the training partition.
            n_classes: Number of classes present in the partition, one for tasks without single
                labels.

        Returns:
            The rounded up fraction of `n_samples` but at least one sample per class for `frac`,
            the count for `n`, the count times `n_classes` for `shots` and `n_samples` for `full`.
            The result is not capped at `n_samples`.
        """
        if self.kind == "frac":
            return max(math.ceil(self.value * n_samples), n_classes)
        if self.kind == "n":
            return int(self.value)
        if self.kind == "shots":
            return int(self.value) * n_classes
        return n_samples


FULL = Budget(kind="full")


class TaskSpec(BaseModel):
    """Declarative definition of a downstream task.

    A task fixes a dataset, a target, a train, validation and test partition and a primary
    metric. Its `hash` changes whenever any field changes, so results obtained under different
    definitions are never ranked together.

    Attributes:
        id: Identifier of two or more slash separated components in lower case letters, digits,
            dots, underscores and hyphens.
        description: Nonempty description of the task.
        dataset: Key of the registered dataset the task is defined on.
        kind: Structure of the prediction target.
        classes: Names of the classes, at least two, indexed as in the targets.
        primary_metric: Name of the metric that ranks encoders on this task.
        budgets: Label budgets evaluated on the task, the full budget alone by default.
        loader: Key under which the loader of the dataset is registered.
        loader_options: Keyword arguments passed to the loader constructor.
        splits: File name of the canonical split, relative to the splits directory.
        splits_sha256: Hexadecimal SHA-256 of the split file, empty while it has not been built.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)+$")
    description: str = Field(min_length=1)
    dataset: str
    kind: TaskKind
    classes: tuple[str, ...] = Field(min_length=2)
    primary_metric: str
    budgets: tuple[Budget, ...] = (FULL,)
    loader: str
    loader_options: dict[str, JsonValue] = Field(default_factory=dict)
    splits: str
    splits_sha256: str = ""

    @property
    def name(self) -> str:
        """Last component of the identifier, unique within the registry."""
        return self.id.rsplit("/", 1)[-1]

    @property
    def hash(self) -> str:
        """First twelve hexadecimal digits of the content hash of every field.

        The hash includes `splits_sha256` and hence the content of the split file.
        """
        return canonical_sha256(self.model_dump(mode="json"))[:12]
