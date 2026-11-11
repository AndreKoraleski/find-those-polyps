"""Generic loaders for datasets organized as class folders, label tables or image and mask folders.

The loaders are configured entirely from the task's `loader_options`, so most datasets need no
code of their own. Sample identifiers are file stems, prefixed by the directly enclosing folder for
stems that occur in several folders, or paths below the root without suffix when `id_mode` is
`path`. Groups are taken from the identifier by an optional regular expression with one capturing
group, and each image is its own group otherwise. A published test partition is read from a text
table of the dataset, optionally filtered on one column, or selected by a regular expression over
identifiers for benchmark defined external test sets. Nested tarballs listed in `expand` are
unpacked before the files are indexed.
"""

import re
from collections import Counter
from collections.abc import Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from find_those_polyps.tasks.loaders.base import FileLoader, register_loader
from find_those_polyps.tasks.loaders.images import (
    IMAGE_SUFFIXES,
    parent,
    read_mask,
    read_rgb,
    read_rows,
    stem,
)
from find_those_polyps.tasks.schema import Sample

if TYPE_CHECKING:
    from find_those_polyps.data.files import DatasetFiles


class OfficialTest(BaseModel):
    """Location of a test partition, published as a table or defined by an identifier pattern.

    Attributes:
        source: Description of the test partition recorded in the split file.
        file: Table listing the test images, relative to the dataset root, or `None` when `match`
            selects them.
        name: Column holding the file names.
        delimiter: Column separator of the table.
        header: Whether the first line of the table names its columns. Without a header the table
            lists one file name per line.
        column: Column on which rows are filtered, or `None` to keep every row.
        value: Value that `column` must hold for a row to be kept.
        match: Regular expression selecting every identifier it matches, used instead of `file`.
        folder: Column holding the folder of each file, for datasets whose file names repeat
            across folders, or `None`.
    """

    model_config = ConfigDict(frozen=True)

    source: str
    file: str | None = None
    name: str = "name"
    delimiter: str = ","
    header: bool = True
    column: str | None = None
    value: str | None = None
    match: str | None = None
    folder: str | None = None

    def key(self, name: str) -> str:
        """Key of a file name, qualified by its folder when `folder` is set.

        Args:
            name: Slash separated name of a dataset file.

        Returns:
            The stem of `name`, prefixed by its directly enclosing folder when `folder` is set.
        """
        return stem(name) if self.folder is None else f"{parent(name)}/{stem(name)}"

    def stems(self, files: "DatasetFiles") -> set[str]:
        """Keys of the files listed in the table, in the form produced by `key`.

        Args:
            files: Files of the dataset holding the table.

        Returns:
            Keys of the listed files of the kept rows, empty when no table is given.
        """
        if self.file is None:
            return set()
        text = files.read_text(self.file)
        if not self.header:
            text = f"{self.name}\n{text}"
        rows = read_rows(text, self.delimiter)
        return {
            stem(row[self.name].strip())
            if self.folder is None
            else f"{row[self.folder].strip()}/{stem(row[self.name].strip())}"
            for row in rows
            if row.get(self.name)
            and (self.column is None or row[self.column].strip() == self.value)
        }


def _group(pattern: re.Pattern[str] | None, sample_id: str) -> str:
    """Group of an identifier, the first capture of `pattern` or the identifier itself.

    Args:
        pattern: Expression with one capturing group searched in `sample_id`, or `None`.
        sample_id: Identifier of the sample.

    Returns:
        The captured text when `pattern` matches, and `sample_id` otherwise.
    """
    match = pattern.search(sample_id) if pattern is not None else None
    return match[1] if match is not None else sample_id


class _Indexed(FileLoader):
    """Shared handling of groups, published test partitions and nested tarballs.

    Subclasses implement `index`, which maps every identifier to its files, and the result is
    computed once and kept in `entries`.

    Args:
        files: Files of the dataset.
        group: Regular expression with one capturing group that extracts the group from an
            identifier, or `None` to make each sample its own group.
        official_test: Fields of an `OfficialTest`, or `None` when no test partition is published.
        expand: Nested tarballs unpacked before indexing.
        **options: Options passed on to `Loader`.
    """

    def __init__(
        self,
        files: "DatasetFiles",
        group: str | None = None,
        official_test: dict[str, Any] | None = None,
        expand: list[str] | None = None,
        **options: Any,
    ) -> None:
        super().__init__(files, **options)
        self.pattern = re.compile(group) if group else None
        self.test = OfficialTest.model_validate(official_test) if official_test else None
        self.tarballs = expand or []

    @cached_property
    def entries(self) -> dict[str, tuple[str, ...]]:
        """Files of every sample keyed by identifier, empty when the files are unavailable."""
        if not self.files.available():
            return {}
        for tarball in self.tarballs:
            self.files.expand(tarball)
        return self.index()

    def index(self) -> dict[str, tuple[str, ...]]:
        """Files of every sample keyed by identifier.

        Returns:
            Names of the files of each sample, its image first.

        Raises:
            NotImplementedError: Always, since subclasses define the index.
        """
        raise NotImplementedError

    def available(self) -> bool:
        """Whether the files are readable and index at least one sample.

        Returns:
            Whether the dataset can be read.
        """
        return self.files.available() and bool(self.entries)

    def ids(self) -> list[str]:
        """Identifiers of all samples.

        Returns:
            Sorted identifiers of the indexed samples.
        """
        return sorted(self.entries)

    def group_of(self, sample_id: str) -> str:
        """Patient, video or lesion of one sample.

        Args:
            sample_id: Identifier of the sample.

        Returns:
            The group extracted by the `group` expression, or the identifier itself.
        """
        return _group(self.pattern, sample_id)

    def groups(self) -> dict[str, str]:
        """Patient, video or lesion of each sample.

        Returns:
            Group of every indexed sample keyed by identifier.
        """
        return {sample_id: self.group_of(sample_id) for sample_id in self.entries}

    def members(self, sample_id: str) -> tuple[str, ...]:
        """Names of the files holding one sample, its image first.

        Args:
            sample_id: Identifier of the sample.

        Returns:
            The indexed files of the sample.
        """
        return self.entries[sample_id]

    def official_test(self) -> tuple[list[str], str] | None:
        """Identifiers of the published test partition and its source.

        A `match` pattern selects the identifiers it matches. Otherwise a sample belongs to the
        test partition when the key of its image, in the form of `OfficialTest.key`, is listed in
        the table.

        Returns:
            Sorted test identifiers with their source, or `None` when no test partition is
            configured.
        """
        if self.test is None:
            return None
        if self.test.match is not None:
            pattern = re.compile(self.test.match)
            chosen = [key for key in self.entries if pattern.search(key)]
        else:
            stems = self.test.stems(self.files)
            chosen = [
                key for key, names in self.entries.items() if self.test.key(names[0]) in stems
            ]
        return sorted(chosen), self.test.source


def _ancestor(name: str, depth: int) -> str:
    """Folder `depth` levels above a file, or an empty string when the name is too shallow.

    Args:
        name: Slash separated file name.
        depth: Number of levels to climb, one for the directly enclosing folder.

    Returns:
        Name of the folder.
    """
    parts = name.split("/")
    return parts[-1 - depth] if len(parts) > depth else ""


@register_loader("folders")
class FolderClassification(_Indexed):
    """Images labelled by the name of an enclosing folder.

    The class folder is the parent of each image, or the ancestor `depth` levels up. Images in
    other folders are ignored. Identifiers are image stems, prefixed by the directly enclosing
    folder for stems that occur more than once, or paths below `root` without suffix when
    `id_mode` is `path`.

    Args:
        files: Files of the dataset.
        root: Directory below which images are searched, empty for the dataset root.
        classes: For each class index, the folder names merged into it.
        depth: Number of levels between an image and its class folder.
        include: Regular expression searched in the full file name of each image, which is kept
            only on a match, or `None` to keep every image.
        id_mode: Form of the identifiers, `stem` or `path`.
        **options: Options of `_Indexed`, namely `group`, `official_test` and `expand`.
    """

    def __init__(
        self,
        files: "DatasetFiles",
        root: str,
        classes: list[list[str]],
        depth: int = 1,
        include: str | None = None,
        id_mode: Literal["stem", "path"] = "stem",
        **options: Any,
    ) -> None:
        super().__init__(files, **options)
        self.root = root
        self.depth = depth
        self.include = re.compile(include) if include else None
        self.id_mode = id_mode
        self.folders = {folder: index for index, names in enumerate(classes) for folder in names}

    def _class(self, name: str) -> int:
        """Class index of an image file from its class folder."""
        return self.folders[_ancestor(name, self.depth)]

    def index(self) -> dict[str, tuple[str, ...]]:
        """Image file of every sample keyed by identifier.

        Returns:
            Name of every image found in a class folder below `root`, as a tuple of one.
        """
        names = [
            n
            for n in self.files.under(self.root, IMAGE_SUFFIXES)
            if _ancestor(n, self.depth) in self.folders
            and (self.include is None or self.include.search(n))
        ]
        if self.id_mode == "path":
            start = len(self.root.rstrip("/")) + 1 if self.root else 0
            return {n[start:].rsplit(".", 1)[0]: (n,) for n in names}
        counts = Counter(stem(n) for n in names)
        return {
            stem(n) if counts[stem(n)] == 1 else f"{_ancestor(n, 1)}/{stem(n)}": (n,) for n in names
        }

    def labels(self) -> dict[str, int]:
        """Class of each sample.

        Returns:
            Class index of every indexed sample keyed by identifier.
        """
        return {key: self._class(names[0]) for key, names in self.entries.items()}

    def decode(self, sample_id: str, contents: Mapping[str, bytes]) -> Sample:
        """Build a classification sample from its image.

        Args:
            sample_id: Identifier of the sample.
            contents: Bytes of the image keyed by file name.

        Returns:
            Sample whose target is the class index of its folder.
        """
        (name,) = self.entries[sample_id]
        return Sample(
            id=sample_id,
            image=read_rgb(contents[name]),
            target=self._class(name),
            group=self.group_of(sample_id),
            path=self.path(name),
        )


@register_loader("table")
class TableClassification(_Indexed):
    """Images labelled by rows of delimited text tables.

    Each row names an image file, matched by stem against the images below `root`. The class is
    either the value of the `label` column mapped through `classes`, or the first class among
    `one_hot` whose columns hold a one. Rows without a class or an image name are skipped, and
    stems listed under different classes are excluded and recorded in `conflicts`.

    Args:
        files: Files of the dataset.
        tables: Tables to read, relative to the dataset root.
        key: Column holding the image file name.
        root: Directory below which images are searched, empty for the dataset root.
        label: Column holding the class value, used together with `classes`.
        classes: For each class index, the values of `label` merged into it.
        one_hot: For each class index, the indicator columns of which any holding a one assigns
            the class. Exactly one of `label` and `one_hot` must be given.
        group_column: Column holding the patient or video, or `None` to use the `group`
            expression of `_Indexed`. A sample whose row lacks the column is its own group.
        delimiter: Column separator of the tables.
        **options: Options of `_Indexed`, namely `group`, `official_test` and `expand`.

    Raises:
        ValueError: If both or neither of `label` and `one_hot` are given.
    """

    def __init__(
        self,
        files: "DatasetFiles",
        tables: list[str],
        key: str,
        root: str = "",
        label: str | None = None,
        classes: list[list[str]] | None = None,
        one_hot: list[list[str]] | None = None,
        group_column: str | None = None,
        delimiter: str = ",",
        **options: Any,
    ) -> None:
        super().__init__(files, **options)
        if (label is None) == (one_hot is None):
            raise ValueError("give either label with classes or one_hot")
        self.tables = tables
        self.key = key
        self.root = root
        self.label = label
        self.values = {v: i for i, values in enumerate(classes or []) for v in values}
        self.one_hot = one_hot or []
        self.group_column = group_column
        self.delimiter = delimiter

    def _row_class(self, row: dict[str, str]) -> int | None:
        """Class index of a table row, or `None` when the row names no known class."""
        if self.label is not None:
            return self.values.get(row.get(self.label, "").strip())
        for index, columns in enumerate(self.one_hot):
            if any(row.get(column, "").strip() in ("1", "1.0") for column in columns):
                return index
        return None

    @cached_property
    def rows(self) -> dict[str, tuple[int, str | None]]:
        """Class and group column value of every labelled stem.

        Stems listed with different classes are excluded, and their names kept in the sorted list
        `conflicts`. When a stem is listed several times its last row provides the group, which is
        `None` when that row lacks the group column or no group column is set.
        """
        found: dict[str, tuple[int, str | None]] = {}
        conflicts: set[str] = set()
        for table in self.tables:
            for row in read_rows(self.files.read_text(table), self.delimiter):
                target = self._row_class(row)
                name = row.get(self.key, "").strip()
                if target is None or not name:
                    continue
                key = stem(name)
                if key in found and found[key][0] != target:
                    conflicts.add(key)
                group = row.get(self.group_column) if self.group_column else None
                found[key] = (target, group)
        self.conflicts = sorted(conflicts)
        return {key: value for key, value in found.items() if key not in conflicts}

    def index(self) -> dict[str, tuple[str, ...]]:
        """Image file of every labelled stem found below `root`.

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
            The value of `group_column` when that column is set, the identifier itself when the
            row lacks that column, and the group of `_Indexed.group_of` otherwise.
        """
        if self.group_column is None:
            return super().group_of(sample_id)
        group = self.rows[sample_id][1]
        return sample_id if group is None else group

    def labels(self) -> dict[str, int]:
        """Class of each sample.

        Returns:
            Class index of every indexed sample keyed by identifier.
        """
        return {key: self.rows[key][0] for key in self.entries}

    def decode(self, sample_id: str, contents: Mapping[str, bytes]) -> Sample:
        """Build a classification sample from its image.

        Args:
            sample_id: Identifier of the sample.
            contents: Bytes of the image keyed by file name.

        Returns:
            Sample whose target is the class index of its table row.
        """
        (name,) = self.entries[sample_id]
        return Sample(
            id=sample_id,
            image=read_rgb(contents[name]),
            target=self.rows[sample_id][0],
            group=self.group_of(sample_id),
            path=self.path(name),
        )


@register_loader("masks")
class MaskSegmentation(_Indexed):
    """Images in `images` paired with binary masks in `masks`.

    Identifiers are image stems. Images without a mask are skipped.

    Args:
        files: Files of the dataset.
        images: Directory below which images are searched.
        masks: Directory below which masks are searched.
        threshold: Lowest grey level, between 0 and 255, counted as foreground.
        mask_name: Regular expression and its replacement rewriting an image stem into the stem of
            its mask, or `None` when both stems are equal.
        **options: Options of `_Indexed`, namely `group`, `official_test` and `expand`.
    """

    def __init__(
        self,
        files: "DatasetFiles",
        images: str,
        masks: str,
        threshold: int = 128,
        mask_name: tuple[str, str] | None = None,
        **options: Any,
    ) -> None:
        super().__init__(files, **options)
        self.images = images
        self.masks = masks
        self.threshold = threshold
        self.rename = (re.compile(mask_name[0]), mask_name[1]) if mask_name else None

    def _mask_stem(self, image_stem: str) -> str:
        """Stem of the mask belonging to an image stem."""
        if self.rename is None:
            return image_stem
        return self.rename[0].sub(self.rename[1], image_stem)

    def index(self) -> dict[str, tuple[str, ...]]:
        """Image and mask file of every image with a mask.

        Returns:
            Names of the image and of its mask keyed by image stem.
        """
        masks = {stem(n): n for n in self.files.under(self.masks, IMAGE_SUFFIXES)}
        found: dict[str, tuple[str, ...]] = {}
        for name in self.files.under(self.images, IMAGE_SUFFIXES):
            mask = masks.get(self._mask_stem(stem(name)))
            if mask is not None:
                found[stem(name)] = (name, mask)
        return found

    def decode(self, sample_id: str, contents: Mapping[str, bytes]) -> Sample:
        """Build a segmentation sample from its image and mask.

        Args:
            sample_id: Identifier of the sample.
            contents: Bytes of the image and mask keyed by file name.

        Returns:
            Sample whose target is a binary mask of shape `(H, W)`.
        """
        image, mask = self.entries[sample_id]
        return Sample(
            id=sample_id,
            image=read_rgb(contents[image]),
            target=read_mask(contents[mask], self.threshold),
            group=self.group_of(sample_id),
            path=self.path(image),
        )
