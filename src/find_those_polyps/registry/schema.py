"""Data models of the versioned task registry and its dataset documentation."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from find_those_polyps.tasks.schema import TaskSpec
from find_those_polyps.utils.hashing import canonical_sha256


class Access(StrEnum):
    """How a dataset is obtained.

    Attributes:
        OPEN: Direct download without registration.
        ACCOUNT: Download after registering an account with the host.
        REQUEST: Access granted on request or after signing an agreement.
        UNAVAILABLE: No longer obtainable from any known host.
    """

    OPEN = "open"
    ACCOUNT = "account"
    REQUEST = "request"
    UNAVAILABLE = "unavailable"


class OverlapKind(StrEnum):
    """Nature of the content shared between two datasets.

    Attributes:
        DUPLICATE: Identical images or frames.
        SAME_PATIENTS: Different images of the same patients or videos.
        SAME_SOURCE: Same institution and period, with shared patients unverified.
    """

    DUPLICATE = "duplicate"
    SAME_PATIENTS = "same_patients"
    SAME_SOURCE = "same_source"


class Overlap(BaseModel):
    """Documented content shared with another dataset.

    The other side is either a registered dataset or a corpus outside the benchmark, such as an
    unlabelled pretraining pool, which then needs a display `name`.

    Attributes:
        dataset: Key of the other dataset or corpus, in lower case letters, digits and hyphens.
        kind: Nature of the shared content.
        evidence: URL or citation supporting the claim.
        note: Optional remark rendered after the warning.
        name: Display name of the other side, required when `dataset` is not registered.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    kind: OverlapKind
    evidence: str = Field(min_length=1)
    note: str = ""
    name: str = ""


class Download(BaseModel):
    """Published file of an openly downloadable dataset.

    Zip archives are streamed member by member and pinned by `index`, the hash of their member
    names, sizes and CRC32 values, together with their byte `size`. Plain files are always
    downloaded whole and must carry a `checksum`.

    Attributes:
        url: Address of the file on the official host.
        filename: Name under which the file is stored locally.
        checksum: Digest of the whole file written as `md5`, `sha1` or `sha256` followed by a
            colon and the hexadecimal value, published by the host or computed at registration.
            Empty when an archive is pinned by `index` alone.
        size: Size of the file in bytes, or `None` when unknown.
        index: Hash of the archive listing written as `sha256` followed by a colon and 64
            hexadecimal digits, or empty.
        extract: Whether the file is a zip archive whose members are read individually.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    filename: str
    checksum: str = Field(default="", pattern=r"^((md5|sha1|sha256):[0-9a-f]+)?$")
    size: int | None = None
    index: str = Field(default="", pattern=r"^(sha256:[0-9a-f]{64})?$")
    extract: bool = True

    @model_validator(mode="after")
    def _pinned(self) -> Self:
        """Require a checksum, or an index for archives.

        Returns:
            The validated download.

        Raises:
            ValueError: If the file has no checksum and is not an archive pinned by an index.
        """
        if not self.checksum and not (self.extract and self.index):
            raise ValueError(f"{self.filename} needs a checksum or, for archives, an index")
        return self


class DatasetCard(BaseModel):
    """Provenance, licence, access and overlaps of a dataset, labelled or not.

    Attributes:
        key: Unique identifier in lower case letters, digits, dots, underscores and hyphens, also
            the name of the dataset directory below the data root.
        name: Display name.
        labelled: Whether the dataset carries labels usable for a downstream task.
        modality: Imaging modality, for instance colonoscopy or capsule endoscopy.
        licence: Licence under which the dataset is distributed.
        access: How the dataset is obtained.
        url: Landing page of the dataset.
        citation: Reference to cite when using the dataset.
        provenance: Description of the clinical origin of the images.
        family: Clinical source pool the dataset is drawn from, the unit at which leakage must be
            reasoned about.
        label_source: Who assigned the labels, for instance histopathology or endoscopist
            consensus, or empty when unknown.
        downloads: Published files of the dataset, empty when it cannot be downloaded directly.
        overlaps: Documented content shared with other datasets or corpora.
        notes: Free text rendered after the provenance.
    """

    model_config = ConfigDict(frozen=True)

    key: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    name: str
    labelled: bool
    modality: str
    licence: str
    access: Access
    url: str
    citation: str
    provenance: str
    family: str
    label_source: str = ""
    downloads: tuple[Download, ...] = ()
    overlaps: tuple[Overlap, ...] = ()
    notes: str = ""


class Registry(BaseModel):
    """Datasets and tasks of the benchmark under a content derived version.

    Attributes:
        version: Counter incremented whenever the content hash changes.
        sha256: Hexadecimal content hash recorded at the last synchronization, compared against
            `content_sha256` to detect stale registries.
        datasets: Cards of every dataset, with unique keys.
        tasks: Definitions of every task, with unique identifiers and names, each on a registered
            dataset.
    """

    model_config = ConfigDict(frozen=True)

    version: int = Field(ge=0)
    sha256: str
    datasets: tuple[DatasetCard, ...] = ()
    tasks: tuple[TaskSpec, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Check that keys and names are unique and that every reference resolves.

        Returns:
            The validated registry.

        Raises:
            ValueError: If dataset keys, task identifiers or task names repeat, if a task names an
                unknown dataset, or if an overlap names an unregistered dataset without a display
                name.
        """
        keys = [dataset.key for dataset in self.datasets]
        if len(set(keys)) != len(keys):
            raise ValueError("dataset keys must be unique")
        ids = [task.id for task in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task ids must be unique")
        names = [task.name for task in self.tasks]
        if len(set(names)) != len(names):
            raise ValueError("task names, the last component of each id, must be unique")
        known = set(keys)
        for task in self.tasks:
            if task.dataset not in known:
                raise ValueError(f"task {task.id!r} names unknown dataset")
        for dataset in self.datasets:
            for overlap in dataset.overlaps:
                if overlap.dataset not in known and not overlap.name:
                    raise ValueError(
                        f"{dataset.key!r} overlaps unknown dataset {overlap.dataset!r} without "
                        "naming it"
                    )
        return self

    def corpora(self) -> set[str]:
        """Keys of the registered datasets and of the corpora they are documented to overlap.

        Returns:
            Every key that a `pretrained_on` declaration can match without being reported as
            overlapping no task.
        """
        keys = {dataset.key for dataset in self.datasets}
        return keys | {o.dataset for dataset in self.datasets for o in dataset.overlaps}

    def content_sha256(self) -> str:
        """Hash of datasets and tasks, excluding the version fields.

        Returns:
            Hexadecimal SHA-256 of the canonical JSON encoding of every field except `version` and
            `sha256`. The task split hashes are included, so the hash covers the split files.
        """
        return canonical_sha256(self.model_dump(mode="json", exclude={"version", "sha256"}))

    def dataset(self, key: str) -> DatasetCard:
        """Card of one dataset.

        Args:
            key: Key of the dataset.

        Returns:
            The card registered under `key`.

        Raises:
            KeyError: If no dataset is registered under `key`.
        """
        for card in self.datasets:
            if card.key == key:
                return card
        raise KeyError(f"unknown dataset {key!r}")

    def task(self, task_id: str) -> TaskSpec:
        """Definition of one task.

        Args:
            task_id: Full identifier of the task.

        Returns:
            The task registered under `task_id`.

        Raises:
            KeyError: If no task is registered under `task_id`.
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)

    def overlapping(self, keys: set[str]) -> set[str]:
        """Datasets in `keys` together with every dataset documented to overlap them.

        Overlaps are followed in both directions but only one step, so overlaps of overlaps are not
        included.

        Args:
            keys: Keys of datasets or corpora.

        Returns:
            Union of `keys` and the keys of every dataset or corpus sharing content with one of
            them.
        """
        related = set(keys)
        for dataset in self.datasets:
            for overlap in dataset.overlaps:
                if dataset.key in keys:
                    related.add(overlap.dataset)
                if overlap.dataset in keys:
                    related.add(dataset.key)
        return related
