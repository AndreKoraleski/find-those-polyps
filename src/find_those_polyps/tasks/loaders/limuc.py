"""Labeled Images for Ulcerative Colitis (Polat et al. 2022).

Images are read from the patient based archive, whose folders hold the Mayo endoscopic score of each
image and whose file names encode the patient. The optional `mapping` option relabels the four
Mayo grades, for instance to remission against active disease. The official test partition is the
set of images also present in the separate test set folder.
"""

import re
from collections.abc import Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Any

from find_those_polyps.tasks.loaders.base import FileLoader, register_loader
from find_those_polyps.tasks.loaders.images import IMAGE_SUFFIXES, parent, read_rgb, stem
from find_those_polyps.tasks.schema import Sample

if TYPE_CHECKING:
    from find_those_polyps.data.files import DatasetFiles

IMAGES_DIR = "patient_based_classified_images"
TEST_DIR = "test_set"
TEST_SOURCE = "official LIMUC test set (Polat et al., Zenodo 5827695)"
_MAYO = re.compile(r"Mayo\s*(\d)")
_PATIENT = re.compile(r"UC_patient_(\d+)_")


@register_loader("limuc")
class LIMUC(FileLoader):
    """Mayo endoscopic score of colonoscopy images from 564 patients.

    Identifiers are file stems and groups are patient numbers.

    Args:
        files: Files of the dataset.
        mapping: Class index of each Mayo grade from 0 to 3, the identity by default.
        **options: Options passed on to `Loader`.

    Attributes:
        mapping: Class index of each Mayo grade.
        conflicts: Sorted stems excluded for carrying contradictory scores, filled when `index` is
            first computed.
    """

    def __init__(self, files: "DatasetFiles", mapping: list[int] | None = None, **options: Any):
        super().__init__(files, **options)
        self.mapping = mapping or [0, 1, 2, 3]
        self.conflicts: list[str] = []

    def available(self) -> bool:
        """Whether the files are readable and hold at least one scored image.

        Returns:
            Whether the dataset can be read.
        """
        return self.files.available() and bool(self.index)

    @cached_property
    def index(self) -> dict[str, tuple[str, int, str]]:
        """File, Mayo score and patient of every image, keyed by file stem.

        The archive files a few byte identical images under two different scores. Such images carry
        contradictory labels and are excluded, and their names are kept in `conflicts`. Files
        outside a Mayo folder or without a patient number in their name are ignored.
        """
        found: dict[str, list[tuple[str, int, str]]] = {}
        for name in self.files.under(IMAGES_DIR, IMAGE_SUFFIXES):
            mayo = _MAYO.search(parent(name))
            patient = _PATIENT.match(name.rsplit("/", 1)[-1])
            if mayo is not None and patient is not None:
                found.setdefault(stem(name), []).append((name, int(mayo[1]), patient[1]))
        self.conflicts = sorted(k for k, v in found.items() if len({e[1] for e in v}) > 1)
        return {k: v[0] for k, v in found.items() if k not in self.conflicts}

    def ids(self) -> list[str]:
        """Identifiers of all samples.

        Returns:
            Sorted file stems.
        """
        return sorted(self.index)

    def groups(self) -> dict[str, str]:
        """Patient of each sample.

        Returns:
            Patient number of every sample keyed by identifier.
        """
        return {key: patient for key, (_, _, patient) in self.index.items()}

    def labels(self) -> dict[str, int]:
        """Class of every image after applying the mapping.

        Returns:
            Class index keyed by identifier.
        """
        return {key: self.mapping[mayo] for key, (_, mayo, _) in self.index.items()}

    def official_test(self) -> tuple[list[str], str]:
        """Identifiers of the official test partition and its source.

        Returns:
            Sorted stems found both in the test set folder and in the patient based archive,
            with the citation of the test set.
        """
        names = {stem(n) for n in self.files.under(TEST_DIR, IMAGE_SUFFIXES)}
        return sorted(names & set(self.index)), TEST_SOURCE

    def members(self, sample_id: str) -> tuple[str, ...]:
        """Image file of one sample.

        Args:
            sample_id: Identifier of the sample.

        Returns:
            The name of the image, as a tuple of one.
        """
        return (self.index[sample_id][0],)

    def decode(self, sample_id: str, contents: Mapping[str, bytes]) -> Sample:
        """Build an ordinal sample from its image.

        Args:
            sample_id: Identifier of the sample.
            contents: Bytes of the image keyed by file name.

        Returns:
            Sample whose target is the mapped Mayo score.
        """
        name, mayo, patient = self.index[sample_id]
        return Sample(
            id=sample_id,
            image=read_rgb(contents[name]),
            target=self.mapping[mayo],
            group=patient,
            path=self.path(name),
        )
