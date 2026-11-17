"""Curated corpora for self supervised pretraining, kept apart from the benchmark datasets.

A corpus is a pool of images, mostly unlabelled, that shares no patient with any test image of
the benchmark. Its card pins every published archive by size and checksum, names the unit each
archive holds, a video or a patient, and lists the units that are held out with the reason, so
that they are never read. The cards live in `data/corpora.json` beside this module and are
maintained by hand, each with the evidence for its choices.

An encoder pretrained on a corpus declares the key of the corpus in `pretrained_on`.
"""

import json
import re
from functools import cache
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from find_those_polyps.registry.schema import Overlap

CORPORA_PATH = Path(__file__).parent / "data" / "corpora.json"
"""Packaged cards of the curated corpora."""


class Archive(BaseModel):
    """Published tarball holding the frames of one unit of a corpus.

    Attributes:
        unit: Video or patient the frames belong to, unique within the corpus.
        url: Address of the archive on the official host.
        filename: Name under which the archive is stored locally.
        checksum: Digest of the whole archive written as `md5`, `sha1` or `sha256` followed by a
            colon and the hexadecimal value.
        size: Size of the archive in bytes.
        frames: Number of frames in the archive, or `None` when unknown.
        fps: Frames per second of the recording, or `None` when unknown.
    """

    model_config = ConfigDict(frozen=True)

    unit: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    url: str
    filename: str
    checksum: str = Field(pattern=r"^(md5|sha1|sha256):[0-9a-f]+$")
    size: int = Field(gt=0)
    frames: int | None = Field(default=None, gt=0)
    fps: float | None = Field(default=None, gt=0)

    def stride(self, fps: float | None) -> int:
        """Gap between kept frame indexes that gives about `fps` frames per second.

        Args:
            fps: Frames to keep per second of recording, or `None` to keep every frame.

        Returns:
            The recording rate divided by `fps` and rounded, at least one, and one when either
            rate is unknown.
        """
        if fps is None or self.fps is None:
            return 1
        return max(1, round(self.fps / fps))


class HeldOut(BaseModel):
    """Units of a corpus that are never read, with the reason.

    Attributes:
        units: Units left out of pretraining.
        reason: Why they are left out.
        evidence: URL or citation that defines the held out units.
    """

    model_config = ConfigDict(frozen=True)

    units: tuple[str, ...] = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class CorpusCard(BaseModel):
    """Provenance, licence, files and held out units of a pretraining corpus.

    Attributes:
        key: Unique identifier in lower case letters, digits and hyphens, declared in
            `pretrained_on` by the encoders trained on the corpus and used as its directory name.
        name: Display name.
        modality: Imaging modality, for instance colonoscopy or capsule endoscopy.
        licence: Licence under which the corpus is distributed.
        url: Landing page of the corpus.
        citation: Reference to cite when using the corpus.
        provenance: Description of the clinical origin of the images.
        family: Clinical source pool the corpus is drawn from, as on dataset cards.
        frame_pattern: Regular expression matched against the base name of every archive member.
            Members that do not match are not frames, and its single group captures the index of
            the frame within its unit.
        archives: Published archives, one per unit, held out units included.
        held_out: Units that are never read.
        overlaps: Documented content shared with benchmark datasets or other corpora.
        notes: Free text on the curation.
    """

    model_config = ConfigDict(frozen=True)

    key: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str
    modality: str
    licence: str
    url: str
    citation: str
    provenance: str
    family: str
    frame_pattern: str
    archives: tuple[Archive, ...] = Field(min_length=1)
    held_out: tuple[HeldOut, ...] = ()
    overlaps: tuple[Overlap, ...] = ()
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Check the units and the frame pattern.

        Returns:
            The validated card.

        Raises:
            ValueError: If units or file names repeat, a held out unit has no archive, or the
                frame pattern does not capture exactly one group.
        """
        units = [archive.unit for archive in self.archives]
        if len(set(units)) != len(units):
            raise ValueError(f"{self.key} names a unit twice")
        names = [archive.filename for archive in self.archives]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.key} names a file twice")
        unknown = self.excluded - set(units)
        if unknown:
            raise ValueError(f"{self.key} holds out unknown units {sorted(unknown)}")
        if re.compile(self.frame_pattern).groups != 1:
            raise ValueError(f"the frame pattern of {self.key} must capture the frame index")
        return self

    @property
    def excluded(self) -> frozenset[str]:
        """Every held out unit."""
        return frozenset(unit for group in self.held_out for unit in group.units)

    def selected(self) -> tuple[Archive, ...]:
        """Archives that pretraining may read.

        Returns:
            The archives of the units that are not held out, in card order.
        """
        excluded = self.excluded
        return tuple(archive for archive in self.archives if archive.unit not in excluded)

    def frame_index(self, member: str) -> int | None:
        """Index of the frame stored under an archive member.

        Args:
            member: Path of the member within its archive.

        Returns:
            The captured index, or `None` when the base name of the member is not a frame.
        """
        match = re.fullmatch(self.frame_pattern, member.rsplit("/", 1)[-1])
        return int(match.group(1)) if match else None


@cache
def load_corpora(path: Path = CORPORA_PATH) -> tuple[CorpusCard, ...]:
    """Read the cards of the curated corpora.

    Args:
        path: JSON file holding a list of cards, the packaged one by default.

    Returns:
        The validated cards in file order.

    Raises:
        ValueError: If two corpora share a key.
    """
    cards = tuple(
        CorpusCard.model_validate(card) for card in json.loads(path.read_text(encoding="utf-8"))
    )
    keys = [card.key for card in cards]
    if len(set(keys)) != len(keys):
        raise ValueError("corpus keys must be unique")
    return cards


def corpus(key: str) -> CorpusCard:
    """Card of one curated corpus.

    Args:
        key: Key of the corpus.

    Returns:
        The card registered under `key`.

    Raises:
        KeyError: If no corpus has that key.
    """
    for card in load_corpora():
        if card.key == key:
            return card
    known = ", ".join(card.key for card in load_corpora())
    raise KeyError(f"unknown corpus {key!r}, known corpora are {known}")
