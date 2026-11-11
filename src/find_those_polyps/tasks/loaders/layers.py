"""Multilabel segmentation masks stored as one page per class.

Every mask file is a multi page image whose page `i` holds the binary mask of class `i`, so that
classes may overlap, as in the EAD2020 artefact segmentation data of Ali et al. (2021). Images and
masks may come from several pairs of folders, and mask stems are mapped to image stems by an
optional regular expression and its replacement.
"""

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from find_those_polyps.tasks.loaders.base import register_loader
from find_those_polyps.tasks.loaders.folders import _Indexed
from find_those_polyps.tasks.loaders.images import IMAGE_SUFFIXES, read_layers, read_rgb, stem
from find_those_polyps.tasks.schema import Sample

if TYPE_CHECKING:
    from find_those_polyps.data.files import DatasetFiles


@register_loader("layered-masks")
class LayeredMasks(_Indexed):
    """Images paired with multi page masks of `layers` classes across `sources` folder pairs.

    Identifiers are image stems, and images without a mask are skipped. When a stem occurs in
    several folder pairs the last pair wins.

    Args:
        files: Files of the dataset.
        sources: Pairs of an image directory and the mask directory that belongs to it.
        layers: Number of classes, one mask page each. Pages beyond the file count as empty.
        threshold: Lowest grey level, between 0 and 255, counted as foreground.
        mask_name: Regular expression and its replacement rewriting a mask stem into the stem of
            its image, the opposite direction of the `masks` loader, or `None` when both stems are
            equal.
        **options: Options of `_Indexed`, namely `group`, `official_test` and `expand`.
    """

    def __init__(
        self,
        files: "DatasetFiles",
        sources: list[tuple[str, str]],
        layers: int,
        threshold: int = 128,
        mask_name: tuple[str, str] | None = None,
        **options: Any,
    ) -> None:
        super().__init__(files, **options)
        self.sources = sources
        self.layers = layers
        self.threshold = threshold
        self.rename = (re.compile(mask_name[0]), mask_name[1]) if mask_name else None

    def _image_stem(self, mask: str) -> str:
        """Stem of the image belonging to a mask file name."""
        if self.rename is None:
            return stem(mask)
        return self.rename[0].sub(self.rename[1], stem(mask))

    def index(self) -> dict[str, tuple[str, ...]]:
        """Image and mask file of every image with a mask.

        Returns:
            Names of the image and of its mask keyed by image stem.
        """
        found: dict[str, tuple[str, ...]] = {}
        for images, masks in self.sources:
            layered = {self._image_stem(n): n for n in self.files.under(masks, IMAGE_SUFFIXES)}
            for name in self.files.under(images, IMAGE_SUFFIXES):
                mask = layered.get(stem(name))
                if mask is not None:
                    found[stem(name)] = (name, mask)
        return found

    def decode(self, sample_id: str, contents: Mapping[str, bytes]) -> Sample:
        """Build a multilabel segmentation sample from its image and layered mask.

        Args:
            sample_id: Identifier of the sample.
            contents: Bytes of the image and mask keyed by file name.

        Returns:
            Sample whose target is a stack of binary masks of shape `(layers, H, W)`.
        """
        image, mask = self.entries[sample_id]
        return Sample(
            id=sample_id,
            image=read_rgb(contents[image]),
            target=read_layers(contents[mask], self.layers, self.threshold),
            group=self.group_of(sample_id),
            path=self.path(image),
        )
