"""Kvasir-SEG polyp segmentation (Jha et al. 2020).

The archive pairs every image in `images` with a mask of the same stem in `masks`, both below the
`Kvasir-SEG` directory. No patient identifiers are released, so each image forms its own group.
"""

from collections.abc import Mapping
from functools import cached_property

from find_those_polyps.tasks.loaders.base import FileLoader, register_loader
from find_those_polyps.tasks.loaders.images import IMAGE_SUFFIXES, read_mask, read_rgb, stem
from find_those_polyps.tasks.schema import Sample

ARCHIVE_DIR = "Kvasir-SEG"


@register_loader("kvasir-seg")
class KvasirSEG(FileLoader):
    """Binary polyp masks of 1000 colonoscopy images.

    Sample identifiers are image stems, and masks are binarized at the default threshold of 128.

    Args:
        files: Files of the dataset.
        **options: Options passed on to `Loader`.
    """

    def available(self) -> bool:
        """Whether the files are readable and hold at least one image with its mask.

        Returns:
            Whether the dataset can be read.
        """
        return self.files.available() and bool(self.pairs)

    @cached_property
    def pairs(self) -> dict[str, tuple[str, str]]:
        """Image and mask file of every sample with both present, keyed by image stem."""
        masks = {stem(n): n for n in self.files.under(f"{ARCHIVE_DIR}/masks", IMAGE_SUFFIXES)}
        images = self.files.under(f"{ARCHIVE_DIR}/images", IMAGE_SUFFIXES)
        return {stem(n): (n, masks[stem(n)]) for n in images if stem(n) in masks}

    def ids(self) -> list[str]:
        """Identifiers of all samples.

        Returns:
            Sorted image stems.
        """
        return sorted(self.pairs)

    def groups(self) -> dict[str, str]:
        """Group of each sample, which is the sample itself.

        Returns:
            Every identifier mapped to itself.
        """
        return {sample_id: sample_id for sample_id in self.pairs}

    def members(self, sample_id: str) -> tuple[str, ...]:
        """Image and mask file of one sample.

        Args:
            sample_id: Identifier of the sample.

        Returns:
            Names of the image and of the mask.
        """
        return self.pairs[sample_id]

    def decode(self, sample_id: str, contents: Mapping[str, bytes]) -> Sample:
        """Build a segmentation sample from its image and mask.

        Args:
            sample_id: Identifier of the sample.
            contents: Bytes of the image and mask keyed by file name.

        Returns:
            Sample whose target is a binary mask of shape `(H, W)`.
        """
        image, mask = self.pairs[sample_id]
        return Sample(
            id=sample_id,
            image=read_rgb(contents[image]),
            target=read_mask(contents[mask]),
            group=sample_id,
            path=self.path(image),
        )
