"""Perceptual image signatures, the search for near duplicates and their confirmation.

The difference hash of Krawetz (2013) compares neighbouring pixels of a grey thumbnail, so that it
barely changes under resizing, recompression and mild colour changes. A signature holds the 64 bit
hash of a 9 by 8 thumbnail and the 256 bit hash of a 17 by 16 thumbnail. Candidate pairs lie within
a few bits of each other in both hashes and are found by exact lookups of the bytes of the coarse
hash, since pairs within seven bits share at least one of its eight bytes.

Hashes alone cannot tell copies from look alike frames that share a fixed border or overlay, such
as capsule frames inside the same octagonal mask, so a candidate is confirmed by the correlation of
grey 32 by 32 thumbnails. Measured on Kvasir-Capsule, LIMUC, GastroHUN and HyperKvasir, resized and
recompressed copies correlate at 0.999 or more and distinct look alike frames at 0.986 or less,
which sets the threshold to 0.995. About half of the neighbouring video frames of HyperKvasir pass
it as well.
"""

from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image

COARSE = 6
"""Largest distance in bits between the coarse hashes of a candidate pair."""

FINE = 16
"""Largest distance in bits between the fine hashes of a candidate pair."""

MAX_DISTANCE = 7
"""Largest coarse distance for which the byte lookup finds every candidate."""

BANDS = 8
"""Number of bytes of the coarse hash, each indexed separately."""

THUMBNAIL = 32
"""Side in pixels of the grey thumbnail compared to confirm a candidate."""

SIMILARITY = 0.995
"""Smallest thumbnail correlation of a confirmed near duplicate."""

Picture = NDArray[np.uint8] | Image.Image
"""RGB image as an array of shape (H, W, 3) or as a Pillow image."""


class Signature(NamedTuple):
    """Coarse and fine difference hashes of one image.

    Attributes:
        coarse: Hash of 64 bits used to find candidates quickly.
        fine: Hash of 256 bits that narrows the candidates.
    """

    coarse: int
    fine: int


def _picture(image: Picture) -> Image.Image:
    """Pillow image of an array or of a Pillow image."""
    return image if isinstance(image, Image.Image) else Image.fromarray(image)


def _hash(picture: Image.Image, width: int, height: int) -> int:
    """Difference hash comparing each pixel of a grey thumbnail with its right neighbour.

    Args:
        picture: Image to hash.
        width: Width of the thumbnail, one more than the number of comparisons per row.
        height: Height of the thumbnail.

    Returns:
        The comparisons packed into an integer of `(width - 1) * height` bits, row by row.
    """
    grey = picture.convert("L").resize((width, height), Image.Resampling.LANCZOS)
    pixels = np.asarray(grey, dtype=np.int16)
    bits = np.packbits((pixels[:, 1:] > pixels[:, :-1]).flatten())
    return int.from_bytes(bits.tobytes(), "big")


def dhash(image: Picture) -> int:
    """Coarse difference hash of an image.

    Args:
        image: RGB image.

    Returns:
        The 64 bit hash of a 9 by 8 grey thumbnail.
    """
    return _hash(_picture(image), 9, 8)


def signature(image: Picture) -> Signature:
    """Coarse and fine difference hashes of an image.

    Args:
        image: RGB image.

    Returns:
        The signature of the image.
    """
    picture = _picture(image)
    return Signature(_hash(picture, 9, 8), _hash(picture, 17, 16))


def thumbnail(image: Picture) -> NDArray[np.float64]:
    """Grey thumbnail of an image, centred and scaled to unit norm.

    Args:
        image: RGB image.

    Returns:
        A vector of `THUMBNAIL` squared values whose dot product with another thumbnail is their
        correlation.
    """
    grey = _picture(image).convert("L").resize((THUMBNAIL, THUMBNAIL), Image.Resampling.BOX)
    values = np.asarray(grey, dtype=np.float64).ravel()
    values -= values.mean()
    norm = float(np.linalg.norm(values))
    return values / norm if norm else values


def similar(first: NDArray[np.float64], second: NDArray[np.float64]) -> bool:
    """Whether two thumbnails correlate enough to confirm a near duplicate.

    Args:
        first: Thumbnail of one image.
        second: Thumbnail of the other image.

    Returns:
        True when their correlation reaches `SIMILARITY`.
    """
    return float(first @ second) >= SIMILARITY


class SignatureIndex:
    """Signatures indexed by each byte of their coarse hash for lookups of near duplicates.

    Attributes:
        signatures: Indexed signatures in their original order.
        bands: For each byte position, the indices of the signatures holding each byte value.
    """

    def __init__(self, signatures: Sequence[Signature]) -> None:
        """Index signatures.

        Args:
            signatures: Signatures to index.
        """
        self.signatures = list(signatures)
        self.bands: list[defaultdict[int, list[int]]] = [defaultdict(list) for _ in range(BANDS)]
        for index, item in enumerate(self.signatures):
            for band in range(BANDS):
                self.bands[band][(item.coarse >> (8 * band)) & 0xFF].append(index)

    def near(
        self, item: Signature, coarse: int = COARSE, fine: int = FINE
    ) -> Iterator[tuple[int, int]]:
        """Candidates near a signature.

        Args:
            item: Signature to look up.
            coarse: Largest distance in bits between coarse hashes.
            fine: Largest distance in bits between fine hashes.

        Yields:
            The index of each candidate and the distance between the fine hashes, in index order.

        Raises:
            ValueError: If `coarse` exceeds `MAX_DISTANCE`, beyond which the lookup misses pairs.
        """
        if not 0 <= coarse <= MAX_DISTANCE:
            raise ValueError(f"coarse distance must lie between 0 and {MAX_DISTANCE}")
        candidates = {
            index
            for band in range(BANDS)
            for index in self.bands[band].get((item.coarse >> (8 * band)) & 0xFF, [])
        }
        for index in sorted(candidates):
            other = self.signatures[index]
            if (item.coarse ^ other.coarse).bit_count() > coarse:
                continue
            gap = (item.fine ^ other.fine).bit_count()
            if gap <= fine:
                yield index, gap


def near_pairs(
    signatures: Sequence[Signature | None], coarse: int = COARSE, fine: int = FINE
) -> Iterator[tuple[int, int]]:
    """Candidate pairs among signatures, each pair once.

    Args:
        signatures: Signature of each image, `None` for images that could not be hashed.
        coarse: Largest distance in bits between coarse hashes.
        fine: Largest distance in bits between fine hashes.

    Yields:
        Pairs of positions in `signatures`, the smaller position first.
    """
    present = [(i, s) for i, s in enumerate(signatures) if s is not None]
    index = SignatureIndex([s for _, s in present])
    for position, (i, item) in enumerate(present):
        for other, _ in index.near(item, coarse, fine):
            if other > position:
                yield i, present[other][0]


def confirmer(load: Callable[[int], Picture | None]) -> Callable[[int, int], bool]:
    """Confirmation of candidate pairs by the correlation of their thumbnails.

    Thumbnails are computed once per image and kept for later pairs.

    Args:
        load: Image at a position, or `None` when it cannot be read.

    Returns:
        A function telling whether the images at two positions are near duplicates. Pairs with an
        unreadable image are not confirmed.
    """
    thumbnails: dict[int, NDArray[np.float64] | None] = {}

    def thumb(position: int) -> NDArray[np.float64] | None:
        if position not in thumbnails:
            image = load(position)
            thumbnails[position] = None if image is None else thumbnail(image)
        return thumbnails[position]

    def confirm(first: int, second: int) -> bool:
        a, b = thumb(first), thumb(second)
        return a is not None and b is not None and similar(a, b)

    return confirm
