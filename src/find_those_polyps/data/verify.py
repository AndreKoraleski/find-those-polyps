"""Checks that the official hosts still serve the files pinned on the dataset cards.

Zip archives pinned by an index are compared with their size and index pins through their central
directory, which a few range requests read, so that no archive is downloaded. Every other file,
which includes archives pinned by a checksum alone, is downloaded into memory and hashed. Every
check is retried with the exponential backoff of the range reader, and a failure is reported for
its file without interrupting the check of the others.
"""

import hashlib
import time
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from find_those_polyps.data.remote import (
    RETRIES,
    TRANSIENT,
    USER_AGENT,
    RangeUnsupportedError,
    open_url,
    read_index,
)
from find_those_polyps.registry.schema import DatasetCard, Download

__all__ = ["Outcome", "Verdict", "verify", "verify_download"]


class Outcome(StrEnum):
    """Result of comparing a published file with its pins.

    Attributes:
        OK: The file matches every pin.
        CHANGED: The file differs from a pin, is no longer a readable zip archive or is no longer
            served with range requests.
        UNREACHABLE: The file could not be read after every retry.
    """

    OK = "ok"
    CHANGED = "changed"
    UNREACHABLE = "unreachable"


class Verdict(BaseModel):
    """Outcome for one file of a dataset card, with what differs or why it could not be read.

    Attributes:
        dataset: Key of the dataset card.
        filename: Name of the published file on the card.
        url: Address the file was read from.
        outcome: Result of the comparison with the pins.
        detail: What differs or why the file could not be read, empty when the file matches.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str
    filename: str
    url: str
    outcome: Outcome
    detail: str = ""

    def __str__(self) -> str:
        """Render the verdict as one line of outcome, dataset and file, then any detail."""
        line = f"{self.outcome:12} {self.dataset}/{self.filename}"
        return f"{line}  {self.detail}" if self.detail else line


def _retrying[T](action: Callable[[], T]) -> T:
    """Run an action, retrying it with exponential backoff after transient failures.

    The waits last 2, 4, 8 and 16 seconds, for at most `RETRIES` attempts in all.

    Args:
        action: Callable without arguments.

    Returns:
        The result of the first attempt that succeeds.

    Raises:
        OSError: If the last attempt fails with an operating system or URL error, which is
            raised again.
        http.client.HTTPException: If the last attempt fails with a malformed HTTP exchange,
            which is raised again.
    """
    failures = 0
    while True:
        try:
            return action()
        except TRANSIENT:
            failures += 1
            if failures == RETRIES:
                raise
            time.sleep(2**failures)


def _download(url: str) -> bytes:
    """Download the whole content of a remote file into memory.

    Args:
        url: Address of the file.

    Returns:
        The bytes of the file.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with open_url(request) as response:
        data: bytes = response.read()
    return data


def _observed(item: Download) -> tuple[int, str, str]:
    """Read the size and digest that a host currently serves for a published file.

    Args:
        item: Published file of a dataset card.

    Returns:
        The observed size in bytes, the pinned digest and the observed digest. For an archive
            pinned by an index the digests are member index hashes, otherwise they are checksums
            prefixed by their algorithm.
    """
    if item.extract and item.index:
        index = _retrying(lambda: read_index(item.url))
        return index.size, item.index, index.digest
    data = _retrying(lambda: _download(item.url))
    algorithm = item.checksum.partition(":")[0]
    return len(data), item.checksum, f"{algorithm}:{hashlib.new(algorithm, data).hexdigest()}"


def verify_download(key: str, item: Download) -> Verdict:
    """Compare one published file of a dataset with its pins.

    Args:
        key: Key of the dataset card.
        item: Published file of the card.

    Returns:
        A verdict that is unreachable after transient failures, changed when the size, the digest,
            the zip structure or range support differ from the pins, and ok otherwise.
    """

    def verdict(outcome: Outcome, detail: str = "") -> Verdict:
        """Verdict for `item` with the given outcome and detail."""
        return Verdict(
            dataset=key, filename=item.filename, url=item.url, outcome=outcome, detail=detail
        )

    try:
        size, pinned, digest = _observed(item)
    except TRANSIENT as error:
        return verdict(Outcome.UNREACHABLE, str(error) or type(error).__name__)
    except RangeUnsupportedError:
        return verdict(Outcome.CHANGED, "range requests are no longer honoured")
    except (ValueError, zipfile.BadZipFile) as error:
        return verdict(Outcome.CHANGED, str(error))
    differences = []
    if item.size is not None and size != item.size:
        differences.append(f"size {item.size} is now {size}")
    if digest != pinned:
        label = "index" if item.extract and item.index else "checksum"
        differences.append(f"{label} {pinned} is now {digest}")
    if differences:
        return verdict(Outcome.CHANGED, ", ".join(differences))
    return verdict(Outcome.OK)


def verify(cards: Iterable[DatasetCard]) -> Iterator[Verdict]:
    """Compare every published file of the given dataset cards with its pins.

    Args:
        cards: Dataset cards to verify.

    Yields:
        One verdict per published file, in card and download order, as soon as it is known.
    """
    for card in cards:
        for item in card.downloads:
            yield verify_download(card.key, item)
