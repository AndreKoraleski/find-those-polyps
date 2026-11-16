"""Local copies of openly downloadable datasets.

`fetch` materializes every file of a dataset under `data_root / key` through the same windowed
range streaming used during evaluation, so files are written as each window arrives and no archive
is stored. With `whole`, and for hosts that ignore range requests, each published file is instead
downloaded whole in one request into `data_root / ".archives" / key` and verified against its
checksum or archive index, then the archives are extracted and the plain files copied. Whole
downloads suit hosts that throttle or refuse the many range requests of streaming. Datasets that
require an account, a request or a signed agreement are never fetched, and the user is pointed to
the access page instead.
"""

import shutil
import zipfile
from pathlib import Path

from tqdm.auto import tqdm

from find_those_polyps.data.files import COMPLETE, RemoteFiles
from find_those_polyps.data.remote import RangeUnsupportedError, checksum, fetch_file
from find_those_polyps.logging import get_logger
from find_those_polyps.registry.schema import Access, DatasetCard, Download
from find_those_polyps.utils.hashing import canonical_sha256

log = get_logger(__name__)

ARCHIVES = ".archives"

__all__ = ["ManualAccessError", "archive_digest", "checksum", "extract", "fetch", "fetch_file"]


class ManualAccessError(RuntimeError):
    """Raised when a dataset cannot be fetched before the user obtains access to it."""


def extract(archive: Path, destination: Path) -> None:
    """Extract a zip archive, refusing members that escape the destination.

    Every member path is checked before anything is written, so an unsafe archive leaves the
    destination untouched.

    Args:
        archive: Local zip archive.
        destination: Directory that receives the members at their archive paths.

    Raises:
        ValueError: If a member path resolves outside `destination`.
    """
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            target = (destination / member).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe path {member!r} in {archive.name}")
        bundle.extractall(destination)


def archive_digest(archive: Path) -> str:
    """Hash of the member names, sizes and CRC32 values of a local zip archive.

    The hash is computed as `ArchiveIndex.digest` computes it for a remote archive, so that a
    whole download is verified against the `index` pin of its dataset card.

    Args:
        archive: Local zip archive.

    Returns:
        The SHA256 of the canonical listing, prefixed by `sha256` and a colon.
    """
    with zipfile.ZipFile(archive) as bundle:
        infos = sorted((i for i in bundle.infolist() if not i.is_dir()), key=lambda i: i.filename)
        listing = [[info.filename, info.file_size, info.CRC] for info in infos]
    return "sha256:" + canonical_sha256(listing)


def _whole(item: Download, archives: Path, destination: Path) -> None:
    """Download one published file whole, then extract it or copy it into the dataset directory.

    Args:
        item: Published file of the dataset card.
        archives: Directory that keeps the downloaded file.
        destination: Dataset directory that receives its content.

    Raises:
        ValueError: If an archive does not match the index pin of the card, in which case it is
            deleted.
    """
    path = fetch_file(item, archives)
    if item.extract and item.index and archive_digest(path) != item.index:
        path.unlink()
        raise ValueError(f"{item.filename} changed on the server, it no longer matches its index")
    if item.extract:
        log.info("extracting", archive=item.filename)
        extract(path, destination)
    else:
        shutil.copy2(path, destination / item.filename)


def fetch(
    card: DatasetCard, data_root: Path, *, keep_archives: bool = False, whole: bool = False
) -> Path:
    """Download every file of an open dataset into its directory under the data root.

    A directory that already holds the completion marker is returned at once. Once every file is
    written, the marker is created with the licence of the dataset as its content.

    Args:
        card: Dataset to download.
        data_root: Directory that holds one directory per dataset key.
        keep_archives: Whether to keep the files downloaded whole. Defaults to `False`.
        whole: Whether to download each published file whole in one request instead of streaming
            its members with range requests. Defaults to `False`.

    Returns:
        The dataset directory `data_root / card.key`.

    Raises:
        ManualAccessError: If the dataset is not open or publishes no file.
        ValueError: If a downloaded file, an archive index or an archive member does not match
            its pin, or an archive member escapes the dataset directory.
    """
    destination = data_root / card.key
    if (destination / COMPLETE).exists():
        return destination
    if card.access is not Access.OPEN or not card.downloads:
        raise ManualAccessError(
            f"{card.name} is not openly downloadable ({card.access}). Obtain it from {card.url} "
            f"and place it under {destination}."
        )
    destination.mkdir(parents=True, exist_ok=True)
    archives = data_root / ARCHIVES / card.key
    if not whole:
        try:
            files = RemoteFiles(card, destination)
            names = files.names()
            for _ in tqdm(files.stream([[n] for n in names]), total=len(names), desc=card.key):
                pass
        except RangeUnsupportedError:
            log.warning("host ignores range requests, downloading whole files", dataset=card.key)
            whole = True
    if whole:
        for item in card.downloads:
            _whole(item, archives, destination)
        if not keep_archives:
            shutil.rmtree(archives, ignore_errors=True)
    (destination / COMPLETE).write_text(card.licence + "\n", encoding="utf-8", newline="\n")
    log.info("dataset ready", dataset=card.key, path=str(destination))
    return destination
