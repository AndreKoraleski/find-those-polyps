"""Fingerprints of every test image, against which pretraining corpora can be checked.

The test images of each task are recorded with the CRC32 and byte size of their file, which identify
identical copies, and with the coarse and fine difference hashes of their perceptual signature,
which find near duplicates. One compressed table per dataset is committed next to the split files,
and only fingerprints are stored, never images.

A corpus is checked by fingerprinting its images in the same way. A near duplicate found by the
hashes is confirmed by comparing thumbnails with the test image itself, read from the local or
streamed dataset, so that look alike frames sharing a border are not reported. When the test image
cannot be read the match is reported as a candidate. Crops, rotations and flips change the hashes
and escape the check.
"""

import gzip
import io
import json
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
from tqdm.auto import tqdm

from find_those_polyps.data.files import DatasetFiles, open_files
from find_those_polyps.logging import get_logger
from find_those_polyps.registry.schema import Registry
from find_those_polyps.registry.store import REGISTRY_PATH, build_task, load_registry, load_split
from find_those_polyps.tasks.loaders.base import FileLoader
from find_those_polyps.tasks.loaders.images import IMAGE_SUFFIXES
from find_those_polyps.tasks.schema import TaskSpec
from find_those_polyps.utils.perceptual import (
    FINE,
    Signature,
    SignatureIndex,
    signature,
    similar,
    thumbnail,
)

log = get_logger(__name__)

MANIFESTS_DIR = REGISTRY_PATH.parent / "manifests"
"""Directory of the committed manifests, one per dataset."""

COLUMNS = ["task", "sample_id", "path", "crc32", "size", "dhash", "dhash_fine"]
"""Columns of a manifest, the hashes in hexadecimal."""

WORKERS = 8
"""Threads fingerprinting the files of a corpus."""

Reader = Callable[[str], bytes | None]
"""Content of a file given its canonical path, or `None` when it cannot be read."""


def manifest_path(dataset: str, directory: Path = MANIFESTS_DIR) -> Path:
    """Path of the manifest of a dataset.

    Args:
        dataset: Key of the dataset.
        directory: Directory of the manifests.

    Returns:
        The compressed CSV file of the dataset.
    """
    return directory / f"{dataset}.csv.gz"


def task_manifest(spec: TaskSpec, data_root: Path, registry: Registry) -> pd.DataFrame:
    """Fingerprint the test images of one task, read from local or streamed data.

    Signatures already computed while building splits are taken from the cache under the data
    root, and only the remaining images are decoded.

    Args:
        spec: Task whose test partition is fingerprinted.
        data_root: Root directory of the local and cached datasets.
        registry: Registry holding the dataset card.

    Returns:
        One row per test image with the columns of `COLUMNS`.

    Raises:
        TypeError: If the loader of the task does not read its samples from files.
    """
    task = build_task(spec, data_root, registry=registry)
    loader = task.loader
    if not isinstance(loader, FileLoader):
        raise TypeError(f"{spec.id} does not read its samples from files")
    cache_path = data_root / ".signatures" / f"{spec.dataset}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    rows = []
    for sample_id in tqdm(task.split.test, desc=spec.id, leave=False):
        image = loader.members(sample_id)[0]
        crc, size = loader.files.fingerprint(image).split(":")
        cached = cache.get(f"{crc}:{size}")
        coarse, fine = Signature(*cached) if cached else signature(loader.load(sample_id).image)
        rows.append(
            {
                "task": spec.id,
                "sample_id": sample_id,
                "path": loader.path(image),
                "crc32": crc,
                "size": int(size),
                "dhash": f"{coarse:016x}",
                "dhash_fine": f"{fine:064x}",
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def read_manifest(path: Path) -> pd.DataFrame:
    """Read a manifest, keeping identifiers and hashes as text.

    Args:
        path: Compressed CSV file.

    Returns:
        The manifest with the columns of `COLUMNS`.
    """
    return pd.read_csv(path, dtype={c: str for c in COLUMNS if c != "size"})


def write_manifest(frame: pd.DataFrame, path: Path) -> None:
    """Write a manifest sorted by task and sample, compressed reproducibly.

    The gzip header carries no time stamp, so that equal manifests are equal files.

    Args:
        frame: Rows with the columns of `COLUMNS`.
        path: Destination file, whose directory is created when missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values(["task", "sample_id"]).reset_index(drop=True)
    text = ordered.to_csv(index=False, lineterminator="\n").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as archive:
        archive.write(text)
    path.write_bytes(buffer.getvalue())


def update_manifests(
    registry: Registry, data_root: Path, tasks: Sequence[TaskSpec], directory: Path = MANIFESTS_DIR
) -> list[Path]:
    """Recompute the fingerprints of tasks in the manifests of their datasets.

    Rows of other tasks of the same datasets are kept.

    Args:
        registry: Registry holding the dataset cards.
        data_root: Root directory of the local and cached datasets.
        tasks: Tasks to fingerprint again.
        directory: Directory of the manifests.

    Returns:
        The manifest files written, one per dataset.
    """
    by_dataset: dict[str, list[TaskSpec]] = defaultdict(list)
    for spec in tasks:
        by_dataset[spec.dataset].append(spec)
    written = []
    for dataset, specs in sorted(by_dataset.items()):
        path = manifest_path(dataset, directory)
        frame = read_manifest(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
        frame = frame[~frame["task"].isin([spec.id for spec in specs])]
        fresh = [task_manifest(spec, data_root, registry) for spec in specs]
        write_manifest(pd.concat([frame, *fresh], ignore_index=True), path)
        written.append(path)
    return written


def stale_manifests(registry: Registry, directory: Path = MANIFESTS_DIR) -> list[str]:
    """Find the tasks whose recorded test images differ from their split file.

    Args:
        registry: Registry holding the tasks.
        directory: Directory of the manifests.

    Returns:
        The identifiers of the tasks whose manifest rows must be recomputed.
    """
    stale = []
    frames: dict[str, pd.DataFrame] = {}
    for spec in registry.tasks:
        path = manifest_path(spec.dataset, directory)
        if spec.dataset not in frames:
            frames[spec.dataset] = (
                read_manifest(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
            )
        frame = frames[spec.dataset]
        recorded = set(frame.loc[frame["task"] == spec.id, "sample_id"])
        if recorded != set(load_split(spec).test):
            stale.append(spec.id)
    return stale


def read_manifests(directory: Path = MANIFESTS_DIR) -> pd.DataFrame:
    """Read the fingerprints of the test images of every task.

    Args:
        directory: Directory of the manifests.

    Returns:
        The rows of every manifest, empty when there is none.
    """
    frames = [read_manifest(path) for path in sorted(directory.glob("*.csv.gz"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def test_reader(data_root: Path, registry: Registry | None = None) -> Reader:
    """Reader of test images by their canonical path, from local or streamed datasets.

    Args:
        data_root: Root directory of the local and cached datasets.
        registry: Registry holding the dataset cards, the packaged one when `None`.

    Returns:
        A function returning the content of the file at a canonical path, or `None` when the
        dataset is unknown or the file cannot be read.
    """
    cards = {card.key: card for card in (registry or load_registry()).datasets}
    opened: dict[str, DatasetFiles] = {}

    def read(path: str) -> bytes | None:
        key, _, name = path.partition("/")
        try:
            if key not in opened:
                opened[key] = open_files(cards[key], data_root)
            return opened[key].read(name)
        except (KeyError, OSError, ValueError):
            return None

    return read


def _corpus_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Image files given directly or found recursively in folders, in sorted order."""
    for path in paths:
        if path.is_dir():
            yield from sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file():
            yield path


class _Print:
    """Fingerprint of one corpus file.

    Attributes:
        crc: CRC32 of the file in hexadecimal.
        size: Size of the file in bytes.
        signature: Perceptual signature, `None` when the file is not an image.
        thumbnail: Grey thumbnail for the confirmation, `None` when the file is not an image.
    """

    def __init__(self, path: Path) -> None:
        """Fingerprint a file.

        Args:
            path: File to read.
        """
        content = path.read_bytes()
        self.crc = f"{zlib.crc32(content):08x}"
        self.size = len(content)
        self.signature: Signature | None = None
        self.thumbnail: NDArray[np.float64] | None = None
        try:
            with Image.open(io.BytesIO(content)) as image:
                picture = image.convert("RGB")
            self.signature = signature(picture)
            self.thumbnail = thumbnail(picture)
        except (OSError, ValueError):
            pass


def check_corpus(
    paths: Iterable[Path],
    distance: int = FINE,
    manifests: pd.DataFrame | None = None,
    read: Reader | None = None,
) -> pd.DataFrame:
    """Find test images in a corpus, as identical files or as near duplicates.

    Args:
        paths: Image files or folders searched recursively.
        distance: Largest distance in bits, from 0 to 256, between the fine hashes of a candidate
            and a test image.
        manifests: Fingerprints of the test images, the committed ones when `None`.
        read: Reader of test images by canonical path, used to confirm candidates by their
            thumbnails. Without it every candidate is reported unconfirmed.

    Returns:
        One row per match with the corpus file, the task, test sample and canonical path it copies,
        the kind of match, `identical`, `near duplicate` or `candidate`, and the distance between
        the fine hashes.

    Raises:
        ValueError: If `distance` lies outside 0 to 256.
    """
    if not 0 <= distance <= 256:
        raise ValueError("distance must lie between 0 and 256")
    known = read_manifests() if manifests is None else manifests
    exact: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (crc, size) in enumerate(zip(known["crc32"], known["size"], strict=True)):
        exact[crc, int(size)].append(index)
    hashes = SignatureIndex(
        [
            Signature(int(coarse, 16), int(fine, 16))
            for coarse, fine in zip(known["dhash"], known["dhash_fine"], strict=True)
        ]
    )
    thumbnails: dict[int, NDArray[np.float64] | None] = {}

    def test_thumbnail(index: int) -> NDArray[np.float64] | None:
        if index not in thumbnails:
            content = None if read is None else read(str(known.at[index, "path"]))
            try:
                image = None if content is None else Image.open(io.BytesIO(content))
                thumbnails[index] = None if image is None else thumbnail(image.convert("RGB"))
            except (OSError, ValueError):
                thumbnails[index] = None
        return thumbnails[index]

    matches = []
    files = list(_corpus_files(paths))
    with ThreadPoolExecutor(WORKERS) as pool:
        prints = pool.map(_Print, files)
        for path, item in tqdm(zip(files, prints, strict=True), total=len(files), leave=False):
            found: dict[int, tuple[str, int]] = dict.fromkeys(
                exact.get((item.crc, item.size), []), ("identical", 0)
            )
            if item.signature is not None and item.thumbnail is not None:
                for index, gap in hashes.near(item.signature, fine=distance):
                    if index in found:
                        continue
                    reference = test_thumbnail(index)
                    if reference is None:
                        found[index] = ("candidate", gap)
                    elif similar(item.thumbnail, reference):
                        found[index] = ("near duplicate", gap)
            for index, (kind, gap) in found.items():
                matches.append(
                    {
                        "file": str(path),
                        "task": known.at[index, "task"],
                        "sample_id": known.at[index, "sample_id"],
                        "path": known.at[index, "path"],
                        "match": kind,
                        "distance": gap,
                    }
                )
    columns = ["file", "task", "sample_id", "path", "match", "distance"]
    return pd.DataFrame(matches, columns=columns)
