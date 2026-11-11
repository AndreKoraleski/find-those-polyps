"""Reading, verifying and versioning the registry file.

The registry records the hash of every split file and a hash of its own content, which covers the
split hashes. `sync_registry` refreshes both and increments the version whenever the content hash
changes, and `load_registry` refuses a registry whose recorded hashes are out of date.
"""

import json
from pathlib import Path

from find_those_polyps.registry.schema import Registry
from find_those_polyps.tasks.loaders import LOADERS, Loader
from find_those_polyps.tasks.schema import TaskSpec
from find_those_polyps.tasks.splits import SplitFile
from find_those_polyps.tasks.task import Task
from find_those_polyps.utils.hashing import file_sha256

DATA_DIR = Path(__file__).parent / "data"
REGISTRY_PATH = DATA_DIR / "registry.json"
SPLITS_DIR = DATA_DIR / "splits"


class StaleRegistryError(RuntimeError):
    """The registry does not match its split files or its recorded hash."""


def _split_sha256(path: Path) -> str:
    """Hexadecimal SHA-256 of a split file, empty while the file has not been built."""
    return file_sha256(path) if path.exists() else ""


def _read(path: Path) -> Registry:
    """Parse and validate the registry file at `path` without checking its hashes."""
    return Registry.model_validate_json(path.read_text(encoding="utf-8"))


def _write(registry: Registry, path: Path) -> None:
    """Write `registry` to `path` as indented UTF-8 JSON with Unix line endings."""
    text = json.dumps(registry.model_dump(mode="json"), indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def load_registry(path: Path = REGISTRY_PATH, *, check: bool = True) -> Registry:
    """Load the registry, refusing it if split files or content hash are stale.

    Split files are looked up in the splits directory next to the registry file.

    Args:
        path: Registry file, the packaged one by default.
        check: Whether to verify the split hashes and the content hash. Tools that rewrite split
            files pass false.

    Returns:
        The validated registry.

    Raises:
        StaleRegistryError: If `check` is true and a split file or the content no longer matches
            its recorded hash.
    """
    registry = _read(path)
    if not check:
        return registry
    splits_dir = path.parent / "splits"
    for task in registry.tasks:
        if _split_sha256(splits_dir / task.splits) != task.splits_sha256:
            raise StaleRegistryError(f"split file of {task.id!r} changed; run `polyps-registry`")
    if registry.content_sha256() != registry.sha256:
        raise StaleRegistryError("registry content changed; run `polyps-registry`")
    return registry


def sync_registry(path: Path = REGISTRY_PATH) -> bool:
    """Refresh split hashes and bump the version if the content changed.

    Args:
        path: Registry file, the packaged one by default. Split files are looked up in the splits
            directory next to it.

    Returns:
        Whether the file was rewritten. The version is incremented by one only when the content
        hash changed.
    """
    registry = _read(path)
    splits_dir = path.parent / "splits"
    tasks = tuple(
        task.model_copy(update={"splits_sha256": _split_sha256(splits_dir / task.splits)})
        for task in registry.tasks
    )
    updated = registry.model_copy(update={"tasks": tasks})
    digest = updated.content_sha256()
    if digest == registry.sha256 and tasks == registry.tasks:
        return False
    version = registry.version + (digest != registry.sha256)
    _write(updated.model_copy(update={"version": version, "sha256": digest}), path)
    return True


def load_split(spec: TaskSpec, splits_dir: Path = SPLITS_DIR) -> SplitFile:
    """Canonical split of a task.

    Args:
        spec: Task whose split file is read.
        splits_dir: Directory of the split files, the packaged one by default.

    Returns:
        The validated split file named by `spec.splits`.
    """
    return SplitFile.model_validate_json((splits_dir / spec.splits).read_text(encoding="utf-8"))


def open_loader(
    spec: TaskSpec,
    data_root: Path,
    registry: Registry | None = None,
    *,
    stream: bool = True,
    cache: bool = True,
) -> Loader:
    """Loader of a task's dataset, local when present and otherwise streamed from its host.

    Args:
        spec: Task whose loader and loader options are used.
        data_root: Directory holding one subdirectory per dataset key, read for local copies and
            used as the stream cache.
        registry: Registry holding the dataset card, or `None` to load the packaged registry, which
            must then be current.
        stream: Whether an open dataset without a local copy may be streamed from its host. When
            false only a local copy is read.
        cache: Whether streamed files are written below `data_root`. When false they are kept in
            memory only.

    Returns:
        The loader registered under `spec.loader`, constructed with `spec.loader_options`.

    Raises:
        KeyError: If no loader is registered under `spec.loader` or the registry lacks the dataset.
    """
    from find_those_polyps.data.files import open_files

    try:
        loader_cls = LOADERS[spec.loader]
    except KeyError:
        raise KeyError(f"no loader registered as {spec.loader!r}") from None
    card = (registry or load_registry()).dataset(spec.dataset)
    files = open_files(card, data_root, stream=stream, cache=cache)
    return loader_cls(files, **spec.loader_options)


def build_task(
    spec: TaskSpec,
    data_root: Path,
    splits_dir: Path = SPLITS_DIR,
    *,
    registry: Registry | None = None,
    stream: bool = True,
    cache: bool = True,
) -> Task:
    """Bind a task definition to its data and canonical split.

    Args:
        spec: Task to instantiate.
        data_root: Directory holding one subdirectory per dataset key.
        splits_dir: Directory of the split files, the packaged one by default.
        registry: Registry holding the dataset card, or `None` to load the packaged registry.
        stream: Whether an open dataset without a local copy may be streamed from its host.
        cache: Whether streamed files are written below `data_root`.

    Returns:
        The task with its loader and split file.
    """
    loader = open_loader(spec, data_root, registry, stream=stream, cache=cache)
    return Task(spec=spec, loader=loader, split=load_split(spec, splits_dir))
