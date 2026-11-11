"""Encoded features kept for reuse, keyed by the canonical path of their image.

Global features are kept in memory, so that tasks sharing images within one evaluation encode them
once. With a directory, every feature is also written to shards on disk, so that later evaluations
of the same encoder read them instead of encoding the images again. The directory is named after
the fingerprint of the encoder, so that changing its weights or code starts a new one. A shard is
written once 4096 global or 256 dense features are pending, and whatever remains on `flush`.
"""

import uuid
from collections import defaultdict
from pathlib import Path

import torch

from find_those_polyps.logging import get_logger

log = get_logger(__name__)

SHARD_ROWS = {False: 4096, True: 256}


class FeatureStore:
    """Features of one encoder, optionally persisted under a directory.

    Shards live in the `global` and `dense` subdirectories and are read lazily through memory
    mapping. Images without a canonical path are neither stored nor looked up.

    Attributes:
        directory: Folder of the shards, `None` to keep global features in memory only.
        memory: Global features encoded during this process, keyed by image path.
    """

    def __init__(self, directory: Path | None = None) -> None:
        """Open the store without reading any shard yet.

        Args:
            directory: Folder of the shards, created on the first write, or `None` to keep
                global features in memory only.
        """
        self.directory = directory
        self.memory: dict[str, torch.Tensor] = {}
        self._disk: dict[bool, dict[str, tuple[torch.Tensor, int]]] = {}
        self._pending: dict[bool, list[tuple[str, torch.Tensor]]] = defaultdict(list)

    def _kind(self, dense: bool) -> Path:
        """Subdirectory holding the shards of dense or global features."""
        assert self.directory is not None
        return self.directory / ("dense" if dense else "global")

    def _index(self, dense: bool) -> dict[str, tuple[torch.Tensor, int]]:
        """Rows of every shard on disk, read lazily through memory mapping.

        Unreadable shards are skipped with a warning. The index is built on first use and then
        extended by every shard written.

        Args:
            dense: Whether to index dense instead of global features.

        Returns:
            The shard tensor and row of each stored image path.
        """
        if dense not in self._disk:
            index: dict[str, tuple[torch.Tensor, int]] = {}
            if self.directory is not None and self._kind(dense).is_dir():
                for shard in sorted(self._kind(dense).glob("*.pt")):
                    try:
                        data = torch.load(shard, mmap=True, weights_only=True)
                    except Exception as error:
                        log.warning("unreadable feature shard", shard=str(shard), error=str(error))
                        continue
                    for row, path in enumerate(data["paths"]):
                        index[path] = (data["values"], row)
            self._disk[dense] = index
        return self._disk[dense]

    def get(self, path: str, dense: bool) -> torch.Tensor | None:
        """Feature of the image at a canonical path.

        Args:
            path: Canonical path of the image relative to the data root, empty when unknown.
            dense: Whether to look up the dense feature map instead of the global embedding.

        Returns:
            A tensor of shape (D,) or (D, h, w), or `None` if the image was never stored or
            `path` is empty. Features still pending are found only when global.
        """
        if not path:
            return None
        if not dense and path in self.memory:
            return self.memory[path]
        found = self._index(dense).get(path)
        if found is None:
            return None
        values, row = found
        return values[row]

    def put(self, paths: list[str], values: torch.Tensor, dense: bool) -> None:
        """Keep freshly encoded features, writing a shard once enough have accumulated.

        Args:
            paths: Canonical path of each image, those left empty being skipped.
            values: Features with one row per path, of shape (B, D) or (B, D, h, w).
            dense: Whether the features are dense maps.
        """
        for path, value in zip(paths, values, strict=True):
            if not path:
                continue
            if not dense:
                self.memory[path] = value
            if self.directory is not None:
                self._pending[dense].append((path, value))
        if len(self._pending[dense]) >= SHARD_ROWS[dense]:
            self._write(dense)

    def flush(self) -> None:
        """Write every pending feature to disk."""
        for dense in (False, True):
            if self._pending[dense]:
                self._write(dense)

    def _write(self, dense: bool) -> None:
        """Write the pending features of one kind to a new shard, atomically, and index them."""
        pending = self._pending.pop(dense)
        paths = [path for path, _ in pending]
        values = torch.stack([value for _, value in pending])
        directory = self._kind(dense)
        directory.mkdir(parents=True, exist_ok=True)
        shard = directory / f"{uuid.uuid4().hex}.pt"
        partial = shard.with_suffix(".part")
        torch.save({"paths": paths, "values": values}, partial)
        partial.replace(shard)
        index = self._index(dense)
        for row, path in enumerate(paths):
            index[path] = (values, row)
