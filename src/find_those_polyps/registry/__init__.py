"""Versioned registry of datasets and downstream tasks.

The package exposes the data models of the registry file together with the functions that load,
verify, version and instantiate it.
"""

from find_those_polyps.registry.schema import (
    Access,
    DatasetCard,
    Download,
    Overlap,
    OverlapKind,
    Registry,
)
from find_those_polyps.registry.store import (
    REGISTRY_PATH,
    SPLITS_DIR,
    StaleRegistryError,
    build_task,
    load_registry,
    load_split,
    open_loader,
    sync_registry,
)

__all__ = [
    "REGISTRY_PATH",
    "SPLITS_DIR",
    "Access",
    "DatasetCard",
    "Download",
    "Overlap",
    "OverlapKind",
    "Registry",
    "StaleRegistryError",
    "build_task",
    "load_registry",
    "load_split",
    "open_loader",
    "sync_registry",
]
