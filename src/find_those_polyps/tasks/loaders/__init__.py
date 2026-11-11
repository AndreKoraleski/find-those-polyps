"""Dataset loaders.

Importing this package registers every implementation in `LOADERS` under the key that task
definitions name in their `loader` field.
"""

from find_those_polyps.tasks.loaders import boxes, folders, kvasir_seg, layers, limuc
from find_those_polyps.tasks.loaders.base import LOADERS, FileLoader, Loader, register_loader

__all__ = [
    "LOADERS",
    "FileLoader",
    "Loader",
    "boxes",
    "folders",
    "kvasir_seg",
    "layers",
    "limuc",
    "register_loader",
]
