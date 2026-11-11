"""Content hashing used to version the registry and its split files."""

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    """SHA-256 of the canonical JSON encoding of a value.

    The encoding sorts mapping keys and omits whitespace, so that equal values hash equally
    whatever the order in which their keys were inserted.

    Args:
        value: Value made of JSON serializable types.

    Returns:
        The hexadecimal digest of the UTF-8 encoded JSON text.

    Raises:
        TypeError: If `value` holds an object that JSON cannot encode.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of the bytes of a file.

    Args:
        path: File to hash, read entirely into memory.

    Returns:
        The hexadecimal digest of the file contents.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()
