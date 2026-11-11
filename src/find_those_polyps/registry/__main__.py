"""Synchronize the registry and its documentation.

`polyps-registry` refreshes the split hashes, bumps the version when the content changed and
renders docs/datasets.md, exiting with status one when it rewrote anything. `polyps-registry
--check` writes nothing and only reports whether both are current, for continuous integration.
"""

import argparse
import sys

from find_those_polyps.registry.docs import DOCS_PATH, render, write_docs
from find_those_polyps.registry.store import (
    REGISTRY_PATH,
    StaleRegistryError,
    load_registry,
    sync_registry,
)


def main(argv: list[str] | None = None) -> int:
    """Refresh hashes, bump the version if needed and render the dataset page.

    Args:
        argv: Command line arguments without the program name, or `None` to read `sys.argv`.

    Returns:
        Exit status, one when a file was rewritten or, under `--check`, when the registry or its
        page is stale, and zero otherwise.
    """
    parser = argparse.ArgumentParser(prog="polyps-registry", description=__doc__)
    parser.add_argument("--check", action="store_true", help="report staleness without writing")
    args = parser.parse_args(argv)
    if args.check:
        try:
            registry = load_registry()
        except StaleRegistryError as error:
            print(error, file=sys.stderr)
            return 1
        current = DOCS_PATH.exists() and DOCS_PATH.read_text(encoding="utf-8") == render(registry)
        if not current:
            print("docs/datasets.md is stale, run polyps-registry", file=sys.stderr)
        return int(not current)
    changed = sync_registry()
    registry = load_registry()
    if changed:
        print(f"{REGISTRY_PATH.name} updated to version {registry.version}")
    documented = write_docs(registry)
    if documented:
        print("docs/datasets.md regenerated")
    return int(changed or documented)


if __name__ == "__main__":
    sys.exit(main())
