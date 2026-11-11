"""Command line access to the registered datasets and tasks.

`polyps-data datasets` lists every dataset with its access route and local state, `polyps-data
tasks` lists every task, `polyps-data fetch KEY ...` makes local copies of open datasets, and
`polyps-data index URL ...` prints the size and member index that pin a remote zip archive on a
dataset card. `polyps-data check PATH ...` searches images, for instance a pretraining corpus, for
copies and near duplicates of any test image, and `polyps-data manifest TASK ...` refreshes the
committed fingerprints of the test images. `polyps-data verify KEY ...` checks that the official
hosts still serve the files pinned on the dataset cards and exits with one when any file changed or
could not be read. The data root defaults to `FIND_THOSE_POLYPS_DATA`, else to `./data`.
"""

import argparse
import sys
from fnmatch import fnmatch
from pathlib import Path

from find_those_polyps.data.download import ManualAccessError, fetch
from find_those_polyps.data.files import COMPLETE, INDEX_DIR
from find_those_polyps.data.paths import default_data_root
from find_those_polyps.data.remote import read_index
from find_those_polyps.data.verify import Outcome, verify
from find_those_polyps.registry.manifest import (
    check_corpus,
    stale_manifests,
    test_reader,
    update_manifests,
)
from find_those_polyps.registry.schema import Access, DatasetCard
from find_those_polyps.registry.store import load_registry


def state(card: DatasetCard, root: Path) -> str:
    """Describe whether a dataset is local, partly cached, streamable or needs manual access.

    Args:
        card: Dataset whose local state is described.
        root: Data root that holds one directory per dataset key.

    Returns:
        `local` when the dataset directory carries the completion marker, `cached` when it holds
            the archive indexes of a stream cache, `manual copy` for any other existing directory,
            `streamable` for an open dataset with published files, and the access route followed
            by `access` otherwise.
    """
    directory = root / card.key
    if (directory / COMPLETE).exists():
        return "local"
    if (directory / INDEX_DIR).is_dir():
        return "cached"
    if directory.is_dir():
        return "manual copy"
    if card.access is Access.OPEN and card.downloads:
        return "streamable"
    return f"{card.access} access"


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface.

    Args:
        argv: Arguments without the program name, read from `sys.argv` when `None`.

    Returns:
        The exit status, zero on success and one when a dataset key is unknown, a dataset needs
            manual access, no task is selected, a test image is found or a pinned file changed or
            could not be read.

    Raises:
        SystemExit: When the arguments cannot be parsed or help is requested, as raised by
            `argparse`.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=None, help="data root directory")
    parser = argparse.ArgumentParser(prog="polyps-data", description=__doc__, parents=[common])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "datasets", aliases=["list"], help="list registered datasets", parents=[common]
    )
    commands.add_parser("tasks", help="list registered tasks", parents=[common])
    fetch_parser = commands.add_parser(
        "fetch", help="make local copies of open datasets", parents=[common]
    )
    fetch_parser.add_argument("keys", nargs="*", help="dataset keys, all open datasets if omitted")
    index_parser = commands.add_parser("index", help="print the pins of remote zip archives")
    index_parser.add_argument("urls", nargs="+")
    check_parser = commands.add_parser(
        "check", help="find copies of test images among files or folders of images"
    )
    check_parser.add_argument("paths", nargs="+", type=Path)
    check_parser.add_argument(
        "--distance", type=int, default=16, help="largest fine hash distance of near duplicates"
    )
    check_parser.add_argument("--output", type=Path, default=None, help="CSV file of the matches")
    manifest_parser = commands.add_parser(
        "manifest", help="refresh the fingerprints of test images", parents=[common]
    )
    manifest_parser.add_argument("tasks", nargs="*", help="task identifiers or glob patterns")
    manifest_parser.add_argument("--stale", action="store_true", help="every outdated task")
    verify_parser = commands.add_parser(
        "verify", help="check that the hosts still serve the pinned files"
    )
    verify_parser.add_argument("keys", nargs="*", help="dataset keys, every dataset if omitted")
    args = parser.parse_args(argv)

    root = default_data_root() if args.root is None else args.root
    if args.command == "index":
        for url in args.urls:
            index = read_index(url)
            print(f"{url}\n  size {index.size}\n  index {index.digest}")
            print(f"  members {len(index.members)}")
        return 0

    if args.command == "check":
        return _check(args.paths, args.distance, args.output)
    registry = load_registry()
    cards = {card.key: card for card in registry.datasets}
    if args.command == "manifest":
        stale = set(stale_manifests(registry)) if args.stale else set()
        specs = [
            spec
            for spec in registry.tasks
            if spec.id in stale or any(fnmatch(spec.id, pattern) for pattern in args.tasks)
        ]
        if not specs:
            print("no task selected", file=sys.stderr)
            return 1
        for path in update_manifests(registry, root, specs):
            print(path)
        return 0
    if args.command == "verify":
        return _verify(args.keys, cards)
    if args.command in ("datasets", "list"):
        for card in registry.datasets:
            print(f"{card.key:28} {state(card, root):16} {card.licence[:40]:42} {card.name}")
        return 0
    if args.command == "tasks":
        for task in registry.tasks:
            where = state(cards[task.dataset], root)
            print(f"{task.id:36} {task.kind:15} {where:16} {task.description}")
        return 0

    keys = args.keys or [key for key, card in cards.items() if card.downloads]
    status = 0
    for key in keys:
        if key not in cards:
            print(f"unknown dataset {key!r}", file=sys.stderr)
            status = 1
            continue
        try:
            print(fetch(cards[key], root))
        except ManualAccessError as error:
            print(error, file=sys.stderr)
            status = 1
    return status


def _verify(keys: list[str], cards: dict[str, DatasetCard]) -> int:
    """Print one verdict per published file of the selected datasets.

    Args:
        keys: Dataset keys to verify, every dataset when empty. Unknown keys are reported on
            standard error.
        cards: Registered dataset cards by key.

    Returns:
        Zero when every key is known and every file matches its pins, one otherwise.
    """
    status = 0
    for key in keys:
        if key not in cards:
            print(f"unknown dataset {key!r}", file=sys.stderr)
            status = 1
    selected = [cards[key] for key in keys if key in cards] if keys else list(cards.values())
    for verdict in verify(selected):
        print(verdict, flush=True)
        if verdict.outcome is not Outcome.OK:
            status = 1
    return status


def _check(paths: list[Path], distance: int, output: Path | None) -> int:
    """Report the test images found among image files or folders.

    Args:
        paths: Image files or folders searched recursively.
        distance: Largest distance in bits, from 0 to 256, between the fine hashes of a near
            duplicate and a test image.
        output: CSV file that receives every match, or `None` to write no file.

    Returns:
        One when any file copies or nearly copies a test image, zero otherwise.

    Raises:
        ValueError: If `distance` lies outside 0 to 256.
    """
    matches = check_corpus(paths, distance, read=test_reader(default_data_root()))
    if output is not None:
        matches.to_csv(output, index=False, lineterminator="\n")
    if matches.empty:
        print("no test image found")
        return 0
    files = matches["file"].nunique()
    print(f"{files} files copy or nearly copy test images")
    counts = matches.groupby(["task", "match"]).size().reset_index(name="count")
    for row in counts.itertuples(index=False):
        print(f"  {row.task:36} {row.match:16} {row.count}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
