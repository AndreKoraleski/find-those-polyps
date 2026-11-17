"""Command line access to the curated pretraining corpora.

`polyps-corpora list` shows every corpus with its licence, its readable and held out units and
what the cache holds. `polyps-corpora fetch KEY` reads the units of a corpus once, so that later
training reads them from the cache. Archives are streamed by default, and `--whole` downloads and
verifies each one before it is read, with `--keep-archives` to leave them on disk. `--fps` sets
how many frames are kept per second of recording, `all` keeping every frame. `polyps-corpora
verify KEY` checks that the host still serves every pinned archive at its pinned size. The data
root defaults to `FIND_THOSE_POLYPS_DATA`, else to `./data`.
"""

import argparse
import sys
from pathlib import Path

from find_those_polyps.data.paths import default_data_root
from find_those_polyps.data.remote import TRANSIENT, RangeUnsupportedError, remote_size
from find_those_polyps.pretraining.corpora import CorpusCard, corpus, load_corpora
from find_those_polyps.pretraining.reader import CorpusReader, fetch


def _fps(text: str) -> float | None:
    """Frames per second from the command line, `None` for `all`."""
    return None if text == "all" else float(text)


def _summary(card: CorpusCard, root: Path, fps: float | None) -> str:
    """One line on a corpus, its units, its size and its cache."""
    reader = CorpusReader(card, root, fps=fps)
    units = reader.units()
    counts = [reader.expected(archive) for archive in units]
    frames = "unknown" if any(c is None for c in counts) else f"{sum(filter(None, counts)):,}"
    cached = sum(reader.cached(archive) for archive in units)
    size = sum(archive.size for archive in units) / 1e9
    return (
        f"{card.key:20} {card.licence:10} units {len(units)} of {len(card.archives)} "
        f"({len(card.excluded)} held out), {size:.1f} GB to read, {frames} frames kept, "
        f"{cached} units cached  {card.name}"
    )


def _verify(card: CorpusCard) -> int:
    """Print one verdict per archive of a corpus, held out units included.

    Args:
        card: Corpus to check.

    Returns:
        Zero when every archive is served at its pinned size, one otherwise.
    """
    status = 0
    for archive in card.archives:
        try:
            size = remote_size(archive.url)
        except (RangeUnsupportedError, *TRANSIENT) as error:
            print(f"unreachable {archive.filename} {error}")
            status = 1
            continue
        if size != archive.size:
            print(f"changed     {archive.filename} {size} bytes, expected {archive.size}")
            status = 1
        else:
            print(f"ok          {archive.filename}")
    return status


def main(argv: list[str] | None = None) -> int:
    """Run the `polyps-corpora` command.

    Args:
        argv: Arguments without the program name, read from `sys.argv` when `None`.

    Returns:
        The exit status, zero on success and one when a corpus or unit is unknown or a pinned
        archive changed or could not be reached.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=None, help="data root directory")
    common.add_argument(
        "--fps", type=_fps, default=1.0, help="frames kept per second of recording, or all"
    )
    parser = argparse.ArgumentParser(prog="polyps-corpora", description=__doc__, parents=[common])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list the curated corpora", parents=[common])
    fetch_parser = commands.add_parser(
        "fetch", help="cache the kept frames of a corpus", parents=[common]
    )
    fetch_parser.add_argument("key", help="corpus key")
    fetch_parser.add_argument("--units", nargs="+", default=None, help="units, all by default")
    fetch_parser.add_argument(
        "--whole",
        action="store_true",
        help="download and verify each archive before reading it instead of streaming it",
    )
    fetch_parser.add_argument(
        "--keep-archives", action="store_true", help="keep the archives downloaded whole"
    )
    fetch_parser.add_argument("--workers", type=int, default=4, help="units read at once")
    verify_parser = commands.add_parser(
        "verify", help="check that the host still serves the pinned archives"
    )
    verify_parser.add_argument("key", help="corpus key")
    args = parser.parse_args(argv)

    root = default_data_root() if args.root is None else args.root
    if args.command == "list":
        for card in load_corpora():
            print(_summary(card, root, args.fps))
        return 0
    try:
        card = corpus(args.key)
    except KeyError as error:
        print(error.args[0], file=sys.stderr)
        return 1
    if args.command == "verify":
        return _verify(card)
    reader = CorpusReader(
        card, root, fps=args.fps, download=args.whole, keep_archives=args.keep_archives
    )
    try:
        count = fetch(reader, args.units, args.workers)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"{count} units read into {reader.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
