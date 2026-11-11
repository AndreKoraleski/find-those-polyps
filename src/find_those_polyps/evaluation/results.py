"""Regeneration of the leaderboard outputs from its CSV file.

`find-those-polyps results [--leaderboard PATH]` recomputes ranks and stale flags and rewrites the
CSV file, the workbook and the Markdown summary, for instance after their format changed, without
evaluating anything.
"""

import argparse
import sys
from pathlib import Path

from find_those_polyps.evaluation.leaderboard import Leaderboard
from find_those_polyps.evaluation.runner import EvaluationConfig


def main(argv: list[str] | None = None) -> int:
    """Rewrite the outputs derived from the leaderboard CSV.

    Args:
        argv: Command line arguments without the program name, `sys.argv[1:]` when `None`.

    Returns:
        0 once the outputs are rewritten.

    Raises:
        SystemExit: If the arguments are invalid or no leaderboard exists at the given path, with
            status 2.
    """
    parser = argparse.ArgumentParser(prog="find-those-polyps results", description=__doc__)
    default = EvaluationConfig.model_fields["leaderboard"].default
    parser.add_argument("--leaderboard", type=Path, default=default, help=f"default {default}")
    args = parser.parse_args(argv)
    board = Leaderboard(args.leaderboard)
    if not board.path.exists():
        parser.error(f"no leaderboard at {board.path}")
    board.refresh()
    print(f"rewrote {board.path}, {board.workbook} and {board.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
