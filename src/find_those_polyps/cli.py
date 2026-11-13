"""Single entry point dispatching to the command line tools of the benchmark.

`find-those-polyps evaluate ...` is `polyps-evaluate`, `find-those-polyps data ...` is
`polyps-data`, `find-those-polyps split ...` is `polyps-split`, `find-those-polyps registry` is
`polyps-registry`, `find-those-polyps results` rewrites the leaderboard outputs and
`find-those-polyps ratings` rates the tasks and metrics from the per image records.
"""

import sys
from collections.abc import Callable
from importlib import import_module

COMMANDS = {
    "evaluate": ("find_those_polyps.evaluation.__main__", "evaluate an encoder on the benchmark"),
    "data": ("find_those_polyps.data.__main__", "list, stream index or fetch datasets"),
    "split": ("find_those_polyps.registry.build", "build the canonical split of a task"),
    "registry": ("find_those_polyps.registry.__main__", "refresh registry hashes and docs"),
    "results": ("find_those_polyps.evaluation.results", "rewrite the workbook and summary"),
    "ratings": ("find_those_polyps.evaluation.ratings", "rate tasks and metrics from records"),
}


def usage() -> str:
    """Usage text listing every command with a short description.

    Returns:
        The usage line followed by one line per command in `COMMANDS`.
    """
    lines = ["usage: find-those-polyps COMMAND [ARGS]", "", "commands"]
    lines += [f"  {name:10} {help_text}" for name, (_, help_text) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the command named by the first argument with the remaining arguments.

    The usage text goes to the standard output when help is requested, and to the standard error
    when the command is missing or unknown.

    Args:
        argv: Command name and its arguments, `sys.argv[1:]` when `None`.

    Returns:
        The exit status of the command, 0 after a request for help and 2 for a missing or unknown
        command.
    """
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help") or args[0] not in COMMANDS:
        print(usage(), file=sys.stdout if args[:1] in (["-h"], ["--help"]) else sys.stderr)
        return 0 if args[:1] in (["-h"], ["--help"]) else 2
    command: Callable[[list[str]], int] = import_module(COMMANDS[args[0]][0]).main
    return command(args[1:])


if __name__ == "__main__":
    sys.exit(main())
