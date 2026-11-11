"""Command line evaluation of an encoder built by a factory.

The encoder is named as `module:factory`, for instance
`find_those_polyps.baselines:ImageNetResNet50`, or as `path/to/file.py:factory`, and the factory is
called without arguments. Modules are also searched in the current directory. Selection and
execution options mirror the fields of `EvaluationConfig`, and datasets absent from the data root
are streamed from their official hosts unless `--no-stream` is given.
"""

import argparse
import importlib
import importlib.util
import math
import sys
from pathlib import Path

from pydantic import ValidationError

from find_those_polyps.data.__main__ import state
from find_those_polyps.evaluation.leaderboard import LeaderboardRow
from find_those_polyps.evaluation.runner import (
    EvaluationConfig,
    evaluate,
    resolve_protocols,
    select_tasks,
)
from find_those_polyps.logging import configure_logging
from find_those_polyps.models.encoder import Encoder
from find_those_polyps.registry.store import load_registry


class EncoderReferenceError(ValueError):
    """The encoder reference does not name a factory returning an encoder."""


def _module(name: str) -> object:
    """Module named by a dotted import path or by the path of a Python file.

    The current directory is added to the import path, so that local modules are found.

    Args:
        name: Dotted module name, or a path ending in `.py` executed as a fresh module.

    Returns:
        The imported module.

    Raises:
        EncoderReferenceError: If the file does not exist or the module cannot be imported.
    """
    if name.endswith(".py"):
        path = Path(name).resolve()
        if not path.is_file():
            raise EncoderReferenceError(f"no such file {name!r}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise EncoderReferenceError(f"cannot import {name!r}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        raise EncoderReferenceError(f"cannot import {name!r}, {error}") from error


def load_encoder(reference: str) -> Encoder:
    """Instantiate the encoder named by `module:factory` or `file.py:factory`.

    Args:
        reference: Module or file followed by the name of a factory callable without arguments,
            split at the last colon.

    Returns:
        The encoder returned by the factory.

    Raises:
        EncoderReferenceError: If the reference is malformed, the module cannot be imported, the
            factory does not exist or it returns something other than an `Encoder`.
    """
    module_name, _, attribute = reference.rpartition(":")
    if not module_name or not attribute:
        raise EncoderReferenceError(
            f"expected module:factory or file.py:factory, got {reference!r}"
        )
    factory = getattr(_module(module_name), attribute, None)
    if factory is None:
        raise EncoderReferenceError(f"{module_name!r} has no attribute {attribute!r}")
    encoder = factory()
    if not isinstance(encoder, Encoder):
        raise EncoderReferenceError(
            f"{reference} returned {type(encoder).__name__}, not an Encoder"
        )
    return encoder


def parser() -> argparse.ArgumentParser:
    """Argument parser of the command.

    Options left unset default to `None`, so that the defaults of `EvaluationConfig` apply.

    Returns:
        The parser of `polyps-evaluate`, described by the module docstring.
    """
    defaults = EvaluationConfig.model_fields
    result = argparse.ArgumentParser(
        prog="polyps-evaluate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add = result.add_argument
    add("encoder", help="module:factory or path/to/file.py:factory returning an Encoder")
    add("--tasks", nargs="+", default=["*"], help="task identifiers or glob patterns, default all")
    add("--exclude-tasks", nargs="+", default=[], help="task identifiers or patterns to skip")
    add("--exclude-datasets", nargs="+", default=[], help="dataset keys to skip")
    add(
        "--pretrained-on",
        nargs="+",
        default=[],
        help="pretraining datasets besides those the encoder declares, skipping overlapping tasks",
    )
    add("--protocols", nargs="+", default=None, help="protocols to run, default all")
    add(
        "--seeds",
        type=int,
        default=None,
        help=f"repeats of reduced budgets, default {defaults['seeds'].default}",
    )
    add("--batch-size", type=int, default=None, help=f"default {defaults['batch_size'].default}")
    add("--device", default=None, help="torch device, default cuda when available")
    add("--data-root", type=Path, default=None, help="default FIND_THOSE_POLYPS_DATA or ./data")
    add("--leaderboard", type=Path, default=None, help=f"default {defaults['leaderboard'].default}")
    add(
        "--per-image-dir",
        type=Path,
        default=None,
        help=f"default {defaults['per_image_dir'].default}",
    )
    add("--bootstrap", type=int, default=None, help="test resamples for intervals, 0 disables")
    add("--no-stream", action="store_true", help="read local copies only")
    add("--no-cache", action="store_true", help="keep streamed files in memory only")
    add("--no-feature-cache", action="store_true", help="encode every image again")
    add("--dry-run", action="store_true", help="list the selected tasks and exit")
    add("--resume", action="store_true", help="skip settings already on the leaderboard")
    add("--limit", type=int, default=None, help="smoke test on at most N images per partition")
    add("--log-level", default="INFO")
    return result


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, evaluate the encoder and return the exit status.

    With `--dry-run` the selected tasks are listed with the state of their dataset and nothing is
    evaluated. Otherwise the primary scores on the full budget are printed once every task ran.

    Args:
        argv: Command line arguments without the program name, `sys.argv[1:]` when `None`.

    Returns:
        0 when the evaluation or the listing completes.

    Raises:
        SystemExit: If the arguments are invalid or violate a constraint of `EvaluationConfig`,
            for instance a nonpositive number of seeds, if a protocol, task pattern or dataset key
            is unknown, or if the encoder cannot be loaded, with status 2.
        EvaluationError: If some tasks failed after the others were recorded.
    """
    command = parser()
    args = command.parse_args(argv)
    configure_logging(args.log_level)
    fields = {
        "tasks": tuple(args.tasks),
        "exclude_tasks": tuple(args.exclude_tasks),
        "exclude_datasets": tuple(args.exclude_datasets),
        "pretrained_on": tuple(args.pretrained_on),
        "protocols": None if args.protocols is None else tuple(args.protocols),
        "seeds": args.seeds,
        "batch_size": args.batch_size,
        "device": args.device,
        "data_root": args.data_root,
        "leaderboard": args.leaderboard,
        "per_image_dir": args.per_image_dir,
        "bootstrap": args.bootstrap,
        "stream": False if args.no_stream else None,
        "cache": False if args.no_cache else None,
        "feature_cache": False if args.no_feature_cache else None,
        "resume": True if args.resume else None,
        "limit": args.limit,
    }
    try:
        config = EvaluationConfig.model_validate({k: v for k, v in fields.items() if v is not None})
    except ValidationError as error:
        problems = (f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
        command.error("invalid options, " + ", ".join(problems))
    registry = load_registry()
    try:
        resolve_protocols(config.protocols)
        specs = select_tasks(registry, config)
        if args.dry_run:
            cards = {card.key: card for card in registry.datasets}
            for spec in specs:
                print(f"{spec.id:32} {state(cards[spec.dataset], config.data_root):16} {spec.kind}")
            print(f"{len(specs)} tasks, protocols {', '.join(config.protocols)}")
            return 0
        encoder = load_encoder(args.encoder)
    except ValueError as error:
        command.error(str(error))
    rows = evaluate(encoder, config, registry=registry)
    print(summary(rows))
    return 0


def summary(rows: list[LeaderboardRow]) -> str:
    """Table of the primary score with its interval for every task and protocol.

    Only rows on the full label budget are listed, sorted by task and protocol. The interval is
    left blank when a bound is `None` or NaN, as when the bootstrap was disabled.

    Args:
        rows: Leaderboard rows produced by the evaluation.

    Returns:
        A header line followed by one line per row on the full budget.
    """
    lines = [f"{'task':32} {'protocol':11} {'metric':16} {'score':>7}  interval"]
    for row in sorted(rows, key=lambda r: (r.task, r.protocol)):
        if row.budget != "full":
            continue
        low, high = row.score_ci_low, row.score_ci_high
        bounds = ""
        if low is not None and high is not None and not (math.isnan(low) or math.isnan(high)):
            bounds = f"[{low:.3f}, {high:.3f}]"
        score = f"{row.score:7.3f}"
        lines.append(f"{row.task:32} {row.protocol:11} {row.primary_metric:16} {score}  {bounds}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
