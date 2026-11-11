"""Construction of canonical split files from local or streamed data.

`polyps-split TASK ... [--missing] [--root DIR] [--seed N] [--force] [--redraw]` adopts the
published test partition of the dataset when its loader exposes one and otherwise holds out a
stratified fifth of the groups. Rebuilding an existing split keeps its test partition verbatim
unless `--redraw` asks for a new one. Validation takes a tenth of all samples by group. The file is
written next to the registry, whose version and documentation are then synchronized.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from fnmatch import fnmatch
from pathlib import Path

from tqdm.auto import tqdm

from find_those_polyps.data.paths import default_data_root
from find_those_polyps.logging import get_logger
from find_those_polyps.registry.docs import write_docs
from find_those_polyps.registry.schema import Registry
from find_those_polyps.registry.store import SPLITS_DIR, load_registry, open_loader, sync_registry
from find_those_polyps.tasks.loaders.base import FileLoader
from find_those_polyps.tasks.loaders.images import read_rgb
from find_those_polyps.tasks.schema import TaskSpec
from find_those_polyps.tasks.splits import (
    BENCHMARK_SOURCE,
    LOSSES,
    SplitFile,
    make_split,
    merge_duplicates,
    merge_near_duplicates,
)
from find_those_polyps.utils.perceptual import Picture, Signature, confirmer, signature

log = get_logger(__name__)


def build_split(
    spec: TaskSpec,
    data_root: Path,
    seed: int = 0,
    registry: Registry | None = None,
    *,
    stream: bool = True,
    previous: SplitFile | None = None,
) -> SplitFile:
    """Compute the canonical split of a task from its dataset, local or streamed.

    The test partition is never altered once fixed. A published one is adopted verbatim, and one the
    benchmark chose earlier, given as `previous`, is kept verbatim on a rebuild. Only training and
    validation are rebuilt around it. Byte identical images, recognized by the CRC32 and size of
    their files, share a group, and near duplicates share a group once their thumbnails confirm the
    candidates found by their perceptual signatures. Every sample sharing a group with the test
    partition is excluded, as are copies carrying conflicting labels outside it. Excluded
    identifiers are recorded in the split file.

    Args:
        spec: Task to split.
        data_root: Root directory of the local and cached datasets.
        seed: First seed of the fold selection.
        registry: Registry holding the dataset card, the packaged one when `None`.
        stream: Whether to stream the dataset when no local copy exists.
        previous: Split committed earlier, whose test partition is kept when the benchmark chose it.

    Returns:
        The split.

    Raises:
        FileNotFoundError: If the dataset is neither local nor streamable.
    """
    loader = open_loader(spec, data_root, registry, stream=stream)
    if not loader.available():
        raise FileNotFoundError(f"{spec.dataset} not found under {data_root}")
    ids = loader.ids()
    group_of = loader.groups()
    label_of = loader.labels()
    groups = [group_of[i] for i in ids]
    labels = None if label_of is None else [label_of[i] for i in ids]
    conflicts: set[str] = set()
    stages = [list(groups)]
    if isinstance(loader, FileLoader):
        prints = [loader.files.fingerprint(loader.members(i)[0]) for i in tqdm(ids, leave=False)]
        groups, conflicts = merge_duplicates(ids, groups, prints, labels)
        stages.append(list(groups))
        before = len(set(groups))
        files = loader.files

        def load(position: int) -> Picture | None:
            try:
                return read_rgb(files.read(loader.members(ids[position])[0]))
            except (OSError, ValueError, KeyError):
                return None

        hashes = signatures(loader, ids, prints, data_root)
        groups = merge_near_duplicates(groups, hashes, confirmer(load))
        stages.append(list(groups))
        largest = max(Counter(groups).values(), default=0)
        merged = before - len(set(groups))
        log.info("near duplicates grouped", task=spec.id, merged=merged, largest_group=largest)
    official = loader.official_test()
    kept = previous is not None and not official and previous.test_source == BENCHMARK_SOURCE
    if official:
        fixed: list[str] | None = official[0]
    else:
        fixed = list(previous.test) if kept and previous is not None else None
    dropped = conflicts - set(fixed or ())
    keep = [index for index, sample_id in enumerate(ids) if sample_id not in dropped]
    split = make_split(
        [ids[i] for i in keep],
        [groups[i] for i in keep],
        None if labels is None else [labels[i] for i in keep],
        test=fixed if fixed is not None else 0.2,
        test_source=official[1] if official else BENCHMARK_SOURCE,
        seed=seed,
        excluded=sorted(dropped),
        kept=kept,
    )
    return split.model_copy(update={"losses": losses(ids, stages, split, dropped)})


def losses(
    ids: Sequence[str], stages: Sequence[Sequence[str]], split: SplitFile, dropped: set[str]
) -> dict[str, int]:
    """Count the samples left out of a split for each reason of `LOSSES`.

    A sample sharing a group with the test partition is attributed to the earliest grouping stage at
    which it does, so that a patient shared with a test image counts as such even when the image is
    also a copy.

    Args:
        ids: Identifier of each sample.
        stages: Group of each sample after each grouping stage, by metadata, by identical files and
            by confirmed near duplicates, of which the later ones may be absent.
        split: Split whose excluded samples are counted.
        dropped: Copies excluded for their conflicting labels.

    Returns:
        The number of lost samples for each reason, every reason included.
    """
    position = {sample_id: index for index, sample_id in enumerate(ids)}
    test = [position[i] for i in split.test]
    shared = [{stage[i] for i in test} for stage in stages]
    counts = dict.fromkeys(LOSSES, 0)
    reasons = LOSSES[: len(stages)]
    for sample_id in split.excluded:
        if sample_id in dropped:
            counts["conflicting labels"] += 1
            continue
        index = position[sample_id]
        for reason, stage, groups in zip(reasons, stages, shared, strict=True):
            if stage[index] in groups:
                counts[reason] += 1
                break
    return counts


def signatures(
    loader: FileLoader, ids: Sequence[str], prints: Sequence[str], data_root: Path
) -> list[Signature | None]:
    """Compute the perceptual signature of the image of every sample.

    Signatures are cached under the data root by the CRC32 and size of the image file, so that tasks
    sharing images and later builds compute each signature once.

    Args:
        loader: Loader of the samples.
        ids: Identifier of each sample.
        prints: CRC32 and size of the image file of each sample.
        data_root: Root directory holding the `.signatures` cache.

    Returns:
        The signature of each sample, `None` where the image could not be decoded.
    """
    cache = data_root / ".signatures" / f"{loader.files.key}.json"
    known: dict[str, list[int]] = (
        json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    )
    key_of = dict(zip(ids, prints, strict=True))
    missing = [i for i in ids if key_of[i] not in known]
    for sample in tqdm(loader.stream(missing), total=len(missing), leave=False):
        known[key_of[sample.id]] = list(signature(sample.image))
    if missing:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, sort_keys=True), encoding="utf-8")
    return [Signature(*known[key_of[i]]) if key_of[i] in known else None for i in ids]


def write_split(split: SplitFile, path: Path) -> None:
    """Write a split file in a stable, diff friendly layout.

    Args:
        split: Split to write.
        path: Destination file.
    """
    text = json.dumps(split.model_dump(mode="json"), indent=1, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    """Build the split files of the selected tasks, then refresh the registry and its docs.

    Args:
        argv: Command line arguments, those of the process when `None`.

    Returns:
        Zero on success. Argument errors exit through argparse.
    """
    parser = argparse.ArgumentParser(prog="polyps-split", description=__doc__)
    parser.add_argument("tasks", nargs="*", help="task identifiers or glob patterns")
    parser.add_argument("--missing", action="store_true", help="every task without a split file")
    parser.add_argument("--root", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing splits, invalidating results"
    )
    parser.add_argument(
        "--redraw",
        action="store_true",
        help="draw anew the test partitions the benchmark chose, instead of keeping them",
    )
    args = parser.parse_args(argv)
    registry = load_registry(check=False)
    specs = [
        spec
        for spec in registry.tasks
        if any(fnmatch(spec.id, pattern) for pattern in args.tasks)
        or (args.missing and not (SPLITS_DIR / spec.splits).exists())
    ]
    if not specs:
        parser.error("no task selected")
    existing = [spec.id for spec in specs if (SPLITS_DIR / spec.splits).exists()]
    if existing and not args.force:
        parser.error(
            f"canonical splits exist for {', '.join(existing)}. Changing them invalidates every "
            "published result on these tasks, pass --force to do so anyway"
        )
    root = default_data_root() if args.root is None else Path(args.root)
    for spec in specs:
        path = SPLITS_DIR / spec.splits
        previous = (
            SplitFile.model_validate_json(path.read_text(encoding="utf-8"))
            if path.exists() and not args.redraw
            else None
        )
        split = build_split(spec, root, args.seed, registry, previous=previous)
        write_split(split, SPLITS_DIR / spec.splits)
        sizes = f"train {len(split.train)} val {len(split.val)} test {len(split.test)}"
        print(f"{spec.id} {sizes} ({split.test_source})")
    sync_registry()
    write_docs(load_registry())
    return 0


if __name__ == "__main__":
    sys.exit(main())
