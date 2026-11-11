"""Human readable dataset documentation rendered from the registry.

The page lists every dataset and task, the sizes of the canonical partitions, the documented leakage
between datasets, the source families and the provenance of each dataset. It is generated into
docs/datasets.md and never edited by hand.
"""

from collections import defaultdict
from pathlib import Path

from find_those_polyps.registry.schema import DatasetCard, OverlapKind, Registry
from find_those_polyps.registry.store import SPLITS_DIR
from find_those_polyps.tasks.schema import TaskSpec
from find_those_polyps.tasks.splits import SplitFile

DOCS_PATH = Path(__file__).parents[3] / "docs" / "datasets.md"

_KIND_TEXT = {
    OverlapKind.DUPLICATE: "contains images from",
    OverlapKind.SAME_PATIENTS: "contains frames of the same patients or videos as",
    OverlapKind.SAME_SOURCE: "shares hospital and period with",
}


def warnings(registry: Registry) -> list[str]:
    """Leakage warnings naming every dataset or corpus that shares content with a dataset.

    Args:
        registry: Registry whose documented overlaps are described.

    Returns:
        One Markdown sentence per overlap, sorted alphabetically, naming both sides, the nature of
        the overlap, its optional note and its evidence.
    """
    cards = {card.key: card for card in registry.datasets}
    lines = []
    for card in registry.datasets:
        for overlap in card.overlaps:
            other = cards[overlap.dataset].name if overlap.dataset in cards else overlap.name
            note = f" ({overlap.note})" if overlap.note else ""
            lines.append(
                f"**{card.name}** {_KIND_TEXT[overlap.kind]} **{other}** (`{overlap.dataset}`)"
                f"{note}. Evidence {overlap.evidence}"
            )
    return sorted(lines)


def _row(card: DatasetCard, tasks: list[str]) -> str:
    """Markdown table row summarizing one dataset.

    Args:
        card: Dataset to describe.
        tasks: Identifiers of the tasks defined on the dataset.

    Returns:
        Row of key, name, labelled flag, access, licence, family and tasks.
    """
    cells = [
        f"`{card.key}`",
        card.name,
        "yes" if card.labelled else "no",
        card.access.value,
        card.licence,
        f"`{card.family}`",
        ", ".join(f"`{t}`" for t in tasks) or "none yet",
    ]
    return "| " + " | ".join(cells) + " |"


def _task_row(task: TaskSpec, splits_dir: Path | None) -> str:
    """Markdown table row summarizing one task and its canonical split.

    Args:
        task: Task to describe.
        splits_dir: Directory of the split files, or `None` to report every split as not built.

    Returns:
        Row of identifier, kind, class count, primary metric, partition sizes, lost images and
        test source. The sizes read "not built" when the split file is missing.
    """
    path = None if splits_dir is None else splits_dir / task.splits
    if path is not None and path.exists():
        split = SplitFile.model_validate_json(path.read_text(encoding="utf-8"))
        sizes = f"{len(split.train)} / {len(split.val)} / {len(split.test)}"
        lost = str(len(split.excluded))
        reasons = [f"{count} {reason}" for reason, count in split.losses.items() if count]
        if reasons:
            lost += f" ({', '.join(reasons)})"
        source = "benchmark" if split.test_source == "find-those-polyps" else split.test_source
    else:
        sizes, lost, source = "not built", "", ""
    cells = [
        f"`{task.id}`",
        task.kind.value,
        str(len(task.classes)),
        f"`{task.primary_metric}`",
        sizes,
        lost,
        source,
    ]
    return "| " + " | ".join(cells) + " |"


def render(registry: Registry, splits_dir: Path | None = SPLITS_DIR) -> str:
    """Markdown page listing datasets, tasks, families and leakage warnings.

    Args:
        registry: Registry to document.
        splits_dir: Directory of the split files that provide the partition sizes, or `None` to
            report every split as not built. Defaults to the packaged split directory.

    Returns:
        Full text of the page with Unix line endings, ending in a single newline.
    """
    tasks: dict[str, list[str]] = defaultdict(list)
    for task in registry.tasks:
        tasks[task.dataset].append(task.id)
    families: dict[str, list[str]] = defaultdict(list)
    for card in registry.datasets:
        families[card.family].append(card.key)
    lines = [
        "# Datasets",
        "",
        f"Generated from registry version {registry.version} (`{registry.sha256[:12]}`). Do not",
        "edit by hand, run `polyps-registry` instead.",
        "",
        "Images are never redistributed. Every dataset is openly downloadable and streamed from",
        "its official host during evaluation, or copied whole with `polyps-data fetch KEY`.",
        "",
        "| Key | Name | Labelled | Access | Licence | Family | Tasks |",
        "|---|---|---|---|---|---|---|",
        *(_row(card, tasks[card.key]) for card in registry.datasets),
        "",
        "## Tasks",
        "",
        "Partition sizes are train, validation and test. The test partition is the published",
        "one when a source is named, and a group stratified fifth chosen by the benchmark",
        "otherwise. It is never altered once fixed. Lost images are left out of training and",
        "validation because they share a patient, video or sequence with a test image (same",
        "group), copy a test image byte for byte (identical copy), are confirmed near duplicates",
        "of a test image (near duplicate), or are copies whose labels contradict each other",
        "(conflicting labels).",
        "",
        "| Task | Kind | Classes | Primary metric | Partitions | Lost | Test source |",
        "|---|---|---|---|---|---|---|",
        *(_task_row(task, splits_dir) for task in registry.tasks),
        "",
        "## Leakage warnings",
        "",
        "An encoder pretrained on a dataset or corpus named here has seen images of the dataset",
        "it overlaps. Pass its key to `pretrained_on` and the benchmark excludes every",
        "overlapping task.",
        "",
        *(f"- {line}" for line in warnings(registry)),
        "",
        "## Source families",
        "",
        "Datasets in one family come from the same clinical source pool. Patient separation across",
        "members of a family is undocumented unless an overlap entry states otherwise.",
        "",
        *(
            f"- `{family}` {', '.join(f'`{k}`' for k in keys)}"
            for family, keys in sorted(families.items())
        ),
        "",
        "## Provenance",
        "",
    ]
    for card in registry.datasets:
        lines += [
            f"### {card.name}",
            "",
            card.provenance,
            "",
            f"Citation {card.citation}. {card.url}",
        ]
        if card.label_source:
            lines.append(f"Labels by {card.label_source}.")
        if card.notes:
            lines.append(card.notes)
        lines.append("")
    return "\n".join(lines)


def write_docs(registry: Registry, path: Path = DOCS_PATH, splits_dir: Path | None = None) -> bool:
    """Write the dataset page unless its current content is identical.

    Args:
        registry: Registry to document.
        path: Destination of the page, docs/datasets.md of the repository by default. Missing
            parent directories are created.
        splits_dir: Directory of the split files, or `None` for the packaged split directory.

    Returns:
        Whether the file was written, that is whether its content changed.
    """
    text = render(registry, SPLITS_DIR if splits_dir is None else splits_dir)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return True
