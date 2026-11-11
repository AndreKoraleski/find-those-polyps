"""Evaluation of an encoder on every selected task and protocol.

Predictors are fitted on the training partition under each label budget, model selection uses the
validation partition, and every reported number, aggregated or per image, is measured on the
canonical test partition.
"""

import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from difflib import get_close_matches
from fnmatch import fnmatch
from math import nan
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from find_those_polyps.data.paths import default_data_root
from find_those_polyps.evaluation.features import Features, extract
from find_those_polyps.evaluation.leaderboard import Leaderboard, LeaderboardRow
from find_those_polyps.evaluation.protocols import PROTOCOLS, DirectProtocol, Protocol
from find_those_polyps.evaluation.records import (
    classification_records,
    detection_records,
    write_records,
)
from find_those_polyps.evaluation.store import FeatureStore
from find_those_polyps.logging import get_logger
from find_those_polyps.metrics import MASKED, Predictions, metric_names, score
from find_those_polyps.metrics.bootstrap import (
    Statistic,
    averaged,
    detection_statistic,
    mean_statistic,
    prediction_statistic,
)
from find_those_polyps.metrics.bootstrap import interval as bootstrap_interval
from find_those_polyps.metrics.detection import match, summarize
from find_those_polyps.metrics.segmentation import per_image
from find_those_polyps.models.encoder import Encoder
from find_those_polyps.registry.schema import Registry
from find_those_polyps.registry.store import build_task, load_registry
from find_those_polyps.tasks.budgets import budget_indices
from find_those_polyps.tasks.loaders.base import FileLoader
from find_those_polyps.tasks.schema import FULL, Budget, Split, TaskKind, TaskSpec
from find_those_polyps.tasks.task import LazySamples, Task

log = get_logger(__name__)

SINGLE_LABEL = frozenset({TaskKind.CLASSIFICATION, TaskKind.ORDINAL})


def default_device() -> str:
    """Default torch device of an evaluation.

    Returns:
        `cuda` when a CUDA device is available, else `cpu`.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


class EvaluationError(RuntimeError):
    """Some tasks failed, while the results of the others were recorded on the leaderboard."""


class EvaluationConfig(BaseModel):
    """Task selection and execution settings of an evaluation.

    Tasks are selected by glob patterns over their identifiers. Datasets listed in `pretrained_on`
    are excluded together with every dataset documented to overlap them, which prevents scoring an
    encoder on images it has seen. The confidence interval of each score comes from resampling the
    test set, the statistic being the mean over seeds of the primary metric on the same resampled
    images.

    Attributes:
        tasks: Glob patterns over task identifiers selecting the tasks, all of them by default.
            A pattern matching no registered task is an error.
        exclude_tasks: Glob patterns of tasks to skip among the selected ones, none by default.
        exclude_datasets: Keys of registered datasets whose tasks are skipped, none by default.
        pretrained_on: Keys of datasets seen in pretraining besides those the encoder declares.
            Tasks on them and on every dataset documented to overlap them are skipped.
        protocols: Names of the protocols to run, every registered protocol by default.
        seeds: Repeats of every reduced budget and of stochastic protocols, 5 by default. Full
            budgets of deterministic protocols run once.
        batch_size: Number of images encoded together, 64 by default.
        device: Torch device of the encoder and the probes, `cuda` when available and `cpu`
            otherwise.
        data_root: Folder of the datasets, `FIND_THOSE_POLYPS_DATA` or `data` by default.
        leaderboard: CSV file of the leaderboard, `results/leaderboard.csv` by default.
        per_image_dir: Folder of the per image results, `results/per_image` by default and
            `None` to write none.
        stream: Whether datasets missing under `data_root` are streamed from their official
            host, true by default.
        cache: Whether streamed files are kept under `data_root`, true by default.
        feature_cache: Whether the features of every image are kept in a `.features` folder
            under `data_root`, keyed by the fingerprint of the encoder, so that later runs of
            the same encoder do not encode the images again. True by default.
        resume: Whether to skip the settings already on the leaderboard for the same model and
            the same task and protocol definitions, false by default.
        bootstrap: Number of test resamples of the confidence interval of each score, 1000 by
            default and zero to disable it.
        limit: Maximum number of samples kept in each partition for a quick smoke test, whose
            results go to a `smoke` folder beside the leaderboard and the per image results.
            `None`, the default, keeps every sample.
    """

    model_config = ConfigDict(frozen=True)

    tasks: tuple[str, ...] = ("*",)
    exclude_tasks: tuple[str, ...] = ()
    exclude_datasets: tuple[str, ...] = ()
    pretrained_on: tuple[str, ...] = ()
    protocols: tuple[str, ...] = tuple(PROTOCOLS)
    seeds: PositiveInt = 5
    batch_size: PositiveInt = 64
    device: str = Field(default_factory=default_device)
    data_root: Path = Field(default_factory=default_data_root)
    leaderboard: Path = Path("results/leaderboard.csv")
    per_image_dir: Path | None = Path("results/per_image")
    stream: bool = True
    cache: bool = True
    feature_cache: bool = True
    resume: bool = False
    bootstrap: int = Field(default=1000, ge=0)
    limit: PositiveInt | None = None


def _unknown(kind: str, names: Iterable[str], known: Iterable[str]) -> ValueError:
    """Error naming unknown entries together with the closest known ones.

    Args:
        kind: Plural noun naming the entries in the message, for instance `protocols`.
        names: Unknown entries.
        known: Valid entries, all listed when there are at most twelve and none is close.

    Returns:
        The error, to be raised by the caller.
    """
    choices = sorted(known)
    parts = []
    for name in sorted(names):
        close = get_close_matches(name, choices, n=3, cutoff=0.4)
        if close:
            parts.append(f"{name!r} (did you mean {', '.join(close)})")
        elif len(choices) <= 12:
            parts.append(f"{name!r} (choose from {', '.join(choices)})")
        else:
            parts.append(repr(name))
    return ValueError(f"unknown {kind} {', '.join(parts)}")


def select_tasks(registry: Registry, config: EvaluationConfig) -> list[TaskSpec]:
    """Registered tasks admitted by the configuration.

    Unknown dataset keys and task patterns that match no registered task are errors, since they
    usually are typing mistakes that would silently change the evaluation. A pretraining dataset
    that neither is registered nor overlaps a registered one shares no image with any task, which
    is logged with the closest known keys in case of a typing mistake. Every skipped task is
    logged with the reason.

    Args:
        registry: Registry of datasets and tasks.
        config: Settings holding the task patterns and the excluded and pretraining datasets.

    Returns:
        The specifications of the selected tasks in registry order.

    Raises:
        ValueError: If an excluded dataset key is not registered or a task pattern matches no
            registered task.
    """
    known = {dataset.key for dataset in registry.datasets}
    unknown = set(config.exclude_datasets) - known
    if unknown:
        raise _unknown("dataset keys", unknown, known)
    corpora = registry.corpora()
    for key in sorted(set(config.pretrained_on) - corpora):
        close = get_close_matches(key, sorted(corpora), n=3, cutoff=0.6)
        log.warning("pretraining dataset overlaps no task", dataset=key, closest=close)
    ids = [spec.id for spec in registry.tasks]
    unmatched = [p for p in config.tasks if not any(fnmatch(i, p) for i in ids)]
    if unmatched:
        raise _unknown("task patterns", unmatched, ids)
    contaminated = registry.overlapping(set(config.pretrained_on))
    selected = []
    for spec in registry.tasks:
        if not any(fnmatch(spec.id, pattern) for pattern in config.tasks):
            continue
        reason = None
        if any(fnmatch(spec.id, pattern) for pattern in config.exclude_tasks):
            reason = "task excluded"
        elif spec.dataset in config.exclude_datasets:
            reason = "dataset excluded"
        elif spec.dataset in contaminated:
            reason = "dataset overlaps pretraining data"
        if reason:
            log.info("skipping task", task=spec.id, reason=reason)
        else:
            selected.append(spec)
    return selected


def resolve_protocols(names: Sequence[str]) -> list[Protocol]:
    """Default protocol instances for the given names.

    Args:
        names: Names of registered protocols.

    Returns:
        The protocol instances in the order of `names`.

    Raises:
        ValueError: If a name is not a registered protocol.
    """
    unknown = set(names) - set(PROTOCOLS)
    if unknown:
        raise _unknown("protocols", unknown, PROTOCOLS)
    return [PROTOCOLS[name] for name in names]


def score_run(
    spec: TaskSpec, predictions: Predictions, test: Features
) -> tuple[dict[str, float], pd.DataFrame]:
    """Aggregated test metrics and the per image records of one run.

    Segmentation metrics are means of per image values, detection metrics are computed over the
    matches of every image, and the other kinds are scored on the probability matrix.

    Args:
        spec: Task the predictions were made for.
        predictions: Class probabilities of shape (N, C), an iterable of (C, H, W) probability
            maps or one table of scored boxes of shape (M, 6) per image, depending on the task
            kind.
        test: Test features whose targets, identifiers and paths are used.

    Returns:
        The value of every metric and one record per test image, starting with its identifier
        and path.

    Raises:
        TypeError: If the type of `predictions` does not match the task kind.
    """
    n_classes = len(spec.classes)
    if spec.kind in MASKED:
        if isinstance(predictions, np.ndarray):
            raise TypeError("segmentation expects an iterable of probability maps")
        masks = [np.asarray(t, dtype=np.int64) for t in test.targets]
        maps = cast(Iterable[NDArray[np.float32]], predictions)
        multilabel = spec.kind is TaskKind.MULTILABEL_SEGMENTATION
        records = pd.DataFrame(per_image(maps, masks, n_classes, multilabel=multilabel))
        names = metric_names(spec.kind, n_classes)
        metrics = {name: float(records[name].mean()) for name in names if name in records}
    elif spec.kind is TaskKind.DETECTION:
        if isinstance(predictions, np.ndarray):
            raise TypeError("detection expects one table of scored boxes per image")
        boxes = [np.asarray(t, dtype=np.float64).reshape(-1, 5) for t in test.targets]
        matched = [match(p, t, n_classes) for p, t in zip(predictions, boxes, strict=True)]
        metrics = summarize(matched, n_classes)
        records = detection_records(matched)
    else:
        if not isinstance(predictions, np.ndarray):
            raise TypeError(f"{spec.kind} expects a probability matrix")
        metrics = score(spec.kind, predictions, test.targets, n_classes)
        records = classification_records(spec, predictions, test.targets)
    records.insert(0, "sample_id", list(test.ids) or range(len(records)))
    records.insert(1, "path", list(test.paths) or "")
    return metrics, records


def budget_validation(val: Features, budget: Budget, spec: TaskSpec, seed: int) -> Features:
    """Validation subset matching a reduced label budget, so that selection uses no more labels.

    The budget rule is applied to the validation partition, for instance five shots per class at
    `shots:5`, which keeps the total of labels proportional to the budget. The full partition is
    kept under the full budget and when the budget covers it.

    Args:
        val: Validation features.
        budget: Label budget of the run.
        spec: Task, whose kind decides whether the draw is stratified by class.
        seed: Seed of the draw, the seed of the run.

    Returns:
        The validation features restricted to the drawn samples.
    """
    n_val = len(val.targets)
    if budget.is_full or budget.size(n_val, len(spec.classes)) >= n_val:
        return val
    labels = val.labels() if spec.kind in SINGLE_LABEL else None
    return val.subset(budget_indices(budget, n_val, labels, seed))


def primary_statistic(
    spec: TaskSpec, predictions: Predictions, records: pd.DataFrame, test: Features
) -> Statistic:
    """Primary metric of the task on resampled test samples.

    Args:
        spec: Task whose primary metric is measured.
        predictions: Predictions of the run on the test partition.
        records: Per image records of the run, whose primary metric column is averaged for
            segmentation tasks.
        test: Test features with the targets.

    Returns:
        A statistic mapping resampled test indices to the primary metric.
    """
    if spec.kind in MASKED:
        return mean_statistic(records[spec.primary_metric].to_numpy(dtype=float))
    if spec.kind is TaskKind.DETECTION:
        tables = cast(Sequence[NDArray[np.float64]], predictions)
        return detection_statistic(spec.primary_metric, tables, test.targets, len(spec.classes))
    assert isinstance(predictions, np.ndarray)
    return prediction_statistic(
        spec.kind, spec.primary_metric, predictions, test.targets, len(spec.classes)
    )


def declaration(encoder: Encoder, config: EvaluationConfig) -> tuple[str, ...] | None:
    """Pretraining datasets declared by the encoder and the configuration.

    Args:
        encoder: Encoder whose card may declare pretraining datasets.
        config: Settings listing further pretraining datasets.

    Returns:
        The sorted union of both declarations, or `None` when the encoder declared nothing and
        the configuration lists no dataset.

    Raises:
        pydantic.ValidationError: If the metadata of the encoder is malformed.
    """
    declared = encoder.card.pretrained_on
    if declared is None and not config.pretrained_on:
        return None
    return tuple(sorted(set(declared or ()) | set(config.pretrained_on)))


def describe(declared: tuple[str, ...] | None) -> str:
    """Leaderboard text of a pretraining declaration.

    Args:
        declared: Pretraining datasets, or `None` when undeclared.

    Returns:
        `undeclared` for `None`, `none` for an empty declaration and the keys joined by `+`
        otherwise.
    """
    if declared is None:
        return "undeclared"
    return "+".join(declared) if declared else "none"


def feature_grid(features: Features) -> str:
    """Height and width of dense feature maps written as `hxw`, empty for global features.

    Args:
        features: Features of shape (N, D) or (N, D, h, w).

    Returns:
        The grid size, for instance `14x14`, or an empty string.
    """
    if features.values.ndim != 4:
        return ""
    return "x".join(str(side) for side in features.values.shape[2:])


def _summarize(
    runs: list[dict[str, float]],
    *,
    encoder: Encoder,
    pretraining: str,
    grid: str,
    spec: TaskSpec,
    protocol: Protocol,
    budget: Budget,
    registry_version: int,
    interval: tuple[float, float],
) -> LeaderboardRow:
    """Leaderboard row aggregating the runs of one setting.

    Direct protocols report their own metrics and primary metric, and the other protocols those of
    the task. Standard deviations use one degree of freedom and are NaN for a single run.

    Args:
        runs: Metrics of every seed.
        encoder: Evaluated encoder.
        pretraining: Leaderboard text of the pretraining declaration.
        grid: Height and width of the dense feature maps, empty for global protocols.
        spec: Evaluated task.
        protocol: Protocol of the runs.
        budget: Label budget of the runs.
        registry_version: Version of the registry the task was taken from.
        interval: Bootstrap interval of the primary score, NaN bounds when disabled.

    Returns:
        The row, stamped with the current time.
    """
    if isinstance(protocol, DirectProtocol):
        names, primary = protocol.metrics, protocol.primary
    else:
        names, primary = metric_names(spec.kind, len(spec.classes)), spec.primary_metric
    values = {name: np.array([run[name] for run in runs]) for name in names}
    return LeaderboardRow(
        model=encoder.name,
        description=encoder.description,
        pretraining=pretraining,
        input_size=encoder.image_size,
        feature_grid=grid,
        task=spec.id,
        task_hash=spec.hash,
        dataset=spec.dataset,
        protocol=protocol.name,
        protocol_hash=protocol.hash,
        budget=str(budget),
        primary_metric=primary,
        runs=len(runs),
        registry_version=registry_version,
        evaluated_at=datetime.now(UTC),
        means={name: float(v.mean()) for name, v in values.items()},
        stds={name: float(v.std(ddof=1)) if len(v) > 1 else nan for name, v in values.items()},
        score_ci_low=interval[0],
        score_ci_high=interval[1],
    )


def _reusable_labels(task: Task) -> dict[str, int] | None:
    """Labels of a single label task whose samples are files, which allow reusing features.

    Args:
        task: Task bound to its data.

    Returns:
        The class index of every sample, or `None` when the task does not qualify.
    """
    if task.spec.kind not in SINGLE_LABEL or not isinstance(task.loader, FileLoader):
        return None
    return task.loader.labels()


def evaluate_task(
    encoder: Encoder,
    task: Task,
    protocols: Sequence[Protocol],
    config: EvaluationConfig,
    registry_version: int,
    *,
    store: FeatureStore | None = None,
    done: frozenset[tuple[str, ...]] = frozenset(),
) -> list[LeaderboardRow]:
    """Every protocol and budget of one task, with features extracted once per partition.

    Features are taken from `store` when present and added to it otherwise, under the canonical
    path of their image. Single label tasks read stored global features without decoding their
    images. Protocols that do not apply to the task kind, dense protocols for encoders without
    dense features and budgets that cover the whole training partition are skipped, and direct
    protocols that are not budgeted run on the full budget only. Reduced budgets and stochastic
    protocols are repeated over `config.seeds` seeds. Per image records are written once every
    protocol ran, unless `config.per_image_dir` is `None`.

    Args:
        encoder: Encoder to evaluate.
        task: Task bound to its data.
        protocols: Protocols to run on the task.
        config: Execution settings.
        registry_version: Version of the registry recorded on every row.
        store: Features already encoded, read and extended in place when given.
        done: Settings to skip, each given as task, task hash, protocol, protocol hash and budget.

    Returns:
        One leaderboard row per protocol and budget evaluated.
    """
    spec = task.spec
    device = torch.device(config.device)
    n_classes = len(spec.classes)
    cache: dict[tuple[bool, Split], Features] = {}
    labels_of = _reusable_labels(task) if store is not None else None

    def encode(samples: LazySamples, dense: bool, split: Split) -> Features:
        """Features of the given samples of one partition, encoded unless stored."""
        return extract(
            encoder,
            samples,
            dense=dense,
            batch_size=config.batch_size,
            device=device,
            description=f"{spec.id} {split}",
            store=store,
        )

    def stored(split: Split) -> Features:
        """Global features of a partition read from the store, encoding only the missing images."""
        assert store is not None and labels_of is not None
        assert isinstance(task.loader, FileLoader)
        loader = task.loader
        ids = task.split.ids(split)
        paths = [loader.path(loader.members(i)[0]) for i in ids]
        missing = [i for i, path in zip(ids, paths, strict=True) if store.get(path, False) is None]
        if missing:
            encode(LazySamples(loader, missing), False, split)
        values = [store.get(path, False) for path in paths]
        return Features(
            values=torch.stack([value for value in values if value is not None]),
            targets=tuple(labels_of[i] for i in ids),
            ids=tuple(ids),
            paths=tuple(paths),
        )

    def features(split: Split, dense: bool) -> Features:
        """Features of a partition, computed once per partition and kind."""
        if (dense, split) not in cache:
            reuse = not dense and labels_of is not None
            cache[dense, split] = (
                stored(split) if reuse else encode(task.samples(split), dense, split)
            )
        return cache[dense, split]

    rows: list[LeaderboardRow] = []
    records: list[pd.DataFrame] = []
    for protocol in protocols:
        if spec.kind not in protocol.kinds:
            continue
        if protocol.dense and not encoder.has_dense:
            log.warning(
                "encoder lacks dense features, define postprocess_dense to evaluate it",
                task=spec.id,
                protocol=protocol.name,
            )
            continue
        fixed = isinstance(protocol, DirectProtocol) and not protocol.budgeted
        budgets = [
            budget
            for budget in ((FULL,) if fixed else spec.budgets)
            if (spec.id, spec.hash, protocol.name, protocol.hash, str(budget)) not in done
        ]
        if not budgets:
            log.info("already evaluated", task=spec.id, protocol=protocol.name)
            continue
        train, val, test = (features(s, protocol.dense) for s in Split)
        labels = train.labels() if spec.kind in SINGLE_LABEL else None
        n_train = len(train.targets)
        for budget in budgets:
            if not budget.is_full and budget.size(n_train, n_classes) >= n_train:
                log.info("budget covers training set", task=spec.id, budget=str(budget))
                continue
            repeats = config.seeds if protocol.stochastic or not budget.is_full else 1
            runs = []
            statistics: list[Statistic] = []
            for seed in range(repeats):
                subset = train.subset(budget_indices(budget, n_train, labels, seed))
                held_out = budget_validation(val, budget, spec, seed)
                if isinstance(protocol, DirectProtocol):
                    metrics, run_records = protocol.measure(
                        subset, held_out, test, spec, seed=seed, device=device
                    )
                    statistic = protocol.statistic(run_records)
                    run_records.insert(0, "sample_id", list(test.ids) or range(len(run_records)))
                    run_records.insert(1, "path", list(test.paths) or "")
                else:
                    predictions = protocol.predict(
                        subset, held_out, test, spec, seed=seed, device=device
                    )
                    metrics, run_records = score_run(spec, predictions, test)
                    statistic = primary_statistic(spec, predictions, run_records, test)
                statistics.append(statistic)
                runs.append(metrics)
                run_records.insert(0, "protocol", protocol.name)
                run_records.insert(1, "budget", str(budget))
                run_records.insert(2, "seed", seed)
                records.append(run_records)
            interval = (
                bootstrap_interval(
                    averaged(statistics), len(test.targets), resamples=config.bootstrap
                )
                if config.bootstrap
                else (nan, nan)
            )
            summary = _summarize(
                runs,
                encoder=encoder,
                pretraining=describe(declaration(encoder, config)),
                grid=feature_grid(test) if protocol.dense else "",
                spec=spec,
                protocol=protocol,
                budget=budget,
                registry_version=registry_version,
                interval=interval,
            )
            log.info(
                "evaluated",
                task=spec.id,
                protocol=protocol.name,
                budget=str(budget),
                metric=summary.primary_metric,
                mean=round(summary.score, 4),
                std=round(summary.stds[summary.primary_metric], 4),
            )
            rows.append(summary)
    if not rows:
        log.warning("no protocol applies to this task and encoder", task=spec.id)
    if records and config.per_image_dir is not None:
        write_records(config.per_image_dir, spec, encoder.name, records)
    return rows


def _prepare(spec: TaskSpec, registry: Registry, config: EvaluationConfig) -> Task | None:
    """Task bound to its data, or `None` with a warning when its split or data are missing.

    Args:
        spec: Specification of the task.
        registry: Registry the task belongs to.
        config: Settings giving the data root and whether to stream and cache datasets.

    Returns:
        The task ready to load its samples, or `None` when it cannot be evaluated.
    """
    if not spec.splits_sha256:
        log.warning("split file not built", task=spec.id)
        return None
    task = build_task(
        spec, config.data_root, registry=registry, stream=config.stream, cache=config.cache
    )
    if not task.loader.available():
        log.warning("dataset not available", task=spec.id, root=str(config.data_root))
        return None
    return task


def shrink(task: Task, limit: int) -> Task:
    """Task restricted to at most `limit` evenly spaced samples of each partition.

    Args:
        task: Task to shrink.
        limit: Maximum number of samples per partition.

    Returns:
        A copy of the task with a reduced split.
    """

    def spread(ids: tuple[str, ...]) -> tuple[str, ...]:
        """Identifiers taken at a regular step from the first, truncated to `limit` entries."""
        step = max(1, len(ids) // limit)
        return ids[::step][:limit]

    split = task.split.model_copy(
        update={name: spread(getattr(task.split, name)) for name in ("train", "val", "test")}
    )
    return task.model_copy(update={"split": split})


def smoke_paths(config: EvaluationConfig) -> EvaluationConfig:
    """Configuration writing smoke test results apart from real ones.

    Args:
        config: Settings of the smoke test.

    Returns:
        A copy whose leaderboard lies in a `smoke` folder beside the original one and whose per
        image results go to a `smoke` folder beside the original folder.
    """
    leaderboard = config.leaderboard.parent / "smoke" / config.leaderboard.name
    per_image = None if config.per_image_dir is None else config.per_image_dir.parent / "smoke"
    return config.model_copy(update={"leaderboard": leaderboard, "per_image_dir": per_image})


def feature_directory(encoder: Encoder, config: EvaluationConfig) -> Path | None:
    """Directory of the features of an encoder cached under the data root, if enabled.

    Computing the fingerprint reads every weight of the encoder.

    Args:
        encoder: Encoder whose name and fingerprint name the directory.
        config: Settings giving the data root and whether the feature cache is enabled.

    Returns:
        The folder `.features/<name>-<fingerprint>` under the data root, with the first sixteen
        hexadecimal digits of the fingerprint, or `None` when the cache is disabled.
    """
    if not config.feature_cache:
        return None
    directory = config.data_root / ".features" / f"{encoder.card.name}-{encoder.fingerprint()[:16]}"
    log.info("feature cache", directory=str(directory))
    return directory


def completed(leaderboard: Leaderboard, model: str) -> frozenset[tuple[str, ...]]:
    """Settings already evaluated for a model, keyed by their task and protocol definitions.

    Args:
        leaderboard: Leaderboard to read.
        model: Name of the encoder.

    Returns:
        The task, task hash, protocol, protocol hash and budget of every row of the model.
    """
    return frozenset(
        (row.task, row.task_hash, row.protocol, row.protocol_hash, row.budget)
        for row in leaderboard.read()
        if row.model == model
    )


def evaluate(
    encoder: Encoder,
    config: EvaluationConfig | None = None,
    *,
    registry: Registry | None = None,
    tasks: Sequence[Task] | None = None,
    **overrides: Any,
) -> list[LeaderboardRow]:
    """Evaluate an encoder and record the results on the leaderboard.

    Settings come from `config` updated by keyword `overrides`, for instance
    `evaluate(encoder, exclude_datasets=("etis",))`. The pretraining datasets declared by the
    encoder are added to those of the configuration. Explicit `tasks` bypass the registry
    selection. Tasks whose split is not built or whose data are neither local nor streamable are
    skipped with a warning. The rows of each task are written to the leaderboard as soon as it
    finishes, and pending features are flushed to the feature cache after every task.

    Args:
        encoder: Encoder to evaluate.
        config: Base settings, the defaults of `EvaluationConfig` when `None`.
        registry: Registry of datasets and tasks, read with `load_registry` when `None`.
        tasks: Tasks bound to their data to evaluate instead of the registry selection.
        **overrides: Fields of `EvaluationConfig` replacing those of `config`.

    Returns:
        The leaderboard rows of every task evaluated.

    Raises:
        pydantic.ValidationError: If the settings or the metadata of the encoder are invalid.
        ValueError: If a protocol, a task pattern or an excluded dataset key is unknown.
        EvaluationError: If some tasks failed, once every other task has been recorded.
    """
    card = encoder.card
    base = (config or EvaluationConfig()).model_dump()
    config = EvaluationConfig.model_validate(base | overrides)
    config = config.model_copy(update={"pretrained_on": declaration(encoder, config) or ()})
    protocols = resolve_protocols(config.protocols)
    registry = registry or load_registry()
    if config.limit is not None:
        config = smoke_paths(config)
        log.warning("smoke test on a few samples per partition", limit=config.limit)
    leaderboard = Leaderboard(config.leaderboard)
    done = completed(leaderboard, card.name) if config.resume else frozenset()
    pending: Sequence[Task | TaskSpec] = (
        tasks if tasks is not None else select_tasks(registry, config)
    )
    log.info("evaluation started", model=card.name, tasks=len(pending))
    rows: list[LeaderboardRow] = []
    failed: list[str] = []
    store = FeatureStore(feature_directory(encoder, config))
    for item in pending:
        name = item.spec.id if isinstance(item, Task) else item.id
        started = time.monotonic()
        try:
            task = item if isinstance(item, Task) else _prepare(item, registry, config)
            if task is None:
                continue
            if config.limit is not None:
                task = shrink(task, config.limit)
            task_rows = evaluate_task(
                encoder, task, protocols, config, registry.version, store=store, done=done
            )
        except Exception:
            log.exception("task failed", task=name)
            failed.append(name)
            continue
        finally:
            store.flush()
        leaderboard.upsert(task_rows)
        rows.extend(task_rows)
        seconds = round(time.monotonic() - started)
        log.info("task finished", task=name, rows=len(task_rows), seconds=seconds)
    log.info("evaluation finished", model=card.name, rows=len(rows), failed=len(failed))
    if failed:
        raise EvaluationError(f"tasks failed after the others were recorded: {failed}")
    return rows
