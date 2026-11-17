"""Training loop of the pretraining recipes.

`train` optimizes a method on the frames of curated corpora with AdamW, a linear warm up and a
cosine decay of the learning rate, mixed precision on accelerators, and a checkpoint after every
epoch, from which an interrupted run resumes. `export` then writes the trained backbone with the
corpora it has seen, which `find_those_polyps.pretraining.encoder` reads back as a benchmark
encoder. Weights are written beside the checkpoints and never enter the repository.
"""

import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from find_those_polyps.logging import get_logger
from find_those_polyps.pretraining.dataset import CorpusDataset
from find_those_polyps.pretraining.methods.base import Method, Schedule

log = get_logger(__name__)

CHECKPOINT = "checkpoint.pt"
"""File of the latest checkpoint in the output directory."""

UNDECAYED = ("pos_embed", "cls_token", "reg_token", "mask_token")
"""Parameters that weight decay skips besides biases and normalization weights."""


def parameter_groups(method: Method, weight_decay: float) -> list[dict[str, Any]]:
    """Trainable parameters of a method, split by whether weight decay applies.

    Args:
        method: Method being trained.
        weight_decay: Weight decay of the matrices.

    Returns:
        Two groups for an optimizer, the weights with decay and the biases, normalization
        weights, layer scales and tokens without.
    """
    decayed: list[torch.nn.Parameter] = []
    plain: list[torch.nn.Parameter] = []
    for name, parameter in method.named_parameters():
        if not parameter.requires_grad:
            continue
        skip = parameter.ndim <= 1 or any(token in name for token in UNDECAYED)
        (plain if skip else decayed).append(parameter)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def learning_rate(schedule: Schedule, step: int, steps_per_epoch: int) -> float:
    """Learning rate of a step, a linear warm up followed by a cosine decay.

    Args:
        schedule: Schedule of the recipe.
        step: Index of the step, from zero.
        steps_per_epoch: Number of steps of one epoch.

    Returns:
        The learning rate of the step.
    """
    total = schedule.epochs * steps_per_epoch
    warmup = schedule.warmup_epochs * steps_per_epoch
    if step < warmup:
        return schedule.lr * (step + 1) / warmup
    share = (step - warmup) / max(1.0, total - warmup)
    return schedule.min_lr + (schedule.lr - schedule.min_lr) * (1 + math.cos(math.pi * share)) / 2


def _move(batch: Any, device: torch.device) -> Any:
    """Move a tensor or a list of tensors to a device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return [view.to(device, non_blocking=True) for view in batch]


def _save(path: Path, state: dict[str, Any]) -> None:
    """Write a checkpoint through a `.part` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    torch.save(state, partial)
    partial.replace(path)


def train(
    method: Method,
    dataset: CorpusDataset,
    schedule: Schedule,
    output: Path,
    *,
    device: str | torch.device = "cpu",
    workers: int = 0,
    amp: bool = True,
    steps_per_epoch: int | None = None,
    log_every: int = 50,
) -> Path:
    """Train a method on the frames of a dataset and checkpoint it after every epoch.

    The transform of the dataset is set to the views of the method. A checkpoint found in the
    output directory is resumed from the epoch after the one it closed. A streamed unit that the
    dataset skips shortens its epoch, and the learning rate then follows the steps actually
    taken.

    Args:
        method: Method to train, moved to `device`.
        dataset: Frames to train on, whose epoch is set before every pass.
        schedule: Schedule of the recipe.
        output: Directory of the checkpoint.
        device: Torch device of the training. Defaults to the processor.
        workers: Worker processes of the data loader. Defaults to zero.
        amp: Whether to use mixed precision when the device is an accelerator. Defaults to
            `True`.
        steps_per_epoch: Steps of one epoch, the frames of the dataset divided by the batch size
            when `None`, which also caps longer epochs.
        log_every: Number of steps between two log lines. Defaults to 50.

    Returns:
        The path of the checkpoint written after the last epoch.

    Raises:
        ValueError: If the length of an epoch is neither given nor known from the corpus cards,
            or is shorter than one batch, or the checkpoint in `output` is of another method.
        FloatingPointError: If the loss stops being finite.
    """
    target = torch.device(device)
    if steps_per_epoch is None:
        frames = dataset.expected()
        if frames is None:
            raise ValueError("the corpora do not give their length, pass steps_per_epoch")
        steps_per_epoch = frames // schedule.batch_size
    if steps_per_epoch < 1:
        raise ValueError("an epoch holds less than one batch")
    total = schedule.epochs * steps_per_epoch
    dataset.transform = method.views()
    method.to(target)
    optimizer = torch.optim.AdamW(
        parameter_groups(method, schedule.weight_decay), lr=schedule.lr, betas=schedule.betas
    )
    mixed = amp and target.type == "cuda"
    scaler = torch.amp.GradScaler(target.type, enabled=mixed)
    path = output / CHECKPOINT
    epoch, step = 0, 0
    if path.exists():
        state = torch.load(path, map_location=target, weights_only=True)
        if state["name"] != method.name:
            raise ValueError(f"{path} holds a {state['name']} run, not a {method.name} one")
        method.load_state_dict(state["method"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        epoch, step = int(state["epoch"]), int(state["step"])
        log.info("resuming", checkpoint=str(path), epoch=epoch, step=step)
    trainable = [p for p in method.parameters() if p.requires_grad]
    while epoch < schedule.epochs:
        dataset.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_size=schedule.batch_size,
            num_workers=workers,
            drop_last=True,
            pin_memory=target.type == "cuda",
        )
        method.train()
        started, running, taken = time.perf_counter(), 0.0, 0
        for batch in loader:
            if taken == steps_per_epoch:
                break
            rate = learning_rate(schedule, step, steps_per_epoch)
            for group in optimizer.param_groups:
                group["lr"] = rate
            method.progress = step / total
            with torch.autocast(target.type, enabled=mixed):
                loss = method(_move(batch, target))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"the loss is {loss.item()} at step {step}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            method.before_step(step / total)
            if schedule.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(trainable, schedule.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            taken += 1
            method.after_step(min(1.0, step / total))
            running += loss.item()
            if taken % log_every == 0:
                log.info(
                    "training",
                    method=method.name,
                    epoch=epoch,
                    step=step,
                    loss=round(running / log_every, 4),
                    lr=rate,
                )
                running = 0.0
        epoch += 1
        log.info(
            "epoch finished",
            method=method.name,
            epoch=epoch,
            steps=taken,
            seconds=round(time.perf_counter() - started, 1),
        )
        _save(
            path,
            {
                "name": method.name,
                "method": method.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "step": step,
            },
        )
    return path


def export(method: Method, path: Path, *, name: str, pretrained_on: Sequence[str]) -> Path:
    """Write the trained backbone with what a benchmark encoder must state about it.

    Args:
        method: Trained method, whose exported network is saved.
        path: Destination file, in the safetensors format.
        name: Name of the encoder on the leaderboard at 224 pixels.
        pretrained_on: Keys of every corpus the backbone has seen, the corpus of its starting
            weights included.

    Returns:
        `path`.
    """
    network = method.exported()
    state = {key: value.detach().cpu().contiguous() for key, value in network.state_dict().items()}
    metadata = {
        "name": name,
        "method": method.name,
        "pretrained_on": json.dumps(list(pretrained_on)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, path, metadata=metadata)
    log.info("backbone exported", path=str(path), name=name, pretrained_on=list(pretrained_on))
    return path
