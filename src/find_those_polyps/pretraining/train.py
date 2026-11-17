"""Command line of the pretraining recipes.

`polyps-pretrain METHOD` trains the ViT-B/14 backbone with one of the methods, `dino`, `mae` or
`moco`, on curated corpora, and writes a checkpoint after every epoch and the trained backbone at
the end, under `runs/pretraining/NAME`. The backbone continues the DINOv2 weights unless `--init
random` starts it from random weights, and each method brings the schedule of its recipe for
either start, which `--epochs`, `--batch-size` and `--lr` override. The corpora are read from
their hosts while training, streamed by default, with the options of `polyps-corpora`. Running
the same command again resumes from the last checkpoint.
"""

import argparse
import sys
from pathlib import Path

import torch

from find_those_polyps.data.paths import default_data_root
from find_those_polyps.evaluation.runner import default_device
from find_those_polyps.logging import configure_logging, get_logger
from find_those_polyps.pretraining.architecture import START_CORPUS, backbone
from find_those_polyps.pretraining.corpora import corpus
from find_those_polyps.pretraining.dataset import CorpusDataset
from find_those_polyps.pretraining.methods import METHODS
from find_those_polyps.pretraining.reader import CorpusReader
from find_those_polyps.pretraining.trainer import export, train

log = get_logger(__name__)


def _fps(text: str) -> float | None:
    """Frames per second from the command line, `None` for `all`."""
    return None if text == "all" else float(text)


def main(argv: list[str] | None = None) -> int:
    """Run the `polyps-pretrain` command.

    Args:
        argv: Arguments without the program name, read from `sys.argv` when `None`.

    Returns:
        The exit status, zero on success and one when a corpus is unknown.
    """
    parser = argparse.ArgumentParser(prog="polyps-pretrain", description=__doc__)
    add = parser.add_argument
    add("method", choices=sorted(METHODS), help="self supervised method")
    add("--corpora", nargs="+", default=["real-colon-train"], help="corpus keys")
    add("--init", choices=["dinov2", "random"], default="dinov2", help="starting weights")
    add("--name", default=None, help="name of the encoder, vitb14-METHOD by default")
    add("--output", type=Path, default=None, help="default runs/pretraining/NAME")
    add("--epochs", type=int, default=None, help="default of the recipe")
    add("--batch-size", type=int, default=None, help="default of the recipe")
    add("--lr", type=float, default=None, help="peak rate per 256 frames, default of the recipe")
    add("--steps-per-epoch", type=int, default=None, help="default the frames over the batch")
    add("--fps", type=_fps, default=1.0, help="frames kept per second of recording, or all")
    add("--root", type=Path, default=None, help="data root directory")
    add("--no-stream", action="store_true", help="read cached units only")
    add("--download", action="store_true", help="download archives whole instead of streaming")
    add("--keep-archives", action="store_true", help="keep the archives downloaded whole")
    add("--no-cache", action="store_true", help="do not keep the frames read")
    add("--shuffle-buffer", type=int, default=256, help="frames held to shuffle a stream")
    add("--workers", type=int, default=8, help="worker processes of the data loader")
    add("--device", default=None, help="torch device, default cuda when available")
    add("--no-amp", action="store_true", help="train in full precision")
    add("--seed", type=int, default=0)
    add("--log-every", type=int, default=50, help="steps between log lines")
    add("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        cards = [corpus(key) for key in args.corpora]
    except KeyError as error:
        print(error.args[0], file=sys.stderr)
        return 1
    method_type = METHODS[args.method]
    schedule = method_type.recipe(args.init)
    changes = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "base_lr": args.lr,
    }
    schedule = schedule.model_copy(update={k: v for k, v in changes.items() if v is not None})
    name = args.name or f"vitb14-{args.method}" + ("" if args.init == "dinov2" else "-scratch")
    output = args.output or Path("runs") / "pretraining" / name
    root = default_data_root() if args.root is None else args.root
    readers = [
        CorpusReader(
            card,
            root,
            fps=args.fps,
            stream=not args.no_stream,
            cache=not args.no_cache,
            download=args.download,
            keep_archives=args.keep_archives,
        )
        for card in cards
    ]
    dataset = CorpusDataset(readers, shuffle_buffer=args.shuffle_buffer, seed=args.seed)
    torch.manual_seed(args.seed)
    method = method_type(backbone(args.init, drop_path=schedule.drop_path))
    log.info("pretraining", method=args.method, init=args.init, name=name, **schedule.model_dump())
    train(
        method,
        dataset,
        schedule,
        output,
        device=args.device or default_device(),
        workers=args.workers,
        amp=not args.no_amp,
        steps_per_epoch=args.steps_per_epoch,
        log_every=args.log_every,
    )
    seen = ([START_CORPUS] if args.init == "dinov2" else []) + [card.key for card in cards]
    export(method, output / f"{name}.safetensors", name=name, pretrained_on=seen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
