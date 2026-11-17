# Pretraining

The benchmark scores frozen encoders, and an encoder pretrained on images of a test patient scores
too well. This page lists the corpora that the team's own pretraining may read and the recipes
that train on them. Each corpus is large, mostly unlabelled, and shares no patient with any test
image of the benchmark. Their cards live in
[`pretraining/data/corpora.json`](../src/find_those_polyps/pretraining/data/corpora.json), where
every archive is pinned by size and checksum and every choice carries its evidence.

Images are never redistributed. A corpus is read from its official host, under its own licence,
and results obtained with it cite its authors.

## Corpora

| Key | Corpus | Units read | Frames | To download | Licence |
|---|---|---|---|---|---|
| `real-colon-train` | REAL-Colon training and validation videos | 48 of 60 videos | 2,166,076, or 79,139 at one per second | 725 GB | CC BY 4.0 |

### REAL-Colon

Sixty full colonoscopies, one per patient, from four clinical studies of fifteen videos each,
three centres in the United States, one in Italy, the University Hospital of St. Poelten in Austria
and Sano Hospital in Japan, recorded with Olympus and Fujifilm endoscopes
([Biffi et al. 2024](https://doi.org/10.1038/s41597-024-03359-0)). No benchmark dataset comes from
these centres, and every video is one patient, so a video is the unit at which nothing leaks.

The twelve official test videos, the last three of every study, are held out and never read. The
split is the one of the authors, in
[`export_coco_format.py`](https://github.com/cosmoimd/real-colon-dataset/blob/7735f377c3f6f912188d9a50d2e79d470538c3bd/export_coco_format.py).
REAL-Colon is the cleanest candidate for a held out video and detection test, and keeping its test
videos unseen lets it join the benchmark later without disqualifying the encoders trained here.

Consecutive frames of a video are nearly identical, so frames are subsampled by their index, to
one per second of recording by default. The annotation archives of the release are not read, since
pretraining uses no labels.

## Pools that were left out

| Pool | Why it is not a corpus |
|---|---|
| HyperKvasir unlabelled images and videos | Same hospital and period as Kvasir-SEG, Kvasir v2, Nerthus and the labelled HyperKvasir images, and the source of images of PolypDB, GastroVision and Kvasir-Instrument. No patient identifiers are released. |
| Kvasir-Capsule unlabelled frames | Frames of the same videos as the Kvasir-Capsule tasks, which WCEBleedGen also draws from. Usable only once every test frame is traced back to its video. |
| Capsule Vision 2024 | Contains Kvasir-Capsule. |
| PolypDiag | Built from the HyperKvasir videos and LDPolypVideo. |
| BKAI-IGH NeoPolyp | Reported to overlap the Vietnamese centre of PolypDB. |
| GastroNet-5M | The images are not released, only weights. |
| LDPolypVideo | States no licence and no source hospital. |
| SUN | Obtained by request under terms that forbid redistribution. |
| Galar | Clean and CC BY 4.0, but published as 7z archives that cannot be streamed. Tracked in issue 44. |
| Training partitions of benchmark datasets | CAD-SEL, GastroHUN and LIMUC separate their patients, but they are small, and an encoder trained on them would lose their tasks until overlaps are declared per partition. |

The evidence for the overlaps is in the leakage warnings of [datasets.md](datasets.md).

## Reading a corpus

A unit is read from the first source that can serve it, the local cache, a whole archive on disk,
or the official host.

```python
from torch.utils.data import DataLoader

from find_those_polyps.pretraining import CorpusDataset, CorpusReader, corpus

reader = CorpusReader(corpus("real-colon-train"), fps=1.0)
dataset = CorpusDataset([reader], transform=augment, shuffle_buffer=256)
loader = DataLoader(dataset, batch_size=64, num_workers=8)
for epoch in range(epochs):
    dataset.set_epoch(epoch)
    for batch in loader:
        ...
```

| Option | Default | Effect |
|---|---|---|
| `stream` | on | A unit that is not local is decompressed while it downloads, so training starts at once. |
| `download` | off | The archive of a unit is downloaded whole and verified before it is read, instead of streamed. |
| `keep_archives` | off | Archives downloaded whole stay under `.archives/` in the data root. |
| `cache` | on | Kept frames are written under `corpora/` in the data root, so later epochs read no host. |
| `fps` | 1 | Frames kept per second of recording, `None` for every frame. |

While some unit is not cached, an epoch visits every unit once in a random order, the workers of
the data loader each read different units, and frames leave through a shuffle buffer. Once every
unit is cached, an epoch draws the frames of all units in one random order. A connection that
drops is resumed where it stopped, and a streamed unit whose host still cannot be read after every
retry is skipped for the epoch and logged.

Every archive read from a host is hashed as it passes and compared with its pin once read whole. A
mismatch discards the frames cached from it and stops the run. A unit counts as cached only after
that check, so an interrupted pass is read again. A unit cached at a finer rate serves coarser
requests.

The first pass over `real-colon-train` downloads 725 GB whatever the rate, since the archives are
compressed streams. At one frame per second the cache holds about 79 thousand frames, roughly 20 GB.

## Fetching ahead of training

```bash
uv run polyps-corpora list
uv run polyps-corpora fetch real-colon-train --fps 1 --workers 4
uv run polyps-corpora fetch real-colon-train --whole --keep-archives --units 001-001 001-002
uv run polyps-corpora verify real-colon-train
```

`fetch` caches the kept frames of every unit, or of the named ones, streaming by default and
downloading whole archives with `--whole`. `verify` checks that the host still serves every pinned
archive at its pinned size, without downloading it.

## Recipes

A recipe trains one backbone with one self supervised method on the corpora above. The backbone is
a ViT-B/14 with the architecture of the DINOv2 reference, so a trained backbone is evaluated
exactly as that encoder is. It continues the DINOv2 weights by default, and `--init random` starts
it from random weights. Every method carries a schedule for either start.

| Method | Objective | Network kept | Continued | From random weights |
|---|---|---|---|---|
| `dino` | A student predicts the output of a momentum teacher on another crop, two large crops and eight small ones ([Caron et al. 2021](https://arxiv.org/abs/2104.14294)) | teacher | 20 epochs, rate 5e-5, decay 0.04 | 100 epochs, rate 5e-4, decay 0.04 |
| `mae` | A light decoder reconstructs the pixels of the three quarters of the patches hidden from the backbone ([He et al. 2022](https://arxiv.org/abs/2111.06377)) | backbone | 50 epochs, rate 1.5e-5, decay 0.05 | 400 epochs, rate 1.5e-4, decay 0.05 |
| `moco` | The backbone picks the momentum embedding of the other view of its frame among those of the batch ([Chen et al. 2021](https://arxiv.org/abs/2104.02057)) | backbone | 20 epochs, rate 1.5e-5, decay 0.1 | 100 epochs, rate 1.5e-4, decay 0.1 |

Rates are peak learning rates for 256 frames and scale with the batch, 256 by default. All three
use AdamW with a linear warm up and a cosine decay, and mixed precision on accelerators. The
schedules from random weights follow the papers with fewer epochs, and the continued ones lower the
rate tenfold. No run has tuned them yet, so they are starting points.

```bash
uv run polyps-pretrain dino
uv run polyps-pretrain mae --init random --epochs 100 --batch-size 128
uv run polyps-pretrain moco --fps 5 --download --workers 12
```

The corpora are read while training with the options above, `--no-stream`, `--download`,
`--keep-archives`, `--no-cache` and `--fps`. A run writes a checkpoint after every epoch and the
trained backbone at the end, under `runs/pretraining/NAME`, which git ignores. Running the same
command again resumes from the last checkpoint.

### Evaluating a trained backbone

The exported file records the name of the encoder and every corpus its weights have seen, the
corpus of the starting weights included. A team encoder is a factory around it, evaluated like any
other, at 224 pixels and at 448 on the dense tasks.

```python
from find_those_polyps.pretraining.encoder import PretrainedViT


def build() -> PretrainedViT:
    return PretrainedViT("runs/pretraining/vitb14-dino/vitb14-dino.safetensors")


def build_448() -> PretrainedViT:
    return PretrainedViT("runs/pretraining/vitb14-dino/vitb14-dino.safetensors", image_size=448)
```

### Adding a method

A method subclasses `Method` in `find_those_polyps.pretraining.methods`. It sets its `name` and
its two schedules, `scratch` and `continued`, returns the augmentation of one frame from `views`
and the loss of a batch of views from `forward`, and joins `METHODS`. `before_step` and
`after_step` are where a method adjusts gradients or updates a momentum copy, and `exported`
names the network to keep when it is not the backbone.

## What a pretrained encoder declares

An encoder trained on a corpus lists the key of the corpus in `pretrained_on`, for instance
`("lvd-142m", "real-colon-train")` when it continues DINOv2 weights. The recipes record this in the
exported file, and the benchmark shows the declaration on every leaderboard row. Before publishing,
the cached frames are searched for copies of test images.

```bash
uv run polyps-data check data/corpora/real-colon-train --output matches.csv
```

## Adding a corpus

A pool becomes a corpus only when its patients can be told apart from those of every benchmark
test image, by a different source or by identifiers that allow holding units out.

1. Add its card to `corpora.json`, with the licence, the citation, the provenance and one archive
   per unit, a video or a patient, each pinned by the size and checksum its host publishes.
2. Hold out every unit that shares a patient with a benchmark dataset or that a later benchmark
   task should test on, with the reason and its evidence.
3. Record the overlaps with other datasets, and list the corpus in the table above.
4. Run `polyps-corpora verify KEY`, fetch a unit, and search its frames with `polyps-data check`.

Archives must be tarballs, plain or compressed, since they are read as streams.
