# find-those-polyps

A benchmark of unsupervised and self-supervised representation learning methods for colonoscopy
and gastrointestinal endoscopy imaging. An encoder is treated as a black box. The benchmark
extracts frozen features, measures them directly in the zero-shot setting, fits standard few-shot
and full-data probes on the training partition of each task, selects hyperparameters on the
validation partition and reports every number on the canonical test partition, so that results
remain comparable with the literature.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). On Windows and Linux, PyTorch is
installed from the CUDA 13.0 wheel index.

```bash
uv sync
```

This installs the command `find-those-polyps` together with the shortcuts `polyps-evaluate`,
`polyps-data`, `polyps-split` and `polyps-registry`.

## Quick start

```bash
export FIND_THOSE_POLYPS_DATA=/path/to/data   # defaults to ./data
uv run polyps-data tasks
uv run polyps-evaluate find_those_polyps.baselines:ImageNetResNet50 --tasks "cls/*" --dry-run
uv run polyps-evaluate find_those_polyps.baselines:ImageNetResNet50 --tasks "cls/*"
```

`--dry-run` lists the selected tasks and how their data will be read, and `--limit 16` runs a smoke
test on sixteen images per partition in a few minutes, written to `results/smoke/` so that it never
mixes with real results. Two reference encoders are provided, an untrained ResNet-50 as the lower
bound and a supervised ImageNet ResNet-50 as the customary transfer learning reference.

No dataset has to be downloaded beforehand. Images of open datasets are streamed from their
official hosts while the encoder runs, and the fetched files are kept under the data root for later
runs.

## Tasks and data

The benchmark currently defines 35 tasks on 14 openly accessible labelled datasets.

- **Classification:** HyperKvasir findings, landmarks, bowel preparation and oesophagitis grade, GastroVision in its 22 official classes, Kvasir v2, GastroHUN stomach stations, Kvasir-Capsule, CAD-SEL subepithelial lesions across centres, WCEBleedGen bleeding and LIMUC remission.
- **Ordinal grading:** LIMUC Mayo score, HyperKvasir ulcerative colitis grade, Nerthus bowel preparation and the AI-KODA capsule cleanliness scores.
- **Segmentation:** Kvasir-SEG, PolypDB in five imaging modalities, Kvasir-Instrument and WCEBleedGen, and the EAD2020 imaging artefacts, whose five classes overlap.
- **Detection:** the eight binary segmentation tasks on their own partitions, with one box per connected mask region, the way the boxes published with Kvasir-SEG were derived, except WCEBleedGen, whose published boxes are used together with its frames without bleeding. In addition seven kinds of Kvasir-Capsule findings and eight kinds of EAD2020 artefacts with their published boxes.

Every task, with its partition sizes and the source of its test partition, is listed in
[docs/datasets.md](docs/datasets.md). The same page documents every dataset with its licence,
provenance and source family, and the images it shares with other datasets and with common
pretraining corpora. The page is generated from the public, versioned registry
[`registry.json`](src/find_those_polyps/registry/data/registry.json).

Images are never redistributed. Streaming reads the central directory of each official zip archive
with HTTP range requests and then fetches the files of the samples in use, in windows of
neighbouring files, so that the encoder processes one window while the next downloads. Every file
is verified against the CRC32 of its archive, and every archive is pinned by its size and by a hash
of its member list. Nested tarballs are unpacked as they download.

```bash
uv run polyps-data datasets           # every dataset with its local state
uv run polyps-data fetch kvasir-seg   # optional complete local copy
uv run polyps-data verify             # check that every host still serves the pinned files
```

`polyps-data verify` compares the size and member index of every archive, and the checksum of every
plain file, with the pins on the dataset cards. It prints ok, changed or unreachable for each file
and exits with one unless all files are unchanged. A weekly workflow runs it, so that a host moving
or altering a file is noticed early.

## Evaluating an encoder

An encoder computes `postprocess(forward(preprocess(image)))`. The default preprocessing resizes to
`image_size` and scales to `[0, 1]`, and every stage can be overridden. A name and a one sentence
description are mandatory, since they identify the model on the leaderboard.

```python
import torch
from find_those_polyps import TorchEncoder, evaluate, normalize, patch_grid, resize

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
encoder = TorchEncoder(
    model.forward_features,
    name="dinov2-vits14",
    description="DINOv2 ViT-S/14 pretrained on LVD-142M.",
    preprocess=lambda image: normalize(resize(image, 224)),
    postprocess=lambda out: out["x_norm_clstoken"],
    postprocess_dense=lambda out: patch_grid(out["x_norm_patchtokens"]),
    pretrained_on=("lvd-142m",),
)
evaluate(encoder, exclude_datasets=("hyperkvasir-labeled",))
```

The same is available from the command line for any `module:factory` or `path/to/file.py:factory`,
with options mirroring `EvaluationConfig`. Modules in the working directory are found.

```bash
uv run polyps-evaluate my_encoder.py:build --pretrained-on hyperkvasir-unlabeled
```

Several encoders can be evaluated in one run, by passing a list to `evaluate` or several
references to the command. Tasks then run one at a time, grouped by dataset, and the images of each
partition are read once and handed to every encoder, so a streamed dataset is downloaded once
whatever the number of encoders. Each encoder keeps its own pretraining declaration, task
selection and feature cache, and all of them must fit on the device together.

```bash
uv run polyps-evaluate find_those_polyps.baselines:RandomResNet50 find_those_polyps.baselines:ImageNetResNet50
```

An encoder declares in `pretrained_on` the keys of the datasets and corpora it was pretrained on,
an empty tuple when it was not pretrained, and the option of the same name adds to it. Every task
whose dataset is documented to overlap a declared one is excluded, and the leaderboard records the
declaration of each model. `--no-stream` restricts the evaluation to local copies, `--no-cache`
keeps streamed files in memory only and `--bootstrap 0` skips the confidence intervals. `--resume`
continues an interrupted evaluation, skipping every setting already on the leaderboard for the same
model under the same task and protocol definitions. Tasks sharing images, such as the HyperKvasir
tasks, encode them once per run, and the command ends with a summary of the full budget scores. The
notebook [notebooks/evaluate.ipynb](notebooks/evaluate.ipynb) walks through a full evaluation.

`--workers N` sets the host threads that preprocess images ahead of the device and the processes
that compute the per image segmentation metrics, by default the number of processors up to eight,
and `--prefetch` the batches prepared ahead, two by default. Neither changes any result.
`--preprocess-on-device` resizes images on the accelerator, which helps with large images on fast
devices but changes the features slightly, so they are cached apart from those resized on the host.

## Keeping test images out of pretraining

An encoder is a black box, and nothing in its weights proves which images it was trained on. All
datasets of the benchmark are public, and pretraining on test images, even without their labels,
inflates every score. The benchmark guarantees that its probes never see a test image, since they
are fitted on the training partition, selected on the validation partition and scored once on the
test partition, with copies, near duplicates and images of the same patient kept out of training.
Beyond that, three safeguards make contamination visible.

- Every leaderboard row states what the model declared to be pretrained on, and the summary marks
  models that declared nothing.
- The fingerprints of every test image, the CRC32 and size of its file and a perceptual difference
  hash, are committed in
  [`registry/data/manifests`](src/find_those_polyps/registry/data/manifests), so that pretraining
  pipelines can filter them out. No image is stored.
- `polyps-data check` searches any folder of images for identical copies and near duplicates of
  test images, which survive resizing and recompression, and exits with status one when it finds
  any. Crops, flips and rotations escape the check.

```bash
uv run polyps-data check /path/to/pretraining/corpus --output matches.csv
```

## Protocols and metrics

| Protocol | Setting | Tasks | Reference |
|---|---|---|---|
| `kmeans` | zero shot, no labels | classification, ordinal | k-means with one cluster per class, ranked by AMI |
| `retrieval` | zero shot, labelled gallery | classification, ordinal | cosine retrieval, Musgrave et al. 2020 |
| `knn` | few shot and full | classification, ordinal, multilabel | weighted k nearest neighbours, Wu et al. 2018 |
| `linear` | few shot and full | classification, ordinal, multilabel | logistic regression with validated L2 strength |
| `linear_seg` | few shot and full | segmentation, multilabel segmentation | batch normalization and pointwise convolution, Oquab et al. 2024 |
| `linear_det` | few shot and full | detection | the one stage detector FCOS, Tian et al. 2019, reduced to a linear head on the dense features |

Each task defines label budgets, from the full training partition down to one shot per class.
Subsets are class stratified and nested within each budget family for a given seed, so that
`shots:1` is contained in `shots:5` and `frac:0.01` in `frac:0.1`. Every reduced budget and every
stochastic protocol is repeated over several seeds. Hyperparameters and checkpoints are selected on
a validation subset drawn with the same budget rule, so that a few shot result never benefits from
the full validation labels.

Canonical splits are built from the data and committed. A published test partition is used
verbatim, and samples sharing a patient, video or lesion with it are excluded from training and
validation. Otherwise the benchmark holds out a fifth of the groups, chosen so that training keeps
every class and the test covers as many classes as possible. Byte identical images always fall into
one partition, and identical images with conflicting labels are excluded. Near duplicates, such as
resized or recompressed copies and neighbouring video frames, are recognized by their perceptual
hashes and also fall into one partition, which matters most for datasets without patient
identifiers. A test partition is never altered once committed, and
[docs/datasets.md](docs/datasets.md) counts for every task the images lost to training and
validation, by reason.

Every metric reported in the literature for a task kind is computed, with its reference in
`find_those_polyps.metrics.METRIC_INFO`.

- **Classification:** accuracy, balanced accuracy, top three and top five accuracy, macro and weighted F1, macro precision and recall, MCC, Cohen kappa, AUROC, AUPRC, log loss, Brier score and expected calibration error.
- **Binary tasks additionally:** sensitivity, specificity, predictive values and F1.
- **Ordinal tasks additionally:** quadratic and linear weighted kappa, mean absolute error, accuracy within one grade and Spearman correlation.
- **Segmentation:** Dice, IoU, precision, recall, specificity, F2, the normalized surface Dice of Nikolov et al. 2021 at tolerances of 2% and 1% of the image diagonal, and the 95th percentile Hausdorff and average symmetric surface distances divided by the diagonal, plus the PraNet measures MAE, S-measure, E-measure and weighted F-measure for binary masks. Overlapping masks are scored class by class. The per image results also keep both distances in pixels.
- **Detection:** the COCO measures of Lin et al. 2014, box average precision at an overlap of 0.5, the primary score as in the polyp and instrument detection literature, over overlaps from 0.5 to 0.95 and at 0.75, and recall with a hundred detections, checked against pycocotools.
- **Retrieval:** MAP@R, R-precision, mean average precision and precision at one and five. Retrieval scores depend on the class balance of the gallery, so they compare models within one budget, whose galleries are identical across models, but not budgets with each other.
- **Clustering:** AMI, NMI, ARI and Hungarian matched accuracy. Resampling duplicates images, which always agree with their own cluster, so the bootstrap inflates every clustering score, and the interval of AMI is the bootstrap spread centred on the estimate.

## Results

Every evaluation writes four outputs.

- [`results/leaderboard.csv`](results/leaderboard.csv) holds one row per model, task, protocol and budget, with one column per metric mean and one per standard deviation over seeds, which is empty for single runs. The score also carries a 95% percentile bootstrap interval over test images, recomputing the mean over seeds on every resample, which expresses the uncertainty due to the finite test set. `input_size` is the side in pixels of the images given to the encoder, zero for rows recorded before it was kept, and `feature_grid` the height and width of the dense feature maps under the dense protocols, for instance `14x14`, empty under global ones. Scores depend on resolution, so the reference encoders carry their input size in their name when it is not 224, for instance `imagenet-resnet50-448`.
- `results/leaderboard.xlsx` gives every configuration its own ranked sheet, named like `limuc-mayo.linear.5shot`, with an index sheet linking to all of them and a comparison sheet holding the primary score of every model in every configuration.
- `results/leaderboard.md` summarizes the full budget scores with their intervals, lists the pretraining data and input size of every model, and is readable on GitHub.
- `results/per_image/` holds one CSV per task and model with the prediction for every test image and its canonical path, for error analysis.

`uv run find-those-polyps results` rewrites the workbook and the summary from the CSV, for instance
after merging leaderboards. Rankings are computed by the primary metric among results obtained
under identical task and protocol definitions, which are tracked by content hashes. Results obtained
before a definition changed are kept in the CSV, marked as not `current`, and left out of the
workbook. Overlapping intervals mean that two models are not distinguishable on that test set.

## Limitations

- The duplicate check recognizes identical files, resized and recompressed copies and neighbouring
  video frames that look nearly alike. Crops, flips, rotations and colour changes escape it, both
  when splits are built and when `polyps-data check` searches a corpus, so a training partition may
  still hold such a view of a test image.
- Datasets without patient or video identifiers, Kvasir-SEG among them, are grouped by their images
  alone, so two images of one patient that do not look alike may fall on both sides of a split.
- Eleven of the 35 test partitions are drawn by the benchmark, because no split was published for
  them. Scores on those tasks compare encoders within the benchmark, but not with papers that drew
  their own splits. The test source of every task is listed in
  [docs/datasets.md](docs/datasets.md).
- The pretraining declaration is made by whoever submits an encoder and cannot be verified from its
  weights.
- Probes on frozen features measure how accessible the information is, not what fine tuning would
  reach. The detection probe is a linear head on the dense features, so its average precision lies
  far below that of trained detectors and serves to rank encoders.
- Some test partitions are small. The PolypDB BLI, FICE and LCI tasks hold six or seven test images
  each, and their intervals are wide accordingly.
- Several datasets may be used only for research and education or only non commercially, among them
  EAD2020, Kvasir-SEG, Kvasir v2, Kvasir-Instrument and PolypDB, and HyperKvasir and Kvasir-Capsule
  need permission for commercial use. The benchmark distributes no image, and whoever streams a
  dataset is bound by its licence, given in [docs/datasets.md](docs/datasets.md).
- Streaming depends on the official hosts. `polyps-data verify` and its weekly workflow notice a
  moved or altered file, but cannot restore it.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and for adding datasets, which
usually requires no code.

## Citation

If you use this work, please cite it using the metadata in [CITATION.cff](CITATION.cff), together
with the datasets you evaluate on.

## License

Released under the [MIT License](LICENSE).
