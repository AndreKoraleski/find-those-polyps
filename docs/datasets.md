# Datasets

Generated from registry version 29 (`94251e1ec018`). Do not
edit by hand, run `polyps-registry` instead.

Images are never redistributed. Every dataset is openly downloadable and streamed from
its official host during evaluation, or copied whole with `polyps-data fetch KEY`.

| Key | Name | Labelled | Access | Licence | Family | Tasks |
|---|---|---|---|---|---|---|
| `ai-koda` | AI-KODA | yes | open | CC BY 4.0 | `aiims-capsule` | `ordinal/ai-koda-ov`, `ordinal/ai-koda-vm` |
| `cad-sel` | CAD-SEL | yes | open | CC BY 4.0 | `cad-sel` | `cls/cad-sel` |
| `ead2020` | EAD2020 | yes | open | CC BY-NC-SA 4.0 in the archive and CC BY-NC 3.0 on Mendeley Data, non commercial use only | `endocv-oxford` | `det/ead2020-boxes`, `seg/ead2020-masks` |
| `gastrohun` | GastroHUN | yes | open | CC BY 4.0 | `gastrohun` | `cls/gastrohun` |
| `gastrovision` | GastroVision | yes | open | CC BY 4.0 | `baerum-simula` | `cls/gastrovision` |
| `hyperkvasir-labeled` | HyperKvasir labeled images | yes | open | CC BY 4.0 (competitions and commercial use need permission) | `baerum-simula` | `cls/hyperkvasir`, `cls/hyperkvasir-bbps`, `cls/hyperkvasir-esophagitis`, `cls/hyperkvasir-landmarks`, `ordinal/hyperkvasir-uc` |
| `kvasir-capsule` | Kvasir-Capsule | yes | open | CC BY 4.0 (commercial use needs permission) | `baerum-capsule` | `cls/kvasir-capsule`, `det/kvasir-capsule-boxes` |
| `kvasir-instrument` | Kvasir-Instrument | yes | open | CC BY 4.0 on OSF, research and education only per the Simula page | `baerum-simula` | `det/kvasir-instrument-boxes`, `seg/kvasir-instrument` |
| `kvasir-seg` | Kvasir-SEG | yes | open | research and education only | `baerum-simula` | `det/kvasir-seg-boxes`, `seg/kvasir-seg` |
| `kvasir-v2` | Kvasir v2 | yes | open | research and education only, commercial use needs permission | `baerum-simula` | `cls/kvasir-v2` |
| `limuc` | LIMUC | yes | open | CC BY 4.0 | `limuc-marmara` | `cls/limuc-remission`, `ordinal/limuc-mayo` |
| `nerthus` | Nerthus | yes | open | no licence stated, released by Simula for research | `baerum-simula` | `ordinal/nerthus` |
| `polypdb` | PolypDB | yes | open | research and education use per the authors, no formal licence on OSF | `baerum-simula` | `det/polypdb-bli-boxes`, `det/polypdb-fice-boxes`, `det/polypdb-lci-boxes`, `det/polypdb-nbi-boxes`, `det/polypdb-wli-boxes`, `seg/polypdb-bli`, `seg/polypdb-fice`, `seg/polypdb-lci`, `seg/polypdb-nbi`, `seg/polypdb-wli` |
| `wcebleedgen` | WCEBleedGen | yes | open | CC BY 4.0 | `wcebleedgen` | `cls/wcebleedgen`, `det/wcebleedgen-boxes`, `seg/wcebleedgen-mask` |

## Tasks

Partition sizes are train, validation and test. The test partition is the published
one when a source is named, and a group stratified fifth chosen by the benchmark
otherwise. It is never altered once fixed. Lost images are left out of training and
validation because they share a patient, video or sequence with a test image (same
group), copy a test image byte for byte (identical copy), are confirmed near duplicates
of a test image (near duplicate), or are copies whose labels contradict each other
(conflicting labels).

| Task | Kind | Classes | Primary metric | Partitions | Lost | Test source |
|---|---|---|---|---|---|---|
| `cls/cad-sel` | classification | 5 | `mcc` | 1993 / 284 / 540 | 0 | external centres 2 to 6 of CAD-SEL (benchmark protocol) |
| `cls/gastrohun` | classification | 23 | `mcc` | 5545 / 786 / 1145 | 0 | official GastroHUN patient level test split (Bravo et al. 2025) |
| `cls/gastrovision` | classification | 22 | `mcc` | 5545 / 792 / 1586 | 7 (7 near duplicate) | official GastroVision test split (Jha et al. 2023, DebeshJha/GastroVision abed639) |
| `cls/hyperkvasir` | classification | 23 | `mcc` | 4033 / 1009 / 5338 | 282 (282 near duplicate) | official HyperKvasir two fold split, fold 1 (Borgli et al. 2020, simula/hyper-kvasir 21cc366) |
| `cls/hyperkvasir-bbps` | classification | 2 | `auroc` | 521 / 130 / 897 | 246 (246 near duplicate) | official HyperKvasir two fold split, fold 1 (Borgli et al. 2020, simula/hyper-kvasir 21cc366) |
| `cls/hyperkvasir-esophagitis` | classification | 2 | `auroc` | 265 / 66 / 332 | 0 | official HyperKvasir two fold split, fold 1 (Borgli et al. 2020, simula/hyper-kvasir 21cc366) |
| `cls/hyperkvasir-landmarks` | classification | 6 | `mcc` | 1636 / 409 / 2054 | 5 (5 near duplicate) | official HyperKvasir two fold split, fold 1 (Borgli et al. 2020, simula/hyper-kvasir 21cc366) |
| `cls/kvasir-capsule` | classification | 11 | `mcc` | 14351 / 3658 / 24084 | 5052 (5052 same group) | official Kvasir-Capsule split_1 (Smedsrud et al. 2021, simula/kvasir-capsule 553601d) |
| `cls/kvasir-v2` | classification | 8 | `mcc` | 5593 / 799 / 1600 | 8 (6 near duplicate, 2 conflicting labels) | benchmark |
| `cls/limuc-remission` | classification | 2 | `auroc` | 8498 / 1074 / 1682 | 2 (2 conflicting labels) | official LIMUC test set (Polat et al., Zenodo 5827695) |
| `cls/wcebleedgen` | classification | 2 | `auroc` | 1656 / 237 / 524 | 201 (201 near duplicate) | benchmark |
| `det/ead2020-boxes` | detection | 9 | `box_ap50` | 1615 / 231 / 530 | 155 (155 near duplicate) | benchmark |
| `det/kvasir-capsule-boxes` | detection | 8 | `box_ap50` | 1249 / 118 / 1843 | 947 (947 same group) | official Kvasir-Capsule split_1 (Smedsrud et al. 2021, simula/kvasir-capsule 553601d) |
| `det/kvasir-instrument-boxes` | detection | 2 | `box_ap50` | 410 / 59 / 118 | 3 (3 identical copy) | official Kvasir-Instrument test list (Jha et al. 2021) |
| `det/kvasir-seg-boxes` | detection | 2 | `box_ap50` | 700 / 100 / 200 | 0 | benchmark |
| `det/polypdb-bli-boxes` | detection | 2 | `box_ap50` | 48 / 7 / 7 | 8 (8 same group) | official PolypDB modality wise test split (Jha et al. 2024, DebeshJha/PolypDB d2006bf) |
| `det/polypdb-fice-boxes` | detection | 2 | `box_ap50` | 37 / 4 / 7 | 22 (22 same group) | official PolypDB modality wise test split (Jha et al. 2024, DebeshJha/PolypDB d2006bf) |
| `det/polypdb-lci-boxes` | detection | 2 | `box_ap50` | 39 / 5 / 6 | 10 (10 same group) | official PolypDB modality wise test split (Jha et al. 2024, DebeshJha/PolypDB d2006bf) |
| `det/polypdb-nbi-boxes` | detection | 2 | `box_ap50` | 117 / 15 / 14 | 0 | official PolypDB modality wise test split (Jha et al. 2024, DebeshJha/PolypDB d2006bf) |
| `det/polypdb-wli-boxes` | detection | 2 | `box_ap50` | 2601 / 324 / 358 | 305 (305 same group) | official PolypDB modality wise test split (Jha et al. 2024, DebeshJha/PolypDB d2006bf) |
| `det/wcebleedgen-boxes` | detection | 2 | `box_ap50` | 1656 / 237 / 524 | 201 (201 near duplicate) | benchmark |
| `ordinal/ai-koda-ov` | ordinal | 4 | `quadratic_kappa` | 1515 / 193 / 424 | 4 (4 conflicting labels) | benchmark |
| `ordinal/ai-koda-vm` | ordinal | 4 | `quadratic_kappa` | 1520 / 186 / 422 | 8 (8 conflicting labels) | benchmark |
| `ordinal/hyperkvasir-uc` | ordinal | 6 | `quadratic_kappa` | 336 / 84 / 428 | 3 (3 near duplicate) | official HyperKvasir two fold split, fold 1 (Borgli et al. 2020, simula/hyper-kvasir 21cc366) |
| `ordinal/limuc-mayo` | ordinal | 4 | `quadratic_kappa` | 8498 / 1074 / 1682 | 2 (2 conflicting labels) | official LIMUC test set (Polat et al., Zenodo 5827695) |
| `ordinal/nerthus` | ordinal | 4 | `quadratic_kappa` | 3550 / 775 / 1200 | 0 | benchmark |
| `seg/ead2020-masks` | multilabel_segmentation | 5 | `dice` | 460 / 58 / 119 | 6 (6 near duplicate) | benchmark |
| `seg/kvasir-instrument` | segmentation | 2 | `dice` | 410 / 59 / 118 | 3 (3 identical copy) | official Kvasir-Instrument test list (Jha et al. 2021) |
| `seg/kvasir-seg` | segmentation | 2 | `dice` | 700 / 100 / 200 | 0 | benchmark |
| `seg/polypdb-bli` | segmentation | 2 | `dice` | 48 / 7 / 7 | 8 (8 same group) | official PolypDB modality wise test split (Jha et al. 2024, DebeshJha/PolypDB d2006bf) |
| `seg/polypdb-fice` | segmentation | 2 | `dice` | 37 / 4 / 7 | 22 (22 same group) | official PolypDB modality wise test split (Jha et al. 2024, DebeshJha/PolypDB d2006bf) |
| `seg/polypdb-lci` | segmentation | 2 | `dice` | 39 / 5 / 6 | 10 (10 same group) | official PolypDB modality wise test split (Jha et al. 2024, DebeshJha/PolypDB d2006bf) |
| `seg/polypdb-nbi` | segmentation | 2 | `dice` | 117 / 15 / 14 | 0 | official PolypDB modality wise test split (Jha et al. 2024, DebeshJha/PolypDB d2006bf) |
| `seg/polypdb-wli` | segmentation | 2 | `dice` | 2601 / 324 / 358 | 305 (305 same group) | official PolypDB modality wise test split (Jha et al. 2024, DebeshJha/PolypDB d2006bf) |
| `seg/wcebleedgen-mask` | segmentation | 2 | `dice` | 861 / 123 / 262 | 63 (63 near duplicate) | benchmark |

## Leakage warnings

An encoder pretrained on a dataset or corpus named here has seen images of the dataset
it overlaps. Pass its key to `pretrained_on` and the benchmark excludes every
overlapping task.

- **AI-KODA** shares hospital and period with **WCEBleedGen** (`wcebleedgen`) (both compiled by the same group at AIIMS New Delhi, overlap unverified). Evidence https://doi.org/10.5281/zenodo.10156571
- **GastroVision** contains images from **HyperKvasir unlabeled images** (`hyperkvasir-unlabeled`) (relabeled subset). Evidence https://arxiv.org/pdf/2307.08140
- **GastroVision** contains images from **Kvasir-Instrument** (`kvasir-instrument`) (1 identical images). Evidence byte identical archive members by CRC32 and size, find-those-polyps audit 2026-11-05
- **HyperKvasir labeled images** shares hospital and period with **HyperKvasir unlabeled images** (`hyperkvasir-unlabeled`). Evidence https://pmc.ncbi.nlm.nih.gov/articles/PMC7455694/
- **HyperKvasir labeled images** shares hospital and period with **HyperKvasir videos** (`hyperkvasir-videos`). Evidence https://pmc.ncbi.nlm.nih.gov/articles/PMC7455694/
- **Kvasir v2** contains images from **HyperKvasir labeled images** (`hyperkvasir-labeled`) (14 identical images). Evidence byte identical archive members by CRC32 and size, find-those-polyps audit 2026-11-05
- **Kvasir v2** contains images from **Kvasir-SEG** (`kvasir-seg`) (14 identical images). Evidence byte identical archive members by CRC32 and size, find-those-polyps audit 2026-11-05
- **Kvasir v2** shares hospital and period with **HyperKvasir unlabeled images** (`hyperkvasir-unlabeled`). Evidence https://doi.org/10.1038/s41597-020-00622-y
- **Kvasir-Capsule** contains frames of the same patients or videos as **Kvasir-Capsule unlabeled frames** (`kvasir-capsule-unlabeled`). Evidence https://pmc.ncbi.nlm.nih.gov/articles/PMC8160146/
- **Kvasir-Capsule** shares hospital and period with **HyperKvasir unlabeled images** (`hyperkvasir-unlabeled`) (same hospital, different modality). Evidence https://pmc.ncbi.nlm.nih.gov/articles/PMC8160146/
- **Kvasir-Instrument** contains images from **HyperKvasir labeled images** (`hyperkvasir-labeled`) (1 identical images). Evidence byte identical archive members by CRC32 and size, find-those-polyps audit 2026-11-05
- **Kvasir-Instrument** contains images from **HyperKvasir unlabeled images** (`hyperkvasir-unlabeled`). Evidence https://ar5iv.labs.arxiv.org/html/2011.08065
- **Kvasir-Instrument** contains images from **Kvasir v2** (`kvasir-v2`) (6 identical images). Evidence byte identical archive members by CRC32 and size, find-those-polyps audit 2026-11-05
- **Kvasir-Instrument** contains images from **Kvasir-SEG** (`kvasir-seg`). Evidence https://ar5iv.labs.arxiv.org/html/2011.08065
- **Kvasir-SEG** contains images from **HyperKvasir labeled images** (`hyperkvasir-labeled`) (HyperKvasir segmented images follow Kvasir-SEG). Evidence https://pmc.ncbi.nlm.nih.gov/articles/PMC7455694/
- **Kvasir-SEG** contains images from **Kvasir-Sessile** (`kvasir-sessile`) (strict subset). Evidence https://github.com/DebeshJha/Kvasir-SEG
- **Kvasir-SEG** shares hospital and period with **HyperKvasir unlabeled images** (`hyperkvasir-unlabeled`). Evidence https://pmc.ncbi.nlm.nih.gov/articles/PMC7455694/
- **Kvasir-SEG** shares hospital and period with **HyperKvasir videos** (`hyperkvasir-videos`). Evidence https://pmc.ncbi.nlm.nih.gov/articles/PMC7455694/
- **Nerthus** shares hospital and period with **HyperKvasir unlabeled images** (`hyperkvasir-unlabeled`). Evidence https://pmc.ncbi.nlm.nih.gov/articles/PMC7455694/
- **PolypDB** contains images from **BKAI-IGH NeoPolyp-Small** (`bkai-igh-neopolyp`) (centre C3, pending hash confirmation). Evidence https://arxiv.org/html/2409.00045v2
- **PolypDB** contains images from **HyperKvasir unlabeled images** (`hyperkvasir-unlabeled`) (all 2,588 WLI and 136 NBI images of centre C1). Evidence https://arxiv.org/html/2409.00045v2
- **PolypDB** shares hospital and period with **GastroVision** (`gastrovision`) (Karolinska images). Evidence https://arxiv.org/html/2409.00045v2

## Source families

Datasets in one family come from the same clinical source pool. Patient separation across
members of a family is undocumented unless an overlap entry states otherwise.

- `aiims-capsule` `ai-koda`
- `baerum-capsule` `kvasir-capsule`
- `baerum-simula` `gastrovision`, `hyperkvasir-labeled`, `kvasir-instrument`, `kvasir-seg`, `kvasir-v2`, `nerthus`, `polypdb`
- `cad-sel` `cad-sel`
- `endocv-oxford` `ead2020`
- `gastrohun` `gastrohun`
- `limuc-marmara` `limuc`
- `wcebleedgen` `wcebleedgen`

## Provenance

### AI-KODA

All India Institute of Medical Sciences, New Delhi. 2,173 frames of 28 patients scored with the Korea Canada cleanliness scale.

Citation Handa et al. 2024, figshare, doi 10.6084/m9.figshare.25807915. https://doi.org/10.6084/m9.figshare.25807915
Labels by gastroenterologists.
File names carry the patient initials, which define the groups. No split is released, so the benchmark holds out a stratified fifth of the patients.

### CAD-SEL

Six Chinese centres. 5,165 frames of colorectal subepithelial tumours from 641 patients in five classes, imaged with white light endoscopy and endoscopic ultrasound.

Citation Xu et al. 2026, Scientific Data, doi 10.1038/s41597-026-07331-y. https://doi.org/10.6084/m9.figshare.29945483
Labels by pathology.
Only the white light images are used. No split is released, so the benchmark tests on the external centres two to six and trains on centre one, which measures transfer across centres. Images are grouped by lesion folder.

### EAD2020

Training data of the EndoCV2020 artefact detection and segmentation challenge. 2,531 frames with 31,069 boxes of eight artefact classes, and 643 frames with overlapping masks of five artefact classes.

Citation Ali et al. 2021, Medical Image Analysis, and Ali et al. 2020, Scientific Reports. https://doi.org/10.17632/c7fjbxcgj9.4
Labels by challenge annotators.
No official partition exists, since the labels of the challenge test data were never released. Frames of one sequence share a group, while single frames carry no patient identifier and form a group each, so that near duplicate frames of one patient may fall into different partitions. Blood is left out of segmentation, since only 169 of the 643 masks annotate it.

### GastroHUN

Hospital Universitario Nacional de Colombia. 8,834 images of 387 patients, each labelled by four gastroenterologists with one of 22 stations of a systematic stomach screening protocol or an other class.

Citation Bravo et al. 2025, Scientific Data, doi 10.1038/s41597-025-04401-5. https://doi.org/10.6084/m9.figshare.27308133
Labels by four expert gastroenterologists.
Labels are the triple agreement column, the class chosen by at least three of the four annotators, and images without such agreement are excluded. The official patient level test partition is used verbatim.

### GastroVision

Baerum Hospital, Norway, and Karolinska University Hospital, Sweden. 8,000 images in 27 classes.

Citation Jha et al. 2023, ICML workshop. https://osf.io/84e7f/
Labels by endoscopist.
The official split of the authors, published on GitHub (DebeshJha/GastroVision abed639) rather than in the archive, covers the 22 classes evaluated in the paper and leaves out five rare ones. The benchmark adopts its test partition verbatim. Each image forms its own group, since no patient identifiers are released. File names repeat across class folders for different images, so the official lists are matched by folder and name.

### HyperKvasir labeled images

Baerum Hospital, Vestre Viken, Norway, 2008 to 2016, Olympus and Pentax endoscopes. 10,662 images in 23 classes, official two fold split.

Citation Borgli et al. 2020, Scientific Data 7, 283. https://osf.io/mh9sj/
Labels by endoscopist.
The official two fold split assigns every image to fold 0 or 1. The benchmark tests on fold 1, as the split file defines it, and draws validation from fold 0. No patient or video identifiers are released, so each image is its own group and near duplicate frames of one examination may fall on both sides of the split.

### Kvasir-Capsule

Baerum Hospital, Norway, 2016 to 2018, Olympus EC-S10. 117 videos, 47,238 labeled frames.

Citation Smedsrud et al. 2021, Scientific Data 8, 142. https://osf.io/dv2ag/
Labels by endoscopist.
The archive nests one tarball per class, which is streamed and unpacked whole when first used. The official two fold split covers eleven classes, and the benchmark tests on split_1. Frames are grouped by the video identifier in their names, but seven videos occur in both official folds, so the official test partition is not patient disjoint from training. Frames of the videos shared with the official test fold are excluded from training and validation. Angiectasia, Blood, Lymphangiectasia and Ulcer occur in too few videos to appear in validation as well as training, so selection on validation does not see them. The metadata table on OSF adds one axis aligned box to 4,235 finding frames of 21 videos. The detection task keeps the seven box classes present in both official folds, and fresh blood, lymphangiectasia and ulcer come from too few videos to appear in its validation partition.

### Kvasir-Instrument

Baerum Hospital, Vestre Viken, Norway, 2008 to 2016, Olympus and Pentax endoscopes. 590 frames with tools.

Citation Jha et al. 2021, MMM, doi 10.1007/978-3-030-67835-7_19. https://datasets.simula.no/kvasir-instrument/
Labels by expert annotators.
Images and masks are nested tarballs, unpacked whole when first used. The official test list of 118 images is used verbatim, and each image is its own group.

### Kvasir-SEG

Baerum Hospital, Vestre Viken, Norway, 2008 to 2016, Olympus and Pentax endoscopes. 1,000 polyp images with masks, the polyp class of Kvasir v2.

Citation Jha et al. 2020, MMM. https://datasets.simula.no/kvasir-seg/
Labels by expert annotators.
The PraNet split (Fan et al. 2020) tests on 100 images, but its file list is only distributed inside archives that also carry data under a separate agreement. The benchmark therefore holds out a seeded fifth of the images. Without patient identifiers each image is its own group. The checksum was computed by the benchmark at registration, since none is published.

### Kvasir v2

Vestre Viken Health Trust, Norway. 8,000 images in 8 balanced classes.

Citation Pogorelov et al. 2017, MMSys, doi 10.1145/3083187.3083212. https://datasets.simula.no/kvasir/
Labels by endoscopists.
No official split is released. The benchmark holds out a stratified fifth of the images, each image forming its own group.

### LIMUC

Marmara University, Istanbul, Turkiye, December 2011 to July 2019. 11,276 images, 564 patients, 1,043 colonoscopies.

Citation Polat et al., LIMUC dataset, Zenodo 2022. https://zenodo.org/records/5827695
Labels by two raters and an adjudicator.
Official test set of 1,686 images. Train and validation are split by patient. The archive files 10 byte identical images under two different Mayo scores (for instance UC_patient_130_20 under Mayo 2 and Mayo 3). These contradictory images are excluded, which leaves 1,682 test images.

### Nerthus

Baerum Hospital, Vestre Viken, Norway, 2008 to 2016, Olympus and Pentax endoscopes. 21 videos, 5,525 frames with Boston bowel preparation scores.

Citation Pogorelov et al. 2017, MMSys. https://datasets.simula.no/nerthus/
Labels by endoscopist.
Frames are named bowel_<video>_score_<score>-<n>_<frame>, and every video carries one Boston bowel preparation score. Splits group frames by video, so the test partition holds only a few of the 21 videos and scores vary accordingly. Only two videos carry score 0, so validation has none while training and test have one each.

### PolypDB

Baerum (Norway), Karolinska (Sweden) and Hanoi (Vietnam). 3,934 images with masks.

Citation Jha et al. 2024, arXiv 2409.00045. https://osf.io/pr7ms/
Labels by expert annotators.
Tasks follow the official modality wise test partitions of the authors' repository. Validation is drawn by the benchmark from the remaining images. Hanoi images are grouped by the patient code in their file names. The BLI, LCI and FICE test partitions hold fewer than ten images, so their scores carry large uncertainty.

### WCEBleedGen

1,309 bleeding frames with masks and boxes and 1,309 non bleeding frames, compiled from several sources described only as internet sources, without patient identifiers.

Citation Handa et al. 2023, Zenodo, doi 10.5281/zenodo.10156571. https://doi.org/10.5281/zenodo.10156571
Labels by medical experts.
No split or patient identifiers are released, so each frame is its own group and near duplicate frames may cross the benchmark split. Its sources are not documented, so overlap with other capsule datasets cannot be excluded.
