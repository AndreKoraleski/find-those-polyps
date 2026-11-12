# Leaderboard summary

Primary score on the full label budget with its 95% bootstrap interval over test images.
The best model of each row is in bold. Every budget and metric is in `leaderboard.csv`
and `leaderboard.xlsx`.

| Model | Pretrained on | Input size |
|---|---|---|
| imagenet-resnet50 | imagenet-1k | 224 |
| imagenet-resnet50-448 | imagenet-1k | 448 |
| random-resnet50 | none | 224 |
| random-resnet50-448 | none | 448 |

Models marked undeclared did not state their pretraining data, so they may have seen the
test images. The probes never do.

| Task | Protocol | Metric | imagenet-resnet50 | imagenet-resnet50-448 | random-resnet50 | random-resnet50-448 |
|---|---|---|---|---|---|---|
| `cls/cad-sel` | kmeans | ami | 0.036 [0.019, 0.057] |  | **0.049 [0.030, 0.071]** |  |
| `cls/cad-sel` | knn | mcc | **0.182 [0.093, 0.259]** |  | 0.051 [0.001, 0.101] |  |
| `cls/cad-sel` | linear | mcc | **0.264 [0.180, 0.335]** |  | 0.188 [0.110, 0.274] |  |
| `cls/cad-sel` | retrieval | map_at_r | **0.279 [0.265, 0.293]** |  | 0.260 [0.247, 0.273] |  |
| `cls/gastrohun` | kmeans | ami | 0.195 [0.183, 0.211] |  | **0.225 [0.211, 0.239]** |  |
| `cls/gastrohun` | knn | mcc | 0.397 [0.367, 0.427] |  | **0.464 [0.433, 0.494]** |  |
| `cls/gastrohun` | linear | mcc | **0.594 [0.562, 0.620]** |  | 0.571 [0.541, 0.600] |  |
| `cls/gastrohun` | retrieval | map_at_r | 0.038 [0.036, 0.040] |  | **0.057 [0.054, 0.060]** |  |
| `cls/gastrovision` | kmeans | ami | **0.272 [0.260, 0.287]** |  | 0.179 [0.167, 0.191] |  |
| `cls/gastrovision` | knn | mcc | **0.556 [0.530, 0.583]** |  | 0.362 [0.334, 0.390] |  |
| `cls/gastrovision` | linear | mcc | **0.661 [0.638, 0.687]** |  | 0.474 [0.448, 0.501] |  |
| `cls/gastrovision` | retrieval | map_at_r | **0.109 [0.104, 0.113]** |  | 0.067 [0.064, 0.071] |  |
| `cls/hyperkvasir` | kmeans | ami | **0.576 [0.567, 0.584]** |  | 0.312 [0.304, 0.320] |  |
| `cls/hyperkvasir` | knn | mcc | **0.780 [0.768, 0.792]** |  | 0.590 [0.576, 0.603] |  |
| `cls/hyperkvasir` | linear | mcc | **0.836 [0.826, 0.847]** |  | 0.722 [0.710, 0.734] |  |
| `cls/hyperkvasir` | retrieval | map_at_r | **0.245 [0.240, 0.249]** |  | 0.133 [0.130, 0.136] |  |
| `cls/hyperkvasir-bbps` | kmeans | ami | **0.205 [0.166, 0.249]** |  | 0.021 [0.007, 0.040] |  |
| `cls/hyperkvasir-bbps` | knn | auroc | **0.996 [0.989, 1.000]** |  | 0.942 [0.928, 0.956] |  |
| `cls/hyperkvasir-bbps` | linear | auroc | **0.998 [0.996, 1.000]** |  | 0.988 [0.979, 0.995] |  |
| `cls/hyperkvasir-bbps` | retrieval | map_at_r | **0.586 [0.574, 0.597]** |  | 0.421 [0.410, 0.432] |  |
| `cls/hyperkvasir-esophagitis` | kmeans | ami | **0.001 [-0.003, 0.022]** |  | -0.001 [-0.003, 0.013] |  |
| `cls/hyperkvasir-esophagitis` | knn | auroc | **0.813 [0.766, 0.860]** |  | 0.705 [0.648, 0.759] |  |
| `cls/hyperkvasir-esophagitis` | linear | auroc | **0.839 [0.798, 0.884]** |  | 0.754 [0.702, 0.806] |  |
| `cls/hyperkvasir-esophagitis` | retrieval | map_at_r | **0.361 [0.346, 0.376]** |  | 0.322 [0.308, 0.335] |  |
| `cls/hyperkvasir-landmarks` | kmeans | ami | **0.758 [0.739, 0.779]** |  | 0.370 [0.349, 0.391] |  |
| `cls/hyperkvasir-landmarks` | knn | mcc | **0.954 [0.943, 0.963]** |  | 0.811 [0.791, 0.831] |  |
| `cls/hyperkvasir-landmarks` | linear | mcc | **0.980 [0.973, 0.986]** |  | 0.925 [0.911, 0.938] |  |
| `cls/hyperkvasir-landmarks` | retrieval | map_at_r | **0.518 [0.508, 0.528]** |  | 0.329 [0.320, 0.340] |  |
| `cls/kvasir-capsule` | kmeans | ami | **0.163 [0.159, 0.167]** |  | 0.129 [0.125, 0.134] |  |
| `cls/kvasir-capsule` | knn | mcc | **0.269 [0.260, 0.278]** |  | 0.247 [0.235, 0.258] |  |
| `cls/kvasir-capsule` | linear | mcc | **0.270 [0.261, 0.279]** |  | 0.230 [0.220, 0.241] |  |
| `cls/kvasir-capsule` | retrieval | map_at_r | 0.447 [0.444, 0.450] |  | **0.504 [0.500, 0.507]** |  |
| `cls/kvasir-v2` | kmeans | ami | **0.530 [0.511, 0.546]** |  | 0.398 [0.381, 0.416] |  |
| `cls/kvasir-v2` | knn | mcc | **0.786 [0.764, 0.808]** |  | 0.603 [0.576, 0.629] |  |
| `cls/kvasir-v2` | linear | mcc | **0.870 [0.852, 0.888]** |  | 0.685 [0.660, 0.709] |  |
| `cls/kvasir-v2` | retrieval | map_at_r | **0.252 [0.244, 0.258]** |  | 0.161 [0.156, 0.166] |  |
| `cls/limuc-remission` | kmeans | ami | **0.013 [0.006, 0.025]** |  | 0.000 [-0.001, 0.004] |  |
| `cls/limuc-remission` | knn | auroc | **0.918 [0.900, 0.932]** |  | 0.705 [0.670, 0.734] |  |
| `cls/limuc-remission` | linear | auroc | **0.962 [0.952, 0.971]** |  | 0.784 [0.757, 0.808] |  |
| `cls/limuc-remission` | retrieval | map_at_r | **0.572 [0.561, 0.582]** |  | 0.535 [0.523, 0.545] |  |
| `cls/wcebleedgen` | kmeans | ami | 0.007 [-0.001, 0.027] |  | **0.014 [0.001, 0.041]** |  |
| `cls/wcebleedgen` | knn | auroc | **0.999 [0.999, 1.000]** |  | 0.968 [0.951, 0.981] |  |
| `cls/wcebleedgen` | linear | auroc | **1.000 [0.999, 1.000]** |  | 0.983 [0.974, 0.991] |  |
| `cls/wcebleedgen` | retrieval | map_at_r | **0.408 [0.398, 0.418]** |  | 0.344 [0.334, 0.354] |  |
| `det/ead2020-boxes` | linear_det | box_ap50 | **0.084 [0.077, 0.097]** | 0.076 [0.070, 0.089] | 0.064 [0.057, 0.076] | 0.047 [0.043, 0.057] |
| `det/kvasir-capsule-boxes` | linear_det | box_ap50 | **0.038 [0.033, 0.049]** | 0.033 [0.030, 0.040] | 0.028 [0.022, 0.037] | 0.021 [0.017, 0.028] |
| `det/kvasir-instrument-boxes` | linear_det | box_ap50 | **0.641 [0.585, 0.721]** | 0.465 [0.429, 0.527] | 0.423 [0.353, 0.523] | 0.278 [0.234, 0.338] |
| `det/kvasir-seg-boxes` | linear_det | box_ap50 | **0.484 [0.438, 0.546]** | 0.357 [0.323, 0.407] | 0.164 [0.126, 0.220] | 0.101 [0.078, 0.144] |
| `det/polypdb-bli-boxes` | linear_det | box_ap50 | **0.127 [0.034, 0.368]** | 0.070 [0.017, 0.258] | 0.001 [0.000, 0.005] | 0.003 [0.000, 0.019] |
| `det/polypdb-fice-boxes` | linear_det | box_ap50 | 0.125 [0.039, 0.260] | **0.191 [0.117, 0.478]** | 0.066 [0.001, 0.201] | 0.033 [0.004, 0.096] |
| `det/polypdb-lci-boxes` | linear_det | box_ap50 | **0.341 [0.136, 0.718]** | 0.320 [0.195, 0.532] | 0.079 [0.024, 0.303] | 0.169 [0.018, 0.520] |
| `det/polypdb-nbi-boxes` | linear_det | box_ap50 | 0.181 [0.082, 0.355] | **0.281 [0.224, 0.438]** | 0.043 [0.005, 0.150] | 0.019 [0.001, 0.086] |
| `det/polypdb-wli-boxes` | linear_det | box_ap50 | **0.348 [0.311, 0.400]** | 0.312 [0.278, 0.356] | 0.053 [0.040, 0.074] | 0.040 [0.031, 0.056] |
| `det/wcebleedgen-boxes` | linear_det | box_ap50 | **0.162 [0.134, 0.199]** | 0.143 [0.120, 0.176] | 0.128 [0.105, 0.163] | 0.089 [0.074, 0.113] |
| `ordinal/ai-koda-ov` | kmeans | ami | 0.081 [0.053, 0.113] |  | **0.307 [0.263, 0.352]** |  |
| `ordinal/ai-koda-ov` | knn | quadratic_kappa | 0.503 [0.438, 0.563] |  | **0.814 [0.774, 0.848]** |  |
| `ordinal/ai-koda-ov` | linear | quadratic_kappa | 0.677 [0.625, 0.723] |  | **0.815 [0.780, 0.848]** |  |
| `ordinal/ai-koda-ov` | retrieval | map_at_r | 0.116 [0.108, 0.122] |  | **0.190 [0.182, 0.198]** |  |
| `ordinal/ai-koda-vm` | kmeans | ami | 0.131 [0.092, 0.177] |  | **0.200 [0.161, 0.248]** |  |
| `ordinal/ai-koda-vm` | knn | quadratic_kappa | 0.613 [0.530, 0.683] |  | **0.739 [0.680, 0.790]** |  |
| `ordinal/ai-koda-vm` | linear | quadratic_kappa | 0.665 [0.583, 0.723] |  | **0.739 [0.675, 0.791]** |  |
| `ordinal/ai-koda-vm` | retrieval | map_at_r | 0.213 [0.198, 0.228] |  | **0.283 [0.261, 0.302]** |  |
| `ordinal/hyperkvasir-uc` | kmeans | ami | **0.021 [0.004, 0.042]** |  | 0.001 [-0.012, 0.017] |  |
| `ordinal/hyperkvasir-uc` | knn | quadratic_kappa | **0.356 [0.280, 0.430]** |  | 0.225 [0.137, 0.308] |  |
| `ordinal/hyperkvasir-uc` | linear | quadratic_kappa | **0.416 [0.324, 0.501]** |  | 0.183 [0.083, 0.283] |  |
| `ordinal/hyperkvasir-uc` | retrieval | map_at_r | **0.197 [0.185, 0.208]** |  | 0.166 [0.157, 0.176] |  |
| `ordinal/limuc-mayo` | kmeans | ami | **0.023 [0.015, 0.034]** |  | 0.013 [0.006, 0.021] |  |
| `ordinal/limuc-mayo` | knn | quadratic_kappa | **0.622 [0.581, 0.661]** |  | 0.206 [0.163, 0.250] |  |
| `ordinal/limuc-mayo` | linear | quadratic_kappa | **0.745 [0.716, 0.772]** |  | 0.373 [0.319, 0.421] |  |
| `ordinal/limuc-mayo` | retrieval | map_at_r | **0.210 [0.204, 0.216]** |  | 0.175 [0.170, 0.180] |  |
| `ordinal/nerthus` | kmeans | ami | 0.572 [0.548, 0.596] |  | **0.603 [0.578, 0.629]** |  |
| `ordinal/nerthus` | knn | quadratic_kappa | **0.531 [0.494, 0.568]** |  | 0.444 [0.397, 0.486] |  |
| `ordinal/nerthus` | linear | quadratic_kappa | **0.455 [0.418, 0.495]** |  | 0.428 [0.387, 0.474] |  |
| `ordinal/nerthus` | retrieval | map_at_r | **0.380 [0.365, 0.397]** |  | 0.226 [0.218, 0.234] |  |
| `seg/ead2020-masks` | linear_seg | dice | **0.230 [0.200, 0.260]** | 0.221 [0.200, 0.244] | 0.054 [0.040, 0.069] | 0.055 [0.043, 0.067] |
| `seg/kvasir-instrument` | linear_seg | dice | 0.832 [0.812, 0.849] | **0.864 [0.844, 0.882]** | 0.522 [0.474, 0.573] | 0.547 [0.495, 0.592] |
| `seg/kvasir-seg` | linear_seg | dice | **0.729 [0.703, 0.757]** | 0.728 [0.703, 0.754] | 0.358 [0.324, 0.391] | 0.284 [0.255, 0.316] |
| `seg/polypdb-bli` | linear_seg | dice | 0.355 [0.216, 0.498] | **0.483 [0.312, 0.643]** | 0.049 [0.000, 0.102] | 0.084 [0.000, 0.246] |
| `seg/polypdb-fice` | linear_seg | dice | 0.459 [0.217, 0.700] | **0.563 [0.285, 0.805]** | 0.157 [0.018, 0.334] | 0.115 [0.022, 0.246] |
| `seg/polypdb-lci` | linear_seg | dice | **0.667 [0.524, 0.762]** | 0.660 [0.446, 0.778] | 0.046 [0.014, 0.082] | 0.184 [0.025, 0.349] |
| `seg/polypdb-nbi` | linear_seg | dice | 0.579 [0.461, 0.674] | **0.657 [0.558, 0.742]** | 0.218 [0.131, 0.303] | 0.096 [0.057, 0.138] |
| `seg/polypdb-wli` | linear_seg | dice | 0.632 [0.605, 0.659] | **0.649 [0.625, 0.673]** | 0.142 [0.123, 0.163] | 0.099 [0.084, 0.115] |
| `seg/wcebleedgen-mask` | linear_seg | dice | **0.548 [0.515, 0.581]** | 0.539 [0.507, 0.570] | 0.407 [0.371, 0.442] | 0.436 [0.399, 0.472] |
