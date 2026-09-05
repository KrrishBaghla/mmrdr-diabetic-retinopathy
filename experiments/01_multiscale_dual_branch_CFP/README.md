# Experiment I — Multi-Scale Dual-Branch CNN on CFP

**Question.** Standard CNNs downsample aggressively, and microaneurysms — the earliest sign of DR — are only
a few pixels across. Does preserving early, high-resolution features alongside deep semantic ones improve
grading?

**Architecture.** `MultiScaleDRNet` (MS-LAN): a ResNet-50 split into two paths. One takes `layer1` output
(256 channels, pre-maxpool detail) and compresses it to 512×16×16; the other takes `layer4` output
(2048 channels, global context) and compresses it to the same shape. The two are fused, then split into a
5-way grade head and a 7-way lesion head. `FocalLoss` (γ=2) on the grade head, BCE on the lesion head,
512×512 input.

**Results** (best validation, epoch 14; early-stopped at epoch 22 of 50):

| Metric | Value |
|---|---|
| Grade accuracy | **0.8291** |
| Lesion accuracy (element-wise) | 0.9494 |

The grade accuracy climbs 0.7145 (epoch 1) → 0.8291 (epoch 14), then plateaus around 0.82–0.83. This beats
the plain ResNet-50 multi-task baseline of 0.8233 (see `../../supporting/`), which is the result that
justified carrying the multi-scale design forward into Experiment III on UWF.

**Caveat.** This is a *best-validation* number under an ad-hoc `random_state=42` split, not a held-out test
number — it is not directly comparable to Experiment II's 0.8094, which uses the official test split. The
evaluation cell produced Grad-CAM figures but no numeric test-set F1/QWK/AUC report.

**Figures.** `../../figures/custom_resnet_architecture.jpeg`,
`../../figures/custom_architecture_resnet_visuals_with_heatmap.png`,
`../../figures/resnet_custom_architecture_results.png`.
