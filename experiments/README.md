# Experiments I–XI

The folder numbers here match the **section numbers in the report**
(`../docs/IIT_BHU_Internship_Report.pdf`) exactly — folder `07_…` is the report's Experiment VII, and so on.

| # | Report section | Folder | Modality | Headline |
|---|---|---|---|---|
| I | Multi-Scale Dual-Branch CNN on CFP | `01_multiscale_dual_branch_CFP` | CFP | val acc **0.8291** |
| II | Reproducing the RETFound Foundation-Model Baseline | `02_retfound_foundation_CFP` | CFP | test acc **0.8094**, QWK 0.8838 |
| III | Feature-Pyramid Network with Spatial Attention | `03_fpn_spatial_attention_UWF` | UWF | val acc 0.7448 |
| IV | Clinically-Weighted Loss for High-Risk Lesions | `04_clinically_weighted_loss_UWF` | UWF | test acc 0.7650; RD recall 0.79 |
| V | Asymmetric False-Negative-Penalised Loss | `05_asymmetric_fn_loss_UWF` | UWF | QWK 0.9208, lesion AUROC 0.9504 |
| VI | Supervised Contrastive Learning for Rare Lesions | `06_supervised_contrastive_UWF` | UWF | val acc 0.7645, QWK 0.9220 |
| VII | Learned Uncertainty Task-Weighting | `07_uncertainty_task_weighting_UWF` | UWF | **lesion F1 0.7779, AUROC 0.9605** |
| VIII | Recall-Driven Curriculum Resampling | `08_recall_driven_resampling_UWF` | UWF | **QWK 0.9234**; RD recall 1.000 |
| IX | Shared vs. Separate Lesion Heads | `09_shared_vs_separate_heads_UWF` | UWF | **negative** — separation −0.074 |
| X | Synthetic Minority Generation | `10_synthetic_minority_generation_UWF` | UWF | **negative** — 0.833 vs 0.878 |
| XI | Cross-Dataset Zero-Shot and Few-Shot Generalization | `11_zero_few_shot_transfer` | CFP → APTOS, IDRiD | zero-shot κ up to 0.860 |

Experiments I–III establish the baselines. IV–X all attack one problem — the rare, sight-threatening PDR
lesions (NV, VH, RD) that standard training ignores — from six different angles. XI tests whether any of it
generalises off MMRDR.

Experiments VII, VIII, IX and X reuse the Experiment VI pipeline verbatim — `UWF_SupConDRNet`
(ResNet-50 + FPN + SupCon projection heads, 26.9M params), `AsymmetricFocalLoss`, `apply_uwf_mask` elliptical
masking, AMP + dual-T4 `DataParallel` — varying only the one named lever, so each measured difference is
attributable to that change.

Every notebook is the original file, copied unmodified with its saved outputs intact. Files named
`NOTES_README.md` and `RESEARCH_NOTES.md` are **pre-run planning documents** kept for provenance; where they
disagree with the notebook outputs, the outputs are authoritative. Each folder's `README.md` reports what
actually happened.
