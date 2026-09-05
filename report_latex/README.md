# IEEE-format report (LaTeX source)

`report.tex` — *"Multi-Task Deep-Learning Baselines for Diabetic Retinopathy Grading and Lesion Detection on
the Multimodal MMRDR Dataset"*, ~2,600 lines, 22 figures. Compile with `pdflatex` against the bundled
`ieeecolor.cls` and `generic.sty`. The compiled output is `../docs/IIT_BHU_Internship_Report.pdf`.

**Structure.** Introduction · Materials and Dataset · Experiments I–XI · Conclusion · Bibliography. Each
experiment section follows the same template — Objective, Motivation, Methodology, Experimental Setup,
Dataset, Preprocessing, Architecture, Training Procedure, Hyperparameters, Evaluation Metrics, Results,
Analysis, Discussion, Limitations, Conclusion.

**Numbering.** The `../experiments/NN_…` folders are numbered to match these sections one-to-one — report
Experiment VII is folder `07_uncertainty_task_weighting_UWF`, and so on. Figure filenames in `figures/`
follow the same numbering (`exp9-heads.png` is the Experiment IX head ablation, `exp10-verdict.png` the
Experiment X synthetic-data verdict, `exp11-*` the Experiment XI transfer study).

| Report section | Folder |
|---|---|
| I — Multi-Scale Dual-Branch CNN on CFP | `01_multiscale_dual_branch_CFP` |
| II — Reproducing the RETFound Foundation-Model Baseline on CFP | `02_retfound_foundation_CFP` |
| III — Feature-Pyramid Network with Spatial Attention on UWF | `03_fpn_spatial_attention_UWF` |
| IV — Clinically-Weighted Loss for High-Risk Lesions (UWF) | `04_clinically_weighted_loss_UWF` |
| V — Asymmetric False-Negative-Penalised Loss (UWF) | `05_asymmetric_fn_loss_UWF` |
| VI — Supervised Contrastive Learning for Rare Lesions (UWF) | `06_supervised_contrastive_UWF` |
| VII — Learned Uncertainty Task-Weighting (UWF) | `07_uncertainty_task_weighting_UWF` |
| VIII — Recall-Driven Curriculum Resampling (UWF) | `08_recall_driven_resampling_UWF` |
| IX — Shared vs. Separate Lesion Heads (UWF) | `09_shared_vs_separate_heads_UWF` |
| X — Synthetic Minority Generation (UWF) | `10_synthetic_minority_generation_UWF` |
| XI — Cross-Dataset Zero-Shot and Few-Shot Generalization | `11_zero_few_shot_transfer` |

The report's two summary tables — "Summary of the Ten Experiments" and "UWF Rare-Lesion Experiments —
Validation Metrics" — sit at the end of the Conclusion.
