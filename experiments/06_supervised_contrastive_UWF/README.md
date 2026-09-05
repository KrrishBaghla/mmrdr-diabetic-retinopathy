# Experiment VI — Supervised contrastive learning on lesion representations (UWF)

**Question.** Rather than only fixing the loss on the output, can the *representation* itself be shaped so
that rare-lesion images cluster together before classification?

**Method.** `UWF_SupConDRNet` — an ImageNet ResNet-50 trunk with two SupCon projection heads (512→512→128 on
layer2 for early features, 2048→512→128 on layer4 for deep features) and a 4-level FPN feeding the grade and
lesion heads. `SupConLoss` on both branches is keyed on rare-lesion presence and added to grade CE + lesion
BCE. Dual-GPU `DataParallel`, 60 epochs, patience 10.

**Results** (best in the visible window, epoch 31; early-stopped at 37/60):

| Metric | Val |
|---|---|
| Grade accuracy | **0.7645** |
| Grade quadratic κ | 0.9220 |
| Lesion macro-F1 | 0.7312 |
| Lesion macro-AUROC | 0.9521 |

Training reached 0.8591 accuracy / 0.9595 κ — a ~0.10 train–val gap.

**Caveats.** The notebook resumes from an external checkpoint at epoch 27, so **only epochs 28–37 have saved
outputs**; the first 27 epochs are not recorded in this file. There is also no ablation against a
no-SupCon control, so the contribution of the contrastive term specifically is not isolated here.

**Why it matters downstream.** This pipeline — trunk, losses, masking, checkpointing — is reused verbatim by
Experiments VII, VIII, IX and X, so those experiments each vary exactly one component against a fixed reference.
