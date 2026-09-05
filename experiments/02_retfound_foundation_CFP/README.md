# Experiment II — Reproducing the RETFound foundation-model baseline (CFP)

**Question.** The MMRDR paper's headline numbers come from fine-tuning RETFound, a ViT-Large retinal
foundation model pretrained with masked autoencoding. Can that result be reproduced?

**Method.** `vit_large_patch16_224` run at **512 px** with interpolated positional embeddings, average global
pooling, drop-path 0.2, layer-wise LR decay 0.65 with warmup + cosine schedule, label smoothing 0.1, and a
**full** fine-tune (304.2M trainable parameters). Joint grade CE + lesion BCE (λ=1.0). Official MMRDR `tr`/`ts`
split with a stratified validation carve — **7,559 / 1,334 / 2,225** — seed 42, 30 epochs, peak GPU 13.47 GB.

**Results** (held-out official test set, n=2,225):

| Metric | Paper | Ours @0.5 | 95 % bootstrap CI | Ours @tuned thresholds |
|---|---|---|---|---|
| Grade accuracy | 0.822 | **0.8094** | [0.792, 0.825] | 0.8094 |
| Grade F1 | 0.714 | **0.6920** | [0.665, 0.718] | 0.6920 |
| Lesion accuracy | 0.951 | **0.9476** | [0.943, 0.951] | 0.9415 |
| Lesion F1 | 0.682 | 0.6051 | [0.570, 0.636] | **0.6743** |

**Three of the paper's four values fall inside our confidence interval** — a statistical match. The fourth,
lesion F1, sits outside at a fixed 0.5 threshold but recovers to 0.6743 with per-lesion tuning, so the gap
was threshold placement rather than discrimination.

Additional grade metrics: **QWK 0.8838**, balanced accuracy 0.6787, MCC 0.6735, per-class recall
[0.948, 0.452, 0.665, 0.590, 0.739]. Lesion macro-AUROC 0.9706, macro-PR-AUC 0.7412; mean rare recall 0.5634
at 0.5 (NV 0.669, VH 0.830, RD 0.190), rising to 0.6514 with tuned thresholds.

**One instructive failure inside a good run:** VB/IRMA collapses to **F1 0.000** despite an AUROC of 0.949,
on just 43 test positives. The model ranks that lesion well but no fixed threshold separates it — a support
problem, not a learning problem.

## Why the broken run is in this folder

`retfound_v1_BROKEN_weight_load.ipynb` reports 66.50 % grade accuracy over 40 completed epochs and was
initially taken at face value. Auditing the weight load found **only 4 of 294 pretrained tensors had
transferred** — a silent failure leaving the "foundation model" randomly initialised — and the layer-freeze
loop had never executed (`int("ln_1")` raised, and the exception was swallowed). The corrected run above
loads 294/294 tensors and 288/288 blocks with 0 missing and 0 unexpected keys.

It is kept because the diagnosis is more useful than the number, and because 66.50 % appears in earlier
project documents that this notebook corrects. **Do not cite 66.50 % as a RETFound result.**

**Supporting artefacts.** `metrics_retfound_cfp_ft512.json` (per-class recalls, confusion matrix, per-lesion
AUROC/PR-AUC), `train_retfound_cfp_ft512.csv` (30-epoch curve; val accuracy 0.5915 → 0.8193, κ 0.0763 → 0.9014),
and `../../figures/retfound_ft512_curves.png` / `retfound_ft512_test_breakdown.png`.
