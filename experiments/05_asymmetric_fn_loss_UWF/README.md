# Experiment V — Asymmetric false-negative-penalised loss for PDR (UWF)

**Question.** A missed neovascularisation is far more costly than a false alarm. Can the loss encode that
asymmetry directly, so the model fails safe?

**Method.** `AsymmetricFocalLoss` (γ=2) built on `BCEWithLogits` for numerical stability, with a per-lesion
β vector `[0.7, 0.7, 0.7, 0.8, 0.95, 0.95, 0.95]` — β = 0.95 on the three PDR-defining lesions (NV, VH, RD),
so false negatives on those classes dominate the gradient. Paired with `UWF_MultiScaleDRNet`
(ResNet-50 + FPN + spatial attention) and elliptical UWF masking to strip eyelid/eyelash artefacts.
100 epochs planned, patience 15, Kaggle-resumable checkpointing.

**Loss sanity check** (printed in the notebook): on NV, the false-negative penalty is **0.2536** against a
false-positive penalty of **0.0415** — a ~6.1× asymmetry, exactly as designed.

**Results** (best checkpoint by validation loss, epoch 16; early-stopped at 31/100; val n=2,081):

| Metric | Value |
|---|---|
| Grade accuracy | 0.7578 (peak seen in run: 0.7770) |
| Grade quadratic κ | **0.9208** |
| Lesion macro-F1 | 0.7151 |
| Lesion macro-AUROC | 0.9504 |

Grad-CAM plus weakly-supervised bounding-box extraction ran and produced the localisation figures.

**Caveat.** No per-lesion recall breakdown was printed in this run, so the specific claim "the FN penalty
raised NV/VH/RD recall" is *not* numerically substantiated here — only the aggregate lesion F1 and AUROC are.
Experiment VII supplies that missing breakdown.
