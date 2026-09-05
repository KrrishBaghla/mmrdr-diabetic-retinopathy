# Experiment VII — Learned uncertainty task weighting (UWF)

**Question.** The multi-task loss weights up to this point were a hand-picked guess (`1:1:0.5:0.5`). Can they
be *learned* instead, so the rare-lesion task is not drowned out by the grade task?

**Method.** Homoscedastic uncertainty weighting (Kendall, Gal & Cipolla 2018): each task contributes
`0.5·exp(−s)·L + 0.5·s` with a learnable log-variance `s`. Four tasks (grade, lesion, and both SupCon
branches). The log-vars live in their own AdamW parameter group with no weight decay and a higher LR
(1e-3 vs 1e-4), and `s` is clamped to ±3. Everything else is the Experiment VI pipeline unchanged.

**Results** (best values across the run; full trace in `uncertainty_training_log.csv`):

| Metric | Value | Exp VI baseline |
|---|---|---|
| Grade accuracy | 0.7645 | 0.7645 |
| Grade quadratic κ | 0.9211 | 0.9220 |
| Lesion macro-F1 | **0.7779** | 0.7312 |
| Lesion macro-AUROC | **0.9605** | 0.9521 |
| Lesion PR-AUC | **0.8395** | — |

Rare-lesion recall on validation: **NV 0.891** (26 FN / 238 pos), **VH 0.947** (12 FN / 226), **RD 0.958**
(2 FN / 48). This is the best lesion detection anywhere in the repo.

**Learned weights.** `w_grade` climbed 0.737 → 1.176 over training. The SupCon weights settled near 0.20–0.22.

**Caveats (also in `REPORT.md`).** `w_lesion` pinned to its clamp ceiling (10.04, i.e. `s = −3.0`) for the
*entire* run — so for the task that matters most, the adaptive mechanism saturated and never actually adapted.
The gain is real, but the intended explanation for it is not supported. Also: mild overfitting (val loss flat
while train loss fell), the "best" checkpoint was selected on val loss rather than κ or F1, and the run was
cut short by a session timeout at epoch 30.

`NOTES_README.md` and `RESEARCH_NOTES.md` are pre-run planning documents. `REPORT.md` is the post-run writeup.
