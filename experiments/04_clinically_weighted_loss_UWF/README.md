# Experiment IV — Clinically-weighted lesion loss (UWF)

**Question.** Does weighting the lesion BCE by clinical rarity recover recall on the rare, sight-threatening
lesions?

**Method.** `MultiScaleLesionAttentionNet` (MS-LAN) with class-balanced focal loss on the grade head, and a
`pos_weight` vector on the lesion head derived from inverse frequency and clinical risk:

| MA | HE | IH | CWS | NV | VH | RD |
|---|---|---|---|---|---|---|
| 0.0330 | 0.0596 | 0.1190 | 0.4536 | 0.7712 | 0.7516 | **4.8120** |

A 145× spread between MA and RD. 512 px, batch 8, max 50 epochs, patience 8; 8,323 train / 2,081 test.

**Results** (held-out test, n=2,081; early-stopped at epoch 22/50):

- Grade accuracy **76.50 %**, macro-F1 0.75. Per-class F1: 0 → 0.86, 1 → 0.58, 2 → 0.71, 3 → 0.68, 4 → **0.90**
- Lesion macro-F1 0.56, micro-F1 0.64, macro precision 0.85, macro recall 0.49
- Per-lesion F1: MA 0.83, HE 0.19, IH 0.52, CWS 0.07, NV 0.76, VH 0.86, **RD 0.70 (recall 0.79)**

**Reading.** The heavy `pos_weight` on RD demonstrably worked — RD has the highest recall in the table
despite being the rarest class. But the mid-frequency lesions paid for it: HE collapsed to recall 0.10 and
CWS to 0.04. Rebalancing moved the failure rather than removing it, which is what motivates the asymmetric
loss in Experiment V.
