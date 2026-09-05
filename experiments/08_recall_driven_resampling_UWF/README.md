# Experiment VIII — Recall-driven curriculum resampling (UWF)

**Question.** Fixed oversampling ratios are set once and never revisited. What if the sampler instead
responded to what the model is *currently* getting wrong?

**Method.** `RecallDrivenSampler` — every epoch, each class is re-weighted by its current per-class
validation **recall deficit**, smoothed with an EMA, combined with an effective-number rarity prior, shaped
by a curriculum temperature τ that warms up from 0, and clamped to a maximum repeat factor. Only the sampler
changed; the Experiment VI model and losses are held identical.

The split was also fixed here: stratified on the rare-lesion indicator → **8,323 train / 2,081 val**.

**Sampler unit test** (printed in the notebook): rare classes up-weighted 3.14 vs 0.76 for common, giving a
rare draw fraction of 0.26 against a natural 0.10. τ ran 0.000 (epochs 1–6 warm-up) → 0.800 by epoch 18,
confirming the curriculum actually engaged.

**Results** (18 epochs completed of a planned 60; full trace in `metrics_log.csv`):

| Metric | Epoch 1 | Best in run |
|---|---|---|
| Grade quadratic κ | 0.8518 | **0.9234** (ep 14) |
| Lesion macro-AUROC | 0.9271 | 0.9455 (ep 18) |
| Lesion macro-F1 | 0.6209 | 0.7088 (ep 17) |
| RD recall | 0.8222 | **1.0000** (ep 7 and 8) |

**Reading — inconclusive, leaning positive on the safety metric.** κ 0.9234 vs the 0.9220 baseline is a
match, not a win. RD recall touched 1.000 and stayed in the 0.89–0.98 band mid-run, but NV and VH drifted
slightly *down* (0.96 → 0.90, 0.95 → 0.91). Only 18 of 60 epochs ran, on one seed, with no significance
test — and this split is stratified while the baseline's is not, so the comparison is not strictly like for
like. The folder's own summary calls it "a strong proof-of-concept run, not the final polished result."
