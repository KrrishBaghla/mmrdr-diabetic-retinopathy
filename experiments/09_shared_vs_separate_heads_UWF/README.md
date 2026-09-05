# Experiment IX — Shared vs separate lesion heads (UWF) — a negative result

**Question.** It seems obvious that the rare PDR lesions (NV, VH, RD) deserve their own classifier head
rather than sharing one with the four common lesions. Is that true *once head capacity is held constant*?

Most naive versions of this comparison are confounded: a separate head simply has more parameters. The
design here breaks that confound.

**Method.** `MultiHeadDRNet` — the Experiment VI trunk (ResNet-50 + FPN + SupCon heads, **26,921,797 params,
byte-identical across all variants**) with only `classifier_lesions` swapped:

| Variant | `head_mode` | Head | Params |
|---|---|---|---|
| A | `shared` | `Linear(512, 7)` | 3,591 |
| A′ | `shared_mlp` | `Linear(512,128) → ReLU → Drop → Linear(128,7)` | 66,567 |
| B | `separate` | `Linear(512,4)` common + `MLP(512→128→3)` rare | 68,103 |
| C | `adapter` | shared body + per-group adapters | 74,631 (implemented, not run) |

A′ ≈ B in head budget, so **B − A′ isolates separation** and **A′ − A isolates capacity**.

**Results** (held-out test, frozen `da1ec8e21f22` split, seed 42, 20-epoch budget):

| Variant | Head params | Mean rare recall | Lesion macro-F1 | Grade κ |
|---|---|---|---|---|
| A (shared linear) | 3,591 | 0.884 | **0.707** | 0.914 |
| A′ (shared MLP) | 66,567 | **0.890** | 0.698 | **0.915** |
| B (separate) | 68,103 | 0.816 | 0.689 | 0.886 |

**Decomposition:** capacity (A′ − A) = **+0.006**; separation (B − A′) = **−0.074**.

**Reading.** The dedicated rare head is clearly *worse*, and the small gain from A to A′ shows the effect
that does exist is about capacity, not separation. The rare-lesion bottleneck lives in the shared
representation, not in the head topology — which retrospectively explains why Experiments VII and VIII, both of
which act on the representation and the data exposure, did better than this one.

**Caveats.** Training budgets are not perfectly matched (A and A′ resumed from checkpoints; B trained fresh
and early-stopped at epoch 14), and the pre-registered 60-epoch × 3-seed protocol with paired Wilcoxon
testing was never run. Treat the direction as informative and the magnitude as provisional.

**Contents.** `head_ablation_run.ipynb` is the executed run. `src/` is the modular implementation the
notebook is generated from — `config.py` + `configs/*.yaml` as a single source of truth, `models/`,
`training/`, `evaluation/`, a `run_ablation.py` CLI, and `tests.py` (6 CPU unit tests verifying forward
shapes, exact head parameter counts, identical trunk size, rare-logit gradient routing, loss reduction
consistency, and that tuned thresholds beat a fixed 0.5).
