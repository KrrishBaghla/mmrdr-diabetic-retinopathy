# Uncertainty-Based Multi-Task Loss Weighting (Paper 2, MMRDR)

> **Status:** Planning + documentation + one Kaggle-ready notebook. **No models trained yet.**
> This folder is self-contained. It does **not** modify any existing experiment folder.

---

## 1. What this is (in one sentence)

We replace the **hand-picked, fixed loss weights** used in the mature folder-3 experiment
(`loss = asym_lesion + CE_grade + 0.5*supcon_early + 0.5*supcon_deep`) with **learned, per-task
weights** derived from *homoscedastic task uncertainty* (Kendall, Gal & Cipolla, CVPR 2018), so the
network itself decides how much each task should count instead of a human guessing the coefficients.

## 2. Why this matters for THIS project

The repo is a multi-task diabetic-retinopathy (DR) model on the MMRDR **UWF** modality. Every image
carries two supervised labels:

- **DR grade** — 5-class ordinal severity (0–4).
- **Lesion detection** — 7-way multi-label (`[MA, HE, IH, VB/IRMA, NV, VH, RD]`); indices 4–6 (NV, VH,
  RD) define sight-threatening **Proliferative DR (PDR)**.

Plus two **auxiliary** supervised-contrastive (SupCon) terms that shape the embedding space so
rare-lesion images cluster together.

These four losses have **different units and scales**: cross-entropy over 5 classes, an asymmetric
focal loss over 7 sigmoid outputs, and two contrastive losses on L2-normalised embeddings. Adding them
up requires coefficients. Today those coefficients are **guesses** (`3:1` in MS-LAN, `1:1:0.5:0.5` in
folder 3). Guessing has three concrete failure modes here:

1. **The grade task can swamp the lesion task** (or vice-versa), quietly sacrificing rare-lesion recall
   — which is exactly the clinical-safety metric this project cares about most.
2. **Every new modality / architecture needs the weights re-tuned by hand**, and a proper grid search
   over 4 weights is exponentially expensive on Kaggle's 12-hour GPU budget.
3. The chosen weights are never justified in the write-up, which is a soft spot for a viva/paper.

Uncertainty weighting attacks all three: it turns the coefficients into **learnable parameters trained
by backprop**, at the cost of *one extra scalar per task* (four numbers total).

## 3. The core idea (intuition, not yet math — see RESEARCH_NOTES.md for the derivation)

Model each task's output as a noisy observation with its own **homoscedastic noise level** σ (a property
of the task, not of any single image). A noisy/hard task gets a **larger σ**, and the maths that falls
out of the Gaussian/softmax likelihood automatically **down-weights** that task by `1/σ²` while adding a
`log σ` penalty that **stops σ from running away to infinity** (which would zero-out the task). We learn
`s = log σ²` (log-variance) instead of σ directly, purely for numerical stability, and combine losses as

```
L_total = Σ_i [ 0.5 · exp(-s_i) · L_i  +  0.5 · s_i ]
```

`exp(-s_i)` is the learned weight; `0.5·s_i` is the regulariser that prevents weight collapse. The
optimum is `s_i* = log L_i`, i.e. the weight self-adjusts to `~0.5 / L_i` — noisier (higher-loss) tasks
are trusted less, automatically, without a grid search.

## 4. Why we selected this method (over the alternatives)

| Method | Idea | Fit for our 2–4 task setup | Verdict |
|---|---|---|---|
| **Uncertainty weighting** (Kendall 2018) | learn `log σ²` per task | tiny (4 params), no extra backward pass, drops straight into the existing loop | **PRIMARY** |
| GradNorm (Chen 2018) | balance gradient *norms* to a target rate | needs a per-task backward to read grad norms of a shared layer → slower on T4 | ablation baseline |
| PCGrad (Yu 2020) | project conflicting task gradients | per-task gradients + projection each step, most expensive | ablation baseline |
| DWA (Liu 2019, MTAN) | weight ∝ recent loss-ratio | cheap but heuristic, no built-in anti-collapse term | mention only |
| Manual grid search | try coefficient combinations | exponential in #tasks, burns the GPU budget | the thing we replace |

Uncertainty weighting is the best **effort-to-payoff** choice: near-zero compute overhead, principled,
and it slots into the folder-3 training loop with one new module and one new optimizer param group.
GradNorm and PCGrad are kept as **comparison ablations** to show our learned weights are competitive with
heavier gradient-balancing methods.

## 5. Architecture we build on (and why)

We reuse **`UWF_SupConDRNet`** verbatim from `contrastive_learning_lesions_3/` (ResNet-50 + FPN +
two SupCon projection heads → returns `out_lesions, out_grade, z_early, z_deep`). It is the natural host
because **it already produces ≥3 loss terms whose weights are currently guessed** — the exact problem
this method solves. We reject alternatives: RETFound (ViT-Large) is too heavy for a dual-T4 Kaggle
session; plain ViT/Swin are data-hungry at 512px with tiny lesions; EfficientNet/ConvNeXt are viable but
break repo continuity and the fair comparison against folder-3's real numbers. See IMPLEMENTATION_GUIDE.md.

**The only architectural change** is a small standalone module, `UncertaintyWeightedLoss`, holding one
`nn.Parameter` of shape `[n_tasks]` (the log-variances). It lives **outside** the DataParallel-wrapped
model so its parameters stay on a single device.

## 6. Existing limitations this addresses

- Fixed `1:1:0.5:0.5` weighting in folder 3 is unjustified and untuned per modality.
- The known `alternate_architechture.py` weight bug (`3*loss_g + loss_l` with wrong class counts) is a
  symptom of manual weighting being fragile; learned weights remove one class of such bugs.
- No principled way today to trade grade accuracy against rare-lesion recall.

## 7. Expected benefits

- **Match or beat** folder-3's UWF numbers (Val: Grade Acc 76.3%, Kappa 0.922, Lesion macro-F1 0.738,
  macro-AUC 0.951) **while improving per-rare-class (NV/VH/RD) recall**, which is the headline claim.
- A **learned weight trajectory** we can plot and defend ("the model down-weighted grade after epoch N
  because lesion detection was the bottleneck").
- **No grid search** → cheaper experimentation.

## 8. How it integrates (one-paragraph map)

`UWF_SupConDRNet(model)` → forward gives the 4 task losses → stack them into a length-4 tensor →
`UncertaintyWeightedLoss(losses)` returns `(total_loss, current_weights)` → `total_loss.backward()`
updates both the network **and** the 4 log-variances (they are in a **separate optimizer param group
with `weight_decay=0`**). Each epoch we log the 4 weights to CSV and to the checkpoint. Nothing else in
the folder-3 loop changes.

## 9. Success criteria

1. Notebook runs top-to-bottom unattended on Kaggle (single-GPU **and** dual-T4), resumes from
   checkpoint, and writes a CSV that includes the learned task weights per epoch.
2. The unit test proves the loss down-weights a high-loss synthetic task **without** driving its weight
   to zero (the `log σ` term does its job).
3. On real UWF training (future, gated behind a config flag): learned method ≥ folder-3 baseline on
   quadratic-weighted Kappa **and** lesion macro-PR-AUC, with **higher NV/VH/RD recall**, averaged over
   ≥3 seeds with significance testing.

## 10. Deliverables in this folder

| File | Purpose |
|---|---|
| `README.md` | this file — what/why/selection/benefits |
| `PLAN.md` | milestones, roadmap, files, validation, schedule, risks |
| `RESEARCH_NOTES.md` | literature review + full math derivation + design decisions |
| `IMPLEMENTATION_GUIDE.md` | architecture, data flow, hyperparameters, debugging |
| `STATUS.md` | Completed / In Progress / Pending / Blockers / Next |
| `create_nb.py` | generator that emits the Kaggle notebook (repo convention) |
| `Uncertainty_Weighted_MTL_UWF.ipynb` | the 20-section Kaggle-ready notebook (generated) |

## 11. Honest novelty note

Uncertainty weighting is an **established** method (2018). The novelty here is **not** the weighting rule
itself; it is the **combination**: applying learned homoscedastic weighting to a *four-term* DR
objective that mixes an asymmetric focal lesion loss, ordinal grade CE, and two SupCon auxiliaries, under
an explicit **clinical-safety framing** (does learned balancing protect rare-PDR recall better than
manual weights?). Publishable as an applied/clinical MTL study, not as a new optimisation method. See
RESEARCH_NOTES.md §Novelty.
