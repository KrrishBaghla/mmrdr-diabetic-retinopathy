# Dynamic / Curriculum-Based Resampling for Rare-Lesion Diabetic Retinopathy

**Paper 2 — MMRDR dataset, UWF modality**
**Method name:** Recall-Driven Curriculum Resampling (RDCR)
**Status:** Planning + documentation + Kaggle-ready notebook complete. Not yet trained on real data.

---

## 1. What this experiment is (in one paragraph)

Instead of fixing the oversampling ratio for rare classes once and freezing it (the usual
`WeightedRandomSampler` with inverse-frequency weights), **RDCR recomputes the sampling weights every
epoch from the model's *current* per-class validation recall.** Classes the model is currently missing
(low recall) are sampled harder next epoch; classes it has already mastered are relaxed. A **curriculum
schedule** starts training near-uniform (instance-balanced) and ramps toward fully recall-driven sampling,
so early epochs build good general features before the sampler starts chasing the tail. The whole thing is
a *sampler* change — the backbone, losses, and evaluation are the mature folder-3 pipeline
(`UWF_SupConDRNet` + `AsymmetricFocalLoss` + grade CE + SupCon), so any measured gain is attributable to
the sampling strategy alone.

## 2. Why this problem matters (the project's core pain)

The MMRDR UWF split is brutally imbalanced on exactly the classes that matter most clinically:

| Rare target | Positives (of 10,404) | Prevalence |
|---|---|---|
| RD (retinal detachment) | 238 | 2.3% |
| VB/IRMA | 1,010 | 9.7% |
| NV (neovascularization) | 1,188 | 11.4% |
| VH (vitreous hemorrhage) | 1,219 | 11.7% |
| Grade 3 | 1,377 | 13.2% |
| Grade 4 | 1,553 | 14.9% |

NV, VH, RD are the three lesions that **define Proliferative DR (PDR)** — the sight-threatening stage. A
false negative here can mean preventable blindness. Under standard training these rare positives contribute
so few gradient steps that the model learns to "predict absent" and still score well on accuracy/macro-AUC.
The folder-3 baseline already reaches 0.951 macro lesion AUC but its **per-rare-class recall** (the metric
that actually protects patients) is not what a fixed sampler optimizes. RDCR targets that gap directly.

## 3. Why *this* method (and why it beats the alternatives we already tried)

- **Fixed inverse-frequency oversampling** (static `WeightedRandomSampler`) over-samples a rare class by a
  constant factor forever, even after the model has learned it — wasting capacity and inviting overfitting
  on the same few RD images. RDCR *backs off* once recall recovers.
- **Loss re-weighting only** (folder 1's clinical-risk weights, folder 2/3's `AsymmetricFocalLoss`) fights
  imbalance in the gradient magnitude but the rare image is still shown rarely, so batch statistics
  (BatchNorm) and the optimizer trajectory stay dominated by the majority. Resampling changes *what the
  model sees*; re-weighting changes *how much each sample counts*. They are complementary — RDCR keeps the
  asymmetric loss and adds the sampler on top.
- **Curriculum/self-paced learning** literature (Bengio 2009; Kumar–Packer–Koller 2010) says *ordering*
  matters. **Decoupling** (Kang 2020) shows instance-balanced sampling learns the best *features* and
  class-balancing helps most *late*, applied to the classifier. **Deferred re-weighting** (Cao 2019, LDAM-DRW)
  confirms: balance late, not early. RDCR's warmup-then-ramp schedule operationalizes all three.
- The method is **architecture-agnostic**, so we freeze the backbone and isolate the sampler as the only
  independent variable — a clean, publishable ablation.

## 4. Existing limitations this addresses

1. **No stratified split** in the repo (`train_test_split(random_state=42)` only) — rare lesions can land
   unevenly. RDCR docs mandate a stratified split on the rare-lesion indicator and report mean±std over seeds.
2. **Static imbalance handling** — every prior experiment uses a fixed weighting. None adapt to training dynamics.
3. **Accuracy/AUC-centric evaluation** hides rare-class failure. We adopt per-rare-class recall as the headline.

## 5. Expected benefits & research intuition

The intuition is a feedback controller: *validation recall is the error signal, the sampler is the actuator.*
If the model is blind to RD this epoch, RD images flood the next epoch's batches until recall climbs, then the
pressure eases automatically. EMA smoothing prevents the controller from oscillating; the curriculum warmup
prevents it from over-steering before features exist. Expected outcome: **higher per-rare-class recall
(especially RD/NV/VH) at equal or slightly better quadratic-weighted Kappa**, with the trade-off being a
possible small dip in majority-class precision (acceptable clinically — a false alarm costs a follow-up, a
miss costs sight).

## 6. How it integrates with the existing repo

- **Backbone/loss/metrics:** reused verbatim from `contrastive_learning_lesions_3/` (`UWF_SupConDRNet`,
  `AsymmetricFocalLoss`, `SupConLoss`, grade `CrossEntropyLoss`, `calculate_metrics`, `save/load_checkpoint`,
  `apply_uwf_mask`, the 512×512 pipeline, AMP, DataParallel, ReduceLROnPlateau).
- **Only new code:** a `RecallDrivenSampler` helper that (a) reads per-class validation recall, (b) updates
  an EMA deficit vector, (c) applies the curriculum temperature, (d) produces per-sample weights, and (e)
  builds a fresh `WeightedRandomSampler` each epoch. Plus a validation pass that returns per-class recall.
- **Nothing in the prior folders is modified.** All new artifacts live only in `research_dynamic_resampling/`.

## 7. Current status

- Literature review, math, and design: **done** (`RESEARCH_NOTES.md`).
- Full plan, roadmap, risks: **done** (`PLAN.md`).
- Implementation guide with hyperparameter table and data/training/inference flow: **done**
  (`IMPLEMENTATION_GUIDE.md`).
- Kaggle notebook generated by `create_nb.py` → `Dynamic_Resampling_UWF.ipynb`, 20 sections, runs
  top-to-bottom, heavy training gated behind a config flag, sampler logic unit-tested on synthetic data:
  **done**.
- Real training run on Kaggle: **pending** (out of scope for this planning task).

See `STATUS.md` for the live checklist.

## 8. Success criteria

| # | Criterion | Target |
|---|---|---|
| 1 | Per-rare-class recall (RD, NV, VH) vs folder-3 baseline | **↑ ≥ 5 absolute points** on the worst rare class, at matched threshold |
| 2 | Grade quadratic-weighted Kappa | **≥ 0.922** (no regression vs baseline) |
| 3 | Macro lesion PR-AUC | ≥ baseline |
| 4 | Stability | No sampler-induced divergence; weight ratio stays clamped; results reproducible mean±std over ≥3 seeds |
| 5 | Ablations confirm mechanism | EMA-on > EMA-off; curriculum-on > curriculum-off; recall-driven > static class-balanced |

If criterion 1 is met without violating 2, the method is a success for this project's clinical-safety goal.
