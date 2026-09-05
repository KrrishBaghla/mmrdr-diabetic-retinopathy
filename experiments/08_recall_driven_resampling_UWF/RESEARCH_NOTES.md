# RESEARCH NOTES — Recall-Driven Curriculum Resampling (RDCR)

This is the "why it works" document: the literature it stands on, the exact math, the design decisions, and
the open questions. A new researcher should be able to read only this file and understand the method end to end.

---

## Part A — Literature review

For each paper: what it contributes, why it is relevant to us, how well it fits, implementation complexity,
compute cost, and expected gain if we borrow the idea. Full citations in Part E.

### A1. Bengio, Louradour, Collobert & Weston (2009) — *Curriculum Learning* (ICML)
- **Contribution.** Formalizes the idea that presenting training examples in a meaningful order (easy→hard)
  improves both convergence speed and, for non-convex objectives, the quality of the minimum reached. Frames
  curriculum learning as a continuation method for global optimization.
- **Relevance.** This is the theoretical license for our *schedule*: we do not want to hammer the rarest,
  hardest RD images from step 0. We ramp difficulty (here, "difficulty" = how aggressively we chase the tail).
- **Fit.** High for the schedule idea; the original paper orders by *example difficulty*, we order by
  *class-balancing pressure over time*, which is a curriculum over the sampling distribution rather than over
  individual examples. We cite it as motivation, not as a drop-in.
- **Complexity / compute.** Trivial (a scalar schedule). No extra compute.
- **Expected gain.** Stability and better late-training tail focus.

### A2. Kumar, Packer & Koller (2010) — *Self-Paced Learning for Latent Variable Models* (NeurIPS)
- **Contribution.** Makes the curriculum *self-paced*: instead of a human ordering examples, the model itself
  decides which examples are "easy enough" to include, jointly optimizing the model and the sample selection.
- **Relevance.** RDCR is self-paced in spirit — the model's own validation recall decides which classes need
  more sampling. The "pace" is set by data, not by a hand-built curriculum.
- **Fit.** Conceptual. We borrow the *feedback-from-the-model* principle but apply it at the class level
  (recall per class) rather than the per-example loss level.
- **Complexity / compute.** Low. Compute is one validation pass we already run.
- **Expected gain.** Adaptivity — the sampler tracks the model instead of a fixed prior.

### A3. Shrivastava, Gupta & Girshick (2016) — *Training Region-Based Object Detectors with OHEM* (CVPR)
- **Contribution.** Online Hard Example Mining: within each SGD batch, sort candidate regions by loss and
  backprop only the hardest. Datasets are mostly easy examples; focusing on hard ones lifts detection mAP.
- **Relevance.** The closest ancestor to "focus on what the model gets wrong." But OHEM mines *hard examples*
  by loss; we mine *hard classes* by recall.
- **Fit.** Partial and instructive. Pure OHEM on our data is dangerous: the hardest-loss samples under extreme
  imbalance are often noisy/ambiguous negatives, and per-example hard-mining can destabilize BatchNorm. We
  deliberately choose *class-level recall* as a more stable, clinically interpretable signal, and note OHEM as
  the per-example alternative in the ablation discussion.
- **Complexity / compute.** OHEM proper needs per-sample loss bookkeeping each step; our class-level variant
  is cheaper (once per epoch).
- **Expected gain.** Motivates the "chase the errors" core; we take the idea, not the per-example mechanism.

### A4. Lin, Goyal, Girshick, He & Dollár (2017) — *Focal Loss for Dense Object Detection* (ICCV)
- **Contribution.** Reshapes cross-entropy with a `(1-p_t)^γ` factor that down-weights easy examples, letting
  a dense detector train through massive foreground/background imbalance without sampling heuristics.
- **Relevance.** Focal loss is the *re-weighting* cousin of resampling and is **already in our stack**
  (`AsymmetricFocalLoss`, folder 2/3). RDCR keeps it. The pairing is intentional: focal loss handles
  within-batch easy/hard weighting; the sampler handles *which images enter the batch at all*.
- **Fit.** Direct — we reuse it unchanged.
- **Complexity / compute.** Already implemented. Zero added cost.
- **Expected gain.** Baseline already benefits; RDCR is additive on top.

### A5. Cui, Jia, Lin, Song & Belongie (2019) — *Class-Balanced Loss Based on Effective Number of Samples* (CVPR)
- **Contribution.** Argues raw inverse-frequency over-weights the tail because samples overlap in feature
  space. Defines the **effective number** `E_n = (1 - β^n)/(1 - β)` and re-weights by `(1-β)/(1-β^n)`.
- **Relevance.** We use the effective-number weighting as the **static rarity prior** floor in RDCR's
  per-class importance, so that even when a rare class's recall momentarily spikes, it keeps a minimum
  sampling priority. Also the basis of the **static class-balanced baseline** we ablate against.
- **Fit.** High. Directly plugged in as the `β`-prior term.
- **Complexity / compute.** One line. Negligible.
- **Expected gain.** Better-calibrated tail weighting than raw `1/n`; prevents the deficit signal from ever
  fully abandoning a rare class.

### A6. Cao, Wei, Gaidon, Arechiga & Ma (2019) — *LDAM-DRW* (NeurIPS)
- **Contribution.** (i) A label-distribution-aware margin loss giving rare classes larger margins; (ii)
  **Deferred Re-Weighting (DRW)** — train with normal (instance-balanced) loss first, then switch on
  class-balancing only in a later stage. DRW alone gives most of the benefit.
- **Relevance.** This is the single most important paper for our *schedule design*. It empirically proves that
  balancing **early hurts** (it damages representation learning) and balancing **late helps**. RDCR's warmup
  (τ=0, instance-balanced) then ramp (τ→1, recall-driven) is a *continuous, recall-conditioned* generalization
  of DRW's two-stage switch.
- **Fit.** Very high (schedule); we adopt LDAM as an optional grade-head loss in future work.
- **Complexity / compute.** DRW schedule is trivial; LDAM margin loss is moderate. We adopt only the schedule now.
- **Expected gain.** Large — the warmup is expected to be the difference between a stable run and a sampler that
  overfits the tail from epoch 1.

### A7. Shu, Xie, Yi, Zhao, Zhou, Xu & Meng (2019) — *Meta-Weight-Net* (NeurIPS)
- **Contribution.** Learns the sample-weighting function itself as a tiny MLP (loss → weight) via a
  meta-learning bilevel objective on a small balanced meta-set.
- **Relevance.** The fully-learned, "optimal" end of the spectrum RDCR lives on. We deliberately do **not**
  use it: bilevel meta-optimization roughly doubles compute (an inner+outer loop and a second-order gradient),
  which is infeasible on a Kaggle 2×T4 12-hour budget with 512×512 images and a ResNet-50+FPN.
- **Fit.** Low for now (compute); high as a "future work" north star.
- **Complexity / compute.** High (bilevel, second-order). ~2× train time.
- **Expected gain.** Potentially the best, but not worth the compute risk here. Documented as future work.

### A8. Kang, Xie, Rohrbach, Yan, Gordo, Feng & Kalantidis (2020) — *Decoupling Representation and Classifier* (ICLR)
- **Contribution.** Shows that **instance-balanced sampling learns the best representations**, and that
  long-tailed performance is mostly recovered by re-balancing only the *classifier* afterward (cRT / τ-norm).
- **Relevance.** Two direct consequences for RDCR: (1) justifies starting the curriculum at instance-balanced
  (τ=0) so the ResNet-50 trunk forms good features; (2) suggests a strong *future* variant — freeze the trunk
  after warmup and let RDCR re-balance mainly the heads. For now we keep it simple (whole network trains) but
  the warmup already captures the key insight.
- **Fit.** High (warmup rationale).
- **Complexity / compute.** The decoupled two-stage variant is a modest add; we note it as an ablation/future.
- **Expected gain.** Confirms warmup is not optional; it is the mechanism that protects representation quality.

### A9. Wu, Huang, Liu, Wang & Lin (2020) — *Distribution-Balanced Loss for Multi-Label Long-Tailed* (ECCV)
- **Contribution.** Purpose-built for **multi-label** long-tail. Two ideas: (i) **rebalanced weighting** that
  corrects for the gap between the sampling probability a label *expects* under class-balanced sampling and the
  probability it actually gets (because one image carries several labels); (ii) **negative-tolerant
  regularization** to stop the flood of negative labels from over-suppressing rare positives.
- **Relevance.** This is *the* paper on the exact subtlety we face: our lesion task is multi-label, so
  oversampling for one rare lesion drags along whatever else co-occurs (NV, VH, RD co-occur in PDR). Wu et al.
  formalize why naive per-label class-balanced sampling is biased in multi-label settings and how to correct it.
  We borrow their diagnosis to justify our **max-over-classes** per-sample weight (see Part B), which avoids the
  double-counting their rebalanced weight is designed to fix.
- **Fit.** High conceptually; we adopt the *reasoning*, and their negative-tolerant idea overlaps with what our
  `AsymmetricFocalLoss` already does (down-weighting the easy negative term).
- **Complexity / compute.** Their full loss is moderate; we take the sampling insight, keep our existing loss.
- **Expected gain.** Prevents the multi-label sampler from silently over-weighting PDR-co-occurring images.

### Synthesis
The field converges on three lessons: **(1)** balance the tail (Cui, Cao, Kang); **(2)** but not too early
(Cao-DRW, Kang) — representations first; **(3)** focus on what the model currently fails (OHEM, self-paced,
focal). No prior work, to our knowledge, combines an **online per-class-recall feedback signal** with a
**curriculum ramp** for **multi-label + ordinal medical imaging**. That intersection is RDCR's novelty.

---

## Part B — Mathematical formulation

### B0. Setup and notation
- Training set of `N` images. Image `i` has an ordinal grade `g_i ∈ {0,…,4}` (single-label) and a multi-hot
  lesion vector `y_i ∈ {0,1}^7` over `[MA, HE, IH, VB, NV, VH, RD]`.
- We define a unified set of **K = 12 rebalancing channels**: 5 grade channels + 7 lesion channels.
- `n_k` = number of training samples positive for channel `k` (for grade `g`, count of images with that grade;
  for lesion `c`, count of positive images). These come straight from SHARED_CONTEXT §1.
- The **membership set** of image `i` is `C(i) = { grade channel g_i } ∪ { lesion channels c : y_{i,c}=1 }`.
  A grade-0 image with no lesions has `C(i) = {grade-0 channel}` only.

### B1. Per-class recall and the deficit signal
After the validation pass at the end of epoch `t`, compute per-channel recall `R_k^{(t)} ∈ [0,1]`:
- Grade channel `g`: `R_g = TP_g / (TP_g + FN_g)` from the 5-class confusion matrix (one-vs-rest recall).
- Lesion channel `c`: recall of the positive class of binary head `c` at decision threshold 0.5:
  `R_c = TP_c / (TP_c + FN_c)`.

Define the **class deficit** — how much the model is still failing this class:
```
d_k^{(t)} = 1 - R_k^{(t)}        ∈ [0, 1]
```
Recall 1.0 → deficit 0 (mastered); recall 0.0 → deficit 1 (blind).

### B2. EMA smoothing of the deficit (stability)
Raw per-epoch recall is noisy (small rare-class support: RD has only ~48 val positives at a 20% split).
Smooth with an exponential moving average, momentum `m ∈ [0,1)`:
```
D_k^{(t)} = m · D_k^{(t-1)} + (1 - m) · d_k^{(t)}
```
Initialize `D_k^{(0)} = 1` for all `k` (assume everything is maximally deficient before we have evidence). With
`m = 0.7`, the sampler reacts over ~3 epochs, not 1 — damping oscillation.

### B3. Per-class importance (rarity prior × dynamic deficit)
Blend a **static** effective-number rarity prior (Cui 2019) with the **dynamic** deficit:
```
                 ┌ (1 - β) / (1 - β^{n_k}) ┐^a     ┌            ┐^b
   I_k^{(t)}  =  │  ─────────────────────  │   ·   │ D_k^{(t)}+ε│
                 └   rarity prior (static) ┘        └  deficit   ┘
```
- `β ∈ [0,1)` is the effective-number hyperparameter (e.g. 0.999). The prior is normalized to mean 1 across `k`.
- `a` weights the static rarity floor, `b` weights the dynamic recall signal, `ε` (=1e-3) keeps a mastered
  class from hitting exactly zero importance.
- **Special cases used as ablations:** `a=1, b=0` → static class-balanced (Cui) baseline sampler;
  `a=0, b=1` → pure recall-driven; default `a=0.5, b=1.0` → recall-driven with a rarity floor.

### B4. Per-sample weight — resolving the multi-label subtlety
An image carries several channels; we must collapse `{I_k : k ∈ C(i)}` into one scalar `s_i`. Options:
```
   max :  s_i = max_{k ∈ C(i)} I_k^{(t)}          (DEFAULT)
   sum :  s_i = Σ_{k ∈ C(i)} I_k^{(t)}            (ablation)
   mean:  s_i = mean_{k ∈ C(i)} I_k^{(t)}         (ablation)
```
**Why `max` is the default.** Wu et al. (2020) show that in multi-label long-tail, summing per-label
importances double-counts images that carry several rare labels — and in our data NV, VH, RD strongly
co-occur (PDR), so `sum` would explode the weight of a handful of PDR images and overfit them. `max` says
"sample this image for the single rarest/most-needed signal it can teach," which is stable and still fully
covers PDR images (their max is the RD/NV importance anyway). `sum` is kept as an ablation because it *may*
help precisely because PDR images are the target — we let the experiment decide.

### B5. Curriculum temperature (the schedule)
We interpolate between uniform sampling and full importance-driven sampling with a temperature `τ(t) ∈ [0,1]`:
```
              ┌ 0                                    if t <  t_warm
   τ(t)  =  ─┤ clamp( (t - t_warm)/t_ramp , 0, 1 )   if t ≥ t_warm
              └
```
Apply it as an exponent on the sample weight:
```
   ŝ_i^{(t)} = ( s_i^{(t)} )^{ τ(t) }
```
- At `τ=0`: `ŝ_i = 1` for all `i` → **instance-balanced sampling** (Kang 2020's best-for-representations regime,
  Cao 2019's DRW pre-switch stage).
- At `τ=1`: full recall-driven importance sampling.
- In between: a smooth power-law ramp — no discontinuity, unlike DRW's hard switch.

Defaults: `t_warm = 5` epochs, `t_ramp = 15` epochs (fully recall-driven by epoch 20).

### B6. Clamping and normalization (stop any class from dominating)
Cap how extreme any single sample's weight can get, then normalize to a probability distribution:
```
   ŝ_i  ←  clip( ŝ_i , 1/ρ_max , ρ_max )        (ratio clamp, e.g. ρ_max = 20)
   p_i   =  ŝ_i / Σ_j ŝ_j
```
`ρ_max = 20` means no image is ever more than 20× likelier to be drawn than the mean — this bounds how many
times the ~48 RD training images can be repeated per epoch, which (together with strong augmentation) is our
main defense against oversampling-induced overfitting.

### B7. The sampler
Build a fresh `torch.utils.data.WeightedRandomSampler(weights = ŝ, num_samples = N, replacement = True)` at the
**start of every epoch** using the weights from the previous epoch's validation. `num_samples = N` keeps epoch
length constant (fair comparison to the baseline); `replacement=True` is required for oversampling.

### B8. End-to-end per-epoch update (pseudocode)
```
for t in epochs:
    sampler = WeightedRandomSampler(weights_t, N, replacement=True)   # from epoch t-1 recall
    train_one_epoch(model, DataLoader(train, sampler=sampler, ...))   # AMP, grad-clip, asym loss + CE + SupCon
    R_t   = validate_per_class_recall(model, val_loader)             # 12-vector
    d_t   = 1 - R_t
    D_t   = m*D_{t-1} + (1-m)*d_t                                    # EMA
    I_t   = (prior**a) * (D_t + eps)**b                              # per-class importance
    s     = collapse_per_sample(I_t, memberships, reduce='max')     # per-sample
    tau   = curriculum_tau(t)                                        # schedule
    s_hat = clip(s**tau, 1/rho_max, rho_max)
    weights_{t+1} = s_hat
```
Cost over the baseline: one 12-vector EMA update + one `O(N)` weight computation per epoch. **Negligible.**

---

## Part C — Design decisions & assumptions

- **Signal = validation recall, not training recall.** Training recall is optimistic (the model has seen those
  exact images); validation recall measures true generalization deficit. Assumes the val split is
  representative — hence the docs *require* a stratified split on the rare-lesion indicator.
- **Class-level, not example-level (vs OHEM/Meta-Weight-Net).** Deliberate: class-level recall is stable,
  interpretable, cheap, and clinically meaningful. Example-level hard mining is noisy under extreme imbalance
  and BatchNorm-hostile. Trade-off: we cannot up-weight a *specific* hard image, only its classes.
- **`max` reduction for multi-label.** Justified by Wu et al. (2020) — avoids double-counting co-occurring PDR
  lesions. Assumption: the rarest channel an image carries is the most informative reason to sample it.
- **Curriculum as a power on the weight**, not a linear blend of distributions — keeps the transformation
  monotone and numerically gentle; `s^0 = 1` gives exact uniformity at warmup with no special-casing.
- **EMA over deficit, not over recall directly** — equivalent up to the constant, but framing on deficit makes
  the "mastered → 0 pressure" behavior explicit.
- **Whole network trains throughout** (not decoupled cRT). Simpler, one-stage, Kaggle-friendly. Kang-style
  decoupling logged as future work.
- **Assumption on stability:** with `m=0.7`, `ρ_max=20`, `t_warm=5`, the sampler is a slow controller and will
  not enter a feedback loop where oversampling a class raises its recall, which drops its weight, which drops
  its recall — the EMA and warmup damp exactly this. We monitor per-class recall curves to confirm.

---

## Part D — Observations, future ideas, open questions

- **Open Q1:** Should grade channels and lesion channels share one importance vector, or be balanced
  separately? We currently pool all 12; a variant normalizes the 5 grade priors and 7 lesion priors within
  their own groups so the (larger, single-label) grade task doesn't dominate the (multi-label) lesion task.
  Logged as an ablation.
- **Open Q2:** Threshold sensitivity — recall at a fixed 0.5 threshold couples the signal to calibration. A
  robust variant uses recall at a fixed operating point (e.g. 95% specificity) so the sampler tracks
  discriminability, not threshold placement.
- **Future 1:** Replace the hand-set `(a,b,m,ρ_max)` with a learned Meta-Weight-Net once compute allows.
- **Future 2:** Kang-style decoupling — warmup with instance-balanced sampling, then freeze the trunk and let
  RDCR rebalance only the heads (cheaper, possibly cleaner representations).
- **Future 3:** Combine RDCR with class-aware MixUp/CutMix so the repeated rare images are always seen in fresh
  mixtures — directly counters oversampling overfitting.
- **Observation to verify in training:** we expect the RD deficit curve to start near 1.0, stay high through
  warmup, then fall as the ramp engages — the shape of that curve is the primary evidence the controller works.

---

## Part E — References (proper citations)

1. Y. Bengio, J. Louradour, R. Collobert, J. Weston. **Curriculum Learning.** ICML 2009, pp. 41–48.
2. M. P. Kumar, B. Packer, D. Koller. **Self-Paced Learning for Latent Variable Models.** NeurIPS 2010,
   pp. 1189–1197.
3. A. Shrivastava, A. Gupta, R. Girshick. **Training Region-Based Object Detectors with Online Hard Example
   Mining (OHEM).** CVPR 2016, pp. 761–769.
4. T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár. **Focal Loss for Dense Object Detection.** ICCV 2017,
   pp. 2999–3007.
5. Y. Cui, M. Jia, T.-Y. Lin, Y. Song, S. Belongie. **Class-Balanced Loss Based on Effective Number of
   Samples.** CVPR 2019, pp. 9268–9277.
6. K. Cao, C. Wei, A. Gaidon, N. Arechiga, T. Ma. **Learning Imbalanced Datasets with Label-Distribution-Aware
   Margin Loss (LDAM-DRW).** NeurIPS 2019.
7. J. Shu, Q. Xie, L. Yi, Q. Zhao, S. Zhou, Z. Xu, D. Meng. **Meta-Weight-Net: Learning an Explicit Mapping for
   Sample Weighting.** NeurIPS 2019, pp. 1917–1928.
8. B. Kang, S. Xie, M. Rohrbach, Z. Yan, A. Gordo, J. Feng, Y. Kalantidis. **Decoupling Representation and
   Classifier for Long-Tailed Recognition.** ICLR 2020.
9. T. Wu, Q. Huang, Z. Liu, Y. Wang, D. Lin. **Distribution-Balanced Loss for Multi-Label Classification in
   Long-Tailed Datasets.** ECCV 2020 (Spotlight).

Dataset paper: *A multimodal retinal image dataset for diabetic retinopathy detection using foundation models*
(MMRDR), the basis of Paper 2.
