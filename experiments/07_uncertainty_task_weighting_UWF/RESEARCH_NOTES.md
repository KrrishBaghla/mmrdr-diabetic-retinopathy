# Research Notes — Uncertainty-Based Multi-Task Loss Weighting

This document is the technical backbone: a literature review of every method we considered, a full
mathematical derivation of the uncertainty-weighted loss (from the likelihood down to the practical
log-variance form, including gradients), implementation notes, assumptions, design decisions, and future
ideas. Written to be read start-to-finish by someone new to multi-task learning (MTL).

---

## 1. The problem, stated precisely

In MTL we minimise a weighted sum of task losses:

```
L_total(θ, w) = Σ_i  w_i · L_i(θ)
```

`θ` are the shared network weights; `w_i ≥ 0` are task weights. The losses `L_i` have **different units
and magnitudes** (a 5-class cross-entropy sits around 1.0–1.6 at init; an asymmetric focal loss on 7
sigmoids is much smaller; a SupCon loss depends on batch composition). If one `L_i` is numerically 10×
larger, it dominates the gradient and the other tasks barely train — the classic **task-imbalance /
task-interference** problem. Choosing `w_i` by hand (or by grid search) is fragile and expensive. Every
method below is a different automatic answer for `w_i`.

Our concrete objective (folder 3, the mature baseline) is:

```
L = L_lesion(asym focal) + L_grade(CE) + 0.5·L_supcon_early + 0.5·L_supcon_deep
```

with weights fixed at `(1, 1, 0.5, 0.5)` — chosen by hand, never justified.

---

## 2. Literature review (every important paper)

### 2.1 Kendall, Gal & Cipolla (2018) — *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR 2018, pp. 7482–7491. **[PRIMARY METHOD]**
- **Contribution:** derive task weights from **homoscedastic (task-dependent, input-independent)
  uncertainty**. Each task gets a learned noise scalar σ_i; the weight becomes `1/(2σ_i²)` (regression)
  or `1/σ_i²` (classification), plus a `log σ_i` regulariser that prevents σ from exploding. They show a
  single network learning depth + semantic + instance segmentation that **beats separately-trained
  single-task models** and beats manual/grid weightings.
- **Relevance:** exactly our situation (mixed classification tasks with incomparable loss scales).
  Cheapest possible mechanism — one scalar per task, no extra backward pass.
- **Fit for 2–4 tasks:** ideal. Overhead is 4 extra parameters.
- **Caveat we must respect:** the derivation is cleanest for **regression** (Gaussian likelihood, factor
  `0.5/σ²`); the **classification** case needs an approximation (softmax scaled by `1/σ²`) giving factor
  `1/σ²` + `log σ`. The factor-of-2 difference is minor and gets absorbed into the learning rate; we note
  it but implement one consistent form (see §3).

### 2.2 Liebel & Körner (2018) — *Auxiliary Tasks in Multi-Task Learning.* arXiv:1805.06334. **[the practical form we implement]**
- **Contribution:** the widely-used **"automatic weighted loss"** engineering form of Kendall's idea.
  Replaces the `log σ` regulariser with `log(1 + σ²)` (equivalently a `+1` inside the log). Reason: `log σ`
  is **negative** when σ<1, so the total loss can dip below zero and the regulariser can *reward* shrinking
  σ in a degenerate way; `log(1+σ²)` is always **positive** and monotincreasing, a strictly safer
  regulariser. This is the reference implementation most codebases copy (e.g. the popular
  `AutomaticWeightedLoss` repo).
- **Relevance:** directly addresses the **"negative-loss / clamping"** concern the brief raised, because
  our asymmetric-focal and SupCon losses, while normally ≥0, are combined with a regulariser that must
  stay well-behaved. We adopt `log(1+σ²)`-style safety as a documented option.

### 2.3 Chen, Badrinarayanan, Lee & Rabinovich (2018) — *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.* ICML 2018. **[ablation baseline]**
- **Contribution:** balance the **gradient magnitudes** each task induces on the last shared layer so all
  tasks train at a *similar rate*; a single asymmetry hyper-parameter α controls how strongly to pull
  faster-learning tasks back. Matches grid search with far less compute.
- **Relevance:** a strong, different philosophy (balance gradients, not likelihoods). Good ablation to
  show our learned weights are competitive.
- **Cost/complexity:** needs the **gradient norm of each task w.r.t. a chosen shared layer** every step,
  i.e. per-task partial backwards or `torch.autograd.grad` calls — noticeably slower and fiddlier on a
  T4. Higher implementation risk. → keep as comparison, not primary.

### 2.4 Liu, Johns & Davison (2019) — *End-to-End Multi-Task Learning with Attention (MTAN).* CVPR 2019. **[DWA — mention]**
- **Contribution:** two ideas. (a) MTAN architecture with per-task soft-attention over a shared feature
  pool. (b) **Dynamic Weight Averaging (DWA):** set task weight ∝ the recent **rate of loss decrease**
  (ratio of loss now vs a few steps ago), softmax-normalised. Very cheap — only needs stored loss values.
- **Relevance:** cheapest adaptive scheme; but purely heuristic, **no anti-collapse regulariser**, and
  sensitive to noisy loss curves. We mention it as the "cheap heuristic" reference point; not a headline
  ablation.

### 2.5 Yu, Kumar, Gupta, Levine, Hausman & Finn (2020) — *Gradient Surgery for Multi-Task Learning (PCGrad).* NeurIPS 2020. **[ablation baseline]**
- **Contribution:** attack **conflicting gradients** directly. When two tasks' gradients point in
  opposing directions (negative cosine), **project** each onto the normal plane of the other before the
  update, removing the destructive component. Orthogonal to weighting — it fixes *direction*, not *scale*.
- **Relevance:** best-in-class for task-interference; can be **combined** with uncertainty weighting.
- **Cost/complexity:** requires storing each task's full gradient and pairwise projection every step —
  the most expensive option, meaningful memory/time cost on 512px UWF images. → ablation only, and only if
  interference is observed.

### 2.6 Recent critique / refinement (context, honesty)
- **"Investigating Uncertainty Weighting for Multi-Task Learning: Insights and Analytical Alternative"**
  (IJCV 2025) and **UW-SO (Soft Optimal Uncertainty Weighting)**: document that plain uncertainty
  weighting (UW) can **overfit the weights**, rests on a rigid homoscedastic assumption, and lacks
  theoretical grounding for arbitrary (non-Gaussian) losses. UW-SO derives analytically-optimal weights
  with a softmax + temperature. **Takeaway for us:** treat learned weights as a strong, cheap default but
  (a) exclude them from weight decay, (b) watch for weight overfitting on the small rare-lesion classes,
  and (c) log the trajectory. We cite these as the honest limitations of our chosen method.
- **General MTL surveys** (e.g. *Multi-Task Learning with Deep Neural Networks: A Survey*, Crawshaw 2020;
  Vandenhende et al. 2021 TPAMI survey) frame the space: architecture-based vs optimisation-based MTL,
  and place uncertainty weighting / GradNorm / PCGrad / DWA as the canonical **optimisation-based**
  balancing family. We position our work inside that family.

### 2.7 Domain precedent (DR-specific, for novelty framing)
- Multi-task DR papers exist (e.g. AAAI-2021 *Deep Multi-Task Learning for DR Grading and Lesion
  Segmentation*; recent ViT+CNN multi-task DR pipelines; DeepDR). They confirm **joint grade + lesion
  learning helps**, but they use fixed or hand-tuned task weights and do **not** combine learned
  uncertainty weighting with an asymmetric-focal clinical-safety loss + SupCon auxiliaries. That gap is
  our (modest, applied) novelty.

---

## 3. Full mathematical derivation

### 3.1 Regression case (the clean starting point)
Assume task *i*'s network output `f_i(x; θ)` is the mean of a Gaussian observation with homoscedastic
noise σ_i:

```
p(y_i | f_i(x)) = N(y_i ; f_i(x), σ_i²)
```

The negative log-likelihood of one task is

```
-log p(y_i | f_i(x)) = 1/(2σ_i²) · ||y_i - f_i(x)||²  +  log σ_i  +  const
                     = 1/(2σ_i²) · L_i(θ)             +  log σ_i  +  const
```

where `L_i(θ) = ||y_i - f_i(x)||²` is the ordinary (unweighted) squared-error loss. Summing independent
tasks and dropping the constant gives the **multi-task objective**:

```
L(θ, σ) = Σ_i [ 1/(2σ_i²) · L_i(θ)  +  log σ_i ]                         (Eq. 1)
```

Read Eq. 1 as: **weight = 1/(2σ_i²)** (a large noise σ_i shrinks the task's influence) and
**regulariser = log σ_i** (penalises making σ_i large just to dodge a hard task). The two terms fight,
and their balance point is where the learning happens.

### 3.2 Classification case (softmax scaled by temperature σ²)
For a classification task Kendall scales the softmax logits by `1/σ_i²` (σ_i acts like a temperature):

```
p(y_i | f_i(x), σ_i) = Softmax( (1/σ_i²) · f_i(x) )
```

The negative log-likelihood, after the paper's approximation
`(1/σ_i²) Σ_c exp((1/σ_i²) f_c) ≈ (Σ_c exp(f_c))^{1/σ_i²}` for σ_i near 1, becomes

```
-log p(y_i | f_i, σ_i) ≈ 1/σ_i² · CE_i(θ)  +  log σ_i                     (Eq. 2)
```

where `CE_i` is the ordinary cross-entropy. **Note the difference from Eq. 1:** the classification weight
is `1/σ_i²`, not `1/(2σ_i²)` — a factor of 2. In practice this factor is immaterial: it rescales the
whole task term and is absorbed by the learning rate. We flag it but implement one consistent form.

### 3.3 The numerical-stability trick: learn `s = log σ²`
Optimising σ directly is dangerous: σ must stay **strictly positive**, and `1/σ²` explodes as σ→0. So we
reparameterise with the **log-variance**

```
s_i := log σ_i²          (so σ_i² = exp(s_i),   σ_i free to be any real number)
```

Then `1/(2σ_i²) = 0.5·exp(-s_i)` and `log σ_i = 0.5·s_i`. Substituting into Eq. 1 gives the **practical
implemented form** we use for all tasks (one consistent convention):

```
L_total = Σ_i [ 0.5 · exp(-s_i) · L_i(θ)  +  0.5 · s_i ]                  (Eq. 3)  ← IMPLEMENTED
```

`s_i` is an unconstrained real `nn.Parameter`; `exp(-s_i)` can never be negative; nothing blows up.
The classification variant (Eq. 2) would be `exp(-s_i)·L_i + 0.5·s_i` — same code, drop the 0.5 on the
first term. We use Eq. 3 uniformly (documented choice) and treat the factor-2 as an LR detail.

**Reported task weight** at any time is `w_i = 0.5·exp(-s_i)` — this is the number we log and plot.

### 3.4 Why the regulariser prevents weight collapse (the key result)
Freeze `L_i` and minimise Eq. 3 over `s_i` alone:

```
d/ds_i [ 0.5·exp(-s_i)·L_i + 0.5·s_i ]  =  -0.5·exp(-s_i)·L_i + 0.5  =  0
  ⇒  exp(-s_i) = 1/L_i  ⇒  s_i* = log L_i                                (Eq. 4)
```

So the **optimal log-variance equals the log of the task's loss**, and the optimal weight is

```
w_i* = 0.5·exp(-s_i*) = 0.5 / L_i                                        (Eq. 5)
```

Three consequences, all the point of the method:
1. **Automatic down-weighting:** a task with a large loss `L_i` (noisy/hard) gets a **small** weight
   `0.5/L_i`. Noisier tasks are trusted less — exactly the desired behaviour, with **no grid search**.
2. **No collapse to zero:** the optimum `s_i* = log L_i` is **finite** whenever `L_i` is finite. Without
   the `+0.5·s_i` term, Eq. 3 would be minimised by `s_i → +∞` (weight → 0), i.e. the task would be
   dropped entirely. The regulariser is what makes the minimum finite and keeps every task alive.
3. **Self-consistency during training:** as `L_i` falls, `s_i*` falls, so the weight `0.5/L_i` *rises* —
   a task that becomes easy gets *more* attention, a gentle curriculum.

### 3.5 Gradients w.r.t. the shared weights and w.r.t. `s_i`
- **w.r.t. θ (shared net):** `∂L_total/∂θ = Σ_i 0.5·exp(-s_i) · ∂L_i/∂θ`. Each task's gradient is scaled
  by its current learned weight — precisely the adaptive balancing we want.
- **w.r.t. `s_i` (log-variance):** `∂L_total/∂s_i = -0.5·exp(-s_i)·L_i + 0.5 = 0.5·(1 - exp(-s_i)·L_i)`.
  Intuition: if the *scaled* loss `exp(-s_i)·L_i > 1`, the gradient is negative → `s_i` **decreases** →
  weight **increases** (task is "worth it"). If the scaled loss `< 1`, `s_i` increases → weight decreases.
  Equilibrium at `exp(-s_i)·L_i = 1`, matching Eq. 4. Both θ and `s` are updated by the **same** backward
  pass — no extra gradient computation, unlike GradNorm/PCGrad.

### 3.6 Negative-loss / clamping concern (BCE, focal, SupCon)
Eq. 3's first term `0.5·exp(-s_i)·L_i` is only guaranteed well-behaved (bounded below, convex-ish in
`s_i`) when `L_i ≥ 0`. Our losses:
- **CE (grade):** always ≥0. Safe.
- **Asymmetric focal (lesion):** built on `binary_cross_entropy_with_logits` × non-negative focal factor
  × β∈(0,1) → always ≥0. Safe.
- **SupCon:** `-(τ/0.07)·mean_log_prob_pos`; `log_prob ≤ 0` so the negated mean is ≥0. Safe.
All three are non-negative, so Eq. 3 is safe as written. **Safeguards we still document/offer:**
(a) the **Liebel–Körner** regulariser `+0.5·log(1+exp(s_i))` (softplus of `s_i`), which stays positive
even if `s_i<0`, hardening against any future negative-ish loss; (b) **clamp `s_i` to a sane range**
(e.g. `[-3, 3]`, i.e. σ² ∈ [0.05, 20]) after each step as a belt-and-braces guard; (c) `exp` is computed
in fp32 even under AMP autocast to avoid overflow. See IMPLEMENTATION_GUIDE.md.

---

## 4. Design decisions (and the reasoning)

1. **Which tasks get uncertainty weights?** Primary configuration: learn `s` for the **two supervised
   prediction tasks** (grade, lesion) and keep the **two SupCon auxiliaries at a fixed 0.5**. Rationale:
   Kendall's homoscedastic-noise story is a clean likelihood model for *supervised predictions with
   labels*; SupCon is a *representation-shaping regulariser* whose "observation noise" interpretation is
   shaky, and letting its weight float risks the model muting the auxiliary or over-trusting a
   batch-composition-dependent term. We therefore recommend fixed SupCon as primary, and provide a config
   flag `UNCERTAINTY_ON_SUPCON=True` to learn all **four** `s_i` as an **ablation** (the brief explicitly
   asks to compare grade+lesion-only vs also-SupCon).
2. **Implement Eq. 3 uniformly** (0.5 factor for every task) rather than mixing 0.5 (regression) and 1.0
   (classification) factors. Simpler, matches the reference `AutomaticWeightedLoss`, and the factor is an
   LR detail. Documented in §3.2.
3. **Initialise `s_i = 0`** (σ²=1 → weight 0.5 for every task). A neutral, unopinionated prior; matches
   folder 3's `0.5` on SupCon and puts grade/lesion at equal 0.5 to start, close to the `1:1` they used.
4. **Log-variances get their own optimizer param group with `weight_decay = 0`.** Weight decay is an L2
   pull toward `s_i = 0` (σ²=1, equal weights) — an *unjustified prior* that fights the whole point of
   *learning* the balance. We must let the data set `s`. (Detailed in IMPLEMENTATION_GUIDE.md.)
5. **Loss module lives OUTSIDE `nn.DataParallel`.** DataParallel replicates a module across GPUs and
   scatters inputs; custom scalar `nn.Parameter`s inside a replicated module get duplicated per-device
   and their gradients must be reduced, which is a known footgun. We wrap only `UWF_SupConDRNet` in
   DataParallel, gather its outputs on `cuda:0`, compute the four scalar losses there, and feed them to
   the single-device `UncertaintyWeightedLoss`. Its 4 parameters live on one GPU — clean and correct.
6. **Persist `s` in the checkpoint** (`uw_state_dict`) so resume restores the learned weights, not just
   the network. Also persist to the CSV log every epoch for the trajectory plot.

---

## 5. Assumptions
- Homoscedastic uncertainty: each task's noise is constant across inputs (not per-image). Reasonable and
  standard; heteroscedastic weighting is possible future work but adds a per-image head.
- Losses are non-negative (verified in §3.6).
- The two tasks genuinely share useful structure (grade and lesions are causally linked in DR), so joint
  training is beneficial — supported by the DR-MTL literature (§2.7) and folder-3's own results.

## 6. Future ideas
- **Heteroscedastic / per-image uncertainty** for hard-example mining on rare PDR lesions.
- **UW-SO** (softmax-normalised analytically-optimal weights) as a drop-in upgrade if plain UW overfits.
- **Uncertainty weighting + PCGrad** combined (weight the scale, project away the conflict).
- **Per-lesion-class** uncertainty (7 σ's inside the lesion task) to further protect NV/VH/RD.

## 7. References
1. Kendall, Gal, Cipolla (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene
   Geometry and Semantics.* CVPR 2018, 7482–7491.
   https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html
2. Liebel, Körner (2018). *Auxiliary Tasks in Multi-Task Learning.* arXiv:1805.06334.
   https://arxiv.org/abs/1805.06334
3. Chen, Badrinarayanan, Lee, Rabinovich (2018). *GradNorm.* ICML 2018.
   https://proceedings.mlr.press/v80/chen18a.html
4. Liu, Johns, Davison (2019). *End-to-End Multi-Task Learning with Attention (MTAN + DWA).* CVPR 2019.
   https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html
5. Yu, Kumar, Gupta, Levine, Hausman, Finn (2020). *Gradient Surgery for Multi-Task Learning (PCGrad).*
   NeurIPS 2020. https://arxiv.org/abs/2001.06782
6. *Investigating Uncertainty Weighting for Multi-Task Learning: Insights and Analytical Alternative
   (UW-SO).* IJCV 2025. https://link.springer.com/article/10.1007/s11263-025-02625-x
7. Crawshaw (2020). *Multi-Task Learning with Deep Neural Networks: A Survey.* arXiv:2009.09796.
8. Khosla et al. (2020). *Supervised Contrastive Learning.* NeurIPS 2020 (SupCon, the auxiliary loss we
   keep). https://arxiv.org/abs/2004.11362
9. AAAI 2021. *Deep Multi-Task Learning for Diabetic Retinopathy Grading and Lesion Segmentation.*
   https://ojs.aaai.org/index.php/AAAI/article/view/16388

## 8. Observations to record during experiments (fill in after runs)
- Trajectory of `w_grade` and `w_lesion` over epochs (do they cross? stabilise?).
- Final learned `(s_grade, s_lesion)` vs the manual `(1,1)` — did the model prefer a different ratio?
- Rare-class (NV/VH/RD) recall vs folder-3 baseline.
- Whether `UNCERTAINTY_ON_SUPCON` helped or hurt.
