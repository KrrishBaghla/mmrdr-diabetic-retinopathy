# RESEARCH NOTES — Synthetic Minority Generation

A self-contained literature review + math + design decisions for generating rare-lesion retinal
images to augment the MMRDR (UWF) diabetic-retinopathy classifier. Written to be readable by a
newcomer: every paper says *what it contributed*, *why it matters here*, *how well it fits*, and its
*compute/complexity cost*. All citations were verified via web search (authors/years/venues checked);
none are fabricated. Where a fact could not be verified it is flagged as such.

---

## Part A — The GAN lineage

### A1. GAN (Goodfellow et al., 2014, NeurIPS)
**Contribution.** The original generative adversarial framework: a **generator** `G` maps noise
`z ~ p(z)` to fake images, a **discriminator** `D` tries to tell real from fake, trained as a minimax
game. **Math (the value function):**

```
min_G max_D  V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
```

At the optimum `D` estimates `p_data/(p_data+p_g)` and `G` matches `p_data`.
**Relevance/fit.** The conceptual root; not used directly.
**Cost.** N/A. **Note.** Vanilla GANs are unstable and mode-collapse-prone on small data — the exact
failure mode ADA later fixes.

### A2. DCGAN (Radford, Metz, Chintala, 2015 → ICLR 2016)
**Contribution.** First stable *convolutional* GAN recipe (strided/fractional-strided convs, BatchNorm,
no fully-connected, ReLU/LeakyReLU). Made GAN training reproducible.
**Relevance/fit.** A sensible *tiny-baseline* generator to sanity-check the pipeline (the PoC notebook
uses a compact DCGAN-style generator as the smallest runnable slice). **Not** competitive at 256² on
238 images. **Complexity.** Low. **Compute.** Low, but low fidelity.

### A3. Conditional GAN — cGAN (Mirza & Osindero, 2014)
**Contribution.** Condition `G` and `D` on side information `y` (class label): both receive `y`, so
`G(z,y)` generates class-specific samples.
**Relevance/fit.** Directly relevant — we must condition on **lesion presence** (rare-flag or the
7-dim lesion vector). Conditioning is the mechanism by which we ask for "an RD-positive retina."
**Complexity.** Low add-on. **Compute.** Same as base GAN.

### A4. ACGAN (Odena, Olah, Shlens, 2017, ICML)
**Contribution.** Auxiliary-Classifier GAN: `D` additionally *predicts the class* of its input; the
loss adds a classification term. Improves conditional sample quality and gives a built-in classifier
signal.
**Relevance/fit.** A clean way to make the discriminator also verify "does this fake actually show
RD?" — a guard against off-manifold hallucination. **Complexity.** Low-moderate. **Compute.** Same order.

### A5. StyleGAN2 (Karras et al., 2020, CVPR)
**Contribution.** Style-based generator with a mapping network + AdaIN-style modulation; StyleGAN2
fixed StyleGAN1 artifacts (weight demodulation, path-length regularisation, no progressive growing).
State-of-the-art unconditional fidelity at high resolution.
**Relevance/fit.** The backbone that ADA augments. Excellent fidelity **but data-hungry** — needs
tens of thousands of images at full res; not directly trainable on 238. **Complexity.** High.
**Compute.** High (many GPU-days at full res).

### A6. **StyleGAN2-ADA (Karras, Aittala, Hellsten, Laine, Lehtinen, Aila, 2020, NeurIPS)** ★ KEY
**Contribution.** **Adaptive Discriminator Augmentation.** The core problem with limited data is
**discriminator overfitting**: `D` memorises the tiny real set, its feedback to `G` becomes
meaningless, and training diverges / collapses. ADA applies a **differentiable, invertible
augmentation pipeline** (geometric + color + filtering) to *both* real and fake images shown to `D`,
with the augmentation **probability `p` adapted on the fly** based on an overfitting heuristic (how
confidently `D` separates train reals from held-out). Because augmentations are invertible and applied
to reals *and* fakes, they **do not leak** into `G`'s outputs ("non-leaking augmentation"). Result:
StyleGAN2-quality results with **~an order of magnitude fewer images** (a few thousand).
**Relevance/fit.** ★ This is *the* method for our regime. It is our **FALLBACK/ablation generator** at
256². Purpose-built for limited medical data; well-documented official PyTorch code (NVlabs).
**Complexity.** Moderate (use official repo). **Compute.** 256² on a T4 is trainable but slow
(multi-session with resume; official numbers are for stronger GPUs). **Expected gain.** Strong GAN
baseline; the natural "GAN vs diffusion" comparison.

### A7. Retinal-specific GANs
- **DR-GAN (Zhou, Wang, He, Cui, Shao, IEEE JBHI 2022, vol 26(1):56–66; arXiv:1912.04670, 2019).**
  Conditional GAN that synthesises **high-resolution fundus images controllable by DR grade and
  lesion masks**; the generator is conditioned on structural + lesion masks and adaptive grading
  vectors. Explicitly motivated by *imbalanced high-severity data* and shows downstream gains for DR
  grading + lesion segmentation. **Relevance.** The closest prior art to our exact goal (controllable
  lesion synthesis for DR augmentation). Confirms the approach is credible and publishable.
  **Caveat.** Needs lesion *masks* we don't have (MMRDR gives multi-hot labels, not pixel masks) — so
  we condition on the **label vector**, not masks.
- **"High-fidelity diabetic retina fundus image synthesis from freestyle lesion maps" (PMC9979677,
  2023).** Two-stage: conditional StyleGAN generates lesion maps from a grade, then a SPADE/GauGAN-
  style network renders a fundus image. **Relevance.** Shows a mask→image route; again mask-dependent.
- **RF-GAN (RF-GANs, PMC8598326).** Two-stage retinal fundus synthesis with segmentation masks +
  grading. **Relevance.** More evidence of the GAN-for-DR-augmentation paradigm.
- **DR-GAN-style takeaway.** Retinal-lesion GANs work best with **structural conditioning** (vessel/
  lesion masks). We lack masks, so our conditioning is weaker (label-only) — a real limitation to
  state, and a reason diffusion fine-tuning (which leverages a strong pretrained prior) may cope
  better with weak conditioning than a from-scratch GAN.

---

## Part B — The Diffusion lineage

### B1. **DDPM (Ho, Jain, Abbeel, 2020, NeurIPS)** ★ foundational
**Contribution.** Denoising Diffusion Probabilistic Models. A **forward process** gradually adds
Gaussian noise to an image over `T` steps; a neural network learns the **reverse** (denoising)
process, and sampling starts from pure noise and denoises to an image. Sample quality rivalling
ProgressiveGAN, and a clean training objective.

**Math — forward process** (fixed, no learning): given a variance schedule `β_1..β_T`,
```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) x_{t-1}, β_t I)
```
With `α_t = 1-β_t` and `ᾱ_t = Π_{s≤t} α_s`, you can jump to any `t` in closed form:
```
q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) x_0, (1-ᾱ_t) I)
  ⇒  x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε,   ε ~ N(0, I)
```

**Reverse process** (learned): `p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))`.

**Training objective.** Ho et al. reparameterise μ so the network `ε_θ` predicts the noise, giving
the now-standard simple loss:
```
L_simple = E_{x_0, ε, t} [ || ε - ε_θ( sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε , t) ||^2 ]
```
i.e. "add known noise to a real image, ask the network to predict that noise, minimise MSE." Sampling:
start `x_T ~ N(0,I)`, iterate `x_{t-1} = 1/sqrt(α_t) (x_t - (β_t/sqrt(1-ᾱ_t)) ε_θ(x_t,t)) + σ_t z`.

**Relevance/fit.** The mathematical core of our **primary** approach. But **pixel-space DDPM at 256²
is far too expensive** to train from scratch on 238 images (hundreds of steps × large UNet × few
images → overfit + slow). We use its objective *inside* a **latent** diffusion, fine-tuned.
**Complexity.** Moderate (well-supported by `diffusers`). **Compute.** From scratch: infeasible here.

### B2. DDIM (Song, Meng, Ermon, 2021, ICLR)
**Contribution.** A **non-Markovian**, deterministic sampler that produces good samples in **10–50
steps** instead of 1000, and gives a (near-)invertible latent→image map.
**Relevance/fit.** Essential for *fast sampling* — we must generate hundreds/thousands of synthetic
images to build the pool; 1000-step DDPM sampling would be prohibitive. We sample with DDIM.
**Complexity.** Trivial (a sampler swap). **Compute.** ~20–50× cheaper sampling.

### B3. **Latent Diffusion / Stable Diffusion (Rombach, Blattmann, Lorenz, Esser, Ommer, 2022, CVPR)** ★ KEY
**Contribution.** Run diffusion in the **latent space of a pretrained VAE** instead of pixel space.
A VAE encoder compresses a 256² image to e.g. a 32²×4 latent; the diffusion UNet denoises in that
small space; the VAE decoder renders back to pixels. **~10–100× cheaper** than pixel diffusion at the
same resolution, with a cross-attention conditioning mechanism that supports class/text/label
conditioning **without retraining the whole model**.
**Relevance/fit.** ★ The backbone of our **PRIMARY** approach. Two reasons it fits our constraints:
(1) latent space makes 256² tractable on a T4; (2) a **pretrained** SD checkpoint gives a strong
natural-image prior, so fine-tuning need only *steer* it toward retinas + lesions, not learn imaging
from 238 pictures. **Complexity.** Moderate (via `diffusers`). **Compute.** Fine-tuning with LoRA:
feasible on a T4 for a small number of steps; full fine-tune: borderline.

### B4. Classifier guidance — "Diffusion Models Beat GANs" (Dhariwal & Nichol, 2021, NeurIPS)
**Contribution.** Showed diffusion beats GANs on ImageNet FID, and introduced **classifier guidance**:
push sampling toward a class using gradients of a *separately trained* classifier
`∇_{x_t} log p(y|x_t)`, trading diversity for fidelity via a guidance scale.
**Relevance/fit.** Important context (diffusion > GAN on fidelity/coverage), and an *alternative*
conditioning route. But it needs a **noise-robust classifier**; we prefer classifier-**free** guidance
(next) to avoid training one. **Complexity.** Moderate. **Compute.** Extra classifier.

### B5. **Classifier-Free Guidance (Ho & Salimans, 2021 NeurIPS workshop; arXiv:2207.12598, 2022)** ★ KEY
**Contribution.** Get guidance **without a separate classifier**. During training, randomly **drop the
condition** (replace `y` with a null token) a fraction of the time, so one network learns *both* the
conditional `ε_θ(x_t, y)` and unconditional `ε_θ(x_t, ∅)`. At sampling, extrapolate:
```
ε̂ = ε_θ(x_t, ∅) + w · ( ε_θ(x_t, y) − ε_θ(x_t, ∅) )
```
`w` is the **guidance scale**: `w=0` unconditional, `w=1` plain conditional, `w>1` sharpens toward the
condition (more "definitely RD", less diverse).
**Relevance/fit.** ★ This is *how we condition* — jointly train conditional + unconditional on the
lesion label, then dial `w` to control how strongly a sample expresses RD/NV/VH. `w` is a key ablation
knob. **Complexity.** Low (dropout on the condition). **Compute.** Negligible extra.

### B6. Medical-imaging diffusion
- **"Diffusion Models for Medical Image Analysis: A Comprehensive Survey" (Kazerouni et al., 2022/2023,
  arXiv:2211.07804; Medical Image Analysis).** Broad survey; the go-to reference that diffusion is now
  mainstream in medical imaging (synthesis, segmentation, reconstruction). **Relevance.** Frames our
  work in the current literature and lists prior fundus-diffusion efforts.
- **Medfusion — "A multimodal comparison of latent denoising diffusion probabilistic models and GANs
  for medical image synthesis" (Müller-Franzes et al., Scientific Reports 2023).** Directly compares
  latent diffusion vs GANs on medical images (incl. fundus/CheXpert/histology) and finds diffusion
  matches or beats GANs on fidelity + diversity. **Relevance.** Empirical support for choosing latent
  diffusion as primary over a GAN on medical data.
- **"Generation of Structurally Realistic Retinal Fundus Images with Diffusion Models" (arXiv:
  2305.06813, 2023).** Diffusion generating anatomically plausible vasculature. **Relevance.** Shows
  fundus-specific diffusion is viable.
- **"Generating Realistic Counterfactuals for Retinal Fundus and OCT Images using Diffusion Models"
  (arXiv:2311.11629, 2023).** Diffusion-based counterfactuals ("make this healthy eye show disease").
  **Relevance.** Conceptually close to conditional-on-lesion generation.
- **MICCAI 2024 — "Diversified and Structure-realistic Fundus Image Synthesis for Diabetic Retinopathy
  Lesion Segmentation."** Synthesises DR fundus + lesion masks under **limited labels**, for
  downstream lesion segmentation. **Relevance.** The most recent, most on-point prior art: limited-
  label DR lesion synthesis with a downstream-task evaluation — exactly our paradigm (they do
  segmentation; we do multi-label classification recall). Confirms novelty is incremental-but-real if
  we target UWF + multi-label recall + rigorous TSTR.

### B7. Small-data fine-tuning strategies (how to adapt a big pretrained diffusion to 238 images)
- **LoRA (Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Chen, 2021; ICLR 2022; arXiv:2106.09685).** Freeze
  the pretrained weights; inject trainable **low-rank** matrices `ΔW = B·A` (rank `r` small) into the
  attention layers. Cuts trainable params by orders of magnitude, tiny checkpoints (~MBs), fast.
  **Relevance/fit.** ★ Our primary fine-tuning method — makes SD fine-tuning fit a T4 and resists
  overfitting (few trainable params = strong implicit regulariser on a tiny dataset). **Cost.** Low.
- **DreamBooth (Ruiz, Li, Jampani, Pritch, Rubinstein, Aberman, 2022; CVPR 2023; arXiv:2208.12242).**
  Fine-tune a diffusion model to bind a **rare token** to a *specific concept* from a handful of
  images, with a **prior-preservation loss** to avoid forgetting/overfitting. **Relevance/fit.** A way
  to bind a token like "`<rd-retina>`" to the RD-positive concept from ~238 images; the prior-
  preservation loss directly targets our overfitting risk. Often combined with LoRA (DreamBooth-LoRA).
  **Cost.** Low-moderate.
- **Textual Inversion (Gal, Alaluf, Atzmon, Patashnik, Bermano, Chechik, Cohen-Or, 2022; ICLR 2023;
  arXiv:2208.01618).** Learn a **new embedding vector** for a new concept while *freezing the whole
  model* — even fewer trainable params than LoRA. **Relevance/fit.** The lightest-weight option; a good
  ablation ("how little tuning can still steer toward RD?"). **Cost.** Lowest.
- **From-scratch small latent diffusion vs fine-tune pretrained.** From-scratch on 238 images →
  overfits/mode-collapses, needs far more data; **decision: fine-tune pretrained** (see design
  decision D2).

---

## Part C — Evaluating generative quality (and, crucially, *utility*)

### C1. FID — Fréchet Inception Distance (Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter, 2017, NeurIPS)
Fits Gaussians to InceptionV3 features of real vs generated sets and computes
```
FID = ||μ_r − μ_g||^2 + Tr( Σ_r + Σ_g − 2 (Σ_r Σ_g)^{1/2} )
```
Lower = closer distributions. **Caveats for us:** FID is **biased on small samples** (238 reals →
unstable), uses **ImageNet** Inception features (not retina-tuned), and rewards *matching the training
set* — which for a memorising generator can look "good" while adding no value. A 2025 note
("A Pragmatic Note on Evaluating Generative Models with FID for Retinal Image Synthesis",
arXiv:2502.17160) specifically warns FID is shaky for retinal synthesis. **Use FID as a smoke-test,
not the verdict.**

### C2. KID — Kernel Inception Distance (Bińkowski, Sutherland, Arbel, Gretton, 2018, ICLR)
Squared MMD between Inception features with a polynomial kernel. **Unbiased**, so far more reliable
than FID on **small samples** — better suited to our 238-image regime. **We report KID alongside FID.**

### C3. Precision & Recall for generative models
- **Sajjadi, Bachem, Lucic, Bousquet, Gelly, 2018, NeurIPS** — introduced precision/recall for
  generative models (fidelity vs coverage as two numbers, not one FID).
- **Kynkäänniemi, Karras, Laine, Lehtinen, Aila, 2019, NeurIPS** — **Improved Precision and Recall**
  via k-NN manifolds. **Precision** = fraction of *generated* samples on the *real* manifold (fidelity;
  low ⇒ hallucination/off-manifold). **Recall** = fraction of *real* samples covered by the *generated*
  manifold (diversity; low ⇒ **mode collapse**). **Relevance/fit.** ★ These two numbers directly
  diagnose our two nightmares — **hallucination (precision↓)** and **mode collapse (recall↓)** — which
  a single FID hides. **We report improved P&R.**
- **Inception Score (Salimans et al., 2016)** — older, ImageNet-class-based; **not meaningful** for
  retinas (no matching classes). Mentioned only to say we deliberately **don't** use it.

### C4. ★ Downstream-utility protocol (THE real success metric) — TSTR and TRTS
The literature (and common sense for medical augmentation) says: **generative metrics are proxies;
the real test is whether the synthetic data helps a downstream task.**
- **TSTR (Train on Synthetic, Test on Real)** — train the classifier *only* on synthetic, test on real.
  A pure test of whether the generator captured the class. Good as a *diagnostic* (if TSTR recall is
  near zero, the generator is useless).
- **TRTS (Train on Real, Test on Synthetic)** — sanity/coverage check.
- **Our headline protocol = augmentation, not pure TSTR:** fix the **real** test split
  (`random_state=42`), then train the *same* folder-3 classifier on:
  1. **real only** (baseline = folder-3 result),
  2. **real + classic oversampling** (the non-generative control = Topic 1),
  3. **real + synthetic** (ours),
  and report **per-rare-class recall + PR-AUC + macro-F1 + grade Kappa**, **mean±std over ≥3 seeds**,
  with a **significance test**. Ablate the **amount** of synthetic added: `0 / 0.5× / 1× / 2×` the real
  rare count. This is the section a reviewer will actually judge.

### C5. Medical risks (must be discussed in any paper)
- **Mode collapse** → generator emits near-duplicates → no added variety → recall↓ in Kynkäänniemi P&R.
  Mitigate: ADA / diffusion (more stable), monitor recall metric, visual diversity grids.
- **Hallucinated non-physiological lesions** → synthetic "RD" that is clinically impossible → teaches
  the classifier wrong features. Mitigate: precision metric, **expert plausibility check**, weak
  guidance scale, conditioning on label + operating only on masked retina.
- **Label leakage / shortcut artifacts** → the classifier latches onto a **generator fingerprint**
  (subtle spectral/texture signature) that correlates with the injected label, inflating *training*
  metrics but not *real-test* recall. Mitigate: test **only on real** images; compare real+synthetic vs
  real-only on the *same real test*; inspect whether gains vanish on real test (they will if it's a
  shortcut); consider spectral-analysis / a "real-vs-synthetic" probe classifier.
- **Data governance.** MMRDR is **CC0** → generating and even publishing synthetic derivatives is
  legally clean. **But**: synthetic images **must never be presented or stored as real patient data**;
  every synthetic artifact must be labelled synthetic in filenames/metadata and in any figure caption.

---

## Part D — The recommendation (converged, with reasoning)

**Constraints:** Kaggle T4 (16 GB) / dual-T4, ~12 h sessions, **238 RD** (and ~1.2k NV/VH) positives,
UWF images that need elliptical masking.

**Why not from-scratch high-res GAN or pixel diffusion?** Both need far more than 238 images; a
from-scratch StyleGAN2 or 256² pixel-DDPM on 238 images will **mode-collapse or overfit**, and 512²
is out of the question on a T4 in 12 h. So: **operate at 256², leverage limited-data-specific methods,
and prefer fine-tuning a pretrained prior.**

**PRIMARY — Fine-tune a pretrained Latent Diffusion (Stable-Diffusion-style) with LoRA/DreamBooth at
256², conditioned via classifier-free guidance on lesion presence, on the masked retina region.**
Reasoning:
- **Pretrained prior** (SD) means we don't learn imaging from 238 images — only steer it. This is the
  single biggest lever against tiny-data failure.
- **LoRA** → few trainable params → fits a T4 and self-regularises (harder to overfit 238 images).
- **Diffusion** → no GAN mode collapse; stable training; DDIM for fast sampling.
- **CFG** → clean label conditioning + a guidance knob to trade fidelity vs diversity.
- **256²** → the downstream classifier resizes anyway; we don't need 512² beauty, we need *useful*
  positives. Masked-retina-only keeps the model from wasting capacity on eyelids/vignette.

**FALLBACK / ABLATION — StyleGAN2-ADA at 256².** Purpose-built for limited data, strong and
well-understood, and gives an honest **GAN-vs-diffusion** comparison. If diffusion fine-tuning is too
slow or underperforms, ADA is the backup primary.

**The success metric is downstream rare-class recall on the fixed real test set — not FID.** State
this everywhere.

---

## Part E — Assumptions
- MMRDR CC0 → synthetic generation + publication is permitted (with "synthetic" labelling).
- UWF is the target modality (largest rare-positive counts: RD 238 vs CFP's 93 — CFP's 93 is likely
  *too* few even for fine-tuning; UWF is the realistic target).
- We condition on the **multi-hot lesion label / rare-flag**, **not** pixel masks (MMRDR has no masks).
- 256² is an acceptable generation resolution because the classifier downsamples; **assumption to
  validate**: 256² synthetic retains enough microaneurysm/NV detail to help (small lesions are the
  worry — flag in risks).
- Kaggle internet is ON, so `diffusers`, `torch-fidelity`, and pretrained weights can be pip-installed/
  downloaded (guarded by try/except in the notebook).

## Part F — Future ideas
- Condition on **grade + lesion vector jointly** (generate "grade-4 with RD").
- **ControlNet / mask-conditioned** synthesis if we ever obtain (or Grad-CAM-derive from folder 2) weak
  lesion masks → structural realism like DR-GAN.
- **Latent-space SMOTE** inside the diffusion latent as a cheaper interpolation baseline.
- **Consistency/latent-consistency models** for faster sampling if pool size grows.
- **Uncertainty-aware filtering:** keep only synthetic images the *current* classifier is confidently-
  and-correctly rare-positive on (self-training/curriculum), to auto-reject hallucinations.

## Part G — Design decisions (log)
- **D1.** Target **UWF**, not CFP — CFP RD (93) too small even to fine-tune. ✔
- **D2.** **Fine-tune pretrained latent diffusion**, not from-scratch — pretrained prior is the key
  tiny-data lever. ✔
- **D3.** **LoRA (± DreamBooth prior-preservation)** for parameter-efficiency + implicit regularisation.
  Textual inversion as lightest ablation. ✔
- **D4.** **Classifier-free guidance** on the lesion label; `w` (guidance scale) is an ablation. ✔
- **D5.** **256²** generation, masked-retina-only; classifier resizes anyway. ✔ (validate small-lesion
  retention — risk).
- **D6.** **StyleGAN2-ADA** as fallback + GAN-vs-diffusion ablation. ✔
- **D7.** **Downstream rare-class recall on fixed real test** is the verdict; **KID > FID** for the
  small-sample generative metric; add **Kynkäänniemi P&R** to catch collapse/hallucination. ✔
- **D8.** Synthetic added to **train only**; **test is always 100% real**; every synthetic file labelled
  synthetic. ✔

## References (verified)
1. Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014.
2. Radford, Metz, Chintala, *Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN)*, ICLR 2016.
3. Mirza & Osindero, *Conditional Generative Adversarial Nets*, arXiv:1411.1784, 2014.
4. Odena, Olah, Shlens, *Conditional Image Synthesis with Auxiliary Classifier GANs (ACGAN)*, ICML 2017.
5. Karras et al., *Analyzing and Improving the Image Quality of StyleGAN (StyleGAN2)*, CVPR 2020.
6. **Karras, Aittala, Hellsten, Laine, Lehtinen, Aila, *Training Generative Adversarial Networks with Limited Data (StyleGAN2-ADA)*, NeurIPS 2020.**
7. **Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models (DDPM)*, NeurIPS 2020.**
8. Song, Meng, Ermon, *Denoising Diffusion Implicit Models (DDIM)*, ICLR 2021.
9. **Rombach, Blattmann, Lorenz, Esser, Ommer, *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022 (arXiv:2112.10752).**
10. Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis*, NeurIPS 2021 (arXiv:2105.05233).
11. **Ho & Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS 2021 Workshop / arXiv:2207.12598, 2022.**
12. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022 (arXiv:2106.09685).
13. Ruiz et al., *DreamBooth*, CVPR 2023 (arXiv:2208.12242).
14. Gal et al., *An Image is Worth One Word (Textual Inversion)*, ICLR 2023 (arXiv:2208.01618).
15. **Zhou, Wang, He, Cui, Shao, *DR-GAN: Conditional GAN for Fine-Grained Lesion Synthesis on DR Images*, IEEE JBHI 26(1):56–66, 2022 (arXiv:1912.04670).**
16. *High-fidelity diabetic retina fundus image synthesis from freestyle lesion maps*, 2023 (PMC9979677).
17. RF-GANs, *A Method to Synthesize Retinal Fundus Images Based on GAN*, 2021 (PMC8598326).
18. Kazerouni et al., *Diffusion Models for Medical Image Analysis: A Comprehensive Survey*, Medical Image Analysis, 2023 (arXiv:2211.07804).
19. Müller-Franzes et al., *Medfusion: latent DDPM vs GANs for medical image synthesis*, Scientific Reports, 2023.
20. *Generation of Structurally Realistic Retinal Fundus Images with Diffusion Models*, 2023 (arXiv:2305.06813).
21. *Generating Realistic Counterfactuals for Retinal Fundus and OCT Images using Diffusion Models*, 2023 (arXiv:2311.11629).
22. MICCAI 2024, *Diversified and Structure-realistic Fundus Image Synthesis for DR Lesion Segmentation*.
23. Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)*, NeurIPS 2017.
24. Bińkowski, Sutherland, Arbel, Gretton, *Demystifying MMD GANs (KID)*, ICLR 2018.
25. Sajjadi et al., *Assessing Generative Models via Precision and Recall*, NeurIPS 2018.
26. Kynkäänniemi et al., *Improved Precision and Recall Metric for Assessing Generative Models*, NeurIPS 2019.
27. Salimans et al., *Improved Techniques for Training GANs (Inception Score)*, NeurIPS 2016.
28. *A Pragmatic Note on Evaluating Generative Models with FID for Retinal Image Synthesis*, arXiv:2502.17160, 2025.
