# Synthetic Minority Generation for Rare Diabetic-Retinopathy Lesions

> Research topic (Paper 2, MMRDR): instead of endlessly **oversampling** the same handful of
> rare-lesion images, **generate new, realistic synthetic positives** with a modern generative
> model and use them to augment training — so the downstream classifier finally sees enough
> variety to learn the rarest, most sight-threatening lesions.

---

## 1. What is this? (the one-paragraph version)

Our dataset (MMRDR, UWF modality) is savagely imbalanced for the lesions that matter most.
**Retinal Detachment (RD) has only 238 positive images out of 10,404; Vitreous Hemorrhage (VH)
has 1,219; Neovascularization (NV) has 1,188.** These three (indices 4, 5, 6 in the fixed lesion
vector `[MA, HE, IH, VB/IRMA, NV, VH, RD]`) are exactly the lesions that define **Proliferative
Diabetic Retinopathy (PDR)** — the stage where a missed diagnosis can mean irreversible blindness.
Classic fixes (oversampling, class-weighted loss, focal/asymmetric loss — already explored in
folders 1, 2, 3 of this repo) re-weight or re-show the *same* scarce images. They cannot invent new
appearances. This project asks: **can a generative model manufacture new, plausible RD/NV/VH retinal
images that add genuine variety, and does adding them measurably improve the classifier's recall on
those rare classes on a fixed, real held-out test set?**

This folder is **planning + documentation + one Kaggle-ready proof-of-concept notebook**. It does
**not** train a full generative model here — that is infeasible in a Kaggle session (see below). The
notebook proves the *code path* end-to-end on a tiny slice; the docs lay out the full research
programme.

---

## 2. Why do this? (the problem, concretely)

The rare-class scarcity in numbers (verified from the actual CSVs — see `SHARED_CONTEXT`):

| Lesion | UWF positives | CFP positives | Clinical role |
|---|---|---|---|
| MA (microaneurysm) | 6,938 | 4,398 | common, early |
| HE (hard exudate) | 3,846 | 2,648 | common |
| IH (intraretinal hem.) | 3,849 | 2,580 | common |
| VB/IRMA | 1,010 | 236 | moderate |
| **NV (neovascularization)** | **1,188** | **556** | **PDR-defining** |
| **VH (vitreous hem.)** | **1,219** | **358** | **PDR-defining** |
| **RD (retinal detachment)** | **238** | **93** | **PDR-defining, rarest** |

RD at 238/10,404 (~2.3%) is the headline problem. When a class is this rare:

- **Oversampling** just duplicates the 238 images ~30× to balance a batch. The network memorises
  those exact 238 pictures (their vignetting, their patient, their camera) rather than learning what
  *retinal detachment* looks like in general. This is textbook overfitting-to-the-minority.
- **Class-weighted / focal / asymmetric loss** (folders 1–3) turns up the gradient on rare
  positives. Helpful, but it still only has 238 real appearances to generalise from.

A generative model that can synthesise **new** RD-positive retinas — different vessels, different
disc, different peripheral spread — could give the classifier the variety that neither oversampling
nor loss-reweighting can. That is the research bet.

---

## 3. Why *this* method was selected (generative augmentation) and what we chose

We evaluated the two dominant generative families against our hard constraints — Kaggle T4 (16 GB),
~12 h sessions, and a **tiny** positive set (238 RD images):

- **GAN family** (DCGAN → StyleGAN2 → **StyleGAN2-ADA** → conditional GANs → DR-GAN). ADA (Adaptive
  Discriminator Augmentation, Karras et al., NeurIPS 2020) is *purpose-built for limited data* — it
  is the reference method for training GANs on a few thousand images without the discriminator
  overfitting and the generator collapsing.
- **Diffusion family** (DDPM → DDIM → **Latent Diffusion / Stable Diffusion** → classifier-free
  guidance → medical diffusion). Diffusion is more stable to train and higher-fidelity, but training
  one from scratch on 238 images is hopeless; the viable route is **fine-tuning a pretrained latent
  diffusion model with LoRA/DreamBooth** at a tractable resolution.

**Our recommendation (justified in full in `RESEARCH_NOTES.md` §"Recommendation"):**

- **PRIMARY: fine-tune a pretrained Latent Diffusion (Stable-Diffusion-style) model with LoRA/
  DreamBooth at 256×256, conditioned on lesion presence** (rare-flag / 7-dim lesion vector),
  operating only on the masked retina region (`apply_uwf_mask`). Pretrained priors mean we are *not*
  learning "what a retina looks like" from 238 images — only how to *steer* an existing image prior
  toward RD/NV/VH. LoRA keeps the trainable parameter count tiny (fits a T4), and diffusion avoids
  GAN mode collapse.
- **FALLBACK / ABLATION: StyleGAN2-ADA at 256×256.** Purpose-built for limited data, a strong,
  well-understood baseline, and a fair "GAN vs diffusion" comparison. If diffusion fine-tuning
  underperforms or is too slow, ADA is the backup.

**The success metric is NOT image beauty (FID).** It is **downstream utility**: does training the
folder-3 classifier on *real + synthetic* beat training on *real only* (and on *real + classic
oversampling*) in **rare-class recall** on a **fixed real test split**? FID/KID are secondary,
sanity-check metrics.

---

## 4. Limitations of the *existing* approaches this improves on

| Existing (repo) | Limitation | How generation helps |
|---|---|---|
| Random oversampling (Topic 1) | Re-shows same 238 images → memorisation, no new variety | New appearances add true variety |
| Class-balanced focal loss (folder 1) | Re-weights gradient; still 238 real looks | More effective *positives* to weight |
| Asymmetric FN-penalised loss (folder 2) | Penalises missing RD, but can't teach unseen RD looks | Broader positive support set |
| SupCon on lesions (folder 3, mature) | Best current model; still starved of rare positives | Feeds the *same* classifier more data |

Generation is **complementary**, not a replacement: synthetic images are *added to*, not swapped
for, the real data, and the same losses (asymmetric focal + SupCon) still apply.

---

## 5. Expected benefits (and honest caveats)

**If it works:**
- Higher **per-class recall / PR-AUC for RD, NV, VH** on the fixed real test set (the headline
  clinical-safety metric) vs. the folder-3 baseline and vs. plain oversampling.
- A reusable **conditional rare-lesion generator** for the MMRDR dataset (CC0-licensed → publishing
  synthetic images is legally clean, provided they are never presented as real patient data).
- A **publishable** contribution *if and only if* the rigorous downstream-utility eval shows a real,
  significant gain over simple oversampling (see §7).

**Honest caveats (read `STATUS.md` risks in full):**
- This is the **highest-risk, highest-compute** topic in the Paper-2 programme.
- A generator trained on 238 images can **hallucinate non-physiological lesions** — synthetic
  images that look plausible to a network but are clinically wrong. That would *hurt* the classifier.
- The classifier may learn **synthetic artifacts as shortcuts** (e.g. a subtle GAN fingerprint that
  correlates with "RD") — inflating training metrics but not real-test recall.
- **Synthetic augmentation may simply not beat good oversampling.** That is a real, publishable-in-
  its-own-right negative result, but it is a genuine risk we state up front.

---

## 6. Research intuition (why it *should* help)

Oversampling moves the *same* 238 points around in a batch; the decision boundary can wrap tightly
around them (overfit). A good conditional generator instead **samples from a learned manifold** of
"RD-positive retina" — producing points *near but not identical to* the real ones. Those extra
points fill in the region of feature space around the true rare class, so the classifier learns a
**smoother, better-supported boundary** for RD rather than memorising 238 dots. This is the same
intuition behind SMOTE (interpolating minority samples) but in high-dimensional *image* space, where
naive interpolation is meaningless and a generative model is the principled tool. The catch (§5) is
that the generator must sample *on* the true manifold, not off it — hence FID/KID and expert
plausibility checks as guardrails, and downstream recall as the final arbiter.

---

## 7. Success criteria (precise, pre-registered)

Primary (must hold to claim success):
1. **Rare-class recall gain.** On the **fixed real held-out test split** (`random_state=42`, ideally
   stratified), the folder-3 classifier (`UWF_SupConDRNet`) trained on **real + synthetic** achieves
   **higher mean per-class recall for RD, NV, VH** than the **real-only baseline**, over **≥3 seeds**,
   with a **statistically significant** difference (paired test, e.g. Wilcoxon; report mean±std).
2. **Beats the non-generative baseline.** The gain must **also** exceed **real + classic
   oversampling** (Topic 1) — otherwise generation added cost but no value.
3. **No collateral damage.** Grade quadratic-weighted Kappa and common-lesion F1 must **not
   regress** materially vs. baseline.

Secondary (supporting evidence, not sufficient alone):
4. **Generative quality:** FID/KID between synthetic and real RD/NV/VH pools within a reasonable
   range; **improved precision/recall for generative models** (Kynkäänniemi et al., 2019) to show the
   generator neither collapses (low recall) nor hallucinates off-manifold (low precision).
5. **Expert plausibility:** a small qualitative check that synthetic RD/NV/VH images are not obviously
   non-physiological.

Failure is declared honestly if (1) or (2) does not hold — see `STATUS.md` → Future Work.

---

## 8. How it integrates with the rest of Paper 2

- **Reuses, unchanged:** the UWF dataset loader + `apply_uwf_mask`, the folder-3 classifier
  `UWF_SupConDRNet` (ResNet-50 + FPN + two SupCon heads), `AsymmetricFocalLoss`, `SupConLoss`,
  `calculate_metrics` (grade Acc + quadratic Kappa; lesion macro-F1 + AUROC), the
  `save_checkpoint`/`load_checkpoint` + AMP + `nn.DataParallel` Kaggle template, and the fixed
  `random_state=42` split.
- **Adds, in this folder only:** a conditional generator (LoRA-fine-tuned latent diffusion primary /
  StyleGAN2-ADA fallback), a **synthetic-image pool**, and a **mixed dataloader** that draws
  `real ∪ synthetic` for the *training* set only (the **test set is always 100% real**).
- **Compares against:** folder-3 (mature real baseline) and Topic-1 (resampling) as the natural
  non-generative control.
- **Does NOT modify** any of `clinically_weighted_loss_1/`, `clinical_safety_pdr_2/`,
  `contrastive_learning_lesions_3/`, `findings/`, `notebooks/`, `models/`, `MMRDR/`.

---

## 9. Deliverables in this folder

| File | Purpose |
|---|---|
| `README.md` | This file — what/why/selection/benefits/success criteria. |
| `PLAN.md` | Milestones, roadmap, dependencies, implementation order, timeline, risks. |
| `RESEARCH_NOTES.md` | Full literature review, math (DDPM, CFG, GAN+ADA, latent diffusion), design decisions. |
| `IMPLEMENTATION_GUIDE.md` | Architecture, data/training/inference flow, hyperparameter tables, bottlenecks, debugging. |
| `STATUS.md` | Completed / In progress / Pending / Blockers / Future work / Next tasks. |
| `create_nb.py` | Generator script (repo convention) that emits the Kaggle notebook. |
| `synthetic_minority_generation.ipynb` | The Kaggle-ready proof-of-concept notebook (20 sections). |

---

## 10. Current status (see `STATUS.md` for detail)

**Planning + documentation + PoC notebook: COMPLETE.** Full generator training: **NOT run**
(infeasible in this environment; gated behind a `MODE` flag in the notebook). The notebook runs
top-to-bottom unattended on Kaggle, proving the smallest end-to-end slice: build the model, run a few
optimisation steps on a tiny subset, sample an honestly-labelled (noisy) grid, and compute a FID
smoke-test between two real subsets.
