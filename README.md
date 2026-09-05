# Multi-Task Deep Learning for Diabetic Retinopathy on the MMRDR Dataset

**Research internship — Department of Computer Science and Engineering, IIT (BHU) Varanasi**
Supervised by Prof. Sanjay Kumar Singh · Author: Krrish Baghla

This repository is the consolidated proof of work for a research internship reproducing and extending the
baselines of *"A multimodal retinal image dataset for diabetic retinopathy detection using foundation
models"* (the **MMRDR** dataset paper).

From a single retinal image the models jointly predict two things:

1. **DR severity grade** — 5-class ICDR scale (0 = No DR … 4 = Proliferative DR)
2. **Lesion presence** — a 7-way multi-label vector `[MA, HE, IH, VB/IRMA, NV, VH, RD]`

The last three lesions — **NV** (neovascularisation), **VH** (vitreous haemorrhage) and **RD** (retinal
detachment) — are what clinically define *Proliferative* DR. They are also the rarest labels in the dataset.
**That tension is the thread running through every experiment here:** the classes that matter most
clinically are exactly the ones standard training ignores.

---

## Contents

| Folder | What's in it |
|---|---|
| [`experiments/`](experiments/) | **Experiments I–XI** — folder numbers match the report's section numbers exactly |
| [`report_latex/`](report_latex/) | Compile-ready IEEE-format LaTeX source of the report (~2,600 lines, 22 figures) |
| [`docs/`](docs/) | Final PDF report, progress report, viva Q&A, supervisor meeting notes, speaker script |
| [`figures/`](figures/) | Result plots, Grad-CAM overlays, architecture diagrams, preprocessing proofs |
| [`presentation/`](presentation/) | Presentation decks and speaker notes |
| [`supporting/`](supporting/) | Data-extraction utility, the plain ResNet-50 reference baseline, exploratory OCT work, standalone training scripts |

> **Every notebook is the original file, copied unmodified, with its saved cell outputs intact** — so the
> results below can be read straight off the page without re-running anything. Nothing was regenerated,
> re-executed, or cleaned.

> **What is *not* here.** The MMRDR dataset (~18.6 GB) and all trained checkpoints (`.pth`, 94 MB – 1.2 GB
> each) are excluded — see [Reproducing this work](#reproducing-this-work).

---

## The dataset

MMRDR ships three imaging modalities. The counts below are computed directly from the shipped metadata CSVs:

| Modality | Images | Grade distribution (0/1/2/3/4) | Lesion labels |
|---|---|---|---|
| **CFP** — Colour Fundus Photography (30–60°) | 11,118 | 6,579 / 1,302 / 1,959 / 603 / 675 | yes |
| **UWF** — Ultra-Widefield Fundus (200°) | 10,404 | 3,372 / 1,936 / 2,166 / 1,377 / 1,553 | yes |
| **OCT** — Optical Coherence Tomography | 2,938 | 1,006 / 220 / 1,712 | no |

Lesion-positive counts, in the fixed order `[MA, HE, IH, VB/IRMA, NV, VH, RD]`:

- **CFP:** 4,398 / 2,648 / 2,580 / 236 / 556 / 358 / **93**
- **UWF:** 6,938 / 3,846 / 3,849 / 1,010 / 1,188 / 1,219 / **238**

RD appears in **2.3 %** of UWF images and **0.8 %** of CFP images — a **29×** imbalance against MA in UWF.
That single statistic motivates Experiments IV, V, VII, VIII, IX and X.

---

## The eleven experiments

Folder numbering matches the report's section numbering one-to-one.

| # | Experiment | Modality | Key result | Verdict |
|---|---|---|---|---|
| [I](experiments/01_multiscale_dual_branch_CFP/) | Multi-scale dual-branch CNN | CFP | val acc **0.8291**, lesion acc 0.9494 | beats the 0.8233 ResNet-50 reference |
| [II](experiments/02_retfound_foundation_CFP/) | RETFound ViT-L/16 foundation-model fine-tune @512px | CFP | test acc **0.8094**, QWK 0.8838, lesion acc 0.9476 | reproduces the paper within CI |
| [III](experiments/03_fpn_spatial_attention_UWF/) | Feature-pyramid network + spatial attention @1024px | UWF | val acc 0.7448, lesion acc 0.9066 | establishes the UWF pipeline |
| [IV](experiments/04_clinically_weighted_loss_UWF/) | Clinically-weighted lesion loss (`pos_weight` RD 4.81 vs MA 0.033 — a 145× spread) | UWF | test acc **0.7650**; **RD recall 0.79, highest in the table** | works for RD, at the cost of HE/CWS |
| [V](experiments/05_asymmetric_fn_loss_UWF/) | Asymmetric FN-penalised focal loss, β=0.95 on NV/VH/RD | UWF | **QWK 0.9208**, lesion F1 0.7151, AUROC 0.9504 | strong, stable baseline |
| [VI](experiments/06_supervised_contrastive_UWF/) | Supervised contrastive (SupCon) auxiliary heads | UWF | val acc **0.7645**, QWK 0.9220, lesion F1 0.7312 | becomes the shared pipeline for VII–X |
| [VII](experiments/07_uncertainty_task_weighting_UWF/) | Learned homoscedastic task weights replacing hand-picked `1:1:0.5:0.5` | UWF | **best lesion detection: F1 0.7779, AUROC 0.9605, PR-AUC 0.8395**; NV/VH/RD recall 0.891/0.947/0.958 | positive |
| [VIII](experiments/08_recall_driven_resampling_UWF/) | Recall-driven curriculum resampling | UWF | **QWK 0.9234**, best in repo; **RD recall reached 1.000** | promising but under-powered |
| [IX](experiments/09_shared_vs_separate_heads_UWF/) | Capacity-matched ablation: dedicated rare-lesion head vs shared head | UWF | shared 0.884 · shared-MLP **0.890** · separate 0.816 mean rare recall | **negative** |
| [X](experiments/10_synthetic_minority_generation_UWF/) | Conditional DDPM generating synthetic rare-lesion images | UWF | real **0.8776** vs synthetic 0.8326 vs oversampling 0.8074 mean rare recall | **negative** |
| [XI](experiments/11_zero_few_shot_transfer/) | Zero-/few-shot transfer to unseen APTOS 2019 and IDRiD | CFP → external | zero-shot κ up to **0.860** | generalises |

Experiments I–III establish the baselines. IV–X attack one problem from six angles. XI tests whether any of
it survives off MMRDR.

### The RETFound reproduction (Experiment II)

The most carefully controlled run in the repository: the official MMRDR `tr`/`ts` split with a stratified
validation carve (7,559 / 1,334 / 2,225), seed 42, bootstrap confidence intervals, and a verified weight load
(294/294 tensors).

| Metric | Paper | Ours @0.5 | 95 % bootstrap CI | Ours @tuned |
|---|---|---|---|---|
| Grade accuracy | 0.822 | **0.8094** | [0.792, 0.825] | 0.8094 |
| Grade F1 | 0.714 | **0.6920** | [0.665, 0.718] | 0.6920 |
| Lesion accuracy | 0.951 | **0.9476** | [0.943, 0.951] | 0.9415 |
| Lesion F1 | 0.682 | 0.6051 | [0.570, 0.636] | **0.6743** |

**Three of the paper's four headline numbers fall inside our confidence interval.** The fourth recovers to
0.6743 once per-lesion thresholds are tuned — the gap was threshold placement, not discrimination.

> **An honest correction recorded in this repo.** An earlier RETFound run reported 66.50 % accuracy and was
> initially believed to be a real result. Auditing the weight load revealed that **only 4 of 294 pretrained
> tensors actually transferred** (a silent failure), and the layer-freeze routine had never executed — the
> "foundation model" was effectively a randomly initialised ViT-L. That run is kept as
> `experiments/02_retfound_foundation_CFP/retfound_v1_BROKEN_weight_load.ipynb` as a debugging exhibit.

### Generalisation (Experiment XI)

Three frozen models applied unchanged to external data (grade only — the lesion vocabulary is MMRDR-specific):

**Zero-shot** (quadratic-weighted κ, no adaptation whatsoever):

| Model | APTOS (n=1,831) | IDRiD (n=309) |
|---|---|---|
| RETFound ViT-L (CFP) | **0.860** | 0.593 |
| MS-LAN (CFP) | 0.854 | **0.626** |
| UWF SupCon — *wrong-modality control* | 0.689 | 0.598 |

**Few-shot** — frozen backbone plus a logistic probe on *k* labelled images per class (5 episodes, mean κ):

| Model / dataset | k=0 | k=1 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| RETFound · APTOS | 0.860 | 0.813 | 0.844 | 0.851 | 0.850 |
| MS-LAN · APTOS | 0.854 | 0.691 | 0.749 | 0.770 | 0.776 |
| RETFound · IDRiD | 0.593 | 0.589 | 0.660 | 0.697 | **0.711** |
| MS-LAN · IDRiD | 0.626 | 0.540 | 0.591 | 0.627 | 0.676 |
| UWF control · IDRiD | 0.598 | 0.254 | 0.480 | 0.552 | 0.575 |

RETFound is the best zero-shot transferer and **gains most from a handful of labels on genuinely shifted
data** (IDRiD: 0.593 → 0.711). MS-LAN *loses* under few-shot probing on APTOS (0.854 → 0.776) because its own
trained head was already well matched to that distribution. The deliberate wrong-modality control is
measurably worse throughout — evidence that the representations are modality-specific rather than an
artefact of the evaluation.

---

## What the negative results say

Two experiments failed, and both are reported in full because both are informative.

**Experiment IX — a dedicated rare-lesion head does not help.** The obvious intuition is that rare lesions
deserve their own classifier. Comparing A′ (shared MLP head, 66,567 params) with B (separate rare head,
68,103 params) holds head capacity nearly constant, so the comparison isolates *separation* from *capacity*.
Capacity contributed +0.006 mean rare recall; separation contributed **−0.074**. The rare-lesion bottleneck
lives in the shared representation, not in the head topology.

**Experiment X — synthetic data does not beat real data here, and the FID says why.** A conditional DDPM
trained from scratch at 64 px produced samples with FID ≈ 200 (64 px) and ≈ 247 (256 px) against a
real-vs-real floor of ≈ 27. Adding them to training dropped mean rare recall from 0.8776 to 0.8326, and
collapsed the target class RD from 0.839 to 0.710. Plain oversampling was worse still on recall (0.8074).
The failure localises to *generation fidelity*, not to the idea of synthetic augmentation — which is why
`experiments/10_synthetic_minority_generation_UWF/03_planned_lora_ldm_followup_UNRUN.ipynb` exists: a LoRA
fine-tune of Stable Diffusion at 512 px that reruns the identical protocol. It has never been run and is
included as code only, clearly marked.

---

## Reading the numbers honestly

A few caveats that matter if you plan to quote anything above.

- **Split types are mixed.** Experiment II and Experiment X use frozen, stratified, signature-checked splits.
  Experiment IV and the ResNet-50 reference report held-out test numbers. Everything else reports
  **best-validation** numbers under an ad-hoc `train_test_split(random_state=42)`. The tables label which is
  which; they should not be merged into a single column.
- **CFP and UWF are not comparable to each other** — different fields of view, different difficulty. UWF sits
  ~7 points below CFP consistently across three independent experiments, which is itself a finding.
- **Most experiments are single-seed.** Experiments VIII, IX and X explicitly pre-registered a ≥3-seed
  protocol with significance testing; compute limits meant only VIII and X got partial runs, and IX got one
  seed at 20 epochs. Where a result is under-powered, the experiment folder says so.
- **Experiment VII's mechanism partly saturated:** the learned lesion weight pinned to its clamp ceiling
  (10.04) for the whole run, so the adaptive weighting never actually adapted for that task. The gain is
  real; the intended explanation for it is not.
- **Experiment VIII ran 18 of a planned 60 epochs**, and its RD recall of 1.000 occurred at two individual
  epochs rather than as a stable plateau.

---

## Reproducing this work

**Data.** MMRDR is not redistributed here. The original figshare download URLs are in
`docs/dataset_source_links.txt`; the extraction utility is
`supporting/data_extraction_provenance.ipynb`. Expected layout:

```
MMRDR/
├── MMRDR-CFP/  img/  FP.csv      # note: FP.csv, not CFP.csv
├── MMRDR-UWF/  img/  UWF.csv
└── MMRDR-OCT/  img/  OCT.csv
```

CSV columns: `type`, `image`, `grade` (0–4), `lesion` (a stringified 7-element multi-hot vector in the fixed
order `[MA, HE, IH, VB/IRMA, NV, VH, RD]`), `lr` (laterality).

**Environment.** Python 3.12, PyTorch + torchvision, pandas, scikit-learn, opencv, tqdm. Experiments VI–XI
were run on Kaggle dual-T4 (2×16 GB) with AMP and `nn.DataParallel`; the notebooks auto-detect the dataset
path and fall back to single-GPU cleanly. Experiment XI additionally needs the public APTOS 2019 and IDRiD
Disease Grading datasets.

**Checkpoints.** Excluded (94 MB – 1.2 GB each). Every notebook retrains from scratch, or from a torchvision
ImageNet / RETFound MAE initialisation.

**Report.** `report_latex/` compiles with `pdflatex` against the bundled `ieeecolor.cls`; the compiled output
is `docs/IIT_BHU_Internship_Report.pdf`.

---

## Techniques implemented

Multi-task learning (grade + multi-label lesion) · focal loss · class-balanced focal loss · asymmetric
FN-penalised focal loss · clinically-weighted BCE · supervised contrastive learning (SupCon) · homoscedastic
uncertainty task weighting · recall-driven curriculum resampling · feature pyramid networks · CBAM spatial
attention · multi-scale dual-branch CNNs · graph convolution over CNN feature grids · vision transformers
(ViT-L/16) and MAE foundation-model fine-tuning with layer-wise LR decay · denoising diffusion models with
classifier-free guidance · LoRA · Grad-CAM explainability and weakly-supervised bounding-box extraction ·
per-class threshold tuning · bootstrap confidence intervals · quadratic-weighted Cohen's κ.

---

## Acknowledgements

Conducted at the Indian Institute of Technology (BHU), Varanasi, under the supervision of
Prof. Sanjay Kumar Singh. The MMRDR dataset is the property of its original authors and is used here under
its published terms for academic research.
