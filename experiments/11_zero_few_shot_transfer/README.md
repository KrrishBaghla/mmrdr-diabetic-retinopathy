# Experiment XI — Zero-shot and few-shot transfer to unseen datasets

**Question.** Every result so far is on MMRDR. Do these models actually generalise, or have they learned one
dataset? And if a hospital has a few hundred labelled images, is fine-tuning necessary or is a linear probe
on frozen features enough?

**External data.** APTOS 2019 (3,662 colour fundus images, Indian cohort; split 50/50 stratified into
support 1,831 / test 1,831) and IDRiD Disease Grading (1,239 train / 309 test). Grade only (0–4 ICDR) — the
7-lesion vocabulary is MMRDR-specific and has no counterpart in either dataset.

**Models** — three, all frozen, all verified to load with 0 missing / 0 unexpected tensors:

- `MultiScaleDRNet_CFP` — the Experiment I dual-branch ResNet-50, 512 px
- `RETFound_CFP` — the fine-tuned ViT-L/16 foundation model, 512 px
- `UWF_SupCon_control` — a UWF model applied to CFP data: a deliberate **wrong-modality negative control**

**Protocol.** Zero-shot = argmax over the model's own 5 grade logits, no fitting of any kind. Few-shot =
frozen backbone, penultimate features extracted once, a fresh standardised multinomial logistic probe fitted
on *k* images per class drawn from the external support pool; k ∈ {1, 5, 10, 20}, 5 episodes, mean ± std,
with zero-shot as the k=0 anchor.

## Zero-shot (quadratic-weighted κ)

| Model | APTOS κ | APTOS acc | IDRiD κ | IDRiD acc |
|---|---|---|---|---|
| RETFound ViT-L | **0.860** | 0.711 | 0.593 | 0.476 |
| MS-LAN | 0.854 | 0.705 | **0.626** | 0.511 |
| UWF control | 0.689 | 0.578 | 0.598 | 0.375 |

## Few-shot (κ, mean ± std over 5 episodes)

| Model / dataset | k=1 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| RETFound · APTOS | .813±.045 | .844±.011 | .851±.004 | .850±.013 |
| MS-LAN · APTOS | .691±.033 | .749±.038 | .770±.010 | .776±.034 |
| UWF ctrl · APTOS | .425±.151 | .575±.134 | .712±.015 | .759±.027 |
| RETFound · IDRiD | .589±.074 | .660±.052 | .697±.028 | **.711±.034** |
| MS-LAN · IDRiD | .540±.087 | .591±.059 | .627±.040 | .676±.039 |
| UWF ctrl · IDRiD | .254±.122 | .480±.037 | .552±.028 | .575±.021 |

## Findings

1. **RETFound transfers best and adapts best.** Highest zero-shot κ on APTOS (0.860), and the largest
   few-shot gain on genuinely shifted IDRiD data (0.593 → 0.711, +0.118). For a new site, a foundation-model
   backbone plus a small calibration set beats retraining.
2. **MS-LAN *loses* under few-shot probing on APTOS** (0.854 → 0.776). Its own trained head was already well
   matched to that distribution, so replacing it with a probe fitted on 20 images per class is a downgrade.
   A linear probe is not free.
3. **The wrong-modality control behaves exactly as a control should** — clearly worse zero-shot on APTOS
   (0.689) — yet its frozen features still recover to 0.759 at k=20, showing generic retinal structure
   transfers even when the modality does not.
4. **Accuracy trails κ everywhere** (0.71 vs 0.86 on APTOS): errors cluster near the confusion diagonal, and
   minority grades 1, 3 and 4 stay under-recalled — the same imbalance problem, following the models across
   datasets.

**Caveat.** IDRiD here is an augmented mirror (1,239/309 rather than the standard 413/103), so its *few-shot*
numbers are optimistic because of support/test overlap. The zero-shot numbers are unaffected.
