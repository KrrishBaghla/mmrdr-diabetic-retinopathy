# Meeting Notes: Handling Rare, High-Risk Lesions in DR Detection

**One-line framing to open with:**
"All three of these experiments attack the same clinical problem — our model was ignoring the rarest but most dangerous lesions — using three progressively stronger techniques. Folder 3 is the one that actually got trained and shows results."

---

## 0. The Problem (say this first, 30 seconds)

Our base model (`MultiScaleLesionAttentionNet`, dual-branch ResNet-50) predicts two things per retinal image:
- **DR Grade**: severity 0 (none) to 4 (Proliferative DR)
- **7 Lesions** (multi-label): `MA, HE, IH, CWS/VB-IRMA, NV, VH, RD`

The last three — **NV (Neovascularization), VH (Vitreous Hemorrhage), RD (Retinal Detachment)** — are the
lesions that *define* **Proliferative DR (PDR)**, the sight-threatening stage. They're also the *rarest*
labels in the dataset (e.g. in UWF: RD appears in only 238 of ~10,400 images, vs. 6,938 for MA).

**The clinical risk:** a standard loss function (plain BCE/cross-entropy) treats every mistake equally.
Missing a rare RD case (a **false negative**) is treated the same as a false alarm on a common MA (a
**false positive**) — even though missing RD can mean a patient goes blind. Statistically, the model just
learns to ignore rare classes because it can get a low loss by mostly predicting "not present."

**The three folders = three different fixes for this one problem**, tried in sequence:

| # | Folder | Idea | Status |
|---|--------|------|--------|
| 1 | `clinically_weighted_loss_1` | Weight the loss by clinical danger, not just rarity | Implemented, training run interrupted early — no final numbers |
| 2 | `clinical_safety_pdr_2` | Asymmetric loss: punish missed detections harder than false alarms | Math + architecture designed and unit-tested on dummy data — not trained on real images |
| 3 | `contrastive_learning_lesions_3` | Force the network to *cluster* rare-lesion images together in feature space | Fully trained (37 epochs, real metrics) — most mature result |

---

## 1. `clinically_weighted_loss_1` — Clinically-Weighted Loss

**The idea:** Don't just weight lesions by how rare they are statistically — weight them by how dangerous
they are *clinically*, using risk scores an ophthalmologist would assign.

**The math (simple version):**
1. Standard rarity weight = `1 / frequency` (rarer lesion → bigger weight)
2. Multiply by a hand-assigned **clinical risk multiplier**:
   `MA=1, HE=1, IH=2, CWS=2, NV=4, VH=4, RD=5`
3. Normalize so the average weight across all 7 classes is 1 (keeps training stable)

**What this produced (real numbers, computed from the actual UWF dataset):**

| Lesion | MA | HE | IH | CWS | NV | VH | RD |
|--------|-----|-----|-----|-----|-----|-----|-----|
| Final weight | 0.033 | 0.060 | 0.119 | 0.454 | 0.771 | 0.752 | **4.812** |

**Talking point:** RD ends up weighted **~145x more** than MA in the loss function — even though pure
rarity alone wouldn't fully justify that, the clinical multiplier pushes it further. This is the "risk
matrix" trick: statistics tells you what's rare, a domain expert tells you what's dangerous, and you
combine both.

**Status to be upfront about:** the training loop was started with this loss plugged into the real model
and real dataloader, but the notebook only shows ~23 of 1,041 batches completing in epoch 1 — it was
stopped early (looks like a scoped test run, not a full training pass). No final accuracy/F1 numbers exist
for this one yet. Present it as "the weighting scheme is built and validated numerically; full training is
the next step," not as a finished result.

---

## 2. `clinical_safety_pdr_2` — Asymmetric Cost-Sensitive Loss

**The idea:** Go a step further than reweighting — explicitly penalize **false negatives** on PDR lesions
much more than **false positives**. Rationale stated up front: missing PDR is "disastrous" (can cause
blindness); a false alarm just means "an unnecessary follow-up appointment."

**The math (this is the one worth writing on the board):**

$$
L_{asym} =
\begin{cases}
-\beta \, (1-p_t)^\gamma \log(p_t) & \text{if lesion is present (penalizes missing it)} \\
-(1-\beta)\, p_t^\gamma \log(1-p_t) & \text{if lesion is absent (penalizes false alarms)}
\end{cases}
$$

- $p_t$ = model's predicted probability that the lesion is present
- $\gamma=2$ = standard focal-loss "focusing" term (down-weights easy/obvious cases so the model spends
  effort on hard ones)
- $\beta$ = the asymmetry knob. Set **high (0.95)** for NV/VH/RD (be very afraid of missing these), and
  **lower (0.7–0.8)** for the less critical lesions (MA/HE/IH).

**Sanity check they ran (and it's a nice thing to show):** two synthetic test cases —
1. Model misses an NV lesion (predicts 11% probability when it's actually present) → high loss
2. Model false-alarms an NV lesion (predicts 88% when it's actually absent) → lower loss

The assertion `FN penalty > FP penalty` passed — proof the loss function behaves the way a clinician would
want, before ever touching real training data.

**Architecture designed alongside it:** `UWF_MultiScaleDRNet` — ResNet-50 + a **Feature Pyramid Network**
(combines features from 4 different depths of the network, so both tiny and large lesions get seen) +
**Spatial Attention** (a small learned mask that tells the network "look here" — helpful since UWF images
are 200° wide and a lot of that is empty/artifact). Also includes a Grad-CAM → bounding-box pipeline for
explainability (draws a box around what the model "looked at"), plus UWF-specific pre-processing (an
elliptical mask that crops out eyelash/eyelid artifacts at the image border).

**Status to be upfront about:** this notebook is a **design + math validation** deliverable — the loss and
architecture are written and the dummy-data test passes, but it has not yet been run against the real MMRDR
images (the Grad-CAM section is explicitly a stub: "not runnable without trained weights"). Frame this as
the methodology/proof-of-concept step.

---

## 3. `contrastive_learning_lesions_3` — Supervised Contrastive Learning (most complete result)

**The idea:** Instead of only fixing the *loss weighting*, change what the network's internal features look
like. Explicitly pull the feature vectors of "has a rare PDR lesion" images close together, and push
"doesn't have one" images apart — *before* the final classification layer even sees them. The intuition:
if rare-lesion images already cluster together in feature space, the classifier's job downstream becomes
much easier.

**The math:**

$$
L_{sup,i} = \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}
$$

- $z_i$ = the image's feature embedding (a vector, after L2-normalizing)
- $P(i)$ = other images in the same batch that share the same rare-lesion status (both have NV/VH/RD, or
  both don't)
- $A(i)$ = everyone else in the batch
- $\tau$ (temperature) = 0.1 here — controls how "sharp" the pulling/pushing is

**Applied twice** — once on "early" features (layer 2 of ResNet-50, more local/texture detail) and once on
"deep" features (layer 4, more global/semantic) — so the clustering effect happens at both levels of visual
abstraction.

**Combined training objective:**
`total_loss = lesion_loss (asymmetric focal, reused from folder 2) + grade_loss (cross-entropy) + 0.5×supcon_early + 0.5×supcon_deep`

This is the folder where the three ideas actually converge: rarity-aware weighting (folder 1's spirit),
the asymmetric FN-focused loss (folder 2's `AsymmetricFocalLoss`, reused verbatim), and the new contrastive
term, all in one training run.

**This one actually trained — real results to report:**

Trained on Kaggle (2 GPUs), resumed from a checkpoint at epoch 27, ran to epoch 37 out of a planned 60
before early stopping triggered (patience = 10 epochs with no improvement):

| Metric | Train | Validation |
|---|---|---|
| Grade Accuracy | 85.9% | 76.3% |
| Grade Quadratic Kappa | 0.960 | 0.922 |
| Lesion F1 (macro) | 0.739 | 0.738 |
| Lesion AUC (macro) | 0.956 | 0.951 |

**How to talk about these numbers:**
- **Kappa ~0.92 on validation** is strong — this metric accounts for *how far off* a wrong grade prediction
  is (confusing Grade 3 with 4 is a small error; confusing 0 with 4 is a big one), so a high kappa means the
  model's mistakes are "close," not wild.
- **Lesion AUC ~0.95** (train and val nearly identical) shows the model ranks presence-vs-absence of lesions
  very well and isn't overfitting on this metric.
- **The train/val accuracy gap (86% vs 76%)** is worth naming honestly as mild overfitting — a good, honest
  thing to flag rather than hide, and a natural segue to "next steps" (more regularization / more data /
  earlier stopping).
- They also ran **Grad-CAM on a real validation image with a confirmed rare lesion (NV)** and got a
  bounding box around the model's attention — this is your visual "proof it's not a black box" slide.

---

## Suggested order to present in the meeting

1. **State the problem once** (Section 0) — rare + dangerous lesions get ignored by standard losses. This
   is the thread connecting all three folders; say it before naming any folder.
2. **Folder 1** — the simplest fix (reweight the loss by clinical risk). Show the weight table. Be honest:
   training was only smoke-tested, not completed.
3. **Folder 2** — the more principled fix (asymmetric FN/FP penalty), with the math on the board and the
   dummy-data proof that it behaves correctly. Mention the FPN + attention architecture briefly. Be honest:
   design/validation stage, not yet trained on real data.
4. **Folder 3** — the payoff: combines ideas from 1 & 2 with contrastive learning, actually trained, real
   metrics, real Grad-CAM image. Spend the most time here since it's the strongest result.
5. **Close with next steps**: finish training folders 1 & 2 to completion so all three can be compared
   apples-to-apples on the same metrics folder 3 already reports (Kappa, macro-F1, macro-AUC).

## If asked "why three separate approaches instead of just one?"
Good answer: each is a different lever on the same problem — *loss weighting* (1), *loss shape/asymmetry*
(2), and *representation/feature geometry* (3) — and folder 3 shows they're not mutually exclusive: it
reuses folder 2's loss function directly inside its combined objective. The progression is deliberate:
each step tries to explain *why* the previous one might still fall short before adding a new mechanism.

## If asked "what's next?"
- Run folders 1 and 2 to full completion so they have real quantitative results to compare against folder 3.
- Standardize evaluation across all three (folder 3 already tracks Kappa/F1/AUC — do the same for 1 & 2).
- Address the train/val accuracy gap in folder 3 (regularization, more augmentation, or longer patience).
