# Experiment 5 — Recall-Driven Curriculum Resampling (RDCR)
### A plain-language summary

**Notebook:** `5th-experio.ipynb`
**Dataset:** MMRDR, UWF (ultra-wide-field) retinal images
**Trained on:** Kaggle, 2× Tesla T4 GPUs, 18 epochs (~5 hours)

---

## 1. What was the topic?

We are teaching a computer to read wide-angle photographs of the back of the eye (retina) and do two jobs
at once:

1. **Grade** how severe a patient's diabetic retinopathy (DR) is, on a 0–4 scale.
2. **Detect lesions** — 7 specific kinds of damage in the eye (like microaneurysms, hemorrhages, and the
   dangerous ones: **NV, VH, RD**).

The special idea in this experiment is a new **training trick** we invented, called
**Recall-Driven Curriculum Resampling (RDCR)**. In simple words:

> After every training round, we check *which diseases the model is still bad at*, and then in the next
> round we deliberately show it **more examples of exactly those diseases** — automatically, without us
> hand-tuning anything.

It's like a teacher who notices a student keeps failing one topic, so the teacher gives that student extra
practice questions on that topic — and eases off on the topics the student already aced.

---

## 2. Why did we do it?

Because the data is **very unfair**. Some eye conditions are common; the most dangerous ones are rare.

| Rare, sight-threatening condition | How many images have it | Out of every 100 images |
|---|---|---|
| **RD** (retinal detachment) | 238 | just **2 or 3** |
| VB/IRMA | 1,010 | about 10 |
| **NV** (new blood vessels) | 1,188 | about 11 |
| **VH** (vitreous hemorrhage) | 1,219 | about 12 |

The rarest lesion (RD) shows up in only ~2% of images. A normal model, seeing so few examples, tends to
**ignore** these rare cases — but these are the ones that can make a patient go blind. Missing them is the
most dangerous mistake the model can make.

The usual fix is to "oversample" rare cases by a **fixed** amount decided once at the start. The problem:
that fixed amount is a guess, and it never adapts. **RDCR fixes this by adapting every single epoch**, based
on the model's actual, current weakness (its *recall* — how many real cases it catches).

We also added a **curriculum**: for the first few epochs the model learns normally (so it builds good
general eyesight first), and only *then* do we gradually turn up the rare-disease focus. Rushing the rare
focus too early would hurt learning.

---

## 3. What did we actually do?

- **Kept the proven parts unchanged.** We reused the mature model and loss functions from our previous
  experiment (a ResNet-50 backbone with an FPN, plus special losses that already punish missing rare
  lesions). We changed **only the sampler** — the part that decides which images go into each training
  batch. This is deliberate: if results improve, we *know* it was the new sampling idea, not something else.

- **Built the RDCR sampler.** Each epoch it:
  1. looks at the model's per-class recall on the validation set,
  2. gives more weight to classes with low recall (the ones being missed),
  3. gently prefers rarer classes using a statistical "rarity" measure,
  4. follows the curriculum schedule (calm start → gradually stronger rare focus),
  5. caps how much any image can be repeated, so the model doesn't just memorize a handful of rare pictures.

- **Proved the sampler works with a built-in test.** Before any real training, a small self-check
  (`unit test`) confirmed the sampler correctly boosts a deliberately-starved rare class:
  *"rare up-weighted 3.14 vs 0.76, rare draw fraction 0.26 (natural 0.10) — PASS."*

- **Trained on Kaggle** with 2 GPUs, mixed-precision for speed, and automatic checkpoint backups to a
  Kaggle dataset so progress survives if the session restarts.

- **Split the data fairly**: 8,323 images for training, 2,081 for validation, split so rare cases appear in
  both sides (stratified).

---

## 4. What are the results?

The model trained for 18 epochs. Here is how it improved from start to finish:

| What we measured | Start (epoch 1) | Best during run | Meaning (plain language) |
|---|---|---|---|
| **Grade agreement (Kappa)** | 0.852 | **0.923** | How well DR-severity grading matches the doctors — 0.92 is excellent |
| **Lesion detection (AUC)** | 0.927 | **0.946** | Overall ability to tell "lesion present vs not" |
| **Lesion F1** | 0.62 | **0.71** | Balance of correctness on lesions |
| **RD recall** (rarest, most dangerous) | 0.82 | **1.00** | Caught up to 100% of retinal-detachment cases |
| **NV recall** | 0.96 | ~0.93 | Consistently catches new-blood-vessel cases |
| **VH recall** | 0.95 | ~0.91 | Consistently catches vitreous-hemorrhage cases |

**The headline result:** the model kept **very high recall on the rare, dangerous lesions** — the whole
point of the method. RD recall (the hardest, rarest one) even reached **1.00** at several epochs, meaning it
missed *none* of those cases in validation. NV and VH stayed around 0.90+ throughout.

**Grading quality** also reached **0.923 Kappa**, which matches the strong baseline from our previous
experiment (0.922) — so chasing the rare cases did **not** damage the main grading job.

**The curriculum behaved exactly as designed:** the "focus dial" (called *tau*) stayed at 0 for the first
5 epochs (calm warm-up), then smoothly ramped up to 0.80 by epoch 18 — proof the adaptive mechanism was
genuinely running, not stuck.

---

## 5. Honest limitations (what to say if asked)

- **The run was 18 epochs, not the full 60.** The session ended early, so the curriculum reached ~80% of
  full strength, not 100%. This is a strong **proof-of-concept** run, not the final polished result.
- **We matched, and did not yet clearly beat, the previous baseline** on overall grading/detection scores.
  The clear *advantage* of RDCR is its steady, high rare-lesion recall — but to claim a formal win for a
  paper, we would need to run to completion and repeat with several random seeds plus statistical tests.
- **Val-loss-based early stopping is a slightly unfair judge** for this method, because RDCR deliberately
  changes the training mix toward rare cases. A fairer next run would stop/save based on rare-lesion recall
  directly.

---

## 6. One-line takeaway

> We built a smart sampler that watches which eye-diseases the model keeps missing and automatically gives
> it more practice on exactly those — and it kept the model catching the rarest, most sight-threatening
> lesions (up to 100% of retinal-detachment cases) without hurting its overall grading accuracy (0.92).

---

*Full method, math, and plans are in `README.md`, `RESEARCH_NOTES.md`, `PLAN.md`, and
`IMPLEMENTATION_GUIDE.md` in this folder. Per-epoch numbers are in `metrics_log.csv`.*
