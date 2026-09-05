# Experiment X — Synthetic minority generation (UWF) — a negative result

**Question.** Reweighting and resampling only redistribute the rare images that exist. Can we *manufacture*
more of them — and does that beat plain oversampling on a fixed, 100 %-real test set?

This is the most methodologically careful experiment in the repo: a frozen 3-way stratified split with a
signature (`da1ec8e21f22`) and a **pre-registered success criterion** set before any result was seen.

## Stage 1 — the generator (`01_conditional_ddpm_generator_UWF.ipynb`)

A conditional DDPM trained from scratch: `SmallCondUNet`, 3.86M params at 64×64, classifier-free guidance on
a rare/not-rare flag, masked retina, AdamW 2e-4 with cosine schedule. Loss fell 0.0810 → 0.0166 over 41
epochs (26,650 of 32,500 steps before the session was cut); a sample-diversity proxy rose rather than fell,
so there was no mode collapse.

## Stage 2 — the verdict (`02_synthetic_vs_oversample_verdict.ipynb`)

Split: 7,282 train / 1,561 val / 1,561 test, rare fraction ≈ 0.149 in every fold. Three regimes trained with
the identical classifier and loss, per-lesion thresholds tuned on validation, seed 42.

**Generative quality gate** — the images never got close to the real distribution:

| Resolution | FID (synth vs real) | FID (real vs real) floor | KID |
|---|---|---|---|
| 64 px | **200.02** | 26.80 | 0.2564 |
| 256 px | **246.88** | 29.23 | 0.3264 |

**Held-out real test set:**

| Metric | real | oversample | synth |
|---|---|---|---|
| Mean rare recall | **0.8776** | 0.8074 | 0.8326 |
| recall NV / VH / RD | 0.894 / 0.900 / **0.839** | 0.806 / 0.842 / 0.774 | 0.888 / 0.900 / **0.710** |
| Rare macro-F1 | 0.6868 | **0.7138** | 0.6559 |
| Rare macro-AUPRC | 0.6531 | **0.7969** | 0.6648 |
| Grade QWK | **0.9020** | 0.8802 | 0.8922 |

## Verdict: false

`verdict_summary.json` records `"verdict": false`. Synthetic augmentation lost to real-only on mean rare
recall and **collapsed on the very class it targeted** (RD, −0.129), while also regressing the guardrail
metrics. Plain oversampling was worst on recall but best on AUPRC and macro-F1 — a threshold-placement
difference, not a real discrimination gain.

The FID table localises the cause: at ~200–247 against a ~27 floor, the 64 px samples were never close
enough to the real distribution to teach the classifier anything. That diagnosis is actionable, which is why
`03_planned_lora_ldm_followup_UNRUN.ipynb` exists — a LoRA fine-tune of Stable Diffusion at 512 px that
reruns this identical protocol. **It has never been run** and is included as code only.

**Caveats.** Single seed, `QUICK_PASS` budget, descriptive only, no significance test (the protocol calls
for ≥3 seeds).
