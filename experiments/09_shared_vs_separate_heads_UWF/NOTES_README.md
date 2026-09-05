# Experiment 07 — Ablation: Single Shared Head vs. Separate Heads for Rare Lesions

**Modality:** MMRDR-UWF · **Task:** 5-grade DR severity + 7-way multi-label lesion detection
`[MA, HE, IH, VB/IRMA, NV, VH, RD]` · **Rare/PDR lesions:** NV, VH, RD (indices 4,5,6; RD only 2.3% positive).

## Research question

> Does giving the rare, sight-threatening PDR lesions their **own** classifier head — instead of predicting
> all 7 lesions from **one shared** head — improve rare-lesion recall and overall multi-label performance,
> **once head capacity is held constant**?

This answers a strong reviewer question and is designed as a publication-quality, capacity-controlled ablation.

## Design in one table

Same ResNet-50 + FPN trunk and 512-d pooled feature for every variant; **only the lesion head changes.**

| Variant | `head_mode` | Lesion head | Head params | Role |
|---|---|---|---|---|
| **A**  | `shared`     | `Linear(512,7)`                                      | 3,591  | current-pipeline baseline (minimal capacity) |
| **A′** | `shared_mlp` | `Linear(512,128)→ReLU→Drop→Linear(128,7)`            | 66,567 | **capacity-matched control for B** |
| **B**  | `separate`   | common `Linear(512,4)` + rare `MLP(512→128→3)`       | 68,103 | **the hypothesis** |
| **C**  | `adapter`    | shared body + per-group adapters                     | 74,631 | soft-split middle ground |

**The crux:** A′ (66,567) ≈ B (68,103) in head budget, so **B − A′ isolates *separation*** and **A′ − A**
measures *capacity*. Pre-registered test: paired **Wilcoxon (B vs A′)** on per-seed **mean rare-recall**
over seeds {42, 1, 2023}, α=0.05, one-sided; common-lesion macro-F1 is the negative-transfer guardrail.

## What is reused vs. new

**Reused unchanged** from the mature folder-3/5 pipeline (so the head is the sole independent variable):
`UWF_SupConDRNet` trunk (ResNet-50 + FPN), `AsymmetricFocalLoss` (β `[.7,.7,.7,.8,.95,.95,.95]`),
`SupConLoss`, `apply_uwf_mask`, AMP, grad-clip 2.0, AdamW(1e-4, wd 1e-4), `ReduceLROnPlateau`, 60 epochs,
early-stop patience 10, seed fixing, DataParallel, Kaggle-resumable checkpointing.

**New, applied uniformly to every variant** (fair improvements, not the variable under test):
1. **Config-selectable lesion head** — one `MultiHeadDRNet(head_mode=…)` class.
2. **Per-lesion threshold optimization** on validation (F1-max), frozen before test — no prior experiment
   here did this (all used a fixed 0.5).
3. **Model selection / early stopping on a clinical composite** (`0.5·mean_rare_recall + 0.5·macro_F1`),
   not val loss.
4. **Frozen rare-stratified 3-way split** (train/val/test 70/15/15), RD guaranteed in every fold.
5. **Statistics**: seed aggregation + paired significance + capacity-vs-separation decomposition.

Uncertainty task-weighting is deliberately **out of scope** (it would add a confound; exp-4 saw its log-var
hit the clamp). Task weights are fixed and equal across variants.

## Folder structure

```
experiment_07_shared_vs_separate_heads/
├── notebook.ipynb            # self-contained Kaggle notebook (generated — do not hand-edit)
├── create_nb.py              # regenerates notebook.ipynb by embedding the modules below
├── run_ablation.py           # local orchestrator: sweep variants × seeds, aggregate, test significance
├── config.py                 # merges configs/*.yaml into one flat CFG (single source of truth)
├── tests.py                  # CPU dummy-data unit tests (no GPU / no dataset needed)
├── configs/                  # data / model / loss / train / eval / kaggle YAML
├── models/                   # heads.py (4 variants) + multihead_drnet.py (trunk + selectable head)
├── training/                 # env, losses, data, checkpoint (+Kaggle autosave), train_loop
├── evaluation/               # metrics, thresholds, stats, plots
├── results/                  # metrics_<variant>_s<seed>.json, summary_table.csv
├── plots/  logs/  checkpoints/
└── README.md
```

The notebook's code cells are **embedded from these modules** (internal imports stripped), so the modules
are the single source of truth — edit a module, re-run `create_nb.py`, and the notebook updates. No
duplicated logic.

## How to run

### Locally (CPU/GPU), full matrix
```powershell
Diabetic_env\Scripts\Activate.ps1
python -m experiment_07_shared_vs_separate_heads.run_ablation --variants shared shared_mlp separate --seeds 42 1 2023
# quick single run:   ... run_ablation --variants separate --seeds 42 --epochs 3
# aggregate only:     ... run_ablation --aggregate_only
```

### Unit tests (seconds, no GPU/data)
```powershell
python -m experiment_07_shared_vs_separate_heads.tests
```

### On Kaggle (GPU T4×2 — just run all cells)
1. Attach the MMRDR dataset (the notebook glob-autodetects `**/UWF.csv`). Accelerator: **GPU T4 ×2**.
2. Add **Secrets** `KAGGLE_USERNAME` and `KAGGLE_KEY` → checkpoints auto-version into the dataset
   `krrish1008/checkpoint-heads-ablation` every 2 epochs and on each new best (survives the 12-h idle wipe).
3. **Run all cells.** Training is on by default (`CFG['run_training']=True`), batch auto-bumps to 16 on the
   two T4s, and §16 loops the whole `variants × seeds` matrix — **skipping** any `(variant, seed)` already
   finished and **auto-resuming** any half-trained one. A `TIME_BUDGET_HRS=11` guard stops before the 12-h
   cutoff so checkpoints are always backed up.
4. **`QUICK_PASS=True` (default, in §16):** a reduced-epoch, single-seed pass so **one run-all produces a
   complete A/A′/B comparison** (§17 table). For the publication-grade run set **`QUICK_PASS=False`**
   (full 60-epoch × 3-seed matrix): re-run the notebook across sessions with the checkpoint dataset
   attached — it resumes and skips until every `results/metrics_*.json` exists, then §17 emits the final
   table + paired significance test. Because ResNet-50 @512px is ~10 min/epoch even on 2×T4, the full
   matrix spans several sessions by design (same as the prior experiments).

### Regenerate the notebook after editing modules
```powershell
python experiment_07_shared_vs_separate_heads\create_nb.py
```

## Checkpointing & autosave

Each epoch writes `last_<tag>.pth` and (on a new best selection score) `best_<tag>.pth`, storing
model/optimizer/scheduler/epoch/best-score/**frozen thresholds**/RNG states — full bit-for-bit resume.
`push_to_dataset` versions them into the Kaggle Dataset; credentials come from `~/.kaggle/kaggle.json`,
env vars, or Kaggle Secrets (**never hard-coded**), and every backup failure is non-fatal.

## Reading the results

- **B > A′ (rare-recall↑), common-F1 flat** → separation, not parameters, drives the gain → recommend the
  separate head (fewer missed PDR lesions at matched precision).
- **B ≈ A′, A′ > A** → the benefit is just capacity → prefer the simpler shared MLP (publishable negative
  result).
- **B rare-recall↑ but common-F1↓** → a Pareto tradeoff (negative transfer) → consider the adapter (C).

Baselines to beat (folder-3 UWF): grade quadratic-κ 0.922, lesion macro-F1 0.738, macro-AUC 0.951.
