# Report — Uncertainty-Weighted Multi-Task Learning (UWF)

**Notebook:** `Uncertainty_Weighted_MTL_UWF1.ipynb`
**Modality:** UWF (ultra-wide-field retinal images)
**Setup used:** 4-task version (`UNCERTAINTY_ON_SUPCON = True`)
**Result files:** `best_model (1).pth`, `checkpoint (2).pth`, `uncertainty_training_log.csv`

---

## 1. What is this about? (the topic)

The model does **two jobs at once** on an eye image:

1. **Grade** the diabetic retinopathy severity (a number from 0 to 4).
2. **Find lesions** — mark which of 7 damage types are present. Three of them (NV, VH, RD)
   are the *dangerous* ones that define the worst stage of the disease (Proliferative DR).

When you train one model on several jobs at once, you have to decide **how much each job counts**.
Normally a human picks these numbers by hand (guesswork). This experiment instead lets the model
**learn those numbers by itself** using a known method (Kendall/Gal/Cipolla, 2018). The idea: the
model figures out which task is "harder" and balances its attention automatically, so the rare but
important lesions don't get ignored.

## 2. What were we trying to do? (the goal)

Beat / match our earlier baseline (folder 3) while keeping the model **safe** — meaning it should
rarely miss the dangerous lesions. In numbers, the targets were:

- Grade accuracy around **76.3%**
- Grade agreement score (kappa) around **0.922**
- Lesion score (F1) around **0.738**, lesion AUC around **0.951**
- ...and **raise** the recall (catch-rate) on the dangerous NV / VH / RD lesions.

## 3. How we did it (in short)

- Backbone: a ResNet-50 based network with a feature pyramid (reused from folder 3, unchanged).
- The model outputs a grade + the 7 lesions, plus two helper signals ("SupCon").
- On top, an **auto-balancer** learns a weight for each task instead of us fixing them.
- Trained on Kaggle GPUs. Checkpoints were auto-backed-up to a Kaggle dataset so an idle
  restart could resume from where it left off.

## 4. What did we get? (the results)

Best validation numbers during the run, next to the target we wanted to beat:

| Thing measured                     | Target (baseline) | What we got | Outcome        |
|------------------------------------|-------------------|-------------|----------------|
| Grade accuracy                     | 76.3%             | ~76.5%      | Matched        |
| Grade agreement (kappa)            | 0.922             | ~0.921      | Matched        |
| Lesion score (F1)                  | 0.738             | **0.778**   | **Better** ✅   |
| Lesion AUC                         | 0.951             | **0.961**   | **Better** ✅   |
| Lesion PR-AUC (hard-case score)    | —                 | ~0.833      | Strong         |
| Dangerous-lesion catch-rate (NV)   | —                 | ~0.86–0.91  | High           |
| Dangerous-lesion catch-rate (VH)   | —                 | ~0.94–0.97  | High           |
| Dangerous-lesion catch-rate (RD)   | —                 | ~0.96       | High           |

**In plain words:** the grade-scoring stayed just as good as before, the lesion-finding got
**clearly better**, and the model catches almost all of the dangerous lesions. That is exactly
what we were hoping for.

## 5. Did it go okay?

**Yes — it went well.** The main goal was met: same grade quality, better lesion detection, and a
high safety catch-rate on the worst lesions.

## 6. Weak spots (being honest)

1. **The auto-balancer only half-worked.** It learned the balance nicely for the *grade* task
   (its weight moved from 0.74 up to ~1.0 during training). But for the *lesion* task, the weight
   shot to its **maximum allowed value (10.04) and stayed stuck there** the whole run. So for the
   lesion task, the "learning the balance" part didn't really get a chance to work — it just hit
   the ceiling. Fixable by raising the cap (`S_CLAMP`) and re-running.
2. **A little overfitting.** The training loss kept dropping but the validation loss stopped
   improving, meaning the model started memorising the training data. The scores held up, so it's
   not serious, but training even longer would not help.
3. **"Best model" was picked by loss, not by the scores we care about.** Because validation loss and
   the clinical scores disagreed here, the saved "best" model may not be the very best on kappa/F1.
   Picking the best epoch from the CSV log (by kappa or F1) is safer.

## 7. Overall rating

**~800 / 1000.** A solid, useful result that beat the baseline where it matters and kept the model
safe. Held back from a top score only by the two issues above (the lesion weight hitting its ceiling,
and mild overfitting).

## 8. If we continue

- Raise `S_CLAMP` so the lesion weight isn't artificially capped.
- Choose the best model by **kappa / F1 / dangerous-lesion recall** instead of validation loss.
- Optionally run the simpler 2-task version (`UNCERTAINTY_ON_SUPCON = False`) as the clean headline
  result and keep this 4-task run as the comparison (ablation).
