"""Per-lesion decision-threshold optimization.

Thresholds are chosen INDEPENDENTLY per lesion by sweeping a grid and maximizing F1 (primary) or
Youden's J (secondary, sensitivity-oriented). Fit on the VALIDATION set only, frozen, then applied to
test -- so test never informs the thresholds. The identical procedure runs for every variant, which
is why any difference is attributable to the head, not to threshold tuning. No prior experiment in
this repo did this (all used a fixed 0.5), so it is a fair, uniformly-applied improvement.
"""
import numpy as np


def _grid(cfg):
    return np.arange(cfg.get('thr_min', 0.05), cfg.get('thr_max', 0.95) + 1e-9,
                     cfg.get('thr_step', 0.01))


def _best_threshold_for_lesion(scores, y, grid, objective):
    """Return the grid threshold maximizing the objective for one lesion. 0.5 if no positives."""
    y = y.astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5                                   # degenerate column -> neutral default
    best_thr, best_obj = 0.5, -1.0
    for t in grid:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        if objective == 'youden':
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            obj = tpr - fpr
        else:                                        # f1
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            obj = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if obj > best_obj:
            best_obj, best_thr = obj, float(t)
    return best_thr


def optimize_thresholds(probs, targets, cfg, objective=None):
    """Fit one threshold per lesion on (validation) probs/targets. Returns a length-7 list."""
    probs = np.asarray(probs, dtype=np.float64)
    y = (np.asarray(targets) > 0.5).astype(int)
    objective = objective or cfg.get('thr_objective', 'f1')
    grid = _grid(cfg)
    return [_best_threshold_for_lesion(probs[:, j], y[:, j], grid, objective)
            for j in range(probs.shape[1])]
