"""Clinical metric suite for the ablation.

All lesion metrics accept a per-lesion threshold vector (from evaluation.thresholds) rather than a
fixed 0.5, and the rare recall (NV/VH/RD) is surfaced as the headline. Grade metrics are reported for
sanity even though the grade head is frozen across variants.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

LESION_NAMES = ['MA', 'HE', 'IH', 'VB/IRMA', 'NV', 'VH', 'RD']
RARE_LESION_IDX = [4, 5, 6]
COMMON_LESION_IDX = [0, 1, 2, 3]


def _binarize(probs, thresholds):
    probs = np.asarray(probs, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64).reshape(1, -1)
    return (probs >= thr).astype(int)


def per_lesion_stats(probs, targets, thresholds):
    """Per-lesion precision/recall/F1/balanced-accuracy/AUROC/AUPRC at the given thresholds."""
    probs = np.asarray(probs, dtype=np.float64)
    y = (np.asarray(targets) > 0.5).astype(int)
    pred = _binarize(probs, thresholds)
    out = {}
    for j, name in enumerate(LESION_NAMES):
        yj, pj, sj = y[:, j], pred[:, j], probs[:, j]
        tp = int(((pj == 1) & (yj == 1)).sum())
        fp = int(((pj == 1) & (yj == 0)).sum())
        fn = int(((pj == 0) & (yj == 1)).sum())
        tn = int(((pj == 0) & (yj == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        f1 = (2 * prec * recall / (prec + recall)) if (prec and recall and prec + recall > 0) else 0.0
        bal_acc = np.nanmean([recall, spec]) if (tp + fn) > 0 else float('nan')
        try:
            auroc = roc_auc_score(yj, sj) if len(np.unique(yj)) > 1 else float('nan')
        except ValueError:
            auroc = float('nan')
        auprc = average_precision_score(yj, sj) if yj.sum() > 0 else float('nan')
        out[name] = {'precision': prec, 'recall': recall, 'specificity': spec, 'f1': f1,
                     'balanced_acc': bal_acc, 'auroc': auroc, 'auprc': auprc,
                     'support': int(yj.sum())}
    return out


def lesion_metrics(probs, targets, thresholds):
    """Aggregate lesion metrics: macro/micro F1, macro balanced-acc, macro AUROC/AUPRC, and the
    rare / common breakdowns used by the selection score and the paper tables."""
    per = per_lesion_stats(probs, targets, thresholds)
    y = (np.asarray(targets) > 0.5).astype(int)
    pred = _binarize(probs, thresholds)
    names = LESION_NAMES

    macro_f1 = float(np.mean([per[n]['f1'] for n in names]))
    micro_f1 = f1_score(y, pred, average='micro', zero_division=0)
    macro_bal_acc = float(np.nanmean([per[n]['balanced_acc'] for n in names]))
    macro_auroc = float(np.nanmean([per[n]['auroc'] for n in names]))
    macro_auprc = float(np.nanmean([per[n]['auprc'] for n in names]))

    rare_recall = {names[k]: per[names[k]]['recall'] for k in RARE_LESION_IDX}
    mean_rare_recall = float(np.nanmean([per[names[k]]['recall'] for k in RARE_LESION_IDX]))
    common_f1 = float(np.mean([per[names[k]]['f1'] for k in COMMON_LESION_IDX]))
    rare_f1 = float(np.mean([per[names[k]]['f1'] for k in RARE_LESION_IDX]))

    return {
        'per_lesion': per,
        'macro_f1': macro_f1, 'micro_f1': float(micro_f1), 'macro_balanced_acc': macro_bal_acc,
        'macro_auroc': macro_auroc, 'macro_auprc': macro_auprc,
        'mean_rare_recall': mean_rare_recall, 'rare_recall': rare_recall,
        'common_macro_f1': common_f1, 'rare_macro_f1': rare_f1,
    }


def grade_metrics(preds, targets):
    gp, gt = np.asarray(preds), np.asarray(targets)
    return {
        'accuracy': float(accuracy_score(gt, gp)),
        'balanced_acc': float(balanced_accuracy_score(gt, gp)),
        'quadratic_kappa': float(cohen_kappa_score(gt, gp, weights='quadratic')),
        'mcc': float(matthews_corrcoef(gt, gp)) if len(np.unique(gt)) > 1 else 0.0,
        'per_class_recall': recall_score(gt, gp, labels=list(range(5)), average=None,
                                         zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(gt, gp, labels=list(range(5))).tolist(),
    }


def selection_score(lm, cfg):
    """Clinical composite used for model selection / early stopping (higher is better)."""
    return (cfg.get('sel_rare_weight', 0.5) * lm['mean_rare_recall']
            + cfg.get('sel_f1_weight', 0.5) * lm['macro_f1'])


def full_report(grade_preds, grade_targets, lesion_probs, lesion_targets, thresholds, cfg):
    """Everything for the results JSON: grade block + lesion block + composite + thresholds."""
    lm = lesion_metrics(lesion_probs, lesion_targets, thresholds)
    report = {
        'head_mode': cfg.get('head_mode'),
        'seed': cfg.get('seed'),
        'thresholds': list(np.asarray(thresholds, dtype=float)),
        'grade': grade_metrics(grade_preds, grade_targets),
        'lesion': {k: v for k, v in lm.items() if k != 'per_lesion'},
        'lesion_per_class': lm['per_lesion'],
        'selection_score': selection_score(lm, cfg),
    }
    return report
