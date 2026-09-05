"""Publication plots: loss/learning curves, ROC & PR curves, confusion matrices, and the headline
rare-recall comparison across variants. All save to the plots/ directory; every function no-ops
gracefully if its inputs are missing so the notebook always renders.
"""
import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:                                    # pragma: no cover
    plt = None

from sklearn.metrics import precision_recall_curve, roc_curve

LESION_NAMES = ['MA', 'HE', 'IH', 'VB/IRMA', 'NV', 'VH', 'RD']
RARE_LESION_IDX = [4, 5, 6]


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_learning_curves(csv_path, out_dir, tag=''):
    """Train/val loss + rare-recall trajectory from a train_<tag>.csv log."""
    if plt is None or not os.path.exists(csv_path):
        return None
    import pandas as pd
    d = pd.read_csv(csv_path)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(d['epoch'], d['train_loss'], label='train', color='#4C78A8')
    ax[0].plot(d['epoch'], d['val_loss'], label='val', color='#E45756')
    ax[0].set_title(f'Loss curves {tag}'); ax[0].set_xlabel('epoch'); ax[0].set_ylabel('loss'); ax[0].legend()
    for c, col in zip(['recall_NV', 'recall_VH', 'recall_RD'], ['#54A24B', '#EECA3B', '#B279A2']):
        if c in d:
            ax[1].plot(d['epoch'], d[c], marker='o', ms=3, label=c.replace('recall_', ''), color=col)
    ax[1].plot(d['epoch'], d['mean_rare_recall'], color='#333', lw=2, ls='--', label='mean')
    ax[1].set_title(f'Rare-lesion recall {tag}'); ax[1].set_xlabel('epoch'); ax[1].set_ylabel('recall')
    ax[1].set_ylim(-0.05, 1.05); ax[1].legend()
    fig.tight_layout()
    return _save(fig, out_dir, f'learning_curves_{tag}.png')


def plot_roc_pr(lesion_probs, lesion_targets, out_dir, tag='', idxs=RARE_LESION_IDX):
    """ROC and PR curves for the selected (rare, by default) lesions."""
    if plt is None:
        return None
    probs = np.asarray(lesion_probs); y = (np.asarray(lesion_targets) > 0.5).astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for j in idxs:
        if y[:, j].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y[:, j], probs[:, j])
        ax[0].plot(fpr, tpr, label=LESION_NAMES[j])
        prec, rec, _ = precision_recall_curve(y[:, j], probs[:, j])
        ax[1].plot(rec, prec, label=LESION_NAMES[j])
    ax[0].plot([0, 1], [0, 1], ls=':', color='grey')
    ax[0].set_title(f'ROC {tag}'); ax[0].set_xlabel('FPR'); ax[0].set_ylabel('TPR'); ax[0].legend()
    ax[1].set_title(f'PR {tag}'); ax[1].set_xlabel('recall'); ax[1].set_ylabel('precision'); ax[1].legend()
    fig.tight_layout()
    return _save(fig, out_dir, f'roc_pr_{tag}.png')


def plot_confusion(grade_cm, out_dir, tag=''):
    if plt is None or grade_cm is None:
        return None
    cm = np.asarray(grade_cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(cm.shape[0]):
        for k in range(cm.shape[1]):
            ax.text(k, i, int(cm[i, k]), ha='center', va='center',
                    color='white' if cm[i, k] > cm.max() / 2 else 'black')
    ax.set_title(f'Grade confusion {tag}'); ax.set_xlabel('pred'); ax.set_ylabel('true')
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    return _save(fig, out_dir, f'confusion_grade_{tag}.png')


def plot_variant_comparison(summary_rows, out_dir, name='variant_comparison.png'):
    """Grouped bars: mean rare-recall and macro-F1 (with std error bars) across variants."""
    if plt is None or not summary_rows:
        return None
    labels = [r['head_mode'] for r in summary_rows]
    rr = [r['mean_rare_recall_mean'] for r in summary_rows]
    rr_e = [r['mean_rare_recall_std'] for r in summary_rows]
    f1 = [r['macro_f1_mean'] for r in summary_rows]
    f1_e = [r['macro_f1_std'] for r in summary_rows]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(labels)), 4.5))
    ax.bar(x - w / 2, rr, w, yerr=rr_e, capsize=4, label='mean rare-recall', color='#E45756')
    ax.bar(x + w / 2, f1, w, yerr=f1_e, capsize=4, label='macro-F1', color='#4C78A8')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.0); ax.set_title('Shared vs Separate heads -- rare recall & macro-F1'); ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, name)
