"""Aggregate per-seed result JSONs into a summary table and run the pre-registered significance tests.

The primary comparison is Model B (separate) vs Model A' (shared_mlp, capacity-matched) on per-seed
mean rare-recall -- a paired Wilcoxon signed-rank test (t-test as secondary). Common-lesion macro-F1
is the negative-transfer guardrail. A'-A quantifies the pure capacity effect separately.
"""
import glob
import json
import os

import numpy as np

VARIANT_ORDER = ['shared', 'shared_mlp', 'separate', 'adapter']
VARIANT_LABEL = {'shared': "A (Shared-Linear)", 'shared_mlp': "A' (Shared-MLP, capacity-matched)",
                 'separate': "B (Separate rare head)", 'adapter': "C (Adapter)"}

# metric_key -> (path in report dict). Flat access helpers below.
_METRICS = ['mean_rare_recall', 'macro_f1', 'micro_f1', 'macro_auprc', 'macro_auroc',
            'macro_balanced_acc', 'common_macro_f1', 'rare_macro_f1']


def load_reports(results_dir):
    """Return {head_mode: [report, ...]} from every metrics_*.json in results_dir."""
    reports = {}
    for path in sorted(glob.glob(os.path.join(results_dir, 'metrics_*.json'))):
        with open(path) as f:
            rep = json.load(f)
        reports.setdefault(rep.get('head_mode', 'unknown'), []).append(rep)
    return reports


def _vec(reports, head_mode, metric):
    """Per-seed values of a lesion metric for one variant, ordered by seed."""
    reps = sorted(reports.get(head_mode, []), key=lambda r: r.get('seed', 0))
    return np.array([r['lesion'][metric] for r in reps], dtype=float), [r.get('seed') for r in reps]


def summary_table(reports):
    """Mean +/- std of each metric per variant. Returns list of row dicts (for CSV / display)."""
    rows = []
    for hm in VARIANT_ORDER:
        if hm not in reports:
            continue
        row = {'variant': VARIANT_LABEL[hm], 'head_mode': hm, 'n_seeds': len(reports[hm])}
        # params / cost from the first report (identical across seeds)
        r0 = reports[hm][0]
        row['head_params'] = r0.get('params', {}).get('head')
        row['grade_kappa'] = np.mean([r['grade']['quadratic_kappa'] for r in reports[hm]])
        for m in _METRICS:
            vals = np.array([r['lesion'][m] for r in reports[hm]], dtype=float)
            row[f'{m}_mean'] = float(np.nanmean(vals))
            row[f'{m}_std'] = float(np.nanstd(vals))
        rows.append(row)
    return rows


def paired_tests(reports, a='shared_mlp', b='separate', metric='mean_rare_recall'):
    """Paired B-vs-A' test on `metric` across shared seeds. Wilcoxon primary, t-test secondary."""
    va, seeds_a = _vec(reports, b, metric)          # b = separate (treatment)
    vb, seeds_b = _vec(reports, a, metric)          # a = shared_mlp (control)
    # align on common seeds
    common = [s for s in seeds_a if s in seeds_b]
    if len(common) < 2:
        return {'error': f'need >=2 shared seeds; have {common}'}
    ia = {s: i for i, s in enumerate(seeds_a)}
    ib = {s: i for i, s in enumerate(seeds_b)}
    treat = np.array([va[ia[s]] for s in common])
    ctrl = np.array([vb[ib[s]] for s in common])
    diff = treat - ctrl
    res = {'metric': metric, 'treatment': b, 'control': a, 'seeds': common,
           'treatment_mean': float(np.mean(treat)), 'control_mean': float(np.mean(ctrl)),
           'mean_diff': float(np.mean(diff))}
    try:
        from scipy.stats import wilcoxon, ttest_rel
        if np.allclose(diff, 0):
            res['wilcoxon_p'] = 1.0
        else:
            res['wilcoxon_p'] = float(wilcoxon(treat, ctrl, alternative='greater').pvalue)
        res['ttest_p_onesided'] = float(ttest_rel(treat, ctrl).pvalue / 2 if np.mean(diff) > 0
                                        else 1 - ttest_rel(treat, ctrl).pvalue / 2)
    except ImportError:
        res['note'] = 'scipy unavailable; reporting effect size only'
    res['cohens_dz'] = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)) if len(diff) > 1 else 0.0
    return res


def capacity_vs_separation(reports, metric='mean_rare_recall'):
    """Decompose the effect: (A' - A) = capacity contribution, (B - A') = separation contribution."""
    def m(hm):
        v, _ = _vec(reports, hm, metric)
        return float(np.nanmean(v)) if len(v) else float('nan')
    a, ap, b = m('shared'), m('shared_mlp'), m('separate')
    return {'metric': metric, 'A_shared': a, 'Aprime_shared_mlp': ap, 'B_separate': b,
            'capacity_effect_(Aprime-A)': ap - a, 'separation_effect_(B-Aprime)': b - ap}


def write_summary_csv(reports, out_path):
    import csv
    rows = summary_table(reports)
    if not rows:
        return None
    keys = list(rows[0].keys())
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return out_path
