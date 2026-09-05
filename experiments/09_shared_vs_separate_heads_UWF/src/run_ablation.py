"""Local orchestrator for the shared-vs-separate-heads ablation matrix.

Runs {variants} x {seeds}, writing per-run metrics_<variant>_s<seed>.json into results/, then prints
the aggregated summary table and the pre-registered paired significance test (B vs A').

Examples
--------
    # full primary matrix (A, A', B) x (42, 1, 2023)
    Diabetic_env\\Scripts\\python.exe -m experiment_07_shared_vs_separate_heads.run_ablation \
        --variants shared shared_mlp separate --seeds 42 1 2023

    # one quick run
    ...run_ablation --variants separate --seeds 42 --epochs 3

    # only aggregate existing result JSONs (no training)
    ...run_ablation --aggregate_only
"""
import argparse
import json
import os

import pandas as pd

from .config import load_config
from .training.data import find_uwf_csv, three_way_split
from .training.train_loop import train_variant
from .evaluation.stats import (load_reports, summary_table, paired_tests, capacity_vs_separation,
                               write_summary_csv)

HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = {'ckpt': os.path.join(HERE, 'checkpoints'),
        'logs': os.path.join(HERE, 'logs'),
        'results': os.path.join(HERE, 'results'),
        'plots': os.path.join(HERE, 'plots')}


def run_matrix(variants, seeds, overrides):
    csv_path = find_uwf_csv()
    if csv_path is None:
        raise SystemExit("UWF.csv not found; cannot train. (Aggregation still works with --aggregate_only.)")
    df = pd.read_csv(csv_path)
    img_dir = os.path.dirname(csv_path)
    base = load_config(**overrides)
    train_df, val_df, test_df = three_way_split(df, base)      # frozen split, reused by every run
    for hm in variants:
        for sd in seeds:
            cfg = load_config(head_mode=hm, seed=sd, **overrides)
            print(f"\n{'=' * 70}\nRUN  head_mode={hm}  seed={sd}\n{'=' * 70}")
            train_variant(cfg, train_df, val_df, test_df, img_dir, DIRS)


def aggregate():
    reports = load_reports(DIRS['results'])
    if not reports:
        print("No result JSONs found in results/. Run the matrix first.")
        return
    rows = summary_table(reports)
    print("\n===== SUMMARY (mean over seeds) =====")
    for r in rows:
        print(f"{r['variant']:38s} | rare-recall {r['mean_rare_recall_mean']:.3f}"
              f"+/-{r['mean_rare_recall_std']:.3f} | macro-F1 {r['macro_f1_mean']:.3f}"
              f"+/-{r['macro_f1_std']:.3f} | PR-AUC {r['macro_auprc_mean']:.3f}"
              f" | common-F1 {r['common_macro_f1_mean']:.3f} | head params {r['head_params']:,}")
    csv_out = write_summary_csv(reports, os.path.join(DIRS['results'], 'summary_table.csv'))
    print("summary ->", csv_out)
    print("\n===== SIGNIFICANCE (B separate vs A' shared_mlp) =====")
    for metric in ['mean_rare_recall', 'macro_f1', 'common_macro_f1']:
        print(metric, '->', json.dumps(paired_tests(reports, metric=metric), default=float))
    print("\n===== CAPACITY vs SEPARATION decomposition =====")
    print(json.dumps(capacity_vs_separation(reports), indent=2, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variants', nargs='+', default=['shared', 'shared_mlp', 'separate'],
                    choices=['shared', 'shared_mlp', 'separate', 'adapter'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 1, 2023])
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--no_backup', action='store_true', help='disable Kaggle dataset autosave')
    ap.add_argument('--aggregate_only', action='store_true', help='skip training, just summarize results/')
    args = ap.parse_args()

    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    overrides = {'run_training': True}
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.no_backup:
        overrides['backup_to_dataset'] = False

    if not args.aggregate_only:
        run_matrix(args.variants, args.seeds, overrides)
    aggregate()


if __name__ == '__main__':
    main()
