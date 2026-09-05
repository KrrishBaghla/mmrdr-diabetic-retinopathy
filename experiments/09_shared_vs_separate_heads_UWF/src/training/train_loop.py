"""Training / validation loop for one ablation run (one head_mode x one seed).

Everything except the lesion head is held identical to the proven exp-3/5 spine: AdamW(1e-4, wd 1e-4),
ReduceLROnPlateau on val loss, AMP, grad-clip 2.0, 60 epochs. Two deliberate, uniformly-applied
upgrades over prior experiments: (1) per-lesion thresholds re-optimized on validation each epoch, and
(2) model selection / early stopping on a CLINICAL composite (rare-recall + macro-F1), not val loss.
"""
import csv
import gc
import json
import os
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .env import device, seed_everything, get_rng_state, LESION_NAMES, RARE_LESION_IDX
from .losses import build_criteria, compute_total_loss
from .checkpoint import (save_checkpoint, load_checkpoint, find_resume_checkpoint, push_to_dataset,
                         restore_log)
from .data import make_loaders, split_signature
from ..models.multihead_drnet import build_model, count_params, count_head_params
from ..evaluation.metrics import lesion_metrics, grade_metrics, selection_score, full_report
from ..evaluation.thresholds import optimize_thresholds


def run_one_epoch(model, loader, optimizer, scaler, criteria, cfg, train=True):
    """One pass. Returns (mean_loss, grade_preds, grade_targets, lesion_probs, lesion_targets)."""
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    gp, gt, lp, lt = [], [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, grades, lesions, rare, _ in tqdm(loader, desc='train' if train else 'val', leave=False):
            images = images.to(device, non_blocking=True)
            grades = grades.to(device, non_blocking=True)
            lesions = lesions.to(device, non_blocking=True)
            rare = rare.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=cfg['use_amp']):
                outputs = model(images)
                loss = compute_total_loss(outputs, (grades, lesions, rare), criteria, cfg)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            bs = images.size(0)
            total_loss += loss.item() * bs
            n += bs
            if not train:                                   # train-pass preds are unused; skip to save RAM
                og, ol = outputs[1], outputs[0]
                gp.extend(torch.argmax(og, 1).detach().cpu().numpy())
                gt.extend(grades.cpu().numpy())
                lp.extend(torch.sigmoid(ol).detach().float().cpu().numpy())
                lt.extend(lesions.cpu().numpy())
            del outputs, loss
    return total_loss / max(n, 1), gp, gt, np.array(lp), np.array(lt)


def train_variant(cfg, train_df, val_df, test_df, img_dir, dirs):
    """Train one variant end-to-end; save best/last checkpoints; return the final test report dict.

    `dirs` = {'ckpt': ..., 'logs': ..., 'results': ...}. `cfg` must contain head_mode and seed.
    """
    tag = f"{cfg['head_mode']}_s{cfg['seed']}"
    cfg['ckpt_dir'] = dirs['ckpt']
    seed_everything(cfg['seed'])

    train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df, img_dir, cfg)
    sig = split_signature(train_df, val_df, test_df)

    model = build_model(cfg, device)
    n_head = count_head_params(model)
    n_total = count_params(model)
    print(f"[{tag}] head params {n_head:,} | total params {n_total:,} | split_sig {sig}")

    criteria = build_criteria(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler(enabled=cfg['use_amp'])

    start_epoch, best_score, best_thr = 0, float('-inf'), [0.5] * 7
    if cfg.get('resume'):
        rp = find_resume_checkpoint(dirs['ckpt'], tag)
        if rp:
            start_epoch, best_score, saved_thr = load_checkpoint(rp, model, optimizer, scheduler, device)
            if saved_thr:
                best_thr = saved_thr

    restore_log(dirs['logs'], tag, start_epoch)          # rebuild the full curve across resumed sessions
    log_path = os.path.join(dirs['logs'], f'train_{tag}.csv')
    with open(log_path, 'a', newline='') as f:
        if os.stat(log_path).st_size == 0:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'macro_f1', 'macro_auprc',
                                    'mean_rare_recall', 'recall_NV', 'recall_VH', 'recall_RD',
                                    'grade_kappa', 'select_score', 'lr', 'sec'])

    patience, t_start = 0, time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, cfg['epochs']):
        # graceful wall-clock stop: if we would run past the session budget, save & bail WITHOUT writing
        # the results JSON, so this variant is resumed (not skipped) on the next run-all.
        if cfg.get('stop_time') and time.time() > cfg['stop_time']:
            print(f"[{tag}] time budget reached at epoch {epoch + 1}; checkpoint saved, will resume next "
                  f"session (re-run the notebook with the checkpoint dataset attached).")
            if cfg.get('backup_to_dataset'):
                push_to_dataset(cfg, extra_files=[log_path], reason=f"{tag} budget-stop ep{epoch + 1}")
            return None
        t0 = time.time()
        tr_loss, *_ = run_one_epoch(model, train_loader, optimizer, scaler, criteria, cfg, train=True)
        val_loss, gp, gt, lp, lt = run_one_epoch(model, val_loader, optimizer, scaler, criteria, cfg,
                                                  train=False)
        # (1) optimize thresholds on validation, (2) score with the clinical composite
        thr = optimize_thresholds(lp, lt, cfg)
        lm = lesion_metrics(lp, lt, thr)
        gm = grade_metrics(gp, gt)
        score = selection_score(lm, cfg)
        elapsed = time.time() - t0

        is_best = score > best_score
        if is_best:
            best_score, best_thr, patience = score, thr, 0
        else:
            patience += 1

        gap = tr_loss - val_loss                            # overfitting monitor (train << val => overfit)
        warn = "  [!] OVERFIT WARNING (train-val loss gap large)" if gap < -0.15 else ""
        rr = lm['rare_recall']
        star = "   <-- NEW BEST" if is_best else ""
        print(f"\n[{tag}] Epoch {epoch + 1:02d}/{cfg['epochs']}  lr {optimizer.param_groups[0]['lr']:.1e}"
              f"  {elapsed:.0f}s{star}")
        print(f"   loss  train {tr_loss:.4f}  val {val_loss:.4f}  (best_score {best_score:.4f},"
              f" patience {patience}/{cfg['patience']}){warn}")
        print(f"   lesion macro-F1 {lm['macro_f1']:.3f}  PR-AUC {lm['macro_auprc']:.3f}"
              f"  AUROC {lm['macro_auroc']:.3f}  bal-acc {lm['macro_balanced_acc']:.3f}")
        print(f"   grade  kappa {gm['quadratic_kappa']:.3f}  acc {gm['accuracy']:.3f}")
        print(f"   RARE recall  NV {_f(rr['NV'])}  VH {_f(rr['VH'])}  RD {_f(rr['RD'])}"
              f"   (mean {lm['mean_rare_recall']:.3f}, HEADLINE)")

        scheduler.step(val_loss)
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'best_score': best_score, 'thresholds': best_thr,
                 'head_mode': cfg['head_mode'], 'seed': cfg['seed'], 'rng': get_rng_state()}
        save_checkpoint(state, is_best, dirs['ckpt'], tag)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f"{tr_loss:.4f}", f"{val_loss:.4f}",
                                    f"{lm['macro_f1']:.4f}", f"{lm['macro_auprc']:.4f}",
                                    f"{lm['mean_rare_recall']:.4f}", _r(rr['NV']), _r(rr['VH']),
                                    _r(rr['RD']), f"{gm['quadratic_kappa']:.4f}", f"{score:.4f}",
                                    f"{optimizer.param_groups[0]['lr']:.2e}", f"{elapsed:.0f}"])
        if cfg.get('backup_to_dataset') and ((epoch + 1) % cfg.get('backup_every', 2) == 0 or is_best):
            push_to_dataset(cfg, extra_files=[log_path], reason=f"{tag} ep{epoch + 1} score {score:.4f}")
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if patience >= cfg['patience']:
            print(f"[{tag}] Early stopping at epoch {epoch + 1}.")
            break

    # ---- final evaluation on TEST with the frozen best-validation thresholds ----
    best_path = os.path.join(dirs['ckpt'], f'best_{tag}.pth')
    if os.path.exists(best_path):
        load_checkpoint(best_path, model, device=device)
    te_loss, tgp, tgt, tlp, tlt = run_one_epoch(model, test_loader, optimizer, scaler, criteria, cfg,
                                                 train=False)
    report = full_report(tgp, tgt, tlp, tlt, best_thr, cfg)
    report['test_loss'] = te_loss
    report['params'] = {'head': int(count_head_params(model)), 'total': int(count_params(model))}
    report['train_time_sec'] = round(time.time() - t_start, 1)
    report['peak_gpu_mem_mb'] = (round(torch.cuda.max_memory_allocated() / 1e6, 1)
                                 if device.type == 'cuda' else None)
    report['split_signature'] = sig
    out_json = os.path.join(dirs['results'], f'metrics_{tag}.json')
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f"[{tag}] TEST  macro-F1 {report['lesion']['macro_f1']:.3f}  "
          f"mean-rare-recall {report['lesion']['mean_rare_recall']:.3f}  "
          f"kappa {report['grade']['quadratic_kappa']:.3f}  -> {out_json}")
    if cfg.get('backup_to_dataset'):
        push_to_dataset(cfg, extra_files=[out_json], reason=f"{tag} final")
    return report


def _f(v):
    return f"{v:.2f}" if v == v else "n/a"          # v==v is False only for NaN


def _r(v):
    return f"{v:.4f}" if v == v else "nan"
