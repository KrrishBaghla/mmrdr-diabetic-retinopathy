"""Checkpointing + resume + autosave to a Kaggle Dataset (survives idle-session wipes).

A checkpoint stores model/optimizer/scheduler state, the epoch, the best selection score, the frozen
per-lesion thresholds, the head_mode/seed, and RNG states -- everything needed to resume a run
bit-for-bit. `push_to_dataset` versions the checkpoints into a Kaggle Dataset every few epochs and on
each new best; credentials come from ~/.kaggle/kaggle.json, env vars, or Kaggle Secrets (never
hard-coded).
"""
import glob
import json
import os
import shutil
import subprocess

import torch


def save_checkpoint(state, is_best, ckpt_dir, tag, filename='checkpoint.pth'):
    """Write per-run last checkpoint and, on a new best, a best checkpoint. `tag` = <head_mode>_s<seed>."""
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, f'last_{tag}.pth')
    torch.save(state, last_path)
    torch.save(state, os.path.join(ckpt_dir, filename))        # generic name for resume autodetect
    if is_best:
        best_path = os.path.join(ckpt_dir, f'best_{tag}.pth')
        torch.save(state, best_path)
        print(f"  [*] new best -> {best_path}")
    return last_path


def find_resume_checkpoint(ckpt_dir, tag):
    """Newest TAG-SPECIFIC checkpoint: local working copy first, then any attached Kaggle input dataset.
    Tag-specific only (never the generic checkpoint.pth) so looping over variants can't cross-resume."""
    local = os.path.join(ckpt_dir, f'last_{tag}.pth')
    if os.path.exists(local):
        return local
    hits = sorted(glob.glob(f'/kaggle/input/**/last_{tag}.pth', recursive=True),
                  key=os.path.getmtime, reverse=True)
    return hits[0] if hits else None


def restore_log(logs_dir, tag, start_epoch):
    """On resume, seed the working CSV with the backed-up per-epoch rows from the attached checkpoint
    dataset, so the learning curve accumulates across sessions instead of fragmenting (Kaggle wipes
    /kaggle/working, so a fresh session would otherwise log only its own epochs). Keeps only rows with
    epoch <= start_epoch (the epochs the restored checkpoint has actually completed) so the subsequent
    append -- which starts at start_epoch+1 -- can never duplicate a row. No-op on a fresh start."""
    local = os.path.join(logs_dir, f'train_{tag}.csv')
    if start_epoch <= 0 or os.path.exists(local):
        return
    hits = sorted(glob.glob(f'/kaggle/input/**/train_{tag}.csv', recursive=True),
                  key=os.path.getmtime, reverse=True)
    if not hits:
        return
    import csv
    with open(hits[0], newline='') as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return
    header, kept = rows[0], [r for r in rows[1:] if r and r[0].isdigit() and int(r[0]) <= start_epoch]
    os.makedirs(logs_dir, exist_ok=True)
    with open(local, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(kept)
    print(f"[*] Restored {len(kept)} prior log rows for {tag} (epochs <= {start_epoch}) from {hits[0]}.")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    """Restore state. Returns (start_epoch, best_score, thresholds) or (0, -inf, None) if absent."""
    if not path or not os.path.exists(path):
        print(f"[!] No checkpoint at {path}; starting fresh.")
        return 0, float('-inf'), None
    ck = torch.load(path, map_location=device, weights_only=False)
    (model.module if hasattr(model, 'module') else model).load_state_dict(_strip(ck['state_dict']))
    if optimizer is not None and ck.get('optimizer'):
        optimizer.load_state_dict(ck['optimizer'])
    if scheduler is not None and ck.get('scheduler'):
        scheduler.load_state_dict(ck['scheduler'])
    print(f"[*] Resumed from {path} at epoch {ck['epoch']} (best_score {ck.get('best_score', 0):.4f}).")
    return ck['epoch'], ck.get('best_score', float('-inf')), ck.get('thresholds')


def _strip(state_dict):
    """Drop a leading 'module.' so a DataParallel-saved checkpoint loads into a bare module too."""
    if all(k.startswith('module.') for k in state_dict):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


# ------------------------------------------------------------------ Kaggle Dataset autosave
def ensure_kaggle_auth():
    """Authenticate the kaggle CLI without hard-coding the key. Priority: kaggle.json > env >
    Kaggle Secrets (KAGGLE_USERNAME / KAGGLE_KEY)."""
    if os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        return True
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        return True
    try:
        from kaggle_secrets import UserSecretsClient
        us = UserSecretsClient()
        os.environ['KAGGLE_USERNAME'] = us.get_secret('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = us.get_secret('KAGGLE_KEY')
        return True
    except Exception as e:                                       # noqa: BLE001 (best-effort, non-fatal)
        print('   [backup] no Kaggle creds -- add KAGGLE_USERNAME + KAGGLE_KEY as Secrets:', str(e)[:100])
        return False


def push_to_dataset(cfg, extra_files=(), reason=''):
    """Version the run's checkpoints/logs into the configured Kaggle Dataset. No-op if disabled or
    unauthenticated; all failures are non-fatal so training never crashes on a backup hiccup."""
    if not cfg.get('backup_to_dataset'):
        return
    if not ensure_kaggle_auth():
        return
    slug = cfg['backup_dataset_slug']
    ckpt_dir = cfg['ckpt_dir']
    workdir = os.path.join(ckpt_dir, 'ckpt_backup')
    os.makedirs(workdir, exist_ok=True)
    staged = 0
    for src in list(glob.glob(os.path.join(ckpt_dir, '*.pth'))) + list(extra_files):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(workdir, os.path.basename(src)))
            staged += 1
    if staged == 0:
        print('   [backup] nothing to push yet')
        return
    with open(os.path.join(workdir, 'dataset-metadata.json'), 'w') as fh:
        json.dump({'title': slug.split('/')[-1], 'id': slug, 'licenses': [{'name': 'CC0-1.0'}]}, fh)
    ver = subprocess.run(['kaggle', 'datasets', 'version', '-p', workdir, '-m', reason or 'auto',
                          '--dir-mode', 'zip'], capture_output=True, text=True)
    if ver.returncode != 0:
        crt = subprocess.run(['kaggle', 'datasets', 'create', '-p', workdir, '--dir-mode', 'zip'],
                             capture_output=True, text=True)
        ok, out = crt.returncode == 0, (crt.stdout or crt.stderr)
    else:
        ok, out = True, (ver.stdout or ver.stderr)
    tail = out.strip().splitlines()[-1][:160] if out.strip() else ''
    print('   [backup]', 'OK' if ok else 'FAILED', '->', slug, '|', tail)
