"""Data pipeline -- UWF CSV loading, elliptical masking, augmentation, and a FROZEN three-way
rare-stratified split (train/val/test).

The split is generated once with `split_seed` (constant across all runs) and stratified on the
rare-lesion indicator so RD (only 238 positives) is guaranteed to appear in every fold. The model
seed varies per run but never changes which images land where -- decoupling "which data" from
"which init" so all variants train and evaluate on identical images.
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

RARE_LESION_IDX = [4, 5, 6]
LESION_NAMES = ['MA', 'HE', 'IH', 'VB/IRMA', 'NV', 'VH', 'RD']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_uwf_csv():
    """Locate UWF.csv: Kaggle glob autodetect first, then local repo fallbacks. None if absent."""
    for pat in ['/kaggle/input/**/UWF.csv', '/kaggle/input/**/MMRDR-UWF/UWF.csv']:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    # `__file__` is absent inside a Jupyter cell (the notebook embeds this function), so fall back to cwd
    here = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    for cand in ['../MMRDR/MMRDR-UWF/UWF.csv', 'MMRDR/MMRDR-UWF/UWF.csv',
                 '../../MMRDR/MMRDR-UWF/UWF.csv',
                 os.path.join(here, '..', '..', 'MMRDR', 'MMRDR-UWF', 'UWF.csv')]:
        if os.path.exists(cand):
            return cand
    return None


def parse_lesion(s):
    """Parse the stringified 7-hot lesion vector into a float32 binary array (>0 -> 1)."""
    s = str(s).strip().strip('[]')
    vals = [float(x) for x in s.split(',') if x.strip() != '']
    v = np.array(vals, dtype=np.float32)
    if v.shape[0] != 7:
        v = np.zeros(7, dtype=np.float32)
    return (v > 0).astype(np.float32)


def apply_uwf_mask(pil_img, tightness=0.98):
    """Elliptical crop that strips eyelid/eyelash/vignette artifacts from UWF images (folder-2/3)."""
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_img
    largest = max(contours, key=cv2.contourArea)
    if len(largest) >= 5:
        e = cv2.fitEllipse(largest)
        se = (e[0], (e[1][0] * tightness, e[1][1] * tightness), e[2])
        mask = np.zeros_like(gray)
        cv2.ellipse(mask, se, 255, -1)
        return Image.fromarray(cv2.bitwise_and(img, img, mask=mask))
    return pil_img


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


class UWF_Dataset(Dataset):
    """Returns (image, grade:int, lesion:[7], has_rare:float, idx). Pre-extracts numpy columns to
    avoid the per-worker DataFrame memory creep (#13246)."""

    def __init__(self, df, img_dir, transform=None, img_size=512):
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
        # Fixed-width unicode array (a contiguous C buffer), NOT an object array of Python str. A str
        # object array leaks host RAM across epochs: every worker __getitem__ bumps the shared str's
        # refcount, forcing a copy-on-write page copy that persistent_workers never releases (#13246).
        self.paths = np.asarray(df['image'].astype(str).tolist())
        self.grades = df['grade'].to_numpy().astype(np.int64)
        self.lesions = np.stack([parse_lesion(x) for x in df['lesion'].values]).astype(np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.paths[idx]) if self.img_dir else ''
        if (not self.img_dir) or (not os.path.exists(img_path)):
            image = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))     # graceful stub
        else:
            image = Image.open(img_path).convert('RGB').resize(
                (self.img_size, self.img_size), Image.Resampling.BILINEAR)
        image = apply_uwf_mask(image)
        if self.transform:
            image = self.transform(image)
        les = self.lesions[idx]
        has_rare = 1.0 if les[RARE_LESION_IDX].sum() > 0 else 0.0
        return (image, int(self.grades[idx]), torch.from_numpy(les.copy()),
                torch.tensor(has_rare, dtype=torch.float32), idx)


def rare_indicator(df):
    return df['lesion'].apply(lambda s: 1 if parse_lesion(s)[RARE_LESION_IDX].sum() > 0 else 0)


def three_way_split(df, cfg):
    """Frozen rare-stratified train/val/test split. `split_seed` fixes the partition for all runs."""
    seed = cfg.get('split_seed', 42)
    strat = rare_indicator(df) if cfg.get('stratify', True) else None
    # 1) hold out the test set
    train_val_df, test_df = train_test_split(
        df, test_size=cfg['test_size'], random_state=seed, stratify=strat)
    # 2) carve val out of the remainder (val fraction expressed relative to the remainder)
    val_frac = cfg['val_size'] / (1.0 - cfg['test_size'])
    strat2 = rare_indicator(train_val_df) if cfg.get('stratify', True) else None
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_frac, random_state=seed, stratify=strat2)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_loaders(train_df, val_df, test_df, img_dir, cfg):
    train_tf, eval_tf = build_transforms()
    sz, bs, nw = cfg['img_size'], cfg['batch_size'], cfg['num_workers']
    # apply_uwf_mask (OpenCV contour+ellipse per image, every epoch) is CPU-heavy, so on dual-T4 the
    # GPUs starve unless the loader prefetches aggressively and keeps workers alive between epochs.
    loader_kw = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=cfg.get('prefetch_factor', 2))
    train_loader = DataLoader(UWF_Dataset(train_df, img_dir, train_tf, sz), batch_size=bs,
                              shuffle=True, drop_last=False, **loader_kw)
    val_loader = DataLoader(UWF_Dataset(val_df, img_dir, eval_tf, sz), batch_size=bs,
                            shuffle=False, **loader_kw)
    test_loader = DataLoader(UWF_Dataset(test_df, img_dir, eval_tf, sz), batch_size=bs,
                             shuffle=False, **loader_kw)
    return train_loader, val_loader, test_loader


def split_signature(train_df, val_df, test_df):
    """A tiny hash of the split membership so runs can assert they used the identical partition."""
    import hashlib
    key = '|'.join(sorted(train_df['image'])) + '#' + '|'.join(sorted(val_df['image'])) \
        + '#' + '|'.join(sorted(test_df['image']))
    return hashlib.md5(key.encode()).hexdigest()[:12]
