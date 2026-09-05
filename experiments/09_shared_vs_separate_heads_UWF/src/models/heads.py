"""Lesion-head variants for the shared-vs-separate ablation.

Every head consumes the SAME 512-d pooled FPN feature and emits logits for the 7 lesions in the
canonical order [MA, HE, IH, VB/IRMA, NV, VH, RD]. Because the common indices (0-3) precede the rare
indices (4-6), the split heads assemble their output by a plain concatenation [common | rare], which
is exactly canonical order and is fully differentiable (no in-place scatter needed).

Only this module changes between ablation variants; the trunk/grade-head/SupCon heads are frozen.
"""
import torch
import torch.nn as nn

COMMON_IDX = (0, 1, 2, 3)     # MA, HE, IH, VB/IRMA
RARE_IDX = (4, 5, 6)          # NV, VH, RD (PDR-defining)


class SharedLinearHead(nn.Module):
    """Model A -- single Linear(512, 7) over all lesions (the current-pipeline baseline)."""

    def __init__(self, in_dim=512, n_lesions=7, **_):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_lesions)

    def forward(self, x):
        return self.fc(x)


class SharedMLPHead(nn.Module):
    """Model A' -- 2-layer MLP over all 7 lesions. The CAPACITY-MATCHED control for Model B: it has
    essentially the same head parameter budget as the separate head but keeps all lesions shared, so
    B - A' isolates *separation* from *capacity*."""

    def __init__(self, in_dim=512, hidden=128, dropout=0.3, n_lesions=7, **_):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_lesions))

    def forward(self, x):
        return self.mlp(x)


class SeparateHeads(nn.Module):
    """Model B -- a linear head for the common lesions [MA,HE,IH,VB] and a dedicated MLP head for the
    rare PDR lesions [NV,VH,RD]. The hypothesis: decoupling the rare classifier frees it from gradient
    domination by the 6-30x more frequent common lesions."""

    def __init__(self, in_dim=512, hidden=128, dropout=0.3, common_idx=COMMON_IDX, rare_idx=RARE_IDX, **_):
        super().__init__()
        assert list(common_idx) == list(range(len(common_idx))), "common indices must be a 0..k prefix"
        assert list(rare_idx) == list(range(len(common_idx), len(common_idx) + len(rare_idx))), \
            "rare indices must directly follow the common ones (so concat == canonical order)"
        self.common_head = nn.Linear(in_dim, len(common_idx))
        self.rare_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, len(rare_idx)))

    def forward(self, x):
        return torch.cat([self.common_head(x), self.rare_head(x)], dim=1)


class AdapterHeads(nn.Module):
    """Model C -- a shared MLP body followed by lightweight per-group adapters (a linear common
    adapter and a small MLP rare adapter). The soft-split middle ground between one shared head and
    two fully separate heads (Houlsby-style task adapters)."""

    def __init__(self, in_dim=512, hidden=128, adapter_dim=64, dropout=0.3,
                 common_idx=COMMON_IDX, rare_idx=RARE_IDX, **_):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.common_adapter = nn.Linear(hidden, len(common_idx))
        self.rare_adapter = nn.Sequential(
            nn.Linear(hidden, adapter_dim), nn.ReLU(), nn.Linear(adapter_dim, len(rare_idx)))

    def forward(self, x):
        h = self.body(x)
        return torch.cat([self.common_adapter(h), self.rare_adapter(h)], dim=1)


def build_lesion_head(head_mode, in_dim=512, hidden=128, adapter_dim=64, dropout=0.3):
    """Factory: map a head_mode string to the corresponding lesion-head module."""
    hm = str(head_mode).lower()
    if hm == 'shared':
        return SharedLinearHead(in_dim=in_dim)
    if hm == 'shared_mlp':
        return SharedMLPHead(in_dim=in_dim, hidden=hidden, dropout=dropout)
    if hm == 'separate':
        return SeparateHeads(in_dim=in_dim, hidden=hidden, dropout=dropout)
    if hm == 'adapter':
        return AdapterHeads(in_dim=in_dim, hidden=hidden, adapter_dim=adapter_dim, dropout=dropout)
    raise ValueError(f"unknown head_mode '{head_mode}' (use shared|shared_mlp|separate|adapter)")
