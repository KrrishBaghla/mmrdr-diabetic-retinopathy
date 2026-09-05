"""Loss functions -- reused verbatim from the mature folder-3 pipeline.

`AsymmetricFocalLoss` penalizes false negatives on the PDR-defining lesions (NV/VH/RD, beta 0.95)
far harder than false positives; `SupConLoss` is supervised contrastive on the binary has-rare-lesion
flag; grade uses cross-entropy. The combined objective is IDENTICAL across every head_mode -- the
split-head forward returns one canonical length-7 logit vector, so the same single-mean AFL applies
to shared and separate variants alike (reduction-consistent).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, beta=(0.7, 0.7, 0.7, 0.8, 0.95, 0.95, 0.95)):
        super().__init__()
        self.gamma = gamma
        # buffer so beta rides to the right device with .to(device) and replicates under DataParallel
        self.register_buffer('beta', torch.tensor(list(beta), dtype=torch.float32))

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = (1 - pt) ** self.gamma * bce
        return torch.where(targets == 1, self.beta * focal, (1 - self.beta) * focal).mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        self_mask = torch.scatter(torch.ones_like(mask), 1,
                                  torch.arange(features.shape[0], device=features.device).view(-1, 1), 0)
        mask = mask * self_mask
        exp_logits = torch.exp(sim) * self_mask
        log_prob = sim - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        denom = mask.sum(1)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        return (-(mask * log_prob).sum(1) / denom).mean()


def build_criteria(cfg, device):
    """Instantiate the three loss objects from the config (identical for every variant)."""
    return {
        'lesion': AsymmetricFocalLoss(gamma=cfg['afl_gamma'], beta=cfg['afl_beta']).to(device),
        'supcon': SupConLoss(temperature=cfg['supcon_temperature']).to(device),
        'grade': nn.CrossEntropyLoss(label_smoothing=cfg.get('grade_label_smoothing', 0.0)),
    }


def compute_total_loss(outputs, targets, criteria, cfg):
    """L = L_lesion + lambda_grade*L_grade + lambda_supcon*(L_supcon_early + L_supcon_deep)."""
    out_lesions, out_grade, z_early, z_deep = outputs
    grades, lesions, rare = targets
    return (criteria['lesion'](out_lesions, lesions)
            + cfg['lambda_grade'] * criteria['grade'](out_grade, grades)
            + cfg['lambda_supcon'] * criteria['supcon'](z_early, rare)
            + cfg['lambda_supcon'] * criteria['supcon'](z_deep, rare))
