"""MultiHeadDRNet -- the mature folder-3 `UWF_SupConDRNet` trunk with a CONFIG-SELECTABLE lesion head.

The backbone (ResNet-50 + FPN), the 512-d pooled feature, the grade head, and the two SupCon
projection heads are byte-identical to the validated baseline. `head_mode` swaps ONLY the lesion
head, so any measured difference is attributable to the head structure alone. `forward` returns the
same 4-tuple signature for every head_mode, keeping the training/eval code entirely variant-agnostic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .heads import build_lesion_head


class MultiHeadDRNet(nn.Module):
    def __init__(self, head_mode='shared', hidden_dim=128, adapter_dim=64, dropout=0.3, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2
        self.layer3, self.layer4 = resnet.layer3, resnet.layer4
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # SupCon projection heads (identical across variants; kept ON for all so their effect is controlled)
        self.proj_early = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128))
        self.proj_deep = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 128))
        # FPN lateral 1x1 convs + top-down fusion
        self.lat_l4 = nn.Conv2d(2048, 512, 1)
        self.lat_l3 = nn.Conv2d(1024, 512, 1)
        self.lat_l2 = nn.Conv2d(512, 512, 1)
        self.lat_l1 = nn.Conv2d(256, 512, 1)
        # the ONE variable component
        self.head_mode = head_mode
        self.classifier_lesions = build_lesion_head(head_mode, 512, hidden_dim, adapter_dim, dropout)
        self.classifier_grade = nn.Linear(512, 5)

    def forward(self, x):
        x = self.stem(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        z_early = self.proj_early(self.global_pool(l2).flatten(1))
        z_deep = self.proj_deep(self.global_pool(l4).flatten(1))
        p4 = self.lat_l4(l4)
        p3 = self.lat_l3(l3) + F.interpolate(p4, size=l3.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.lat_l2(l2) + F.interpolate(p3, size=l2.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.lat_l1(l1) + F.interpolate(p2, size=l1.shape[-2:], mode='bilinear', align_corners=False)
        pooled = self.global_pool(p1).flatten(1)
        return self.classifier_lesions(pooled), self.classifier_grade(pooled), z_early, z_deep


def _unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def count_head_params(model):
    """Trainable parameters in the lesion head only (the ablation's fairness axis)."""
    return sum(p.numel() for p in _unwrap(model).classifier_lesions.parameters())


def count_params(model):
    """Total trainable parameters (trunk + all heads)."""
    return sum(p.numel() for p in _unwrap(model).parameters() if p.requires_grad)


def build_model(cfg, device):
    """Construct the model for cfg['head_mode'], wrap in DataParallel on multi-GPU, move to device."""
    m = MultiHeadDRNet(head_mode=cfg['head_mode'], hidden_dim=cfg.get('hidden_dim', 128),
                       adapter_dim=cfg.get('adapter_dim', 64), dropout=cfg.get('dropout', 0.3),
                       pretrained=cfg.get('pretrained', True))
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
    return m.to(device)
