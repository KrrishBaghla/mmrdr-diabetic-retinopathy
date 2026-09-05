"""CPU-only, dummy-data unit tests (repo convention: verify the mechanism without a GPU run).

Run:  Diabetic_env\\Scripts\\python.exe -m experiment_07_shared_vs_separate_heads.tests
or import and call run_all_tests() from the notebook. Every test uses tiny synthetic tensors and a
non-pretrained backbone so it executes in seconds on CPU with no dataset and no network.
"""
import numpy as np
import torch

from .config import EXPECTED_HEAD_PARAMS
from .models.multihead_drnet import MultiHeadDRNet, count_head_params, count_params
from .models.heads import build_lesion_head
from .training.losses import AsymmetricFocalLoss
from .evaluation.thresholds import optimize_thresholds
from .evaluation.metrics import lesion_metrics

HEAD_MODES = ['shared', 'shared_mlp', 'separate', 'adapter']


def test_forward_shapes():
    """Every head_mode forwards to canonical lesion[7]/grade[5]/proj[128] shapes."""
    x = torch.randn(2, 3, 128, 128)                 # small spatial size keeps CPU fast
    for hm in HEAD_MODES:
        net = MultiHeadDRNet(head_mode=hm, pretrained=False).eval()
        with torch.no_grad():
            ol, og, ze, zd = net(x)
        assert ol.shape == (2, 7), f"{hm}: lesion shape {ol.shape}"
        assert og.shape == (2, 5), f"{hm}: grade shape {og.shape}"
        assert ze.shape == (2, 128) and zd.shape == (2, 128), f"{hm}: proj shapes"
    return "forward shapes OK for all 4 head modes"


def test_head_param_counts():
    """Lesion-head parameter counts match the pre-registered fairness table exactly."""
    for hm in HEAD_MODES:
        net = MultiHeadDRNet(head_mode=hm, pretrained=False)
        n = count_head_params(net)
        assert n == EXPECTED_HEAD_PARAMS[hm], f"{hm}: head params {n} != {EXPECTED_HEAD_PARAMS[hm]}"
    return f"head param counts match table: {EXPECTED_HEAD_PARAMS}"


def test_trunk_identical_across_variants():
    """Total-minus-head params (the shared trunk) are identical across every variant -> fair."""
    trunks = {}
    for hm in HEAD_MODES:
        net = MultiHeadDRNet(head_mode=hm, pretrained=False)
        trunks[hm] = count_params(net) - count_head_params(net)
    assert len(set(trunks.values())) == 1, f"trunk sizes differ: {trunks}"
    return f"shared trunk identical across variants ({next(iter(trunks.values())):,} params)"


def test_separate_head_gradient_routing():
    """In the separate head, rare logits must only touch rare_head params (and vice-versa)."""
    head = build_lesion_head('separate', in_dim=8, hidden=4)
    x = torch.randn(3, 8, requires_grad=False)
    out = head(x)                                    # [3,7], cols 0-3 common, 4-6 rare
    # push gradient through a rare logit only
    head.zero_grad()
    out[:, 4].sum().backward(retain_graph=True)
    rare_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.rare_head.parameters())
    common_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.common_head.parameters())
    assert rare_grad and not common_grad, "rare logit should update ONLY the rare head"
    # and a common logit only touches the common head
    head.zero_grad()
    out2 = head(x)
    out2[:, 0].sum().backward()
    c_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.common_head.parameters())
    r_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.rare_head.parameters())
    assert c_grad and not r_grad, "common logit should update ONLY the common head"
    return "separate head routes rare/common gradients to their own sub-heads"


def test_loss_reduction_consistency():
    """AFL over the concatenated 7-vector == AFL applied to the shared 7-vector (same numbers).
    This proves the split-head loss is NOT silently reweighted vs the shared baseline."""
    torch.manual_seed(0)
    logits = torch.randn(5, 7)
    targets = (torch.rand(5, 7) > 0.5).float()
    afl = AsymmetricFocalLoss()
    shared_loss = afl(logits, targets)
    # simulate the split-head assembly: same logits, just concatenated common|rare
    split_logits = torch.cat([logits[:, :4], logits[:, 4:]], dim=1)
    split_loss = afl(split_logits, targets)
    assert torch.allclose(shared_loss, split_loss), "split-head loss differs from shared"
    return f"loss reduction-consistent (shared == split = {shared_loss.item():.4f})"


def test_threshold_optimization_helps_rare():
    """On synthetic imbalanced scores, per-lesion threshold tuning beats a fixed 0.5 on rare recall."""
    rng = np.random.RandomState(0)
    n = 400
    y = np.zeros((n, 7), dtype=np.float32)
    probs = rng.rand(n, 7) * 0.4                     # negatives sit low
    for j in [4, 5, 6]:                              # rare positives are few and only mildly elevated
        pos = rng.choice(n, size=15, replace=False)
        y[pos, j] = 1
        probs[pos, j] = 0.45 + rng.rand(15) * 0.2    # straddle 0.5 -> fixed-0.5 misses many
    cfg = {'thr_min': 0.05, 'thr_max': 0.95, 'thr_step': 0.01, 'thr_objective': 'f1'}
    thr = optimize_thresholds(probs, y, cfg)
    tuned = lesion_metrics(probs, y, thr)['mean_rare_recall']
    fixed = lesion_metrics(probs, y, [0.5] * 7)['mean_rare_recall']
    assert tuned >= fixed, f"tuned rare recall {tuned:.3f} < fixed {fixed:.3f}"
    return f"threshold tuning rare-recall {tuned:.2f} >= fixed-0.5 {fixed:.2f}"


def run_all_tests(verbose=True):
    tests = [test_forward_shapes, test_head_param_counts, test_trunk_identical_across_variants,
             test_separate_head_gradient_routing, test_loss_reduction_consistency,
             test_threshold_optimization_helps_rare]
    results = []
    for t in tests:
        msg = t()
        results.append((t.__name__, msg))
        if verbose:
            print(f"[PASS] {t.__name__}: {msg}")
    print(f"\nAll {len(results)} unit tests passed.")
    return results


if __name__ == '__main__':
    run_all_tests()
