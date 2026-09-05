# Standalone training scripts

Two argparse CLIs implementing architectures referenced elsewhere in this repository. Both resolve `MMRDR/`
relative to their own location, split train/test with a fixed `random_state=42`, and checkpoint the best
model by validation accuracy.

### `alternate_architechture.py` — MS-LAN

`MultiScaleLesionAttentionNet`: a dual-branch ResNet-50. One branch keeps early, pre-maxpool features at
256×256 to preserve microaneurysm-scale detail; the other runs the full trunk for global context. Both are
channel-aligned to 512×16×16 and fused before splitting into a grade head and a 7-lesion head. Trained with
`ClassBalancedFocalLoss`. This is the reference implementation behind
`../../experiments/01_multiscale_dual_branch_CFP/`.

```bash
python alternate_architechture.py --modality cfp --epochs 50 --batch_size 256
```

> **Known issue, left in place deliberately.** The grade class weights are hardcoded from the CFP training
> split as `[6579, 1302, 1959, 603, 1553]`. The last value is **UWF's** grade-4 count; CFP's true count is
> **675**. Recompute per modality before reusing this for grade weighting. The error is documented rather
> than silently patched because the affected baseline numbers were reported with it in place.

### `train_gnn.py` — ResNet-GCN

`ResNetGCNClassifier`: a ResNet-18 backbone whose final feature map is flattened into a 16×16 grid of graph
nodes, with a self-attention-derived adjacency matrix feeding two graph-convolution layers plus a residual
projection to counter over-smoothing. Grade-only, CFP-only, designed to fit in 6 GB of VRAM. An exploratory
direction that was not carried into the main experiment series.

```bash
python train_gnn.py --modality cfp --epochs 25 --batch_size 8
```
