# Experiment III — Feature-Pyramid Network with spatial attention on UWF

**Question.** Ultra-widefield imaging captures up to 200° of retina and reveals peripheral pathology that
standard fundus photography misses — which matters precisely for the sight-threatening lesions of
proliferative DR. But peripheral lesions are tiny relative to that field. Can an explicit feature pyramid
plus a spatial-attention gate find them?

**Architecture.** `UWF_MultiScaleDRNet`: a ResNet-50 backbone split into stem and `layer1`–`layer4`. Four
1×1 lateral convolutions project every stage to a common 512-channel width, and a top-down pathway with
bilinear upsampling fuses them into a high-resolution map (FPN). A CBAM-style spatial-attention module
(channel avg/max pooling → 7×7 convolution → sigmoid gate) reweights that map. After global average pooling,
two linear heads give the lesion (512→7) and grade (512→5) predictions.

**Input handling.** 1024×1024 with an **elliptical field-of-view mask** (tightness 0.98) — UWF images carry
eyelid and eyelash artefacts at the field boundary that must be stripped before the model sees them.
AdamW, AMP with gradient scaling, gradient-norm clipping at 1.0, and gradient accumulation to compensate for
a small physical batch. Up to 50 epochs, early stopping with patience 8 on validation grade accuracy.

**Results** (stopped at epoch 28, best model at epoch 20):

| Metric | Best value | Epoch |
|---|---|---|
| Grade accuracy | **0.7448** | 20 |
| Lesion accuracy (element-wise) | 0.9066 | 25 |

**Reading.** UWF sits about 7 points below CFP (0.7448 vs 0.8291 in Experiment I), and that gap reproduces
consistently across every UWF experiment that follows. The 200° field buys peripheral coverage at the cost
of per-pixel lesion resolution — a real trade-off, not a training artefact.

**Caveats.** Best-validation number under an ad-hoc split; no test-set F1/AUC report was produced. The
Grad-CAM cell ran out of GPU memory before completing.

**Why it matters downstream.** The masking, the FPN trunk and the attention gate established here are what
Experiments IV–X build on.

**Figures.** `../../figures/UWF_results.png`, `UWF_result_1..3.png`, `UWF_dataset.png`,
`preprocessing_proof.png`.
