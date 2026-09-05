# Supporting work

Work that supports the eleven numbered experiments but is not itself a report section.

### `data_extraction_provenance.ipynb`
The figshare download and extraction utility used to assemble `MMRDR/` from the published archives. Kept for
provenance — it documents where the data came from. It is a short utility script, not an experiment.

### `resnet50_multitask_baseline_CFP.ipynb`
The plain ResNet-50 multi-task reference ("Dual Brain": shared trunk, separate grade and lesion heads,
weighted CE + BCE, 512 px). This is the number Experiment I had to beat.

Held-out test set (n=2,224): **grade accuracy 0.8233**, macro-F1 0.7453, mean lesion AUROC 0.9649, best
epoch 11 of 12. Per-class F1: No DR 0.94, Mild 0.56, Moderate 0.68, Severe 0.67, PDR 0.88. Per-lesion AUROC
`[MA, HE, IH, VB/IRMA, NV, VH, RD]` = 0.979 / 0.950 / 0.944 / 0.931 / 0.985 / 0.987 / 0.979. An overfitting
audit in the notebook reports train F1 0.829 against test F1 0.745, a gap of 0.084.

> **Known labelling error in this notebook.** One output cell prints the seven lesion AUROC values under the
> wrong names ("Hemorrhages, Exudates, Cotton Wool, Microaneurysms, IRMA, Venous Beading, Neovascularization").
> The values are correct; the labels are not. The canonical order is `[MA, HE, IH, VB/IRMA, NV, VH, RD]`, as
> printed correctly by a later cell in the same notebook. The notebook is committed unmodified, so the error
> is still visible — read the later cell.

The notebook also contains three earlier abandoned runs (peaking at 0.8076 and 0.8107) and a working
weakly-supervised lesion-localisation demo that emits bounding boxes with confidences. Two cells end in
errors (a CUDA OOM, and a `.numpy()` call on a tensor requiring grad).

### `OCT_volumetric_EXPLORATORY.ipynb`
A 3D volumetric attention classifier treating OCT slices as volumes. **Incomplete** — only 3 of 20 planned
epochs ran (QWK 0.0000 → 0.0389 → 0.4979, macro-F1 0.2456 → 0.4875) before two cells failed on a missing
`patient_id` column and a missing dataset folder.

It is the only OCT artefact in the project and does not correspond to any report section. OCT also carries no
lesion labels and only three grade values (0, 1, 2) out of five, so it does not fit the multi-task
formulation used everywhere else. Included for completeness, not as a result.

### `scripts/`
Two standalone argparse CLIs — the MS-LAN reference implementation and an exploratory ResNet-GCN. See
`scripts/README.md`.
