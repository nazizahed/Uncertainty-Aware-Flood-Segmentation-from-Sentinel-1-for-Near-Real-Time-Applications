# sar-flood-uq

**Lightweight, uncertainty-aware flood segmentation from Sentinel-1 SAR**

Independent research project building on Ghosh et al. (2024), *"Automatic Flood Detection from
Sentinel-1 Data Using a Nested UNet Model and a NASA Benchmark Dataset"*, PFG 92:1–18,
[DOI: 10.1007/s41064-024-00275-1](https://doi.org/10.1007/s41064-024-00275-1).

The project targets three questions:

1. **Efficiency** — where is the accuracy–efficiency Pareto frontier (IoU vs. params / FLOPs /
   latency) for Sentinel-1 flood segmentation on the NASA/IEEE GRSS (ETCI 2021) benchmark?
2. **Uncertainty** — can pixel-wise uncertainty (deep ensembles / MC dropout), made affordable
   by lightweight encoders, be *exploited operationally* via selective prediction
   (risk–coverage analysis), and does it localize known SAR failure modes?
3. **Generalization** — do lightweight + UQ models hold up cross-regionally
   (Spain 2019, Kerala 2018, Bihar 2021, Vietnam 2020)?

## Why PyTorch

The baseline was implemented in TensorFlow/Keras. We use **PyTorch +
[segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)**
because it gives one-line access to UNet/UNet++ with every encoder in the study
(ResNet-34, Inception-v3, EfficientNet-B0–B7, MobileNetV3, MiT/SegFormer), clean hooks for
dropout injection (MC dropout), and easy ONNX/TensorRT export for the deployment analysis.

## Repository layout

```
sar-flood-uq/
├── configs/                    # experiment configs (YAML)
│   ├── baseline_unetpp_b7.yaml       # Phase 1: replication anchor
│   └── lightweight_unet_b0.yaml      # Phase 2: lightweight candidate
├── src/sarflood/
│   ├── data/dataset.py         # ETCI dataset, ratio channel, rotation aug, stratified batches
│   ├── models/                 # model factory + dropout injection + MC-dropout inference
│   ├── training/               # BCE+Dice loss, metrics, training loop
│   ├── uncertainty/            # deep ensembles, calibration (ECE/Brier), risk–coverage/AURC
│   └── evaluation/             # region evaluation + statistics (Wilcoxon, bootstrap, McNemar)
├── scripts/                    # CLI entry points (train / evaluate / predict-uncertainty)
├── notebooks/                  # analysis & visualization notebooks
└── tests/                      # CPU smoke tests
```

## Setup

```bash
git clone <your-fork-url> && cd sar-flood-uq
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Training/validation uses the **NASA IMPACT / IEEE GRSS ETCI 2021 flood detection dataset**
(Sentinel-1 IW, VV+VH, 256×256 tiles). Download links are listed on the competition page
(IEEE GRSS Data Fusion Contest / NASA IMPACT). Expected directory layout
(`src/sarflood/data/dataset.py` is tolerant to naming variants):

```
data/etci/
├── nebraska/        ├── vv/  ├── vh/  └── flood_label/
├── north_alabama/   ├── vv/  ├── vh/  └── flood_label/
├── bangladesh/      ├── vv/  ├── vh/  └── flood_label/
└── florence/        ├── vv/  ├── vh/  └── flood_label/   # held out as validation (8,382 tiles)
```

Notes:
- The released tiles are already preprocessed (border-noise correction, speckle filtering,
  RTC with 30 m DEM, dB scaling, 0–255 grayscale) — no SNAP/hyp3 processing needed for
  training data, only for new cross-regional test events.
- Following Ghosh et al., a third channel `(VV+VH)/(VV−VH)` is computed on the fly
  (`bands: [vv, vh, ratio]` in the config).

## Quickstart

```bash
# Phase 1 — replicate baseline (anchor for all comparisons)
python scripts/train.py --config configs/baseline_unetpp_b7.yaml

# Phase 2 — lightweight model
python scripts/train.py --config configs/lightweight_unet_b0.yaml

# Phase 3 — uncertainty maps (MC dropout, N stochastic passes)
python scripts/predict_uncertainty.py --checkpoint runs/<run>/best.pt --method mc_dropout --passes 20

# Phase 4 — evaluation: metrics + calibration + risk–coverage + stats
python scripts/evaluate.py --checkpoint runs/<run>/best.pt --split val
```

Model selection follows the baseline: the checkpoint with the highest **F1 + mIoU** sum on
validation is kept as `best.pt`.

## Baseline reference numbers (Ghosh et al. 2024, UNet++ EfficientNet-B7)

| Split | Acc | Prec | Rec | F1 | IoU | Kappa |
|---|---|---|---|---|---|---|
| Florence (val) | 98.8 | 89.5 | 89.1 | 89.3 | 75.76 | 81.6 |
| Spain 2019 | 98.8 | 82.8 | 86.4 | 84.5 | 73.0 | 80.5 |
| Kerala 2018 | 97.7 | 84.0 | 87.3 | 85.6 | 74.1 | 80.8 |
| Bihar 2021 | 98.5 | 89.7 | 89.4 | 89.5 | 74.7 | 80.3 |

Replication targets: IoU within ~1–2 points of these on each split.

## Methodological notes (deviations from the baseline, by design)

- **Statistics:** baseline used pixel-level McNemar; we additionally report per-tile Wilcoxon
  signed-rank tests and block-bootstrap CIs (pixel-level tests are inflated by spatial
  autocorrelation). McNemar is still computed for direct comparability.
- **Uncertainty exploitation** is evaluated with risk–coverage curves / AURC and sparsification
  error, not "mask top-X% pixels and recompute precision" (that framing conflates abstention
  with accuracy).
- **Deep ensembles are the primary UQ method** (MC dropout as the cheap baseline), because
  cross-regional testing is exactly the distribution-shift regime where MC dropout degrades.

## Roadmap

- [x] Repo scaffold, data pipeline, model zoo, training loop
- [x] MC dropout / deep ensemble / calibration / risk–coverage tooling
- [ ] Phase 1: baseline replication runs
- [ ] Phase 2: efficiency frontier (params/FLOPs/latency CPU+GPU/quantization)
- [ ] Phase 3: ensemble training, aleatoric/epistemic decomposition
- [ ] Phase 4: cross-regional evaluation + land-cover-stratified uncertainty analysis
- [ ] Phase 5: ablations (bands, dropout placement/rate, ensemble size)

## License

MIT (code). Dataset is subject to the ETCI 2021 / NASA IMPACT terms of use.
