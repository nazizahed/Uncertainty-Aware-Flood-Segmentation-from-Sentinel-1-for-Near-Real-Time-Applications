# Small Models, Honest Maps

**Uncertainty-Aware Flood Segmentation from Sentinel-1 for Near-Real-Time Applications**

[![Tests](https://github.com/nazizahed/Uncertainty-Aware-Flood-Segmentation-from-Sentinel-1-for-Near-Real-Time-Applications/actions/workflows/tests.yml/badge.svg)](https://github.com/nazizahed/Uncertainty-Aware-Flood-Segmentation-from-Sentinel-1-for-Near-Real-Time-Applications/actions/workflows/tests.yml)

> **Development status:** this independent research project is under active
> development. The data pipeline, model factory, training loop, uncertainty
> utilities, evaluation code, configurations, and Colab entry point are
> implemented. Full baseline replication, efficiency benchmarks, ensemble
> experiments, and cross-region results are still pending. This repository does
> not yet claim original model-performance or near-real-time latency results.

The project investigates whether lightweight semantic-segmentation models can
produce useful Sentinel-1 flood maps while making their uncertainty explicit.
It builds on Ghosh et al. (2024), [*Automatic Flood Detection from Sentinel-1
Data Using a Nested UNet Model and a NASA Benchmark
Dataset*](https://doi.org/10.1007/s41064-024-00275-1), while using a PyTorch
implementation and an explicitly staged evaluation plan.

## Research questions

1. **Efficiency:** where is the accuracy-efficiency frontier across model size,
   FLOPs, and CPU/GPU inference latency?
2. **Uncertainty:** do deep ensembles or MC dropout produce calibrated
   uncertainty that supports selective prediction through risk-coverage
   analysis?
3. **Generalization:** how do candidate models and uncertainty estimates behave
   when transferred to geographically distinct flood events?

Near-real-time use is a design objective. It will only be claimed after latency,
throughput, preprocessing, and end-to-end deployment measurements are complete.

## Implemented workflow

```mermaid
flowchart TD
    A["ETCI 2021 VV, VH, labels"] --> B["Paired tiles + ratio channel"]
    B --> C["UNet / UNet++ training"]
    C --> D["Deterministic or stochastic inference"]
    D --> E["Segmentation + calibration metrics"]
    E --> F["Risk-coverage and uncertainty maps"]
```

| Component | Current state |
| --- | --- |
| ETCI event discovery and VV/VH/label pairing | Implemented and synthetically tested |
| Stabilized polarization-ratio channel | Implemented and configurable |
| Flood-stratified batches and 90-degree rotations | Implemented |
| UNet and UNet++ model factory | Implemented |
| BCE + soft-Dice training and Florence validation | Implemented |
| MC-dropout and ensemble inference utilities | Implemented |
| ECE, Brier score, risk-coverage, AURC, and sparsification error | Implemented and tested |
| Full ETCI baseline replication | Pending compute run |
| Efficiency and deployment benchmarks | Planned |
| Cross-region evaluation | Planned; external event preparation required |

## Repository layout

```text
.
|-- configs/                 # Baseline and lightweight experiment definitions
|-- notebooks/               # Colab entry point and notebook guide
|-- scripts/                 # Download, train, evaluate, and UQ commands
|-- src/sarflood/
|   |-- data/                # ETCI discovery, channels, augmentation, sampling
|   |-- models/              # UNet/UNet++ factory and stochastic inference
|   |-- training/            # Losses, metrics, and training loop
|   |-- uncertainty/         # Calibration and selective-prediction analysis
|   `-- evaluation/          # Evaluation orchestration and paired statistics
|-- tests/                   # Synthetic-data and CPU model smoke tests
`-- pyproject.toml           # Package metadata and bounded dependencies
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/nazizahed/Uncertainty-Aware-Flood-Segmentation-from-Sentinel-1-for-Near-Real-Time-Applications.git
cd Uncertainty-Aware-Flood-Segmentation-from-Sentinel-1-for-Near-Real-Time-Applications
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA training, install a PyTorch build compatible with the available GPU
before installing the remaining project dependencies.

## Data

The project uses the [NASA IMPACT / IEEE GRSS ETCI 2021 Flood Detection
dataset](https://nasa-impact.github.io/etci2021/): paired Sentinel-1 IW VV/VH
tiles and binary flood labels from five regions. Follow the official competition
page for access conditions, acknowledgement text, and terms of use.

The optional download helper retrieves a documented community mirror. Review
the official terms before using it:

```bash
python scripts/download_data.py --out data/etci
```

The loader recursively supports both the official event layout and the mirror's
additional `data/<split>/<event>/tiles/` nesting. Event directories must contain
`vv/`, `vh/`, and `flood_label/` subdirectories. Data and generated model
artifacts are excluded from Git.

## Reproducible starting points

```bash
# Quick CPU-level integrity and model tests
python -m pytest -q
python scripts/validate_repository.py

# Phase 1 replication anchor
python scripts/train.py --config configs/baseline_unetpp_b7.yaml

# Lightweight candidate
python scripts/train.py --config configs/lightweight_unet_b0.yaml

# Deterministic Florence evaluation
python scripts/evaluate.py \
  --checkpoint runs/baseline_unetpp_efficientnetb7/best.pt \
  --regions florence \
  --out runs/baseline_unetpp_efficientnetb7/florence.json

# MC-dropout uncertainty preview
python scripts/predict_uncertainty.py \
  --checkpoint runs/baseline_unetpp_efficientnetb7/best.pt \
  --regions florence \
  --method mc_dropout \
  --passes 20 \
  --out outputs/uncertainty/florence
```

The Colab workflow is documented in [`notebooks/README.md`](notebooks/README.md).

## Evaluation design

- Florence is held out from model fitting, following the reference study's
  geographic validation setup.
- Segmentation reporting includes pooled accuracy, precision, recall, F1, IoU,
  kappa, boundary F1, and mean per-tile IoU.
- Calibration uses ECE and Brier score on a bounded, reproducible pixel sample
  to avoid retaining the full validation set in memory.
- Selective prediction uses risk-coverage curves, AURC, and sparsification
  error; lower AURC indicates that abstaining on uncertain pixels reduces risk.
- Model comparisons are planned at tile level with paired Wilcoxon tests and
  bootstrap confidence intervals. Pixel-level McNemar results are retained only
  for comparison with earlier work because spatial autocorrelation inflates
  nominal pixel counts.

## Development roadmap

- [x] Repository and experiment scaffold
- [x] ETCI pairing, augmentation, ratio channel, and stratified sampling
- [x] Model, training, evaluation, calibration, and risk-coverage utilities
- [x] Synthetic dataset tests and CI
- [ ] Baseline replication and recorded environment snapshot
- [ ] Lightweight accuracy-efficiency frontier
- [ ] Deep-ensemble training and uncertainty decomposition
- [ ] Cross-region evaluation and failure-mode analysis
- [ ] Latency, throughput, quantization, and deployment benchmarks
- [ ] Publish versioned results and model cards

## Research integrity and scope

Published values discussed in notebooks are reference values from Ghosh et al.
(2024), not results produced by this repository. Until completed runs and their
artifacts are versioned, the repository should be cited as an ongoing software
and research-method development project.

Code is released under the [MIT License](LICENSE). The dataset and upstream
model weights retain their own licences and usage conditions.
