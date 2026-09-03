# Analysis notebooks

The notebooks are execution entry points for the ongoing research workflow;
they are not evidence that the planned full experiments have been completed.

## Resource-aware execution

The repository now treats lightweight models as the primary experiments. On a
free-tier Colab/Kaggle GPU, begin with:

```bash
python scripts/train.py --config configs/lightweight_unet_b0.yaml
```

The core configs use AMP, physical batch size 8, and gradient accumulation to
reach an effective batch size of 32 without requiring large VRAM. Run the other
core candidates only after the B0 pipeline completes successfully.

For uncertainty evaluation, begin with deterministic inference and then five MC
dropout passes. Escalate to 10 or 20 passes only if the lower-cost setting shows
meaningful improvement in calibration or selective prediction.

The EfficientNet-B7 UNet++ experiment is optional and outside the critical path.
Do not spend a limited free-GPU session on B7 before the lightweight experiments
are secure.

See [`../EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md) for the complete staged
protocol.

## `01_colab_training.ipynb`

This existing Colab-oriented notebook provides setup, dataset download, data
verification, checkpoint persistence, training, and evaluation examples. Some
of its narrative was originally written around the B7 reference-aligned anchor;
for the current research plan, use the resource-aware configurations and command
order in the repository README/experiment plan as authoritative.

Repository notebooks are stored without completed research outputs,
checkpoints, credentials, or personal Drive identifiers. Results must not be
presented as project findings until their run artifacts and configuration are
recorded.
