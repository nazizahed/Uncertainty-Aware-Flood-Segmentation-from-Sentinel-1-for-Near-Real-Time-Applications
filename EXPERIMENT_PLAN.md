# Resource-Aware Experimental Plan

This project is designed to be scientifically useful without requiring dedicated paid GPU infrastructure. The primary study is the reliability-efficiency frontier of lightweight Sentinel-1 flood-segmentation models. The EfficientNet-B7 UNet++ run is optional and must not become a dependency for completing the study.

## Core question

How much segmentation quality and trustworthy uncertainty can lightweight Sentinel-1 flood models provide under geographic shift, and what computational cost is required to obtain it?

The main comparison therefore considers multiple objectives together:

- segmentation quality: F1, pooled IoU, mean per-tile IoU, boundary F1;
- calibration: ECE and Brier score;
- selective prediction: AURC and sparsification error;
- uncertainty quality: deterministic entropy/confidence, MC variance, predictive entropy, expected entropy, mutual information;
- efficiency: parameters, model size, inference latency, peak memory, and eventually FLOPs;
- generalization: performance and calibration degradation on geographically distinct events.

## Compute policy

1. Free-tier Colab/Kaggle GPUs are the target environment.
2. Mixed precision is enabled for GPU training.
3. Physical batch size is kept small and gradient accumulation is used to reach an effective batch of approximately 32.
4. Expensive stochastic inference is evaluated progressively (5, 10, then 20 passes) rather than defaulting to 20 passes everywhere.
5. Deep ensembles begin with three members of lightweight models. Five-member ensembles are optional.
6. UNet++/EfficientNet-B7 is an optional reference-aligned anchor. The project is considered complete without it if resource limits make the run impractical.
7. Full hyperparameter sweeps are avoided. The study uses a controlled common protocol and a small number of scientifically motivated model choices.

## Stage 0 — integrity check

No large training run should begin until these pass:

```bash
python -m pytest -q
python scripts/validate_repository.py
```

Then run a short 1–2 epoch smoke experiment with the B0 configuration before a full training session.

## Stage 1 — core lightweight models

Primary models:

| Config | Role | Expected burden |
| --- | --- | --- |
| `lightweight_unet_b0.yaml` | main compact CNN baseline | low |
| `lightweight_unet_mobilenetv3.yaml` | very efficient CNN candidate | very low |
| `lightweight_segformer_b0.yaml` | MiT-B0 encoder candidate with different inductive bias | low–moderate |
| `lightweight_unet_b2.yaml` | intermediate-capacity candidate | moderate |

All use 256×256 inputs, AMP, physical batch size 8, and four-step gradient accumulation by default (effective batch 32).

Train one seed of every core model first. Do not train repeated seeds until the first-pass frontier is known.

## Stage 2 — deterministic reliability

For every trained core model, evaluate Florence using a single deterministic forward pass.

Record:

- segmentation metrics;
- overall/flood/non-flood calibration;
- deterministic predictive entropy;
- deterministic confidence uncertainty;
- risk-coverage and AURC.

This is the zero-extra-inference-cost uncertainty baseline and must be reported before MC dropout or ensembles.

## Stage 3 — MC dropout budget study

Do not immediately run 20 stochastic passes everywhere. For the strongest two lightweight models, evaluate:

- 5 passes;
- 10 passes;
- 20 passes only if 10 passes still provides meaningful gains.

Compare predictive entropy, expected entropy, mutual information, and variance against the deterministic baseline. Report the improvement in AURC/calibration together with the multiplicative inference cost.

The key question is whether additional stochastic passes provide enough reliability gain to justify their latency.

## Stage 4 — geographic shift

After the core models are trained, prioritize inference on geographically distinct events because this is scientifically valuable and comparatively cheap.

For every shift condition record:

- IoU/F1 degradation relative to the in-distribution reference;
- calibration degradation;
- AURC degradation;
- whether uncertainty increases on failure regions;
- qualitative failure modes.

Where event or source-scene identifiers are available, use them as groups for cluster bootstrap confidence intervals.

## Stage 5 — lightweight deep ensemble

Only after Stage 1 identifies the best lightweight model:

- train three independent seeds of that model;
- evaluate the three-member ensemble;
- compare it with deterministic inference and MC dropout;
- measure both reliability gain and total training/inference cost.

A five-member ensemble is optional and should be attempted only if the three-member curve suggests that more members are scientifically useful.

## Stage 6 — efficiency benchmark

For each retained model measure at minimum:

- trainable parameter count;
- fp32 model size;
- single-image and small-batch inference latency;
- CPU latency;
- GPU latency when available;
- peak GPU memory.

Report warm-up separately and benchmark repeated inference after warm-up. The final analysis should look for a Pareto frontier rather than declaring one universally best model.

## Stage 7 — optional heavy anchor

`baseline_unetpp_b7.yaml` is optional.

Attempt it only after the lightweight study is already secure. It uses a very small physical batch plus gradient accumulation to reduce memory pressure, but training time may still be substantial. Failure to run B7 on free infrastructure is not a failure of the research design.

## Decision rules

To conserve compute:

- stop training clearly dominated models rather than adding arbitrary variants;
- do not repeat seeds for every architecture;
- reserve repeated seeds for the best lightweight candidate and ensemble study;
- escalate MC passes only when lower-pass experiments show measurable value;
- keep B7 outside the critical path.

The intended final result is a reliability-efficiency-generalization study, not a leaderboard-oriented architecture search.
