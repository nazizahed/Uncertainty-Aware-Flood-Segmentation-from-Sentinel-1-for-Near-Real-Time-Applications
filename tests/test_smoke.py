"""CPU smoke test: build a small model, run forward + MC-dropout passes, check metrics."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarflood.models.build import build_model, count_parameters, enable_mc_dropout
from sarflood.models.uncertainty_inference import stochastic_forward_passes, summarize_passes
from sarflood.training.losses import BCEDiceLoss
from sarflood.training.metrics import SegmentationMetrics, eval_score
from sarflood.uncertainty.calibration import brier_score, expected_calibration_error
from sarflood.uncertainty.risk_coverage import aurc, risk_coverage_curve, sparsification_error
from sarflood.evaluation.stats import wilcoxon_per_tile, block_bootstrap_iou_delta


def test_model_forward():
    cfg = {"arch": "unet", "encoder": "resnet18", "encoder_weights": None, "dropout": 0.2}
    model = build_model(cfg, in_channels=3)
    x = torch.rand(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64), y.shape
    assert count_parameters(model) > 0

    # MC dropout: eval mode but dropout active -> stochastic outputs
    passes = stochastic_forward_passes(model, x, n_passes=5)
    assert passes.shape == (5, 2, 1, 64, 64), passes.shape
    s = summarize_passes(passes)
    assert s["mean"].shape == (2, 1, 64, 64)
    # dropout is active: passes must differ
    assert not torch.allclose(passes[0], passes[1]), "MC dropout not stochastic"

    # deterministic in pure eval
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), model(x)), "eval mode not deterministic"

    loss = BCEDiceLoss()(torch.randn(2, 1, 64, 64), (torch.rand(2, 1, 64, 64) > 0.9).float())
    assert torch.isfinite(loss)
    print("model/loss/mc-dropout OK, params:", count_parameters(model))


def test_metrics_and_uq():
    rng = np.random.default_rng(0)
    probs = rng.random(10000)
    labels = (rng.random(10000) > 0.9).astype(float)
    unc = rng.random(10000)

    m = SegmentationMetrics()
    m.update(probs.reshape(1, 100, 100), labels.reshape(1, 100, 100))
    res = m.compute()
    for k in ("accuracy", "precision", "recall", "f1", "iou", "kappa", "boundary_f1"):
        assert np.isfinite(res[k]), k
    assert np.isfinite(eval_score(res))

    assert 0 <= expected_calibration_error(probs, labels) <= 1
    assert 0 <= brier_score(probs, labels) <= 1

    cov, risk = risk_coverage_curve(probs, labels, unc)
    assert len(cov) == len(risk) and np.all(np.diff(cov) <= 0)
    assert aurc(cov, risk) >= 0
    _, _, _, se = sparsification_error(probs, labels, unc)
    assert se >= -1e-12

    # Calibration bins include exact boundary probabilities.
    assert expected_calibration_error(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0

    a = rng.random(200)
    b = a - 0.05 * rng.random(200)
    w = wilcoxon_per_tile(a, b)
    assert w["p_value"] < 0.05
    ci = block_bootstrap_iou_delta(a, b)
    assert ci["ci95"][0] < ci["mean_delta"] < ci["ci95"][1]
    print("metrics/uq/stats OK")


if __name__ == "__main__":
    test_model_forward()
    test_metrics_and_uq()
    print("ALL SMOKE TESTS PASSED")
