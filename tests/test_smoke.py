"""CPU smoke tests for models, uncertainty, metrics, and paired statistics."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarflood.models.build import build_model, count_parameters
from sarflood.models.uncertainty_inference import (
    confidence_uncertainty,
    deterministic_entropy,
    stochastic_forward_passes,
    summarize_passes,
)
from sarflood.training.losses import BCEDiceLoss
from sarflood.training.metrics import SegmentationMetrics, eval_score
from sarflood.uncertainty.calibration import brier_score, expected_calibration_error
from sarflood.uncertainty.risk_coverage import aurc, risk_coverage_curve, sparsification_error
from sarflood.evaluation.stats import (
    grouped_bootstrap_iou_delta,
    tile_bootstrap_iou_delta,
    wilcoxon_per_tile,
)


def test_model_forward_and_uncertainty():
    cfg = {"arch": "unet", "encoder": "resnet18", "encoder_weights": None, "dropout": 0.2}
    model = build_model(cfg, in_channels=3)
    x = torch.rand(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64), y.shape
    assert count_parameters(model) > 0

    passes = stochastic_forward_passes(model, x, n_passes=5)
    assert passes.shape == (5, 2, 1, 64, 64)
    assert not torch.allclose(passes[0], passes[1]), "MC dropout not stochastic"

    s = summarize_passes(passes)
    for key in (
        "mean", "variance", "predictive_entropy", "expected_entropy", "mutual_information"
    ):
        assert s[key].shape == (2, 1, 64, 64)
        assert torch.isfinite(s[key]).all()
    assert torch.all(s["mutual_information"] >= 0)
    assert torch.allclose(s["entropy"], s["predictive_entropy"])

    det_ent = deterministic_entropy(s["mean"])
    det_conf = confidence_uncertainty(s["mean"])
    assert det_ent.shape == det_conf.shape == (2, 1, 64, 64)
    assert torch.all((det_conf >= 0) & (det_conf <= 1))

    with pytest.raises(ValueError):
        stochastic_forward_passes(model, x, n_passes=1)

    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), model(x)), "eval mode not deterministic"

    loss = BCEDiceLoss()(torch.randn(2, 1, 64, 64), (torch.rand(2, 1, 64, 64) > 0.9).float())
    assert torch.isfinite(loss)


def test_metrics_and_uq():
    rng = np.random.default_rng(0)
    probs = rng.random(10000)
    labels = (rng.random(10000) > 0.9).astype(float)
    unc = rng.random(10000)

    m = SegmentationMetrics()
    m.update(probs.reshape(100, 100), labels.reshape(100, 100))
    res = m.compute()
    for key in ("accuracy", "precision", "recall", "f1", "iou", "miou_tiles", "kappa", "boundary_f1"):
        assert np.isfinite(res[key]), key
    assert np.isclose(eval_score(res), res["f1"] + res["miou_tiles"])

    with pytest.raises(ValueError):
        m.update(np.zeros((2, 10, 10)), np.zeros((2, 10, 10)))

    assert 0 <= expected_calibration_error(probs, labels) <= 1
    assert 0 <= brier_score(probs, labels) <= 1

    cov, risk = risk_coverage_curve(probs, labels, unc)
    assert len(cov) == len(risk) and np.all(np.diff(cov) <= 0)
    assert aurc(cov, risk) >= 0
    _, risk_model, risk_oracle, se = sparsification_error(probs, labels, unc)
    assert se >= -1e-12
    assert aurc(cov, risk_oracle) <= aurc(cov, risk_model) + 1e-12

    assert expected_calibration_error(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0

    a = rng.random(200)
    b = a - 0.05 * rng.random(200)
    w = wilcoxon_per_tile(a, b)
    assert w["p_value"] < 0.05

    ci = tile_bootstrap_iou_delta(a, b)
    assert ci["ci95"][0] < ci["mean_delta"] < ci["ci95"][1]

    groups = np.repeat(np.arange(20), 10)
    grouped = grouped_bootstrap_iou_delta(a, b, groups, n_boot=200)
    assert grouped["n_groups"] == 20
    assert grouped["n_tiles"] == 200
    assert grouped["ci95"][0] <= grouped["mean_delta"] <= grouped["ci95"][1]


if __name__ == "__main__":
    test_model_forward_and_uncertainty()
    test_metrics_and_uq()
    print("ALL SMOKE TESTS PASSED")
