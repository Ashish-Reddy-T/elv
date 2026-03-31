"""Tests for ablation orchestration."""

from __future__ import annotations

import pytest

from spatialvlm.eval.ablations import AblationOrchestrator, apply_overrides, set_nested


def test_set_nested_creates_path():
    cfg = {}
    set_nested(cfg, "model.use_gridcell_rope3d", False)
    assert cfg["model"]["use_gridcell_rope3d"] is False


def test_apply_overrides_deep_copy():
    base = {"model": {"use_gatr": True}}
    out = apply_overrides(base, {"model.use_gatr": False})
    assert base["model"]["use_gatr"] is True
    assert out["model"]["use_gatr"] is False


def test_ablation_orchestrator_baseline_and_delta():
    base_cfg = {
        "model": {
            "use_gridcell_rope3d": True,
            "use_gatr": True,
            "use_siglip": True,
            "use_dinov2": True,
        },
        "fusion": {"use_rms_norm_matching": True, "use_typed_attention_bias": True},
    }

    def evaluator(cfg):
        score = 1.0
        if not cfg["model"].get("use_gridcell_rope3d", True):
            score -= 0.2
        if not cfg["model"].get("use_gatr", True):
            score -= 0.1
        if not cfg["fusion"].get("use_rms_norm_matching", True):
            score -= 0.05
        return score

    orchestrator = AblationOrchestrator(base_config=base_cfg, evaluator=evaluator)
    out = orchestrator.run(ablation_ids=["no-gridcell-rope3d", "no-gatr"])
    assert "baseline" in out
    assert out["baseline"].score == 1.0
    assert out["no-gridcell-rope3d"].score == 0.8
    assert out["no-gridcell-rope3d"].delta_vs_baseline == pytest.approx(-0.2)
    assert out["no-gatr"].delta_vs_baseline == pytest.approx(-0.1)
