"""Phase 9 ablation-run specifications and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Phase9RunSpec:
    """Specification for one Phase 9 ablation run."""

    run_id: str
    title: str
    hypotheses: tuple[str, ...]
    config_overrides: dict[str, Any]


def phase9_run_specs() -> list[Phase9RunSpec]:
    """Canonical run matrix aligned with TODO Phase 9."""
    return [
        Phase9RunSpec("full-model", "Full model (all modules)", (), {}),
        Phase9RunSpec(
            "no-gridcell-rope3d",
            "No GridCellRoPE3D (standard M-RoPE)",
            ("H2b",),
            {"model.use_gridcell_rope3d": False},
        ),
        Phase9RunSpec("no-gatr", "No GATr", ("H2a",), {"model.use_gatr": False}),
        Phase9RunSpec(
            "siglip-only",
            "SigLIP only",
            ("H1a",),
            {"model.use_dinov2": False, "model.use_gatr": False},
        ),
        Phase9RunSpec(
            "dinov2-only",
            "DINOv2 only",
            ("H1a",),
            {"model.use_siglip": False, "model.use_gatr": False},
        ),
        Phase9RunSpec(
            "dinov2-pooled-576",
            "DINOv2 pooled to 576",
            ("H1d", "H3e"),
            {"encoder.dinov2_pool_to_queries": True},
        ),
        Phase9RunSpec(
            "concat-fusion",
            "Concatenation fusion (no SVA/cross-attn)",
            ("H3a",),
            {
                "fusion.use_sva": False,
                "fusion.use_gated_cross_attn": False,
                "fusion.concat_fusion_only": True,
            },
        ),
        Phase9RunSpec(
            "no-rms-norm-matching",
            "No RMS norm matching",
            ("H3b",),
            {"fusion.use_rms_norm_matching": False},
        ),
        Phase9RunSpec(
            "no-typed-attn-bias",
            "No typed attention bias",
            ("H3d",),
            {"fusion.use_typed_attention_bias": False},
        ),
        Phase9RunSpec(
            "sft-only",
            "SFT only (no GRPO)",
            ("H5a",),
            {"training.rl.algorithm": "none"},
        ),
        Phase9RunSpec(
            "permutation-test",
            "Permutation test",
            ("H3c",),
            {"eval.run_permutation_test": True},
        ),
        Phase9RunSpec(
            "gt-depth-vs-depth-anything-v2",
            "GT depth vs Depth Anything V2",
            ("H2c",),
            {"depth.source": "depth_anything_v2"},
        ),
        Phase9RunSpec(
            "mean-vs-15pct-aggregation",
            "Mean vs 15th-pct aggregation",
            ("H2e",),
            {"geometry.depth_aggregation": "mean"},
        ),
        Phase9RunSpec(
            "scale-ratio-sweep",
            "Scale ratio sweep (phi vs sqrt(e) vs pow2)",
            ("H2d",),
            {"geometry.gridcell_scale_sweep": ["phi", "sqrt_e", "pow2"]},
        ),
        Phase9RunSpec(
            "grpo-vs-fdpo-vs-sft",
            "GRPO vs fDPO vs SFT-only",
            ("H5b",),
            {"training.comparison_algorithms": ["grpo", "fdpo", "sft"]},
        ),
        Phase9RunSpec(
            "dense-vs-sparse-rewards",
            "Dense vs sparse rewards",
            ("H5c",),
            {"training.reward_mode": "sparse"},
        ),
    ]


def missing_phase9_runs(results_by_id: Mapping[str, Any]) -> list[str]:
    """Return missing run IDs from provided result dictionary."""
    expected = {spec.run_id for spec in phase9_run_specs()}
    present = set(results_by_id.keys())
    return sorted(expected.difference(present))


def phase9_coverage_complete(results_by_id: Mapping[str, Any]) -> bool:
    return len(missing_phase9_runs(results_by_id)) == 0


def permutation_smoking_gun_pass(
    ours_relative_drop: float,
    baseline_relative_drop: float,
    ours_min_drop: float = 0.15,
    baseline_max_drop: float = 0.03,
) -> bool:
    """Check core H3c criterion for the permutation diagnostic."""
    return ours_relative_drop >= ours_min_drop and baseline_relative_drop <= baseline_max_drop
