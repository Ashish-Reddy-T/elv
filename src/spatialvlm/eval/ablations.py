"""Ablation orchestration for controlled module drop experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AblationSpec:
    """One ablation experiment definition."""

    ablation_id: str
    name: str
    hypotheses: tuple[str, ...]
    overrides: dict[str, Any]


@dataclass(frozen=True)
class AblationResult:
    """Result for a single ablation run."""

    ablation_id: str
    score: float
    delta_vs_baseline: float
    metadata: dict[str, Any]


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a dotted config path in-place.

    Example:
      `set_nested(cfg, "model.use_gridcell", False)`
    """
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(config)
    for key, value in overrides.items():
        set_nested(out, key, value)
    return out


def default_ablation_specs() -> list[AblationSpec]:
    """Default ablation matrix aligned to TODO hypotheses."""
    return [
        AblationSpec(
            ablation_id="no-gridcell-rope3d",
            name="No GridCellRoPE3D",
            hypotheses=("H2b", "H4a"),
            overrides={"model.use_gridcell_rope3d": False},
        ),
        AblationSpec(
            ablation_id="no-gatr",
            name="No GATr",
            hypotheses=("H2a",),
            overrides={"model.use_gatr": False},
        ),
        AblationSpec(
            ablation_id="siglip-only",
            name="SigLIP only",
            hypotheses=("H1a",),
            overrides={"model.use_dinov2": False, "model.use_gatr": False},
        ),
        AblationSpec(
            ablation_id="dinov2-only",
            name="DINOv2 only",
            hypotheses=("H1a",),
            overrides={"model.use_siglip": False, "model.use_gatr": False},
        ),
        AblationSpec(
            ablation_id="no-rms-norm-matching",
            name="No RMS norm matching",
            hypotheses=("H3b",),
            overrides={"fusion.use_rms_norm_matching": False},
        ),
        AblationSpec(
            ablation_id="no-typed-attn-bias",
            name="No typed attention bias",
            hypotheses=("H3d",),
            overrides={"fusion.use_typed_attention_bias": False},
        ),
    ]


class AblationOrchestrator:
    """Runs baseline + ablation variants via injected evaluator callback."""

    def __init__(
        self,
        base_config: Mapping[str, Any],
        evaluator: Callable[[Mapping[str, Any]], float],
        specs: Sequence[AblationSpec] | None = None,
    ) -> None:
        self.base_config = deepcopy(dict(base_config))
        self.evaluator = evaluator
        self.specs = list(specs if specs is not None else default_ablation_specs())

    def run(
        self,
        ablation_ids: Sequence[str] | None = None,
    ) -> dict[str, AblationResult]:
        selected = set(ablation_ids) if ablation_ids is not None else None
        baseline_score = float(self.evaluator(self.base_config))

        out: dict[str, AblationResult] = {
            "baseline": AblationResult(
                ablation_id="baseline",
                score=baseline_score,
                delta_vs_baseline=0.0,
                metadata={"overrides": {}},
            )
        }
        for spec in self.specs:
            if selected is not None and spec.ablation_id not in selected:
                continue
            cfg = apply_overrides(self.base_config, spec.overrides)
            score = float(self.evaluator(cfg))
            out[spec.ablation_id] = AblationResult(
                ablation_id=spec.ablation_id,
                score=score,
                delta_vs_baseline=score - baseline_score,
                metadata={"name": spec.name, "hypotheses": list(spec.hypotheses)},
            )
        return out
