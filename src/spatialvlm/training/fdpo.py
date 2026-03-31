"""fDPO (fine-grained DPO) training utilities for preference optimization.

The "fine-grained" aspect: different optimization pressures (beta values) are
applied to descriptive spatial grounding segments vs. logical reasoning segments,
following SpatialReasoner-R1 (Shen et al., NeurIPS 2025).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch.nn.utils import clip_grad_norm_


@dataclass(frozen=True)
class FDPOConfig:
    """Hyperparameters for fine-grained preference optimization.

    The key distinction from standard DPO: separate betas for grounding
    (spatial description) and reasoning (logical inference) segments.
    """

    beta_grounding: float = 0.1
    beta_reasoning: float = 0.05
    learning_rate: float = 5e-7
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    reference_free: bool = False
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class FDPOLossBreakdown:
    """fDPO loss decomposition."""

    total_loss: torch.Tensor
    grounding_loss: torch.Tensor
    reasoning_loss: torch.Tensor
    preference_margin: torch.Tensor
    accuracy: torch.Tensor


def fdpo_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    chosen_ref_logps: torch.Tensor | None,
    rejected_ref_logps: torch.Tensor | None,
    beta_grounding: float,
    beta_reasoning: float,
    segment_mask: torch.Tensor | None = None,
    reference_free: bool = False,
    label_smoothing: float = 0.0,
) -> FDPOLossBreakdown:
    """Compute fDPO objective with segment-specific beta values.

    Parameters
    ----------
    chosen_logps : Tensor[N]
    rejected_logps : Tensor[N]
    chosen_ref_logps : Tensor[N] | None
    rejected_ref_logps : Tensor[N] | None
    beta_grounding : float
        Beta for spatial grounding segments (stricter).
    beta_reasoning : float
        Beta for logical reasoning segments (gentler).
    segment_mask : Tensor[N] | None
        Boolean mask: True = grounding segment, False = reasoning segment.
        If None, all samples use beta_grounding (degrades to standard DPO).
    reference_free : bool
    label_smoothing : float
    """
    for name, tensor in (("chosen_logps", chosen_logps), ("rejected_logps", rejected_logps)):
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be [N], got {tuple(tensor.shape)}")

    if chosen_logps.shape != rejected_logps.shape:
        raise ValueError("chosen_logps and rejected_logps must have same shape.")

    pi_logratios = chosen_logps - rejected_logps  # [N]
    if reference_free:
        ref_logratios = torch.zeros_like(pi_logratios)
    else:
        if chosen_ref_logps is None or rejected_ref_logps is None:
            raise ValueError("Reference log-probs are required unless reference_free=True.")
        if (
            chosen_ref_logps.shape != chosen_logps.shape
            or rejected_ref_logps.shape != chosen_logps.shape
        ):
            raise ValueError("Reference tensors must match chosen/rejected tensor shape.")
        ref_logratios = chosen_ref_logps - rejected_ref_logps

    raw_logratios = pi_logratios - ref_logratios  # [N]

    # Build per-sample beta: grounding segments get beta_grounding, reasoning gets beta_reasoning
    if segment_mask is not None:
        if segment_mask.shape != chosen_logps.shape:
            raise ValueError(
                f"segment_mask shape {tuple(segment_mask.shape)} "
                f"must match logps shape {tuple(chosen_logps.shape)}."
            )
        beta = torch.where(
            segment_mask,
            torch.tensor(beta_grounding, device=chosen_logps.device, dtype=chosen_logps.dtype),
            torch.tensor(beta_reasoning, device=chosen_logps.device, dtype=chosen_logps.dtype),
        )  # [N]
    else:
        beta = torch.full_like(chosen_logps, beta_grounding)

    logits = beta * raw_logratios  # [N]
    pos = functional.logsigmoid(logits)
    neg = functional.logsigmoid(-logits)
    per_sample_loss = -((1.0 - label_smoothing) * pos + label_smoothing * neg)

    # Compute segment-specific losses for logging
    if segment_mask is not None and segment_mask.any() and (~segment_mask).any():
        grounding_loss = per_sample_loss[segment_mask].mean()
        reasoning_loss = per_sample_loss[~segment_mask].mean()
    else:
        grounding_loss = per_sample_loss.mean()
        reasoning_loss = torch.zeros((), device=chosen_logps.device, dtype=chosen_logps.dtype)

    total_loss = per_sample_loss.mean()
    accuracy = (logits > 0).float().mean()

    return FDPOLossBreakdown(
        total_loss=total_loss,
        grounding_loss=grounding_loss,
        reasoning_loss=reasoning_loss,
        preference_margin=logits.mean(),
        accuracy=accuracy,
    )


@dataclass(frozen=True)
class FDPOStepOutput:
    """Scalar summary of one fDPO step."""

    loss: float
    grounding_loss: float
    reasoning_loss: float
    preference_margin: float
    accuracy: float
    grad_norm: float


class FDPOTrainer:
    """Minimal optimizer wrapper for fDPO updates."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: FDPOConfig,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        self.optimizer = optimizer

    def step(self, batch: Mapping[str, Any]) -> FDPOStepOutput:
        required = {"chosen_logps", "rejected_logps"}
        missing = required.difference(batch.keys())
        if missing:
            raise KeyError(f"fDPO batch missing keys: {sorted(missing)}")

        loss = fdpo_loss(
            chosen_logps=batch["chosen_logps"],
            rejected_logps=batch["rejected_logps"],
            chosen_ref_logps=batch.get("chosen_ref_logps"),
            rejected_ref_logps=batch.get("rejected_ref_logps"),
            beta_grounding=self.config.beta_grounding,
            beta_reasoning=self.config.beta_reasoning,
            segment_mask=batch.get("segment_mask"),
            reference_free=self.config.reference_free,
            label_smoothing=self.config.label_smoothing,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.total_loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()

        return FDPOStepOutput(
            loss=float(loss.total_loss.detach().cpu().item()),
            grounding_loss=float(loss.grounding_loss.detach().cpu().item()),
            reasoning_loss=float(loss.reasoning_loss.detach().cpu().item()),
            preference_margin=float(loss.preference_margin.detach().cpu().item()),
            accuracy=float(loss.accuracy.detach().cpu().item()),
            grad_norm=float(grad_norm.detach().cpu().item()),
        )
