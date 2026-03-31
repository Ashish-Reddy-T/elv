"""fDPO training utilities for preference optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch.nn.utils import clip_grad_norm_


@dataclass(frozen=True)
class FDPOConfig:
    """Hyperparameters for preference optimization."""

    beta: float = 0.1
    learning_rate: float = 5e-7
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    reference_free: bool = False
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class FDPOLossBreakdown:
    """fDPO loss decomposition."""

    total_loss: torch.Tensor
    preference_margin: torch.Tensor
    accuracy: torch.Tensor


def fdpo_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    chosen_ref_logps: torch.Tensor | None,
    rejected_ref_logps: torch.Tensor | None,
    beta: float,
    reference_free: bool = False,
    label_smoothing: float = 0.0,
) -> FDPOLossBreakdown:
    """Compute fDPO objective from sequence log-likelihood scores.

    All log-prob tensors are expected shape [N].
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

    logits = beta * (pi_logratios - ref_logratios)
    pos = functional.logsigmoid(logits)
    neg = functional.logsigmoid(-logits)
    loss = -((1.0 - label_smoothing) * pos + label_smoothing * neg).mean()

    accuracy = (logits > 0).float().mean()
    return FDPOLossBreakdown(
        total_loss=loss,
        preference_margin=logits.mean(),
        accuracy=accuracy,
    )


@dataclass(frozen=True)
class FDPOStepOutput:
    """Scalar summary of one fDPO step."""

    loss: float
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
            beta=self.config.beta,
            reference_free=self.config.reference_free,
            label_smoothing=self.config.label_smoothing,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.total_loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()

        return FDPOStepOutput(
            loss=float(loss.total_loss.detach().cpu().item()),
            preference_margin=float(loss.preference_margin.detach().cpu().item()),
            accuracy=float(loss.accuracy.detach().cpu().item()),
            grad_norm=float(grad_norm.detach().cpu().item()),
        )
