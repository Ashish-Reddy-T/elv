"""Stage-1 pre-alignment training utilities (projector-focused SFT)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch.nn.utils import clip_grad_norm_

from spatialvlm.training.sft import set_trainable_by_groups


@dataclass(frozen=True)
class PrealignConfig:
    """Hyperparameters for projector pre-alignment."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    ignore_index: int = -100
    projector_keywords: tuple[str, ...] = ("projector",)
    # When non-None, `trainable_groups` overrides `projector_keywords` and
    # the trainer uses `set_trainable_by_groups`. Stored as a tuple so the
    # dataclass stays hashable/frozen.
    trainable_groups: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PrealignStepOutput:
    """Result from one pre-alignment optimization step."""

    loss: float
    grad_norm: float
    trainable_params: int


def freeze_all_parameters(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_parameters_by_keyword(
    model: torch.nn.Module,
    keywords: Sequence[str],
) -> list[str]:
    """Unfreeze parameters where the full parameter name contains any keyword."""
    touched: list[str] = []
    lowered = tuple(k.lower() for k in keywords)
    for name, param in model.named_parameters():
        if any(key in name.lower() for key in lowered):
            param.requires_grad_(True)
            touched.append(name)
    return touched


def masked_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute token-level cross-entropy with ignore index.

    logits: [B, T, V]
    labels: [B, T]
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be [B,T,V], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be [B,T], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits and labels token dims must match, got {tuple(logits.shape[:2])} vs "
            f"{tuple(labels.shape)}"
        )

    vocab = logits.shape[-1]
    return functional.cross_entropy(
        logits.reshape(-1, vocab),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


class PrealignmentTrainer:
    """Projector-focused trainer for Stage-1 pre-alignment."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: PrealignConfig,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config

        if config.trainable_groups is not None:
            self.trainable_parameter_names = set_trainable_by_groups(
                self.model, config.trainable_groups
            )
            selection_detail = f"groups={tuple(config.trainable_groups)}"
        else:
            freeze_all_parameters(self.model)
            self.trainable_parameter_names = unfreeze_parameters_by_keyword(
                self.model,
                config.projector_keywords,
            )
            selection_detail = f"keywords={config.projector_keywords}"
        if len(self.trainable_parameter_names) == 0:
            raise ValueError(f"No trainable parameters found for pre-alignment. {selection_detail}")

        if optimizer is None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        self.optimizer = optimizer

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def forward_backward(self, batch: Mapping[str, Any], loss_scale: float = 1.0) -> float:
        """Forward + backward without optimizer step (for gradient accumulation).

        Returns the unscaled loss value.
        """
        if "labels" not in batch:
            raise KeyError("Prealignment batch must include `labels`.")

        labels = batch["labels"]
        if not isinstance(labels, torch.Tensor):
            raise TypeError("Batch `labels` must be a torch.Tensor.")

        model_kwargs = {k: v for k, v in batch.items() if k != "labels"}
        outputs = self.model(**model_kwargs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        loss = masked_lm_loss(logits=logits, labels=labels, ignore_index=self.config.ignore_index)

        scaled = loss * loss_scale
        scaled.backward()
        return float(loss.detach().cpu().item())

    def clip_and_step(self) -> float:
        """Clip gradients, optimizer step, zero grads. Returns grad norm."""
        grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm.detach().cpu().item())

    def step(self, batch: Mapping[str, Any]) -> PrealignStepOutput:
        """Full forward + backward + optimizer step (no accumulation)."""
        self.optimizer.zero_grad(set_to_none=True)
        loss_val = self.forward_backward(batch)
        grad_norm_val = self.clip_and_step()
        return PrealignStepOutput(
            loss=loss_val,
            grad_norm=grad_norm_val,
            trainable_params=self.trainable_parameter_count(),
        )
