"""Stage-2 supervised fine-tuning utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch.nn.utils import clip_grad_norm_


@dataclass(frozen=True)
class SFTConfig:
    """Hyperparameters for supervised fine-tuning."""

    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    ignore_index: int = -100
    label_smoothing: float = 0.0
    trainable_keywords: tuple[str, ...] = (
        "projector",
        "gatr",
        "sva",
        "cross_attn",
        "lora",
    )


@dataclass(frozen=True)
class SFTStepOutput:
    """Result from one SFT optimization step."""

    loss: float
    grad_norm: float
    trainable_params: int


def supervised_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy SFT objective.

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
        label_smoothing=label_smoothing,
    )


def set_trainable_by_keyword(
    model: torch.nn.Module,
    keywords: Sequence[str],
) -> list[str]:
    """Freeze all parameters, then unfreeze names containing any keyword."""
    lowered = tuple(k.lower() for k in keywords)
    touched: list[str] = []

    for p in model.parameters():
        p.requires_grad_(False)

    for name, p in model.named_parameters():
        if any(key in name.lower() for key in lowered):
            p.requires_grad_(True)
            touched.append(name)
    return touched


class SFTTrainer:
    """Trainer for Stage-2 supervised fine-tuning."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: SFTConfig,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.trainable_parameter_names = set_trainable_by_keyword(model, config.trainable_keywords)
        if len(self.trainable_parameter_names) == 0:
            raise ValueError(
                f"No trainable parameters selected for SFT. Keywords={config.trainable_keywords}"
            )

        if optimizer is None:
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        self.optimizer = optimizer

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def step(self, batch: Mapping[str, Any]) -> SFTStepOutput:
        if "labels" not in batch:
            raise KeyError("SFT batch must include `labels`.")
        labels = batch["labels"]
        if not isinstance(labels, torch.Tensor):
            raise TypeError("Batch `labels` must be a torch.Tensor.")

        model_kwargs = {k: v for k, v in batch.items() if k != "labels"}
        outputs = self.model(**model_kwargs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        loss = supervised_loss(
            logits=logits,
            labels=labels,
            ignore_index=self.config.ignore_index,
            label_smoothing=self.config.label_smoothing,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()

        return SFTStepOutput(
            loss=float(loss.detach().cpu().item()),
            grad_norm=float(grad_norm.detach().cpu().item()),
            trainable_params=self.trainable_parameter_count(),
        )
