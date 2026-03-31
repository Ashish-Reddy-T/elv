"""Tests for Stage-1 pre-alignment trainer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spatialvlm.training.prealign import PrealignConfig, PrealignmentTrainer, masked_lm_loss


class DummyPrealignModel(nn.Module):
    def __init__(self, vocab_size: int = 13, hidden_dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.backbone = nn.Linear(hidden_dim, hidden_dim)
        self.projector = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        x = self.embed(input_ids)
        x = self.backbone(x)
        x = self.projector(x)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits)


def test_masked_lm_loss_accepts_ignore_index():
    logits = torch.randn(2, 3, 7)
    labels = torch.tensor([[1, -100, 2], [3, 4, -100]])
    loss = masked_lm_loss(logits=logits, labels=labels, ignore_index=-100)
    assert loss.item() > 0


def test_masked_lm_loss_rejects_bad_shapes():
    logits = torch.randn(2, 3, 7)
    labels = torch.randint(0, 7, (2, 4))
    with pytest.raises(ValueError):
        masked_lm_loss(logits=logits, labels=labels)


def test_prealign_trainer_unfreezes_projector_only_and_steps():
    model = DummyPrealignModel()
    trainer = PrealignmentTrainer(
        model=model,
        config=PrealignConfig(projector_keywords=("projector",), learning_rate=1e-3),
    )
    assert all("projector" in name for name in trainer.trainable_parameter_names)
    assert trainer.trainable_parameter_count() > 0

    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "labels": torch.tensor([[1, 2, 3], [4, -100, 6]]),
    }
    out = trainer.step(batch)
    assert out.loss > 0
    assert out.grad_norm >= 0
    assert out.trainable_params == trainer.trainable_parameter_count()


def test_prealign_trainer_raises_when_no_params_selected():
    model = DummyPrealignModel()
    with pytest.raises(ValueError, match="No trainable parameters"):
        PrealignmentTrainer(
            model=model,
            config=PrealignConfig(projector_keywords=("does-not-exist",)),
        )
