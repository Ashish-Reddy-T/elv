"""Tests for Stage-2 SFT trainer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spatialvlm.training.sft import SFTConfig, SFTTrainer, supervised_loss


class DummySFTModel(nn.Module):
    def __init__(self, vocab_size: int = 17, hidden_dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.projector = nn.Linear(hidden_dim, hidden_dim)
        self.gatr_adapter = nn.Linear(hidden_dim, hidden_dim)
        self.lora_block = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        x = self.embed(input_ids)
        x = self.projector(x)
        x = self.gatr_adapter(x)
        x = self.lora_block(x)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits)


def test_supervised_loss_label_smoothing():
    logits = torch.randn(2, 4, 11)
    labels = torch.randint(0, 11, (2, 4))
    loss = supervised_loss(logits=logits, labels=labels, label_smoothing=0.1)
    assert loss.item() > 0


def test_supervised_loss_shape_validation():
    logits = torch.randn(2, 4, 11)
    labels = torch.randint(0, 11, (2, 3))
    with pytest.raises(ValueError):
        supervised_loss(logits=logits, labels=labels)


def test_sft_trainer_step_and_parameter_selection():
    model = DummySFTModel()
    config = SFTConfig(
        trainable_keywords=("projector", "gatr", "lora"),
        learning_rate=1e-3,
    )
    trainer = SFTTrainer(model=model, config=config)
    assert any("projector" in n for n in trainer.trainable_parameter_names)
    assert any("gatr" in n for n in trainer.trainable_parameter_names)
    assert any("lora" in n for n in trainer.trainable_parameter_names)

    out = trainer.step(
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([[1, 2, 3], [4, 5, -100]]),
        }
    )
    assert out.loss > 0
    assert out.grad_norm >= 0
    assert out.trainable_params == trainer.trainable_parameter_count()


def test_sft_trainer_raises_when_no_trainable_selected():
    with pytest.raises(ValueError):
        SFTTrainer(
            model=DummySFTModel(),
            config=SFTConfig(trainable_keywords=("not-found",)),
        )
