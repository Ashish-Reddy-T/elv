"""Tests for fDPO objective and trainer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spatialvlm.training.fdpo import FDPOConfig, FDPOTrainer, fdpo_loss


class DummyFDPOModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))


def test_fdpo_loss_prefers_better_chosen_responses():
    chosen = torch.tensor([2.0, 1.5])
    rejected = torch.tensor([0.0, 0.5])
    chosen_ref = torch.tensor([1.0, 1.0])
    rejected_ref = torch.tensor([0.5, 0.5])
    out = fdpo_loss(
        chosen_logps=chosen,
        rejected_logps=rejected,
        chosen_ref_logps=chosen_ref,
        rejected_ref_logps=rejected_ref,
        beta=0.1,
    )
    assert out.total_loss.item() > 0.0
    assert out.accuracy.item() == pytest.approx(1.0)
    assert out.preference_margin.item() > 0.0


def test_fdpo_loss_reference_free_mode():
    chosen = torch.tensor([1.0, 0.8])
    rejected = torch.tensor([0.2, 0.1])
    out = fdpo_loss(
        chosen_logps=chosen,
        rejected_logps=rejected,
        chosen_ref_logps=None,
        rejected_ref_logps=None,
        beta=0.1,
        reference_free=True,
    )
    assert out.total_loss.item() > 0.0


def test_fdpo_trainer_step():
    model = DummyFDPOModel()
    trainer = FDPOTrainer(model=model, config=FDPOConfig(beta=0.2, learning_rate=1e-3))
    chosen = torch.tensor([1.2, 1.0], requires_grad=True) * model.scale
    rejected = torch.tensor([0.1, 0.2], requires_grad=True) * model.scale
    out = trainer.step(
        {
            "chosen_logps": chosen,
            "rejected_logps": rejected,
            "chosen_ref_logps": torch.tensor([0.3, 0.4]),
            "rejected_ref_logps": torch.tensor([0.2, 0.2]),
        }
    )
    assert out.loss > 0.0
    assert out.accuracy >= 0.0
    assert out.grad_norm >= 0.0
