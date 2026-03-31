"""Tests for GRPO utilities and trainer."""

from __future__ import annotations

import torch
import torch.nn as nn

from spatialvlm.training.grpo import (
    GRPOConfig,
    GRPOTrainer,
    SelectiveSampleReplay,
    approximate_kl,
    compute_group_advantages,
    grpo_loss,
)


class DummyGRPOModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


def test_compute_group_advantages_normalized_per_group():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 8.0]])
    adv = compute_group_advantages(rewards)
    assert adv.shape == rewards.shape
    assert torch.allclose(adv.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_approximate_kl_non_negative():
    new = torch.tensor([[0.0, -0.1], [-0.2, 0.3]])
    ref = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    kl = approximate_kl(new_logprobs=new, ref_logprobs=ref)
    assert kl.item() >= 0.0


def test_grpo_loss_outputs_expected_fields():
    new = torch.tensor([[0.0, -0.1], [0.2, -0.3]], requires_grad=True)
    old = torch.zeros_like(new)
    ref = torch.zeros_like(new)
    adv = torch.tensor([1.0, -1.0])
    out = grpo_loss(
        new_logprobs=new,
        old_logprobs=old,
        ref_logprobs=ref,
        advantages=adv,
        clip_epsilon=0.2,
        kl_beta=0.001,
    )
    assert out.total_loss.ndim == 0
    assert out.clip_fraction.item() >= 0.0
    assert out.kl_loss.item() >= 0.0


def test_selective_sample_replay_keeps_high_advantage_only():
    replay = SelectiveSampleReplay(capacity=10, min_abs_advantage=0.5)
    inserted = replay.add_batch(
        advantages=torch.tensor([0.1, -0.7, 0.8]),
        payloads=[{"id": 0}, {"id": 1}, {"id": 2}],
    )
    assert inserted == 2
    assert len(replay) == 2
    assert len(replay.sample(5)) == 2


def test_grpo_trainer_step():
    model = DummyGRPOModel()
    trainer = GRPOTrainer(
        model=model,
        config=GRPOConfig(group_size=2, replay_advantage_threshold=0.2, learning_rate=1e-3),
    )
    base = torch.tensor([[0.0, -0.1], [0.2, -0.2], [0.0, 0.1], [0.3, -0.4]], requires_grad=True)
    out = trainer.step(
        {
            "new_logprobs": base * model.weight,
            "old_logprobs": torch.zeros_like(base),
            "ref_logprobs": torch.zeros_like(base),
            "rewards": torch.tensor([1.0, 0.0, 0.5, 0.5]),
        }
    )
    assert out.total_loss != 0.0
    assert out.grad_norm >= 0.0
    assert out.replay_inserted >= 0
