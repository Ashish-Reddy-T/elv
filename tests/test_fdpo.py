"""Tests for fDPO objective and trainer with segment-specific betas."""

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
        beta_grounding=0.1,
        beta_reasoning=0.1,
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
        beta_grounding=0.1,
        beta_reasoning=0.05,
        reference_free=True,
    )
    assert out.total_loss.item() > 0.0


def test_fdpo_segment_specific_betas_produce_different_losses():
    """Different betas for grounding vs reasoning should produce different loss contributions."""
    n = 10
    chosen = torch.randn(n)
    rejected = torch.randn(n)

    # All grounding (high beta)
    mask_all_grounding = torch.ones(n, dtype=torch.bool)
    out_grounding = fdpo_loss(
        chosen_logps=chosen,
        rejected_logps=rejected,
        chosen_ref_logps=None,
        rejected_ref_logps=None,
        beta_grounding=0.5,
        beta_reasoning=0.01,
        segment_mask=mask_all_grounding,
        reference_free=True,
    )

    # All reasoning (low beta)
    mask_all_reasoning = torch.zeros(n, dtype=torch.bool)
    out_reasoning = fdpo_loss(
        chosen_logps=chosen,
        rejected_logps=rejected,
        chosen_ref_logps=None,
        rejected_ref_logps=None,
        beta_grounding=0.5,
        beta_reasoning=0.01,
        segment_mask=mask_all_reasoning,
        reference_free=True,
    )

    # With very different betas, losses should differ
    assert out_grounding.total_loss.item() != pytest.approx(
        out_reasoning.total_loss.item(), abs=1e-5
    ), "Segment-specific betas should produce different losses"


def test_fdpo_loss_segment_mask_splits_correctly():
    """Grounding and reasoning loss breakdowns should reflect the segment mask."""
    n = 6
    chosen = torch.tensor([1.0, 1.5, 2.0, 0.5, 0.8, 1.2])
    rejected = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.1, 0.2])
    # First 3 = grounding, last 3 = reasoning
    segment_mask = torch.tensor([True, True, True, False, False, False])

    out = fdpo_loss(
        chosen_logps=chosen,
        rejected_logps=rejected,
        chosen_ref_logps=None,
        rejected_ref_logps=None,
        beta_grounding=0.2,
        beta_reasoning=0.05,
        segment_mask=segment_mask,
        reference_free=True,
    )

    assert out.grounding_loss.item() > 0.0
    assert out.reasoning_loss.item() > 0.0
    # Both should be finite
    assert torch.isfinite(out.grounding_loss)
    assert torch.isfinite(out.reasoning_loss)


def test_fdpo_loss_no_segment_mask_uses_grounding_beta():
    """Without segment_mask, all samples should use beta_grounding."""
    chosen = torch.tensor([1.0, 0.8])
    rejected = torch.tensor([0.2, 0.1])

    out_no_mask = fdpo_loss(
        chosen_logps=chosen,
        rejected_logps=rejected,
        chosen_ref_logps=None,
        rejected_ref_logps=None,
        beta_grounding=0.3,
        beta_reasoning=0.01,
        segment_mask=None,
        reference_free=True,
    )

    out_all_grounding = fdpo_loss(
        chosen_logps=chosen,
        rejected_logps=rejected,
        chosen_ref_logps=None,
        rejected_ref_logps=None,
        beta_grounding=0.3,
        beta_reasoning=0.01,
        segment_mask=torch.ones(2, dtype=torch.bool),
        reference_free=True,
    )

    assert out_no_mask.total_loss.item() == pytest.approx(
        out_all_grounding.total_loss.item(), abs=1e-6
    )


def test_fdpo_trainer_step():
    model = DummyFDPOModel()
    trainer = FDPOTrainer(
        model=model,
        config=FDPOConfig(beta_grounding=0.2, beta_reasoning=0.05, learning_rate=1e-3),
    )
    chosen = torch.tensor([1.2, 1.0], requires_grad=True) * model.scale
    rejected = torch.tensor([0.1, 0.2], requires_grad=True) * model.scale
    out = trainer.step(
        {
            "chosen_logps": chosen,
            "rejected_logps": rejected,
            "chosen_ref_logps": torch.tensor([0.3, 0.4]),
            "rejected_ref_logps": torch.tensor([0.2, 0.2]),
            "segment_mask": torch.tensor([True, False]),
        }
    )
    assert out.loss > 0.0
    assert out.grounding_loss >= 0.0
    assert out.reasoning_loss >= 0.0
    assert out.accuracy >= 0.0
    assert out.grad_norm >= 0.0
