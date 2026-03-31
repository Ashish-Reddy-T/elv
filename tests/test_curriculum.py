"""Tests for reward curriculum utilities."""

from __future__ import annotations

import pytest
import torch

from spatialvlm.training.curriculum import (
    CurriculumPoint,
    RewardCurriculum,
    RewardWeights,
    aggregate_weighted_rewards,
)


def test_reward_curriculum_interpolates_between_points():
    cur = RewardCurriculum(
        points=[
            CurriculumPoint(1, RewardWeights(1.0, 0.0, 0.0, 0.0, 0.0)),
            CurriculumPoint(3, RewardWeights(0.0, 1.0, 0.0, 0.0, 0.0)),
        ]
    )
    w = cur.get_weights(epoch=2)
    assert w.format_weight == pytest.approx(0.5)
    assert w.progress_weight == pytest.approx(0.5)


def test_reward_curriculum_clamps_outside_range():
    cur = RewardCurriculum.default()
    first = cur.get_weights(epoch=1)
    before = cur.get_weights(epoch=0 + 1)
    after = cur.get_weights(epoch=99)
    assert before == first
    assert after == cur.get_weights(epoch=5)


def test_aggregate_weighted_rewards():
    weights = RewardWeights(1.0, 0.5, -2.0, 3.0, 0.25)
    terms = {
        "format": torch.tensor([1.0, 0.0]),
        "progress": torch.tensor([2.0, 2.0]),
        "collision": torch.tensor([0.0, 1.0]),
        "goal": torch.tensor([0.0, 1.0]),
        "consistency": torch.tensor([4.0, 4.0]),
    }
    out = aggregate_weighted_rewards(terms, weights)
    assert out.shape == (2,)
    # sample 0: 1 + 1 + 0 + 0 + 1 = 3
    assert out[0].item() == pytest.approx(3.0)


def test_aggregate_weighted_rewards_missing_key():
    weights = RewardWeights(1.0, 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(KeyError):
        aggregate_weighted_rewards({"format": torch.tensor([1.0])}, weights)
