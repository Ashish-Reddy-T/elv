"""Tests for training reward functions."""

from __future__ import annotations

import pytest
import torch

from spatialvlm.training.curriculum import RewardWeights
from spatialvlm.training.rewards import (
    RewardConfig,
    collision_penalty_from_clearance,
    compute_reward_terms,
    consistency_reward,
    consistency_reward_from_responses,
    format_reward_from_responses,
    goal_reward,
    progress_reward,
    total_reward,
)


def test_format_reward_from_responses():
    responses = [
        "Reasoning: move forward\nAction: FORWARD",
        "Action: LEFT only",
        "reasoning: ok\naction: right",
    ]
    out = format_reward_from_responses(responses)
    assert out.shape == (3,)
    assert out.tolist() == [1.0, 0.0, 1.0]


def test_progress_reward_with_clip_and_nan():
    prev = torch.tensor([5.0, 2.0, float("nan"), 10.0])
    cur = torch.tensor([4.0, 4.5, 1.0, 0.0])
    out = progress_reward(prev, cur, clip_range=(-1.5, 1.5))
    assert out.tolist() == [1.0, -1.5, 0.0, 1.5]


def test_collision_penalty_from_clearance():
    clearance = torch.tensor([0.2, 0.05, float("inf"), -0.1])
    out = collision_penalty_from_clearance(clearance, threshold=0.1, penalty=-2.0)
    assert out.tolist() == [0.0, -2.0, -2.0, -2.0]


def test_goal_reward_requires_stop_and_distance():
    dist = torch.tensor([0.5, 0.5, 2.0])
    stopped = torch.tensor([1, 0, 1], dtype=torch.int64)
    out = goal_reward(dist, stopped, threshold=1.0, reward_value=10.0)
    assert out.tolist() == [10.0, 0.0, 0.0]


def test_consistency_reward():
    pred = ["forward", "turn_left", None]
    executed = ["forward", "turn_right", "forward"]
    out = consistency_reward(pred, executed, mismatch_penalty=-1.0)
    assert out.tolist() == [0.0, -1.0, -1.0]


def test_consistency_reward_from_responses():
    responses = [
        "Reasoning: ...\nAction: TURN_LEFT",
        "Reasoning: ...\nAction: FORWARD",
    ]
    executed = ["turn_left", "turn_left"]
    out = consistency_reward_from_responses(
        responses=responses,
        executed_actions=executed,
        mismatch_penalty=-1.0,
    )
    assert out.tolist() == [0.0, -1.0]


def test_compute_reward_terms_and_total():
    responses = [
        "Reasoning: clear\nAction: FORWARD",
        "Reasoning: uncertain\nAction: LEFT",
    ]
    executed = ["forward", "right"]
    terms = compute_reward_terms(
        responses=responses,
        executed_actions=executed,
        previous_geodesic=torch.tensor([5.0, 3.0]),
        current_geodesic=torch.tensor([4.0, 3.5]),
        clearance=torch.tensor([0.2, 0.05]),
        final_geodesic=torch.tensor([0.7, 2.0]),
        stopped=torch.tensor([1.0, 1.0]),
        config=RewardConfig(),
    )
    assert set(terms.keys()) == {"format", "progress", "collision", "goal", "consistency"}
    for value in terms.values():
        assert value.shape == (2,)

    weights = RewardWeights(
        format_weight=1.0,
        progress_weight=1.0,
        collision_weight=1.0,
        goal_weight=1.0,
        consistency_weight=1.0,
    )
    total = total_reward(terms, weights=weights)
    assert total.shape == (2,)
    assert torch.isfinite(total).all()


def test_compute_reward_terms_raises_on_batch_mismatch():
    with pytest.raises(ValueError):
        compute_reward_terms(
            responses=["Reasoning: x\nAction: FORWARD"],
            executed_actions=["forward", "left"],
            previous_geodesic=torch.tensor([1.0]),
            current_geodesic=torch.tensor([1.0]),
            clearance=torch.tensor([1.0]),
            final_geodesic=torch.tensor([1.0]),
            stopped=torch.tensor([1.0]),
        )
