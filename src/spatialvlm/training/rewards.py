"""Reward functions for Stage-3 policy optimization."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from spatialvlm.training.curriculum import RewardWeights, aggregate_weighted_rewards

_ACTION_RE = re.compile(r"action\s*:\s*([a-zA-Z0-9_\- ]+)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class RewardConfig:
    """Configurable constants for reward computation."""

    format_reward: float = 1.0
    collision_penalty: float = -2.0
    goal_reward: float = 10.0
    consistency_penalty: float = -1.0
    collision_clearance_threshold: float = 0.1
    goal_distance_threshold: float = 1.0
    progress_clip: tuple[float, float] = (-2.0, 2.0)
    required_response_markers: tuple[str, ...] = ("Reasoning:", "Action:")


def _normalize_action(text: str | None) -> str | None:
    if text is None:
        return None
    norm = text.strip().lower().replace("-", "_").replace(" ", "_")
    return norm or None


def _extract_action_from_response(response: str) -> str | None:
    match = _ACTION_RE.search(response)
    if match is None:
        return None
    return _normalize_action(match.group(1))


def _ensure_vector(name: str, value: torch.Tensor, expected_size: int | None = None) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1D tensor, got {tuple(value.shape)}")
    if expected_size is not None and value.shape[0] != expected_size:
        raise ValueError(f"{name} batch size {value.shape[0]} != expected {expected_size}.")


def format_reward_from_responses(
    responses: Sequence[str],
    reward_value: float = 1.0,
    required_markers: Sequence[str] = ("Reasoning:", "Action:"),
    device: torch.device | None = None,
) -> torch.Tensor:
    """Binary reward for response format compliance."""
    out = torch.zeros(len(responses), dtype=torch.float32, device=device)
    markers = [m.lower() for m in required_markers]
    for i, response in enumerate(responses):
        text = response.lower()
        if all(marker in text for marker in markers):
            out[i] = reward_value
    return out


def progress_reward(
    previous_geodesic: torch.Tensor,
    current_geodesic: torch.Tensor,
    clip_range: tuple[float, float] | None = (-2.0, 2.0),
) -> torch.Tensor:
    """Dense reward from geodesic-distance improvement."""
    _ensure_vector("previous_geodesic", previous_geodesic)
    _ensure_vector("current_geodesic", current_geodesic, expected_size=previous_geodesic.shape[0])

    delta = previous_geodesic.float() - current_geodesic.float()
    delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_range is not None:
        delta = delta.clamp(min=clip_range[0], max=clip_range[1])
    return delta


def collision_penalty_from_clearance(
    clearance: torch.Tensor,
    threshold: float = 0.1,
    penalty: float = -2.0,
) -> torch.Tensor:
    """Penalty when clearance is below threshold."""
    _ensure_vector("clearance", clearance)
    c = clearance.float()
    bad = ~torch.isfinite(c) | (c < threshold)
    return torch.where(
        bad,
        torch.full_like(c, penalty),
        torch.zeros_like(c),
    )


def goal_reward(
    final_geodesic: torch.Tensor,
    stopped: torch.Tensor,
    threshold: float = 1.0,
    reward_value: float = 10.0,
) -> torch.Tensor:
    """Terminal reward when agent stops within goal threshold."""
    _ensure_vector("final_geodesic", final_geodesic)
    _ensure_vector("stopped", stopped, expected_size=final_geodesic.shape[0])

    dist = final_geodesic.float()
    stop = stopped.bool()
    success = torch.isfinite(dist) & (dist <= threshold) & stop
    return torch.where(
        success,
        torch.full_like(dist, reward_value),
        torch.zeros_like(dist),
    )


def consistency_reward(
    predicted_actions: Sequence[str | None],
    executed_actions: Sequence[str | None],
    mismatch_penalty: float = -1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Penalty when predicted action does not match executed action."""
    if len(predicted_actions) != len(executed_actions):
        raise ValueError("predicted_actions and executed_actions must have same length.")

    out = torch.zeros(len(predicted_actions), dtype=torch.float32, device=device)
    for i, (pred, exec_) in enumerate(zip(predicted_actions, executed_actions)):
        p = _normalize_action(pred)
        e = _normalize_action(exec_)
        if p is None or e is None or p != e:
            out[i] = mismatch_penalty
    return out


def consistency_reward_from_responses(
    responses: Sequence[str],
    executed_actions: Sequence[str | None],
    mismatch_penalty: float = -1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    predicted = [_extract_action_from_response(r) for r in responses]
    return consistency_reward(
        predicted_actions=predicted,
        executed_actions=executed_actions,
        mismatch_penalty=mismatch_penalty,
        device=device,
    )


def compute_reward_terms(
    responses: Sequence[str],
    executed_actions: Sequence[str | None],
    previous_geodesic: torch.Tensor,
    current_geodesic: torch.Tensor,
    clearance: torch.Tensor,
    final_geodesic: torch.Tensor,
    stopped: torch.Tensor,
    config: RewardConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compute all dense reward terms used in GRPO/fDPO training."""
    cfg = RewardConfig() if config is None else config
    n = len(responses)
    if len(executed_actions) != n:
        raise ValueError("responses and executed_actions must have the same batch size.")

    for name, tensor in (
        ("previous_geodesic", previous_geodesic),
        ("current_geodesic", current_geodesic),
        ("clearance", clearance),
        ("final_geodesic", final_geodesic),
        ("stopped", stopped),
    ):
        _ensure_vector(name, tensor, expected_size=n)

    device = previous_geodesic.device
    terms = {
        "format": format_reward_from_responses(
            responses=responses,
            reward_value=cfg.format_reward,
            required_markers=cfg.required_response_markers,
            device=device,
        ),
        "progress": progress_reward(
            previous_geodesic=previous_geodesic,
            current_geodesic=current_geodesic,
            clip_range=cfg.progress_clip,
        ),
        "collision": collision_penalty_from_clearance(
            clearance=clearance,
            threshold=cfg.collision_clearance_threshold,
            penalty=cfg.collision_penalty,
        ),
        "goal": goal_reward(
            final_geodesic=final_geodesic,
            stopped=stopped,
            threshold=cfg.goal_distance_threshold,
            reward_value=cfg.goal_reward,
        ),
        "consistency": consistency_reward_from_responses(
            responses=responses,
            executed_actions=executed_actions,
            mismatch_penalty=cfg.consistency_penalty,
            device=device,
        ),
    }
    return terms


def total_reward(
    reward_terms: Mapping[str, torch.Tensor],
    weights: RewardWeights,
) -> torch.Tensor:
    """Weighted sum of reward terms using curriculum-compatible weights."""
    return aggregate_weighted_rewards(dict(reward_terms), weights=weights)
