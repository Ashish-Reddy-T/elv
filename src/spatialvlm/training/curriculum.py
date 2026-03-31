"""Curriculum utilities for staged reward weighting during GRPO training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RewardWeights:
    """Scalar weights for dense spatial reward components."""

    format_weight: float
    progress_weight: float
    collision_weight: float
    goal_weight: float
    consistency_weight: float

    def as_dict(self) -> dict[str, float]:
        return {
            "format": self.format_weight,
            "progress": self.progress_weight,
            "collision": self.collision_weight,
            "goal": self.goal_weight,
            "consistency": self.consistency_weight,
        }


@dataclass(frozen=True)
class CurriculumPoint:
    """Anchor point for piecewise-linear curriculum interpolation."""

    epoch: int
    weights: RewardWeights


class RewardCurriculum:
    """Piecewise-linear reward schedule keyed by epoch."""

    def __init__(self, points: list[CurriculumPoint]) -> None:
        if len(points) < 1:
            raise ValueError("RewardCurriculum requires at least one point.")
        ordered = sorted(points, key=lambda x: x.epoch)
        if ordered[0].epoch < 1:
            raise ValueError("Curriculum epochs must start at 1 or later.")
        self._points = ordered

    @classmethod
    def default(cls) -> RewardCurriculum:
        """Default 6-epoch progression emphasizing format -> spatial rewards."""
        return cls(
            points=[
                CurriculumPoint(
                    epoch=1,
                    weights=RewardWeights(
                        format_weight=1.0,
                        progress_weight=0.1,
                        collision_weight=0.1,
                        goal_weight=0.1,
                        consistency_weight=0.0,
                    ),
                ),
                CurriculumPoint(
                    epoch=3,
                    weights=RewardWeights(
                        format_weight=0.4,
                        progress_weight=0.4,
                        collision_weight=0.3,
                        goal_weight=0.5,
                        consistency_weight=0.2,
                    ),
                ),
                CurriculumPoint(
                    epoch=5,
                    weights=RewardWeights(
                        format_weight=0.1,
                        progress_weight=0.7,
                        collision_weight=0.6,
                        goal_weight=1.0,
                        consistency_weight=0.6,
                    ),
                ),
            ]
        )

    def get_weights(self, epoch: int) -> RewardWeights:
        if epoch < 1:
            raise ValueError(f"Epoch must be >= 1, got {epoch}.")

        if epoch <= self._points[0].epoch:
            return self._points[0].weights
        if epoch >= self._points[-1].epoch:
            return self._points[-1].weights

        left = self._points[0]
        right = self._points[-1]
        for i in range(1, len(self._points)):
            candidate = self._points[i]
            if epoch <= candidate.epoch:
                left = self._points[i - 1]
                right = candidate
                break

        span = right.epoch - left.epoch
        ratio = (epoch - left.epoch) / span
        lw = left.weights
        rw = right.weights
        return RewardWeights(
            format_weight=lw.format_weight + ratio * (rw.format_weight - lw.format_weight),
            progress_weight=lw.progress_weight + ratio * (rw.progress_weight - lw.progress_weight),
            collision_weight=lw.collision_weight
            + ratio * (rw.collision_weight - lw.collision_weight),
            goal_weight=lw.goal_weight + ratio * (rw.goal_weight - lw.goal_weight),
            consistency_weight=lw.consistency_weight
            + ratio * (rw.consistency_weight - lw.consistency_weight),
        )


def aggregate_weighted_rewards(
    reward_terms: dict[str, torch.Tensor],
    weights: RewardWeights,
) -> torch.Tensor:
    """Aggregate reward terms into a single scalar reward tensor.

    Expected keys:
      `format`, `progress`, `collision`, `goal`, `consistency`
    """
    expected = {"format", "progress", "collision", "goal", "consistency"}
    missing = expected.difference(reward_terms)
    if missing:
        raise KeyError(f"Missing reward terms: {sorted(missing)}")

    return (
        weights.format_weight * reward_terms["format"]
        + weights.progress_weight * reward_terms["progress"]
        + weights.collision_weight * reward_terms["collision"]
        + weights.goal_weight * reward_terms["goal"]
        + weights.consistency_weight * reward_terms["consistency"]
    )
