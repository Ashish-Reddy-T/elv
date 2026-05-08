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
    wrong_stop_weight: float = 0.0
    proximity_weight: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "format": self.format_weight,
            "progress": self.progress_weight,
            "collision": self.collision_weight,
            "goal": self.goal_weight,
            "consistency": self.consistency_weight,
            "wrong_stop": self.wrong_stop_weight,
            "proximity": self.proximity_weight,
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
        """Default 3-epoch progression: format+anti-STOP → navigation → full spatial.

        proximity_reward replaces the binary goal_reward + wrong_stop_reward with a
        sliding scale: ≤1m→+10, ≤3m→+5, ≤6m→+1, >6m→-1. goal_weight and
        wrong_stop_weight are kept at 0 to avoid double-counting.
        """
        return cls(
            points=[
                CurriculumPoint(
                    epoch=1,
                    weights=RewardWeights(
                        format_weight=0.5,
                        progress_weight=0.3,
                        collision_weight=0.1,
                        goal_weight=0.0,
                        consistency_weight=0.0,
                        wrong_stop_weight=0.0,
                        proximity_weight=1.0,
                    ),
                ),
                CurriculumPoint(
                    epoch=2,
                    weights=RewardWeights(
                        format_weight=0.3,
                        progress_weight=0.5,
                        collision_weight=0.3,
                        goal_weight=0.0,
                        consistency_weight=0.2,
                        wrong_stop_weight=0.0,
                        proximity_weight=0.8,
                    ),
                ),
                CurriculumPoint(
                    epoch=3,
                    weights=RewardWeights(
                        format_weight=0.1,
                        progress_weight=0.7,
                        collision_weight=0.6,
                        goal_weight=0.0,
                        consistency_weight=0.6,
                        wrong_stop_weight=0.0,
                        proximity_weight=1.0,
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
        def lerp(a: float, b: float) -> float:
            return a + ratio * (b - a)

        return RewardWeights(
            format_weight=lerp(lw.format_weight, rw.format_weight),
            progress_weight=lerp(lw.progress_weight, rw.progress_weight),
            collision_weight=lerp(lw.collision_weight, rw.collision_weight),
            goal_weight=lerp(lw.goal_weight, rw.goal_weight),
            consistency_weight=lerp(lw.consistency_weight, rw.consistency_weight),
            wrong_stop_weight=lerp(lw.wrong_stop_weight, rw.wrong_stop_weight),
            proximity_weight=lerp(lw.proximity_weight, rw.proximity_weight),
        )


def aggregate_weighted_rewards(
    reward_terms: dict[str, torch.Tensor],
    weights: RewardWeights,
) -> torch.Tensor:
    """Aggregate reward terms into a single scalar reward tensor.

    Expected keys:
      `format`, `progress`, `collision`, `goal`, `consistency`, `wrong_stop`
    """
    expected = {"format", "progress", "collision", "goal", "consistency", "wrong_stop", "proximity"}
    missing = expected.difference(reward_terms)
    if missing:
        raise KeyError(f"Missing reward terms: {sorted(missing)}")

    return (
        weights.format_weight * reward_terms["format"]
        + weights.progress_weight * reward_terms["progress"]
        + weights.collision_weight * reward_terms["collision"]
        + weights.goal_weight * reward_terms["goal"]
        + weights.consistency_weight * reward_terms["consistency"]
        + weights.wrong_stop_weight * reward_terms["wrong_stop"]
        + weights.proximity_weight * reward_terms["proximity"]
    )
