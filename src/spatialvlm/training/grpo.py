"""Stage-3 GRPO utilities with selective sample replay support."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_


@dataclass(frozen=True)
class GRPOConfig:
    """Core GRPO hyperparameters."""

    group_size: int = 8
    clip_epsilon: float = 0.2
    kl_beta: float = 0.001
    entropy_beta: float = 0.0
    normalize_advantages: bool = True
    advantage_eps: float = 1e-6
    max_grad_norm: float = 1.0
    learning_rate: float = 5e-7
    weight_decay: float = 0.0
    replay_capacity: int = 4096
    replay_advantage_threshold: float = 0.05


@dataclass(frozen=True)
class GRPOLossBreakdown:
    """Structured GRPO loss decomposition."""

    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    entropy_loss: torch.Tensor
    total_loss: torch.Tensor
    clip_fraction: torch.Tensor


@dataclass(frozen=True)
class ReplaySample:
    """Single replay entry for selective sample replay."""

    advantage: float
    payload: dict[str, Any]


class SelectiveSampleReplay:
    """Replay buffer storing high-advantage trajectories only."""

    def __init__(self, capacity: int, min_abs_advantage: float) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self._buffer: deque[ReplaySample] = deque(maxlen=capacity)
        self.min_abs_advantage = float(min_abs_advantage)

    def __len__(self) -> int:
        return len(self._buffer)

    def add_batch(
        self,
        advantages: torch.Tensor,
        payloads: list[dict[str, Any]],
    ) -> int:
        if advantages.ndim != 1:
            raise ValueError(f"advantages must be 1D, got {tuple(advantages.shape)}")
        if len(payloads) != advantages.shape[0]:
            raise ValueError("payload count must match number of advantages.")

        keep = torch.nonzero(advantages.abs() >= self.min_abs_advantage, as_tuple=False).flatten()
        inserted = 0
        for idx in keep.tolist():
            self._buffer.append(
                ReplaySample(
                    advantage=float(advantages[idx].detach().cpu().item()),
                    payload=payloads[idx],
                )
            )
            inserted += 1
        return inserted

    def sample(self, n: int) -> list[ReplaySample]:
        if n <= 0 or len(self._buffer) == 0:
            return []
        n = min(n, len(self._buffer))
        idx = torch.randperm(len(self._buffer))[:n].tolist()
        return [self._buffer[i] for i in idx]


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over all elements, optionally weighted by boolean mask."""
    values_f = values.float()
    if mask is None:
        return values_f.mean()
    if values.shape != mask.shape:
        raise ValueError(f"values and mask shapes must match, got {values.shape} vs {mask.shape}.")
    weight = mask.float()
    denom = weight.sum().clamp_min(1.0)
    return (values_f * weight).sum() / denom


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-6,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute per-group normalized advantages.

    rewards: [G, K], where K is GRPO group size.
    """
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be [groups, group_size], got {tuple(rewards.shape)}")

    centered = rewards - rewards.mean(dim=1, keepdim=True)
    if not normalize:
        return centered

    std = rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return centered / std


def approximate_kl(
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Token-level KL approximation used in PPO-style policy optimization."""
    if new_logprobs.shape != ref_logprobs.shape:
        raise ValueError("new_logprobs and ref_logprobs must have matching shapes.")
    log_ratio = new_logprobs - ref_logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    return masked_mean(kl, mask=mask)


def grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    kl_beta: float,
    entropy_beta: float = 0.0,
    entropy: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> GRPOLossBreakdown:
    """Compute GRPO objective from logprob tensors.

    new_logprobs: [N, T]
    old_logprobs: [N, T]
    ref_logprobs: [N, T]
    advantages: [N]
    mask: [N, T] boolean (optional)
    """
    if new_logprobs.ndim != 2:
        raise ValueError(f"new_logprobs must be [N,T], got {tuple(new_logprobs.shape)}")
    if old_logprobs.shape != new_logprobs.shape or ref_logprobs.shape != new_logprobs.shape:
        raise ValueError("old/ref logprobs must match new_logprobs shape.")
    if advantages.ndim != 1 or advantages.shape[0] != new_logprobs.shape[0]:
        raise ValueError("advantages must be [N] where N matches logprob batch.")
    if mask is not None and mask.shape != new_logprobs.shape:
        raise ValueError("mask must match logprob tensor shape.")

    adv = advantages[:, None]  # [N, 1]
    ratio = torch.exp(new_logprobs - old_logprobs)  # [N, T]
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate_1 = ratio * adv
    surrogate_2 = clipped_ratio * adv
    policy_obj = torch.minimum(surrogate_1, surrogate_2)
    policy_loss = -masked_mean(policy_obj, mask=mask)

    kl_loss = approximate_kl(new_logprobs=new_logprobs, ref_logprobs=ref_logprobs, mask=mask)
    if entropy is None:
        entropy_term = torch.zeros((), device=new_logprobs.device, dtype=new_logprobs.dtype)
    else:
        entropy_term = masked_mean(entropy, mask=mask)

    total = policy_loss + kl_beta * kl_loss - entropy_beta * entropy_term
    clip_fraction = masked_mean((ratio - clipped_ratio).abs() > 1e-8, mask=mask)

    return GRPOLossBreakdown(
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        entropy_loss=entropy_term,
        total_loss=total,
        clip_fraction=clip_fraction,
    )


@dataclass(frozen=True)
class GRPOStepOutput:
    """Serializable scalar summary of one GRPO optimization step."""

    total_loss: float
    policy_loss: float
    kl_loss: float
    entropy: float
    clip_fraction: float
    grad_norm: float
    replay_inserted: int


class GRPOTrainer:
    """Minimal GRPO trainer around pre-computed token logprobs."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: GRPOConfig,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        self.optimizer = optimizer
        self.replay = SelectiveSampleReplay(
            capacity=config.replay_capacity,
            min_abs_advantage=config.replay_advantage_threshold,
        )

    def _build_advantages(self, batch: Mapping[str, Any]) -> torch.Tensor:
        if "advantages" in batch:
            adv = batch["advantages"]
            if not isinstance(adv, torch.Tensor):
                raise TypeError("`advantages` must be a torch.Tensor.")
            if adv.ndim != 1:
                raise ValueError("`advantages` must be 1D [N].")
            return adv

        if "rewards" not in batch:
            raise KeyError("GRPO batch requires either `advantages` or `rewards`.")
        rewards = batch["rewards"]
        if not isinstance(rewards, torch.Tensor):
            raise TypeError("`rewards` must be a torch.Tensor.")
        grouped = rewards.reshape(-1, self.config.group_size)
        adv_grouped = compute_group_advantages(
            rewards=grouped,
            eps=self.config.advantage_eps,
            normalize=self.config.normalize_advantages,
        )
        return adv_grouped.reshape(-1)

    def step(self, batch: Mapping[str, Any]) -> GRPOStepOutput:
        required = {"new_logprobs", "old_logprobs", "ref_logprobs"}
        missing = required.difference(batch.keys())
        if missing:
            raise KeyError(f"GRPO batch missing keys: {sorted(missing)}")

        new_logprobs = batch["new_logprobs"]
        old_logprobs = batch["old_logprobs"]
        ref_logprobs = batch["ref_logprobs"]
        if not all(isinstance(x, torch.Tensor) for x in (new_logprobs, old_logprobs, ref_logprobs)):
            raise TypeError("GRPO logprob entries must be torch.Tensor objects.")

        advantages = self._build_advantages(batch).to(new_logprobs.device)
        mask = batch.get("mask")
        if mask is not None and not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor when provided.")
        entropy = batch.get("entropy")
        if entropy is not None and not isinstance(entropy, torch.Tensor):
            raise TypeError("entropy must be a torch.Tensor when provided.")

        loss = grpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            clip_epsilon=self.config.clip_epsilon,
            kl_beta=self.config.kl_beta,
            entropy_beta=self.config.entropy_beta,
            entropy=entropy,
            mask=mask,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.total_loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()

        payloads = [{"idx": i} for i in range(advantages.shape[0])]
        replay_inserted = self.replay.add_batch(
            advantages=advantages.detach().cpu(), payloads=payloads
        )

        return GRPOStepOutput(
            total_loss=float(loss.total_loss.detach().cpu().item()),
            policy_loss=float(loss.policy_loss.detach().cpu().item()),
            kl_loss=float(loss.kl_loss.detach().cpu().item()),
            entropy=float(loss.entropy_loss.detach().cpu().item()),
            clip_fraction=float(loss.clip_fraction.detach().cpu().item()),
            grad_norm=float(grad_norm.detach().cpu().item()),
            replay_inserted=replay_inserted,
        )
