"""Habitat environment wrapper for RGB + depth observations.

This module supports two usage modes:
1. Wrap an already-created Habitat (or gym-like) environment.
2. Construct a Habitat environment from a config path + overrides.

Core invariant for SpatialVLM:
  RGB and depth observations must be rendered at exactly 518x518 to keep
  pixel-perfect alignment with the DINOv2 patch grid (37x37 at patch size 14).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


def _resolve_habitat_get_config(habitat_module: Any):
    """Resolve Habitat's `get_config` entrypoint across version variants."""
    if hasattr(habitat_module, "get_config"):
        return habitat_module.get_config

    try:
        from habitat.config.default import get_config as legacy_get_config
    except ImportError as exc:  # pragma: no cover - depends on external package version
        raise ImportError(
            "Could not resolve Habitat get_config API in installed habitat package."
        ) from exc
    return legacy_get_config


def require_habitat():
    """Import Habitat lazily and raise a clear message when unavailable."""
    try:
        import habitat
    except ImportError as exc:
        raise ImportError(
            "Habitat is not installed. Install optional deps with "
            "`pip install -e .[habitat]` in your SpatialVLM environment."
        ) from exc
    return habitat


@dataclass
class HabitatEnvConfig:
    """Configuration for constructing a Habitat environment wrapper."""

    config_path: str
    scene_id: str | None = None
    width: int = 518
    height: int = 518
    max_episode_steps: int | None = None
    seed: int | None = None
    extra_overrides: list[str] = field(default_factory=list)

    def build_overrides(self) -> list[str]:
        """Build Habitat override strings, including required sensor resolution."""
        overrides = list(self.extra_overrides)
        overrides.extend(
            [
                f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width={self.width}",
                f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height={self.height}",
                f"habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width={self.width}",
                f"habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height={self.height}",
            ]
        )
        if self.scene_id is not None:
            overrides.append(f"habitat.simulator.scene={self.scene_id}")
        if self.max_episode_steps is not None:
            overrides.append(f"habitat.environment.max_episode_steps={self.max_episode_steps}")
        if self.seed is not None:
            overrides.append(f"habitat.seed={self.seed}")
        return overrides


class HabitatEnvWrapper:
    """Light wrapper around Habitat env with tensorized observations."""

    def __init__(
        self,
        env: Any,
        expected_width: int = 518,
        expected_height: int = 518,
        device: torch.device | None = None,
    ) -> None:
        self.env = env
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.device = torch.device("cpu") if device is None else device

    @staticmethod
    def _resolve_rgb_hw(rgb: torch.Tensor) -> tuple[int, int]:
        """Resolve RGB height/width for HWC/CHW and batched variants."""
        if rgb.ndim == 3:
            if rgb.shape[-1] == 3:  # [H, W, 3]
                return int(rgb.shape[0]), int(rgb.shape[1])
            if rgb.shape[0] == 3:  # [3, H, W]
                return int(rgb.shape[1]), int(rgb.shape[2])
        elif rgb.ndim == 4:
            if rgb.shape[-1] == 3:  # [B, H, W, 3]
                return int(rgb.shape[-3]), int(rgb.shape[-2])
            if rgb.shape[1] == 3:  # [B, 3, H, W]
                return int(rgb.shape[-2]), int(rgb.shape[-1])
        raise ValueError(f"RGB observation must be HWC/CHW or batched variants, got {rgb.shape}.")

    @staticmethod
    def _resolve_depth_hw(depth: torch.Tensor) -> tuple[int, int]:
        """Resolve depth height/width for HW/HW1/1HW and batched variants."""
        if depth.ndim == 2:  # [H, W]
            return int(depth.shape[0]), int(depth.shape[1])
        if depth.ndim == 3:
            if depth.shape[-1] == 1:  # [H, W, 1]
                return int(depth.shape[0]), int(depth.shape[1])
            if depth.shape[0] == 1:  # [1, H, W]
                return int(depth.shape[1]), int(depth.shape[2])
            return int(depth.shape[-2]), int(depth.shape[-1])  # [B, H, W]
        if depth.ndim == 4:
            if depth.shape[1] == 1:  # [B, 1, H, W]
                return int(depth.shape[-2]), int(depth.shape[-1])
            if depth.shape[-1] == 1:  # [B, H, W, 1]
                return int(depth.shape[-3]), int(depth.shape[-2])
        raise ValueError(
            f"Depth observation must be HW/HW1/1HW or batched variants, got {depth.shape}."
        )

    @classmethod
    def from_config(
        cls,
        cfg: HabitatEnvConfig,
        device: torch.device | None = None,
    ) -> HabitatEnvWrapper:
        """Create wrapper by constructing Habitat env from config + overrides."""
        config_path = Path(cfg.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Habitat config file not found: {config_path}")

        habitat = require_habitat()
        get_config = _resolve_habitat_get_config(habitat)
        habitat_cfg = get_config(config_path=str(config_path), overrides=cfg.build_overrides())
        env = habitat.Env(config=habitat_cfg)
        return cls(
            env=env,
            expected_width=cfg.width,
            expected_height=cfg.height,
            device=device,
        )

    def _to_tensor_obs(self, obs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Convert observation mapping values to tensors on configured device."""
        out: dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = torch.as_tensor(v, device=self.device)
        return out

    def validate_observation_resolution(self, obs: Mapping[str, torch.Tensor]) -> None:
        """Ensure rgb/depth tensors are exactly expected resolution."""
        if "rgb" in obs:
            rgb = obs["rgb"]
            h, w = self._resolve_rgb_hw(rgb)
            if (h, w) != (self.expected_height, self.expected_width):
                raise ValueError(
                    "RGB resolution must be "
                    f"{(self.expected_height, self.expected_width)}, got {(h, w)}."
                )

        if "depth" in obs:
            depth = obs["depth"]
            h, w = self._resolve_depth_hw(depth)
            if (h, w) != (self.expected_height, self.expected_width):
                raise ValueError(
                    f"Depth resolution must be {(self.expected_height, self.expected_width)}, "
                    f"got {(h, w)}."
                )

    def reset(self) -> dict[str, torch.Tensor]:
        """Reset environment and return tensorized observations."""
        obs = self.env.reset()
        out = self._to_tensor_obs(obs)
        self.validate_observation_resolution(out)
        return out

    def step(
        self,
        action: int | str | Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], float, bool, dict[str, Any]]:
        """Step environment and normalize return signature."""
        result = self.env.step(action)
        if not isinstance(result, tuple):
            raise ValueError("Env step must return a tuple.")

        if len(result) == 4:
            obs, reward, done, info = result
        elif len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated) or bool(truncated)
        else:
            raise ValueError(f"Unsupported step return length: {len(result)}.")

        out_obs = self._to_tensor_obs(obs)
        self.validate_observation_resolution(out_obs)
        return out_obs, float(reward), bool(done), dict(info)

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()


def extract_rgb_depth(obs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and standardize RGB + depth tensors from observation mapping.

    Returns
    -------
    rgb : Tensor[H, W, 3]
    depth : Tensor[H, W]
    """
    if "rgb" not in obs or "depth" not in obs:
        raise KeyError("Observation must contain both 'rgb' and 'depth' keys.")

    rgb = obs["rgb"]
    depth = obs["depth"]

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"RGB must be [H,W,3], got {tuple(rgb.shape)}.")

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth must be [H,W] (or [H,W,1]), got {tuple(depth.shape)}.")

    return rgb, depth
