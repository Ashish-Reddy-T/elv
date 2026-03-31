"""Tests for Habitat data wrapper utilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import spatialvlm.data.habitat_env as habitat_env
from spatialvlm.data.habitat_env import HabitatEnvConfig, HabitatEnvWrapper, extract_rgb_depth


class FakeEnv:
    def __init__(self, reset_obs, step_result) -> None:
        self._reset_obs = reset_obs
        self._step_result = step_result
        self.closed = False

    def reset(self):
        return self._reset_obs

    def step(self, action):
        _ = action
        return self._step_result

    def close(self):
        self.closed = True


def _make_obs() -> dict[str, np.ndarray]:
    return {
        "rgb": np.zeros((518, 518, 3), dtype=np.uint8),
        "depth": np.ones((518, 518, 1), dtype=np.float32),
    }


def test_build_overrides_includes_required_resolution_and_optional_fields():
    cfg = HabitatEnvConfig(
        config_path="dummy.yaml",
        scene_id="scene.glb",
        max_episode_steps=300,
        seed=7,
        extra_overrides=["foo=bar"],
    )
    overrides = cfg.build_overrides()
    assert "foo=bar" in overrides
    assert "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=518" in overrides
    assert "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=518" in overrides
    assert "habitat.simulator.scene=scene.glb" in overrides
    assert "habitat.environment.max_episode_steps=300" in overrides
    assert "habitat.seed=7" in overrides


def test_from_config_missing_file_raises_before_habitat_import():
    cfg = HabitatEnvConfig(config_path="/tmp/does-not-exist.yaml")
    with pytest.raises(FileNotFoundError):
        HabitatEnvWrapper.from_config(cfg)


def test_from_config_uses_habitat_get_config_and_constructs_env(monkeypatch, tmp_path):
    config_file = tmp_path / "habitat.yaml"
    config_file.write_text("dummy: true", encoding="utf-8")
    captured: dict[str, object] = {}

    class DummyHabitatEnv:
        def __init__(self, config):
            self.config = config

        def reset(self):
            return _make_obs()

    def fake_get_config(*, config_path: str, overrides: list[str]):
        captured["config_path"] = config_path
        captured["overrides"] = overrides
        return {"ok": True, "overrides": overrides}

    dummy_habitat_module = SimpleNamespace(Env=DummyHabitatEnv, get_config=fake_get_config)
    monkeypatch.setattr(habitat_env, "require_habitat", lambda: dummy_habitat_module)

    wrapper = HabitatEnvWrapper.from_config(HabitatEnvConfig(config_path=str(config_file)))
    assert isinstance(wrapper.env, DummyHabitatEnv)
    assert captured["config_path"] == str(config_file)
    assert (
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=518"
        in captured["overrides"]
    )


def test_reset_tensorizes_and_accepts_depth_hw1():
    obs = _make_obs()
    env = FakeEnv(reset_obs=obs, step_result=(obs, 0.0, False, {}))
    wrapper = HabitatEnvWrapper(env=env)
    out = wrapper.reset()
    assert isinstance(out["rgb"], torch.Tensor)
    assert isinstance(out["depth"], torch.Tensor)
    assert out["rgb"].shape == (518, 518, 3)
    assert out["depth"].shape == (518, 518, 1)


def test_step_handles_4_tuple_signature():
    obs = _make_obs()
    env = FakeEnv(reset_obs=obs, step_result=(obs, 1.5, True, {"x": 1}))
    wrapper = HabitatEnvWrapper(env=env)
    out_obs, reward, done, info = wrapper.step(action=0)
    assert out_obs["rgb"].shape == (518, 518, 3)
    assert reward == 1.5
    assert done is True
    assert info == {"x": 1}


def test_step_handles_5_tuple_signature():
    obs = _make_obs()
    env = FakeEnv(reset_obs=obs, step_result=(obs, 0.3, False, True, {"mode": "truncated"}))
    wrapper = HabitatEnvWrapper(env=env)
    _, reward, done, info = wrapper.step(action="forward")
    assert reward == 0.3
    assert done is True
    assert info["mode"] == "truncated"


def test_validate_observation_resolution_raises_on_mismatch():
    bad_obs = {
        "rgb": torch.zeros(500, 500, 3),
        "depth": torch.zeros(500, 500, 1),
    }
    wrapper = HabitatEnvWrapper(
        env=FakeEnv(reset_obs=bad_obs, step_result=(bad_obs, 0.0, False, {}))
    )
    with pytest.raises(ValueError, match="RGB resolution must be"):
        wrapper.reset()


def test_extract_rgb_depth_supports_depth_hw1():
    obs = {
        "rgb": torch.zeros(518, 518, 3),
        "depth": torch.ones(518, 518, 1),
    }
    rgb, depth = extract_rgb_depth(obs)
    assert rgb.shape == (518, 518, 3)
    assert depth.shape == (518, 518)


def test_extract_rgb_depth_errors_when_missing_keys():
    with pytest.raises(KeyError):
        extract_rgb_depth({"rgb": torch.zeros(2, 2, 3)})
