#!/usr/bin/env python3
"""Habitat environment worker for SpatialVLM GRPO rollouts.

Runs under habitat-venv (Python 3.11). Spawned as a subprocess by train_grpo.py.
Communicates via stdin/stdout using pickle serialization.

All print/stderr is forwarded to a log file; stdout is exclusively for pickle.

Commands (received via stdin, pickle):
    {"cmd": "load_scene", "scene_path": str}
    {"cmd": "reset", "start_pos": [x,y,z], "start_rot": [x,y,z,w], "goal_pos": [x,y,z]}
    {"cmd": "step", "action": str}   # MOVE_FORWARD | TURN_LEFT | TURN_RIGHT | STOP
    {"cmd": "quit"}

Responses (sent via stdout, pickle):
    {"ok": True, "rgb": ndarray[H,W,3] uint8, "depth": ndarray[H,W] float32,
     "geodesic": float, "clearance": float, "done": bool, "step_idx": int}
    {"ok": False, "error": str}

Action space:
    MOVE_FORWARD  — 0.25 m forward
    TURN_LEFT     — 15 deg left  (around Y axis)
    TURN_RIGHT    — 15 deg right (around Y axis)
    STOP          — no movement, marks done=True
"""

from __future__ import annotations

import math
import os
import pickle
import sys
from typing import Any

import numpy as np

# ── stdout protection ────────────────────────────────────────────────────────
# habitat_sim writes C-level output directly to fd 1 (stdout) during scene
# loading.  Python-level sys.stdout redirection does NOT prevent this.
# Fix: save fd 1 into a new fd before any habitat import, then point fd 1
# at /dev/null so C-level writes are silently dropped.  All pickle responses
# are sent via the saved fd (_comm_write).
_COMM_FD = os.dup(1)
_comm_write = os.fdopen(_COMM_FD, "wb", buffering=0)
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 1)   # fd 1 → /dev/null (catches C-level writes from habitat)
os.close(_devnull_fd)
sys.stdout = open(os.devnull, "w")  # Python-level stdout also silenced

# Redirect stderr to a log file
_log_path = os.environ.get("HABITAT_WORKER_LOG", "/tmp/habitat_worker.log")
_log_fh = open(_log_path, "a", buffering=1)
sys.stderr = _log_fh


def _log(msg: str) -> None:
    print(f"[habitat_worker] {msg}", file=sys.stderr, flush=True)


def _require_sim():
    try:
        import habitat_sim
        return habitat_sim
    except ImportError:
        _log("habitat_sim not found — is this running in the habitat-venv?")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Discrete action constants
# ---------------------------------------------------------------------------

FORWARD_STEP = 0.25   # metres
TURN_DEG = 15.0       # degrees


# ---------------------------------------------------------------------------
# Quaternion helpers (habitat_sim uses [x, y, z, w] scipy-style quats)
# ---------------------------------------------------------------------------

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two [x,y,z,w] quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)


def _quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (both as numpy arrays)."""
    vq = np.array([v[0], v[1], v[2], 0.0])
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
    rotated = _quat_mul(_quat_mul(q, vq), q_conj)
    return rotated[:3]


def _quat_from_angle_y(angle_rad: float) -> np.ndarray:
    """[x,y,z,w] quaternion for rotation around Y axis."""
    half = angle_rad / 2.0
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)])


def _quat_to_habitat(hs: Any, q: np.ndarray) -> Any:
    """Convert [x,y,z,w] numpy array to habitat_sim quaternion."""
    return hs.utils.common.quat_from_coeffs(q.astype(np.float32))


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------

class HabitatRolloutEnv:
    """Manages a single Habitat sim for interactive rollout generation."""

    FORWARD = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # Habitat: -Z is forward

    def __init__(self, width: int = 518, height: int = 518, hfov: float = 90.0) -> None:
        self._hs = _require_sim()
        self.width = width
        self.height = height
        self.hfov = hfov
        self._sim: Any = None
        self._goal_pos: np.ndarray | None = None
        self._step_idx: int = 0
        self._done: bool = False
        self._current_scene: str = ""

    def load_scene(self, scene_path: str) -> None:
        if scene_path == self._current_scene and self._sim is not None:
            return
        if self._sim is not None:
            self._sim.close()
            self._sim = None
        _log(f"Loading scene: {scene_path}")
        hs = self._hs
        backend_cfg = hs.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.enable_physics = False

        rgb_spec = hs.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = hs.SensorType.COLOR
        rgb_spec.resolution = [self.height, self.width]
        rgb_spec.hfov = self.hfov
        rgb_spec.position = [0.0, 1.25, 0.0]

        depth_spec = hs.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = hs.SensorType.DEPTH
        depth_spec.resolution = [self.height, self.width]
        depth_spec.hfov = self.hfov
        depth_spec.position = [0.0, 1.25, 0.0]

        agent_cfg = hs.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

        cfg = hs.Configuration(backend_cfg, [agent_cfg])
        self._sim = hs.Simulator(cfg)
        self._current_scene = scene_path
        _log(f"Scene loaded: {scene_path}")

    def reset(
        self,
        start_pos: list[float],
        start_rot: list[float],
        goal_pos: list[float],
    ) -> dict[str, Any]:
        if self._sim is None:
            raise RuntimeError("Call load_scene before reset.")
        agent = self._sim.get_agent(0)
        state = agent.get_state()
        state.position = np.array(start_pos, dtype=np.float32)
        state.rotation = _quat_to_habitat(self._hs, np.array(start_rot, dtype=np.float64))
        agent.set_state(state)
        raw_goal = np.array(goal_pos, dtype=np.float32)
        if self._sim.pathfinder.is_loaded:
            snapped = self._sim.pathfinder.snap_point(raw_goal)
            raw_goal = np.array(snapped, dtype=np.float32) if not any(np.isnan(snapped)) else raw_goal
        self._goal_pos = raw_goal
        self._step_idx = 0
        self._done = False
        return self._make_obs()

    def step(self, action: str) -> dict[str, Any]:
        if self._sim is None:
            raise RuntimeError("Call reset before step.")
        if self._done:
            return self._make_obs()

        action = action.strip().upper()
        if action == "STOP":
            self._done = True
        elif action == "MOVE_FORWARD":
            self._do_move_forward()
        elif action == "TURN_LEFT":
            self._do_turn(TURN_DEG)
        elif action == "TURN_RIGHT":
            self._do_turn(-TURN_DEG)
        else:
            _log(f"Unknown action '{action}', treating as STOP")
            self._done = True

        self._step_idx += 1
        return self._make_obs()

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _do_move_forward(self) -> None:
        agent = self._sim.get_agent(0)
        state = agent.get_state()
        q = np.array([
            state.rotation.x, state.rotation.y,
            state.rotation.z, state.rotation.w,
        ], dtype=np.float64)
        forward_world = _quat_rotate_vector(q, self.FORWARD)
        new_pos = np.array(state.position, dtype=np.float32) + (forward_world * FORWARD_STEP).astype(np.float32)
        # Snap to navigable surface (prevents walking through walls)
        snapped = self._sim.pathfinder.snap_point(new_pos)
        state.position = snapped
        agent.set_state(state)

    def _do_turn(self, degrees: float) -> None:
        agent = self._sim.get_agent(0)
        state = agent.get_state()
        q_cur = np.array([
            state.rotation.x, state.rotation.y,
            state.rotation.z, state.rotation.w,
        ], dtype=np.float64)
        q_turn = _quat_from_angle_y(math.radians(degrees))
        q_new = _quat_mul(q_cur, q_turn)
        q_new /= np.linalg.norm(q_new)
        state.rotation = _quat_to_habitat(self._hs, q_new)
        agent.set_state(state)

    def _make_obs(self) -> dict[str, Any]:
        obs = self._sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3].astype(np.uint8)  # [H, W, 3]
        depth_raw = obs["depth"]
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        depth = depth_raw.astype(np.float32)  # [H, W] metric metres

        geodesic = self._get_geodesic()
        clearance = self._get_clearance()

        return {
            "ok": True,
            "rgb": rgb,
            "depth": depth,
            "geodesic": geodesic,
            "clearance": clearance,
            "done": self._done,
            "step_idx": self._step_idx,
        }

    def _get_geodesic(self) -> float:
        if self._goal_pos is None or self._sim is None:
            return float("inf")
        try:
            agent_pos = np.array(self._sim.get_agent(0).get_state().position, dtype=np.float32)
            # habitat-sim 0.3.x removed get_geodesic_distance; use ShortestPath instead
            path = self._hs.ShortestPath()
            path.requested_start = agent_pos
            path.requested_end = self._goal_pos
            found = self._sim.pathfinder.find_path(path)
            return float(path.geodesic_distance) if found else float("inf")
        except Exception:
            return float("inf")

    def _get_clearance(self) -> float:
        if self._sim is None:
            return float("inf")
        try:
            agent_pos = np.array(self._sim.get_agent(0).get_state().position, dtype=np.float32)
            return float(self._sim.pathfinder.distance_to_closest_obstacle(agent_pos, 2.0))
        except Exception:
            return float("inf")


# ---------------------------------------------------------------------------
# Main serve loop
# ---------------------------------------------------------------------------

def _send(obj: Any) -> None:
    pickle.dump(obj, _comm_write, protocol=4)
    _comm_write.flush()


def _recv() -> Any:
    return pickle.load(sys.stdin.buffer)


def serve(env: HabitatRolloutEnv) -> None:
    _log("Worker ready, awaiting commands.")
    while True:
        try:
            cmd = _recv()
        except EOFError:
            _log("stdin closed, exiting.")
            break
        except Exception as exc:
            _log(f"Error reading command: {exc}")
            break

        action = cmd.get("cmd", "")
        try:
            if action == "quit":
                _send({"ok": True})
                break

            elif action == "load_scene":
                env.load_scene(cmd["scene_path"])
                _send({"ok": True})

            elif action == "reset":
                obs = env.reset(
                    start_pos=cmd["start_pos"],
                    start_rot=cmd["start_rot"],
                    goal_pos=cmd["goal_pos"],
                )
                _send(obs)

            elif action == "step":
                obs = env.step(cmd["action"])
                _send(obs)

            else:
                _send({"ok": False, "error": f"Unknown command: {action!r}"})

        except Exception as exc:
            _log(f"Error handling command {action!r}: {exc}")
            _send({"ok": False, "error": str(exc)})


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--log", type=str, default="")
    args = parser.parse_args()

    if args.log:
        os.environ["HABITAT_WORKER_LOG"] = args.log
        global _log_fh
        _log_fh = open(args.log, "a", buffering=1)
        sys.stderr = _log_fh

    env = HabitatRolloutEnv(width=args.width, height=args.height, hfov=args.hfov)
    try:
        serve(env)
    finally:
        env.close()
        _log("Worker exiting.")
        _comm_write.flush()
        _comm_write.close()


if __name__ == "__main__":
    main()
