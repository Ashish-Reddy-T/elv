#!/usr/bin/env python3
"""Render and cache RGB + depth frames from Habitat for SpatialVLM training.

Groups episodes by scene to minimise env reloads. Outputs one .pt file per
rendered step matching the CachedFrameDataset contract:

    rgb:         Tensor[3, 518, 518]  float32, [0, 1]
    depth:       Tensor[518, 518]     float32, metric metres (raw — NOT normalised)
    intrinsics:  dict  {fx, fy, cx, cy, width, height}
    instruction: str
    episode_id:  str
    source:      str   (r2r-ce | rxr-ce | sqa3d)
    step_idx:    int
    caption:     str   (optional — MP3D semantic caption, present when semantic mesh loads)
    target:      str   (SFT only — "Reasoning: ... Action: MOVE_FORWARD/STOP")

Usage
-----
# Pre-alignment frames (~50 K, 3 frames/episode from R2R-CE train split):
python scripts/render_training_frames.py \\
    --dataset r2r-ce \\
    --episode-json data/episodes/R2R_VLNCE_v1-3_train.json.gz \\
    --scene-dir data/scene_datasets/mp3d \\
    --out-dir data/frames/prealign \\
    --stage prealign \\
    --frames-per-episode 3

# SFT frames (all steps, R2R-CE + RxR-CE + SQA3D):
python scripts/render_training_frames.py \\
    --dataset r2r-ce \\
    --episode-json data/episodes/R2R_VLNCE_v1-3_train.json.gz \\
    --scene-dir data/scene_datasets/mp3d \\
    --out-dir data/frames/sft \\
    --stage sft

Requirements
------------
- habitat-sim 3.0  (conda install -c aihabitat habitat-sim withbullet)
- habitat-lab 3.0  (pip install habitat-lab)
- Matterport3D or HM3D scene files in --scene-dir
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Habitat lazy import
# ---------------------------------------------------------------------------

def _require_habitat_sim():
    try:
        import habitat_sim
        return habitat_sim
    except ImportError:
        sys.exit(
            "habitat-sim is not installed.\n"
            "Install with: conda install -c aihabitat habitat-sim withbullet"
        )


# ---------------------------------------------------------------------------
# Semantic caption helpers
# ---------------------------------------------------------------------------

_IGNORE_CATEGORIES: frozenset[str] = frozenset({
    "floor", "ceiling", "wall", "misc", "void", "unknown", "",
    "object", "room", "column", "beam", "railing", "unlabeled", "objects",
})


def _format_object_list(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _region_contains(aabb: Any, point: np.ndarray) -> bool:
    """Check if a point is inside a habitat_sim AABB."""
    try:
        mn = np.array(aabb.min)
        mx = np.array(aabb.max)
        return bool(np.all(point >= mn) and np.all(point <= mx))
    except AttributeError:
        pass
    try:
        center = np.array(aabb.center)
        half = np.array(aabb.sizes) / 2.0
        return bool(np.all(np.abs(point - center) <= half))
    except AttributeError:
        return False


def build_scene_caption(sim: Any, semantic_obs: np.ndarray, agent_pos: np.ndarray) -> str:
    """Generate a brief scene caption from semantic observation and MP3D annotations.

    Returns empty string if semantic data is unavailable.
    """
    semantic_scene = getattr(sim, "semantic_scene", None)
    if semantic_scene is None:
        return ""

    # Count visible pixels per instance ID
    flat = semantic_obs.ravel().astype(np.int32)
    instance_ids, counts = np.unique(flat, return_counts=True)

    objects = getattr(semantic_scene, "objects", [])
    cat_pixels: dict[str, int] = {}
    for iid, count in zip(instance_ids, counts):
        if iid <= 0:
            continue
        try:
            obj = objects[int(iid)]
            if obj is None:
                continue
            name = obj.category.name().lower().strip()
        except (IndexError, AttributeError, TypeError):
            continue
        if name in _IGNORE_CATEGORIES:
            continue
        cat_pixels[name] = cat_pixels.get(name, 0) + int(count)

    # Top 5 most-visible object categories
    top = sorted(cat_pixels.items(), key=lambda x: -x[1])[:5]
    obj_names = [n for n, _ in top]

    # Room type from agent's current region
    room = ""
    for region in getattr(semantic_scene, "regions", []):
        try:
            if _region_contains(region.aabb, agent_pos):
                rname = region.category.name().lower().strip()
                if rname and rname not in _IGNORE_CATEGORIES:
                    room = rname
                    break
        except AttributeError:
            continue

    if not obj_names and not room:
        return ""
    if room and obj_names:
        return f"A {room} with {_format_object_list(obj_names)}."
    if room:
        return f"A {room}."
    return f"A scene with {_format_object_list(obj_names)}."


# ---------------------------------------------------------------------------
# Camera intrinsics from Habitat sensor spec
# ---------------------------------------------------------------------------

def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> dict[str, float | int]:
    """Compute pinhole intrinsics from horizontal field-of-view."""
    hfov_rad = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx  # square pixels assumed (Habitat default)
    return dict(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0, width=width, height=height)


# ---------------------------------------------------------------------------
# Episode JSON loading (handles .json and .json.gz)
# ---------------------------------------------------------------------------

def load_episodes(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        opener = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = open(path, encoding="utf-8")

    with opener as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("episodes", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    raise ValueError(f"Unexpected JSON structure in {path}")


# ---------------------------------------------------------------------------
# SQA3D episode loading (different schema)
# ---------------------------------------------------------------------------

def load_sqa3d_episodes(path: Path) -> list[dict[str, Any]]:
    """Load SQA3D QA pairs and normalise to a common episode schema."""
    if path.suffix == ".gz":
        opener = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = open(path, encoding="utf-8")

    with opener as f:
        data = json.load(f)

    # SQA3D schema: list of QA dicts each with scene_id, position, rotation, question, answer
    episodes = data if isinstance(data, list) else data.get("questions", [])
    out = []
    for ep in episodes:
        out.append(
            {
                "episode_id": ep.get("question_id", ep.get("id", "?")),
                "scene_id": ep["scene_id"],
                "start_position": ep.get("position", [0.0, 0.0, 0.0]),
                "start_rotation": ep.get("rotation", [0.0, 0.0, 0.0, 1.0]),
                "instruction": ep.get("question", ""),
                "answer": ep.get("answer", ""),
                "reference_path": None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def look_at_yaw(from_pos: np.ndarray, to_pos: np.ndarray) -> list[float]:
    """Quaternion [x, y, z, w] that yaws the agent from from_pos toward to_pos.

    Habitat uses Y-up convention. Yaw is rotation around the Y axis.
    """
    dx = to_pos[0] - from_pos[0]
    dz = to_pos[2] - from_pos[2]
    yaw = math.atan2(-dx, -dz)  # atan2 of (-dx, -dz) → Habitat forward=-Z, right=+X
    half = yaw / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]  # [x, y, z, w]


def quat_from_list(q: list[float]):
    """Build habitat_sim.utils.quat_from_coeffs quaternion [x, y, z, w]."""
    hs = _require_habitat_sim()
    mn = hs.utils.common.quat_from_coeffs
    return mn(np.array(q, dtype=np.float32))


# ---------------------------------------------------------------------------
# Habitat sim setup
# ---------------------------------------------------------------------------

def make_sim(
    scene_path: str,
    width: int = 518,
    height: int = 518,
    hfov: float = 90.0,
    enable_semantic: bool = True,
):
    """Create a minimal habitat_sim.Simulator for RGB + depth (+ semantic) rendering."""
    hs = _require_habitat_sim()

    backend_cfg = hs.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = False
    if enable_semantic:
        backend_cfg.load_semantic_mesh = True

    sensor_specs = []

    rgb_spec = hs.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = hs.SensorType.COLOR
    rgb_spec.resolution = [height, width]
    rgb_spec.hfov = hfov
    rgb_spec.position = [0.0, 1.25, 0.0]  # standard Habitat agent eye height
    sensor_specs.append(rgb_spec)

    depth_spec = hs.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = hs.SensorType.DEPTH
    depth_spec.resolution = [height, width]
    depth_spec.hfov = hfov
    depth_spec.position = [0.0, 1.25, 0.0]
    sensor_specs.append(depth_spec)

    if enable_semantic:
        sem_spec = hs.CameraSensorSpec()
        sem_spec.uuid = "semantic"
        sem_spec.sensor_type = hs.SensorType.SEMANTIC
        sem_spec.resolution = [height, width]
        sem_spec.hfov = hfov
        sem_spec.position = [0.0, 1.25, 0.0]
        sensor_specs.append(sem_spec)

    agent_cfg = hs.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    cfg = hs.Configuration(backend_cfg, [agent_cfg])
    return hs.Simulator(cfg)


# ---------------------------------------------------------------------------
# Observation extraction
# ---------------------------------------------------------------------------

def obs_to_tensors(obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract RGB [3, H, W] float32 [0,1] and depth [H, W] float32 metres."""
    rgb_np = obs["rgb"][:, :, :3].astype(np.float32) / 255.0
    rgb = torch.from_numpy(rgb_np).permute(2, 0, 1)  # [3, H, W]

    depth_np = obs["depth"]
    if depth_np.ndim == 3:
        depth_np = depth_np[:, :, 0]
    depth = torch.from_numpy(depth_np.astype(np.float32))  # [H, W]

    return rgb, depth


# ---------------------------------------------------------------------------
# Turn-sequence helpers (must match habitat_worker.py TURN_DEG = 15.0)
# ---------------------------------------------------------------------------

TURN_DEG: float = 15.0


def _normalize_angle(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw_from_points(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    """Yaw (radians) to look from from_pos toward to_pos, Habitat Y-up convention."""
    dx = to_pos[0] - from_pos[0]
    dz = to_pos[2] - from_pos[2]
    return math.atan2(-dx, -dz)


def _yaw_to_quat(yaw: float) -> list[float]:
    """Yaw (radians) → quaternion [x, y, z, w] for Habitat Y-up rotation."""
    half = yaw / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def _build_action_sequence(
    path: list[np.ndarray],
    turn_deg: float = TURN_DEG,
) -> list[tuple[np.ndarray, float, str]]:
    """Convert a reference path into a flat (position, yaw, action) sequence.

    For each consecutive pair of waypoints the agent may need to rotate before
    moving forward.  Each turn step is rendered at the current position with the
    current heading so the model sees what it would actually see mid-turn.

    Returns a list of (position, yaw_radians, action_str) tuples.
    """
    if len(path) == 0:
        return []
    if len(path) == 1:
        return [(path[0], 0.0, "STOP")]

    turn_rad = math.radians(turn_deg)
    steps: list[tuple[np.ndarray, float, str]] = []

    # Heading the agent has after following the previous segment.
    current_yaw = _yaw_from_points(path[0], path[1])

    for i, pos in enumerate(path):
        is_last = i == len(path) - 1

        if is_last:
            steps.append((pos, current_yaw, "STOP"))
            break

        departure_yaw = _yaw_from_points(pos, path[i + 1])

        # Turns needed at this waypoint: only meaningful after the first
        # waypoint where the agent arrives from a previous segment.
        if i > 0:
            diff = _normalize_angle(departure_yaw - current_yaw)
            n_turns = round(abs(diff) / turn_rad)
            if n_turns > 0:
                turn_sign = 1.0 if diff > 0 else -1.0
                action_str = "TURN_LEFT" if diff > 0 else "TURN_RIGHT"
                for k in range(n_turns):
                    # Render at the heading the agent has *before* this turn step.
                    render_yaw = current_yaw + k * turn_sign * turn_rad
                    steps.append((pos, render_yaw, action_str))
                current_yaw = departure_yaw

        steps.append((pos, departure_yaw, "MOVE_FORWARD"))
        current_yaw = departure_yaw

    return steps


# ---------------------------------------------------------------------------
# SFT target string
# ---------------------------------------------------------------------------

def make_sft_target(
    action: str,
    step_idx: int,
    total_steps: int,
    instruction: str,
    caption: str = "",
    is_sqa3d: bool = False,
    answer: str = "",
) -> str:
    scene_prefix = f"{caption} " if caption else ""

    if is_sqa3d:
        scene_ctx = caption.rstrip(".") if caption else "Based on the current viewpoint"
        return f"Reasoning: {scene_ctx}, {answer.lower() or 'the answer is unclear'}. Answer: {answer}"

    step_str = f"step {step_idx + 1} of {total_steps}"
    if action == "STOP":
        return (
            f"Reasoning: {scene_prefix}"
            f"I am at {step_str} and have reached the destination. "
            f"The instruction was: '{instruction}'. "
            f"Action: STOP"
        )
    return (
        f"Reasoning: {scene_prefix}"
        f"I am at {step_str}, following the instruction: "
        f"'{instruction}'. "
        f"Action: {action}"
    )


# ---------------------------------------------------------------------------
# Main rendering logic
# ---------------------------------------------------------------------------

def render_episode(
    sim,
    episode: dict[str, Any],
    out_dir: Path,
    source: str,
    stage: str,
    frames_per_episode: int | None,
    width: int,
    height: int,
    hfov: float,
) -> int:
    """Render frames for one episode. Returns number of frames written."""
    hs = _require_habitat_sim()
    intrinsics = intrinsics_from_hfov(width, height, hfov)

    episode_id = str(episode.get("episode_id", episode.get("id", "?")))
    instruction = episode.get("instruction", episode.get("instruction_text", ""))
    if isinstance(instruction, dict):
        instruction = instruction.get("instruction_text", "")
    answer = str(episode.get("answer", ""))
    is_sqa3d = source == "sqa3d"

    reference_path = episode.get("reference_path")

    safe_ep_id = episode_id.replace("/", "-").replace("\\", "-")

    # Build step sequence.
    # SFT: expand reference path into per-action steps including turns.
    # Prealign / SQA3D: teleport to subsampled waypoints (no turns needed).
    if stage == "sft" and not is_sqa3d and reference_path is not None:
        raw_path = [np.array(p, dtype=np.float64) for p in reference_path]
        action_seq = _build_action_sequence(raw_path)
    else:
        # Prealign / SQA3D: simple waypoint list, no explicit turn steps.
        if is_sqa3d or reference_path is None:
            pos0 = np.array(episode["start_position"], dtype=np.float64)
            rot0 = episode.get("start_rotation", [0.0, 0.0, 0.0, 1.0])
            half = math.atan2(rot0[1], rot0[3])  # extract yaw from quat
            action_seq = [(pos0, half * 2, "STOP")]
        else:
            path = [np.array(p, dtype=np.float64) for p in reference_path]
            seq = []
            for i, pos in enumerate(path):
                if i + 1 < len(path):
                    yaw = _yaw_from_points(pos, path[i + 1])
                else:
                    yaw = _yaw_from_points(path[i - 1], pos) if i > 0 else 0.0
                action = "STOP" if i == len(path) - 1 else "MOVE_FORWARD"
                seq.append((pos, yaw, action))
            action_seq = seq

        # Subsample for pre-alignment
        if frames_per_episode is not None and len(action_seq) > frames_per_episode:
            step = max(1, len(action_seq) // frames_per_episode)
            action_seq = action_seq[::step][:frames_per_episode]

    total = len(action_seq)
    written = 0

    for step_idx, (pos, yaw, action) in enumerate(action_seq):
        # Teleport agent to this position and heading
        agent_state = hs.agent.AgentState()
        agent_state.position = pos
        agent_state.rotation = quat_from_list(_yaw_to_quat(yaw))
        sim.get_agent(0).set_state(agent_state)

        obs = sim.get_sensor_observations()
        rgb, depth = obs_to_tensors(obs)

        caption = ""
        sem_obs = obs.get("semantic")
        if sem_obs is not None:
            if sem_obs.ndim == 3:
                sem_obs = sem_obs[:, :, 0]
            try:
                caption = build_scene_caption(sim, sem_obs, pos)
            except Exception:
                caption = ""

        frame: dict[str, Any] = {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics,
            "instruction": instruction,
            "episode_id": episode_id,
            "source": source,
            "step_idx": step_idx,
        }
        if caption:
            frame["caption"] = caption

        if stage == "sft":
            frame["target"] = make_sft_target(
                action, step_idx, total, instruction, caption, is_sqa3d, answer
            )

        fname = f"ep_{safe_ep_id}_step_{step_idx:04d}.pt"
        torch.save(frame, out_dir / fname)
        written += 1

    return written


def resolve_scene_path(scene_id: str, scene_dir: Path) -> str:
    """Resolve a Habitat scene_id to an absolute .glb path."""
    # scene_id may be e.g. "mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb" or bare "1LXtFkjw3qL"
    candidate = scene_dir / scene_id
    if candidate.exists():
        return str(candidate)

    # Try stripping leading prefix components
    for part in Path(scene_id).parts:
        candidate = scene_dir / part
        if candidate.is_dir():
            # Look for .glb inside
            glbs = list(candidate.glob("**/*.glb"))
            if glbs:
                return str(glbs[0])
        if candidate.with_suffix(".glb").exists():
            return str(candidate.with_suffix(".glb"))

    # Fallback: search recursively by scene name stem
    stem = Path(scene_id).stem
    matches = list(scene_dir.glob(f"**/{stem}.glb"))
    if matches:
        return str(matches[0])

    raise FileNotFoundError(
        f"Scene not found for scene_id='{scene_id}' under {scene_dir}. "
        "Check --scene-dir points to your scene_datasets directory."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["r2r-ce", "rxr-ce", "sqa3d"],
                   help="Which dataset to render")
    p.add_argument("--episode-json", required=True, type=Path,
                   help="Path to episode JSON (or .json.gz)")
    p.add_argument("--scene-dir", required=True, type=Path,
                   help="Root directory of Habitat scene files (.glb)")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory for .pt frame files")
    p.add_argument("--stage", default="prealign", choices=["prealign", "sft"],
                   help="Training stage — sft adds 'target' field to each frame")
    p.add_argument("--frames-per-episode", type=int, default=None,
                   help="Max frames per episode (default: all waypoints). "
                        "Recommended: 3-5 for pre-alignment, None for SFT")
    p.add_argument("--width", type=int, default=518)
    p.add_argument("--height", type=int, default=518)
    p.add_argument("--hfov", type=float, default=90.0,
                   help="Horizontal field of view in degrees (Habitat default: 90)")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Stop after this many episodes (for dry-runs)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip episodes whose first frame .pt already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading episodes from {args.episode_json} ...")
    if args.dataset == "sqa3d":
        episodes = load_sqa3d_episodes(args.episode_json)
    else:
        episodes = load_episodes(args.episode_json)

    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    print(f"  {len(episodes)} episodes loaded")

    # Group by scene to minimise simulator reloads
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_scene[ep["scene_id"]].append(ep)

    print(f"  {len(by_scene)} unique scenes")

    total_frames = 0
    total_episodes = 0

    for scene_idx, (scene_id, scene_episodes) in enumerate(by_scene.items()):
        try:
            scene_path = resolve_scene_path(scene_id, args.scene_dir)
        except FileNotFoundError as e:
            print(f"  [WARN] {e} — skipping {len(scene_episodes)} episodes")
            continue

        print(f"[{scene_idx + 1}/{len(by_scene)}] Scene: {scene_id}  ({len(scene_episodes)} episodes)")

        sim = None
        try:
            sim = make_sim(scene_path, args.width, args.height, args.hfov)

            for ep in scene_episodes:
                episode_id = str(ep.get("episode_id", ep.get("id", "?")))

                if args.skip_existing:
                    safe_ep_id = episode_id.replace("/", "-").replace("\\", "-")
                    first_frame = args.out_dir / f"ep_{safe_ep_id}_step_0000.pt"
                    if first_frame.exists():
                        continue

                n = render_episode(
                    sim=sim,
                    episode=ep,
                    out_dir=args.out_dir,
                    source=args.dataset,
                    stage=args.stage,
                    frames_per_episode=args.frames_per_episode,
                    width=args.width,
                    height=args.height,
                    hfov=args.hfov,
                )
                total_frames += n
                total_episodes += 1

        except Exception as e:
            print(f"  [ERROR] Scene {scene_id}: {e}")
        finally:
            if sim is not None:
                sim.close()

    print(f"\nDone. {total_frames} frames from {total_episodes} episodes → {args.out_dir}")


if __name__ == "__main__":
    main()
