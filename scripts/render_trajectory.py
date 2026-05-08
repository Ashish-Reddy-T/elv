#!/usr/bin/env python3
"""Render an ego-centric video of one navigation trajectory.

Usage:
    python scripts/render_trajectory.py \
        --checkpoint /mnt/home/npant/ceph/elv/checkpoints/grpo-v2/grpo_epoch1_step400.pt \
        --episode-id 1 \
        --out artifacts/eval/traj_ep1.mp4

    # Random episode from a split:
    python scripts/render_trajectory.py \
        --checkpoint ... \
        --split val_seen \
        --out artifacts/eval/traj.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from eval_r2r import load_model
from train_grpo import (
    HABITAT_PYTHON,
    WORKER_SCRIPT,
    HabitatClient,
    episode_scene_path,
    load_r2r_episodes,
    obs_to_model_inputs,
    build_generation_prompt,
    parse_action,
    MAX_STEPS_PER_ROLLOUT,
)
from spatialvlm.utils.camera import CameraIntrinsics


def render_episode(
    model: object,
    tokenizer: object,
    client: HabitatClient,
    episode: dict,
    device: torch.device,
    max_steps: int,
    fps: int,
    out_path: Path,
) -> None:
    try:
        import cv2
    except ImportError:
        raise SystemExit("opencv-python not installed: pip install opencv-python-headless")

    intrinsics = CameraIntrinsics(fx=256.0, fy=256.0, cx=259.0, cy=259.0, width=518, height=518)

    start_pos = episode["start_position"]
    start_rot = episode["start_rotation"]
    ref_path = episode.get("reference_path", [])
    goals = episode.get("goals", [])
    goal_pos = ref_path[-1] if ref_path else (goals[0]["position"] if goals else start_pos)
    instruction = episode.get("instruction", "Navigate to the goal.")
    if isinstance(instruction, dict):
        instruction = instruction.get("instruction_text", "Navigate to the goal.")

    obs = client.reset(start_pos=start_pos, start_rot=start_rot, goal_pos=goal_pos)

    frames: list[np.ndarray] = []

    def annotate(frame: np.ndarray, step: int, action: str, geodesic: float) -> np.ndarray:
        f = frame.copy()
        cv2.putText(f, f"Step {step}  Action: {action}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(f, f"Geo: {geodesic:.1f}m", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(f, instruction[:80], (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return f

    model.eval()
    hf_model = model.backbone.model

    for step in range(max_steps):
        rgb = obs["rgb"]  # [H, W, 3] uint8
        geodesic = obs["geodesic"]

        siglip_px, dinov2_px, depth = obs_to_model_inputs(obs, device)
        input_ids, attn_mask, spatial_start_idx = build_generation_prompt(tokenizer, instruction)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        hf_model.gradient_checkpointing_disable()
        gen_out = model.generate(
            siglip_pixels=siglip_px,
            dinov2_pixels=dinov2_px,
            depth=depth,
            intrinsics=intrinsics,
            input_ids=input_ids,
            spatial_start_idx=spatial_start_idx,
            attention_mask=attn_mask,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        hf_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        prompt_len = input_ids.shape[1]
        response_text = tokenizer.decode(gen_out[0, prompt_len:], skip_special_tokens=True)
        action = parse_action(response_text)

        print(f"  step {step+1:2d} | geo={geodesic:.1f}m | action={action} | {response_text[:80]!r}")

        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frames.append(annotate(frame_bgr, step + 1, action, geodesic))

        if action == "STOP" or obs.get("done", False):
            break

        obs = client.step(action)

    # append final frame
    final_rgb = obs["rgb"]
    final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
    frames.append(annotate(final_bgr, len(frames) + 1, "STOP", obs["geodesic"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".gif":
        from PIL import Image
        rgb_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        rgb_frames[0].save(
            out_path,
            save_all=True,
            append_images=rgb_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
    else:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
    print(f"\nSaved {len(frames)} frames → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="artifacts/eval/trajectory.mp4")
    p.add_argument("--episode-id", type=int, default=None, help="Specific episode ID")
    p.add_argument("--episode-index", type=int, default=0, help="Index into episode list (default: 0)")
    p.add_argument("--split", default="val_seen", choices=["val_seen", "val_unseen", "train"])
    p.add_argument("--episode-json", default=None)
    p.add_argument("--scene-dir", default="/mnt/home/npant/ceph/datasets/habitat/habitat-sim/data/scene_datasets/mp3d")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_ROLLOUT)
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    episode_json = args.episode_json or f"data/episodes/r2r/{args.split}.json.gz"
    episodes = load_r2r_episodes(Path(episode_json))

    if args.episode_id is not None:
        matches = [e for e in episodes if e.get("episode_id") == args.episode_id or e.get("id") == args.episode_id]
        if not matches:
            raise SystemExit(f"Episode id {args.episode_id} not found")
        episode = matches[0]
    else:
        episode = episodes[args.episode_index]

    ep_id = episode.get("episode_id", episode.get("id", "?"))
    print(f"Episode {ep_id} — {episode.get('instruction', {}).get('instruction_text', '')[:80]}")

    model, tokenizer = load_model(args.checkpoint, device, dtype)

    scene_path = episode_scene_path(episode, args.scene_dir)
    client = HabitatClient(habitat_python=HABITAT_PYTHON, worker_script=WORKER_SCRIPT,
                           log_path="artifacts/slurm/habitat_render.log")
    try:
        client.load_scene(scene_path)
        render_episode(model, tokenizer, client, episode, device, args.max_steps, args.fps, Path(args.out))
    finally:
        client.close()


if __name__ == "__main__":
    main()
