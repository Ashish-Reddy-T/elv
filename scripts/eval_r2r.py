#!/usr/bin/env python3
"""R2R val_seen / val_unseen evaluation for SpatialVLM.

Loads a checkpoint, runs greedy rollouts over R2R episodes, and reports
Success Rate (SR) and Success weighted by Path Length (SPL).

Episodes are grouped by scene to minimise scene-load overhead (~60s each).
Results are saved incrementally after each scene so preemption is safe.

Usage:
    python scripts/eval_r2r.py \\
        --checkpoint /mnt/home/npant/ceph/elv/checkpoints/sft_epoch2_step668.pt \\
        --split val_seen \\
        --scene-dir /mnt/home/npant/ceph/datasets/habitat/habitat-sim/data/scene_datasets/mp3d \\
        --out artifacts/eval/r2r_val_seen.json

    # Limit to first N episodes (debug):
    python scripts/eval_r2r.py ... --limit 50
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # scripts/ for train_grpo imports

from train_grpo import (
    HABITAT_PYTHON,
    WORKER_SCRIPT,
    HabitatClient,
    episode_scene_path,
    load_r2r_episodes,
    run_rollout,
)

SUCCESS_THRESHOLD = 3.0  # metres — standard R2R success criterion
FORWARD_STEP_M = 0.25   # metres per MOVE_FORWARD action


def load_model(ckpt_path: str, device: torch.device, dtype: torch.dtype) -> tuple[Any, Any]:
    from spatialvlm.config.model import SpatialVLMConfig
    from spatialvlm.model import SpatialVLM
    from transformers import AutoTokenizer

    model_cfg = SpatialVLMConfig()
    model_cfg.backbone.model_id = "Qwen/Qwen3-VL-8B-Instruct"
    model_cfg.backbone.lora_rank = 32
    model_cfg.backbone.lora_alpha = 64

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    model = SpatialVLM(config=model_cfg, device=device, torch_dtype=dtype,
                       lazy_load_backbone=False)
    model.to(dtype)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys")
    print(f"  Checkpoint loaded (epoch={ckpt.get('epoch', '?')}, step={ckpt.get('step', '?')})")

    model.eval()
    return model, tokenizer


def group_by_scene(episodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        groups[ep.get("scene_id", "unknown")].append(ep)
    return dict(groups)


def path_length_m(actions: list[str]) -> float:
    return sum(1 for a in actions if a == "MOVE_FORWARD") * FORWARD_STEP_M


def eval_episode(
    model: Any,
    tokenizer: Any,
    client: HabitatClient,
    episode: dict[str, Any],
    device: torch.device,
    max_steps: int,
) -> dict[str, Any]:
    t0 = time.time()
    traj = run_rollout(
        model=model,
        tokenizer=tokenizer,
        client=client,
        episode=episode,
        device=device,
        max_steps=max_steps,
        max_new_tokens=64,
        temperature=0.0,  # greedy for eval
    )
    elapsed = time.time() - t0

    final_geodesic = traj["geodesics"][-1] if traj["geodesics"] else float("inf")
    pl = path_length_m(traj["actions"])
    sp = traj["initial_geodesic"]
    success = traj["stopped"] and final_geodesic < SUCCESS_THRESHOLD

    spl_val = 0.0
    if success:
        spl_val = sp / max(pl, sp, 1e-8)

    return {
        "episode_id": episode.get("episode_id", episode.get("id", "?")),
        "scene_id": episode.get("scene_id", ""),
        "success": success,
        "final_geodesic": final_geodesic,
        "initial_geodesic": sp,
        "path_length_m": pl,
        "spl": spl_val,
        "actions": traj["actions"],
        "stopped": traj["stopped"],
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="R2R evaluation for SpatialVLM")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--split", default="val_seen",
                   choices=["val_seen", "val_unseen", "train"])
    p.add_argument("--episode-json", type=str, default=None,
                   help="Override episode JSON path (default: data/episodes/r2r/<split>.json.gz)")
    p.add_argument("--scene-dir", type=str,
                   default="/mnt/home/npant/ceph/datasets/habitat/habitat-sim/data/scene_datasets/mp3d")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path (default: artifacts/eval/r2r_<split>.json)")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N episodes (debug)")
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    episode_json = args.episode_json or f"data/episodes/r2r/{args.split}.json.gz"
    out_path = args.out or f"artifacts/eval/r2r_{args.split}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.checkpoint, device, dtype)

    print(f"Loading episodes from: {episode_json}")
    episodes = load_r2r_episodes(Path(episode_json))
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"  {len(episodes)} episodes")

    scenes = group_by_scene(episodes)
    print(f"  {len(scenes)} unique scenes")

    # Load any partial results already saved (for preemption safety)
    all_results: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    if Path(out_path).exists():
        with open(out_path) as f:
            saved = json.load(f)
        all_results = saved.get("episodes", [])
        done_ids = {str(r["episode_id"]) for r in all_results}
        print(f"  Resuming: {len(done_ids)} episodes already done")

    total_eps = len(episodes)
    evaluated = len(done_ids)

    log_path = str(Path(out_path).parent / f"habitat_worker_eval_{args.split}.log")
    client = HabitatClient(log_path=log_path)

    try:
        for scene_id, scene_eps in scenes.items():
            pending = [ep for ep in scene_eps
                       if str(ep.get("episode_id", ep.get("id", "?"))) not in done_ids]
            if not pending:
                continue

            scene_path = episode_scene_path(pending[0], args.scene_dir)
            print(f"\nScene {scene_id} ({len(pending)} episodes) — loading...")
            t_scene = time.time()
            client.load_scene(scene_path)
            print(f"  Scene loaded in {time.time() - t_scene:.0f}s")

            for ep in pending:
                result = eval_episode(model, tokenizer, client, ep, device, args.max_steps)
                all_results.append(result)
                done_ids.add(str(result["episode_id"]))
                evaluated += 1

                status = "✓" if result["success"] else "✗"
                print(
                    f"  [{evaluated}/{total_eps}] {status} "
                    f"geo={result['final_geodesic']:.1f}m  "
                    f"pl={result['path_length_m']:.2f}m  "
                    f"actions={result['actions']}  "
                    f"({result['elapsed_s']}s)"
                )

                # Incremental save after every episode
                successes = [r["success"] for r in all_results]
                sr = sum(successes) / len(successes)
                spl_avg = sum(r["spl"] for r in all_results) / len(all_results)
                with open(out_path, "w") as f:
                    json.dump({
                        "checkpoint": args.checkpoint,
                        "split": args.split,
                        "episodes_evaluated": len(all_results),
                        "sr": round(sr, 4),
                        "spl": round(spl_avg, 4),
                        "episodes": all_results,
                    }, f, indent=2)

    finally:
        client.close()

    successes = [r["success"] for r in all_results]
    sr = sum(successes) / len(successes) if successes else 0.0
    spl_avg = sum(r["spl"] for r in all_results) / len(all_results) if all_results else 0.0

    print(f"\n{'='*50}")
    print(f"R2R {args.split} results ({len(all_results)} episodes)")
    print(f"  SR  = {sr:.3f}  ({sum(successes)}/{len(successes)})")
    print(f"  SPL = {spl_avg:.3f}")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
