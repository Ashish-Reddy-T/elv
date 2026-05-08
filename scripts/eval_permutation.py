#!/usr/bin/env python3
"""Spatial grounding permutation test for SpatialVLM.

Runs two eval passes on the same episodes:
  1. Baseline: normal spatial tokens → SR_baseline
  2. Permuted: shuffled spatial token positions → SR_permuted

Permutation Sensitivity Index (PSI):
  PSI = max(0, (SR_baseline - SR_permuted) / max(SR_baseline, 1e-8))

A model that genuinely relies on spatial structure should show PSI >= 0.15.
A model that ignores spatial context (or collapses to STOP) will show PSI ~ 0.

Usage:
    python scripts/eval_permutation.py \\
        --checkpoint /mnt/home/npant/ceph/elv/checkpoints/grpo-v2/grpo_epoch1_step400.pt \\
        --limit 100 \\
        --num-permutations 3 \\
        --out artifacts/eval/psi_grpo_v2_step400.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from eval_r2r import load_model, group_by_scene, path_length_m
from train_grpo import (
    HABITAT_PYTHON,
    WORKER_SCRIPT,
    HabitatClient,
    episode_scene_path,
    load_r2r_episodes,
    run_rollout,
)

SUCCESS_THRESHOLD = 3.0
FORWARD_STEP_M = 0.25


def eval_episode(
    model: Any,
    tokenizer: Any,
    client: HabitatClient,
    episode: dict[str, Any],
    device: torch.device,
    max_steps: int,
    permute_spatial: bool,
) -> dict[str, Any]:
    t0 = time.time()

    if permute_spatial:
        # Monkey-patch model.generate for this rollout only
        orig_generate = model.generate

        def _permuted_generate(*args: Any, **kwargs: Any) -> Any:
            kwargs["permute_spatial"] = True
            return orig_generate(*args, **kwargs)

        model.generate = _permuted_generate
        try:
            traj = run_rollout(model, tokenizer, client, episode, device, max_steps)
        finally:
            model.generate = orig_generate
    else:
        traj = run_rollout(model, tokenizer, client, episode, device, max_steps)

    elapsed = time.time() - t0
    final_geo = traj["geodesics"][-1] if traj["geodesics"] else float("inf")
    pl = path_length_m(traj["actions"])
    sp = traj["initial_geodesic"]
    success = traj["stopped"] and final_geo < SUCCESS_THRESHOLD
    spl_val = sp / max(pl, sp, 1e-8) if success else 0.0

    return {
        "episode_id": episode.get("episode_id", episode.get("id", "?")),
        "success": success,
        "final_geodesic": final_geo,
        "spl": spl_val,
        "actions": traj["actions"],
        "elapsed_s": round(elapsed, 1),
    }


def run_split(
    model: Any,
    tokenizer: Any,
    episodes: list[dict[str, Any]],
    scene_dir: str,
    device: torch.device,
    max_steps: int,
    permute_spatial: bool,
    log_path: str,
) -> list[dict[str, Any]]:
    label = "PERMUTED" if permute_spatial else "BASELINE"
    scenes = group_by_scene(episodes)
    results: list[dict[str, Any]] = []

    client = HabitatClient(log_path=log_path)
    try:
        for scene_id, scene_eps in scenes.items():
            scene_path = episode_scene_path(scene_eps[0], scene_dir)
            print(f"\n[{label}] Scene {scene_id} ({len(scene_eps)} eps) — loading...")
            t = time.time()
            client.load_scene(scene_path)
            print(f"  loaded in {time.time() - t:.0f}s")

            for ep in scene_eps:
                r = eval_episode(model, tokenizer, client, ep, device, max_steps, permute_spatial)
                results.append(r)
                n = len(results)
                sr_so_far = sum(x["success"] for x in results) / n
                status = "✓" if r["success"] else "✗"
                print(
                    f"  [{n}/{len(episodes)}] {status} "
                    f"geo={r['final_geodesic']:.1f}m  "
                    f"actions={r['actions']}  "
                    f"SR={sr_so_far:.3f}  ({r['elapsed_s']}s)"
                )
    finally:
        client.close()

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Spatial grounding permutation test")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val_seen", choices=["val_seen", "val_unseen", "train"])
    p.add_argument("--episode-json", default=None)
    p.add_argument("--scene-dir", default="/mnt/home/npant/ceph/datasets/habitat/habitat-sim/data/scene_datasets/mp3d")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--num-permutations", type=int, default=3,
                   help="Number of independent permuted runs to average over")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    ckpt_tag = Path(args.checkpoint).stem
    out_path = args.out or f"artifacts/eval/psi_{ckpt_tag}_{args.split}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    episode_json = args.episode_json or f"data/episodes/r2r/{args.split}.json.gz"
    model, tokenizer = load_model(args.checkpoint, device, dtype)

    print(f"Loading episodes from: {episode_json}")
    episodes = load_r2r_episodes(Path(episode_json))
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"  {len(episodes)} episodes, {len(group_by_scene(episodes))} scenes")

    baseline_log = str(Path(out_path).parent / f"psi_baseline_{args.split}.log")
    permuted_log = str(Path(out_path).parent / f"psi_permuted_{args.split}.log")

    # --- Baseline pass ---
    print(f"\n{'='*60}")
    print(f"BASELINE PASS")
    baseline_results = run_split(
        model, tokenizer, episodes, args.scene_dir,
        device, args.max_steps, False, baseline_log,
    )
    sr_baseline = sum(r["success"] for r in baseline_results) / max(len(baseline_results), 1)
    spl_baseline = sum(r["spl"] for r in baseline_results) / max(len(baseline_results), 1)
    print(f"\nBaseline: SR={sr_baseline:.4f}  SPL={spl_baseline:.4f}")

    # --- Permuted passes ---
    permuted_srs: list[float] = []
    permuted_spls: list[float] = []
    all_permuted_results: list[list[dict[str, Any]]] = []

    for perm_i in range(args.num_permutations):
        print(f"\n{'='*60}")
        print(f"PERMUTED PASS {perm_i + 1}/{args.num_permutations}")
        perm_log = str(Path(out_path).parent / f"psi_permuted{perm_i}_{args.split}.log")
        perm_results = run_split(
            model, tokenizer, episodes, args.scene_dir,
            device, args.max_steps, True, perm_log,
        )
        sr_p = sum(r["success"] for r in perm_results) / max(len(perm_results), 1)
        spl_p = sum(r["spl"] for r in perm_results) / max(len(perm_results), 1)
        permuted_srs.append(sr_p)
        permuted_spls.append(spl_p)
        all_permuted_results.append(perm_results)
        print(f"  Permuted run {perm_i+1}: SR={sr_p:.4f}  SPL={spl_p:.4f}")

    sr_permuted_mean = sum(permuted_srs) / len(permuted_srs)
    spl_permuted_mean = sum(permuted_spls) / len(permuted_spls)

    eps = 1e-8
    psi_sr = max(0.0, (sr_baseline - sr_permuted_mean) / max(sr_baseline, eps))
    psi_spl = max(0.0, (spl_baseline - spl_permuted_mean) / max(spl_baseline, eps))

    spatially_grounded = psi_sr >= 0.15

    print(f"\n{'='*60}")
    print(f"PERMUTATION TEST RESULTS ({len(episodes)} episodes)")
    print(f"  SR  baseline = {sr_baseline:.4f}")
    print(f"  SR  permuted = {sr_permuted_mean:.4f}  (mean of {args.num_permutations} runs: {[f'{s:.3f}' for s in permuted_srs]})")
    print(f"  PSI (SR)     = {psi_sr:.4f}  {'✓ SPATIALLY GROUNDED' if spatially_grounded else '✗ not grounded (threshold=0.15)'}")
    print(f"  SPL baseline = {spl_baseline:.4f}")
    print(f"  SPL permuted = {spl_permuted_mean:.4f}")
    print(f"  PSI (SPL)    = {psi_spl:.4f}")

    output = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "episodes_evaluated": len(episodes),
        "num_permutations": args.num_permutations,
        "sr_baseline": round(sr_baseline, 4),
        "spl_baseline": round(spl_baseline, 4),
        "sr_permuted_mean": round(sr_permuted_mean, 4),
        "spl_permuted_mean": round(spl_permuted_mean, 4),
        "permuted_sr_runs": [round(s, 4) for s in permuted_srs],
        "permuted_spl_runs": [round(s, 4) for s in permuted_spls],
        "psi_sr": round(psi_sr, 4),
        "psi_spl": round(psi_spl, 4),
        "spatially_grounded": spatially_grounded,
        "baseline_episodes": baseline_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
