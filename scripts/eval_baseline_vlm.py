#!/usr/bin/env python3
"""Vanilla Qwen3-VL-8B baseline: RGB + instruction only, no spatial processing.

Loads Qwen3-VL-8B-Instruct directly (no LoRA, no SpatialVLM modules) and runs
the same R2R rollout loop as eval_r2r.py so results are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from train_grpo import (
    HABITAT_PYTHON,
    WORKER_SCRIPT,
    HabitatClient,
    episode_scene_path,
    load_r2r_episodes,
    parse_action,
)
from spatialvlm.data.tokenization import SYSTEM_PROMPT

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
SUCCESS_THRESHOLD = 3.0
FORWARD_STEP_M = 0.25


def load_model(device: torch.device, dtype: torch.dtype) -> tuple[Any, Any]:
    from transformers import AutoProcessor, AutoModelForVision2Seq

    print(f"Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=str(device),
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    return model, processor


def obs_to_pil(obs: dict[str, Any]) -> Image.Image:
    rgb = obs["rgb"]  # [H, W, 3] uint8
    return Image.fromarray(rgb.astype(np.uint8))


def build_prompt(processor: Any, instruction: str, image: Image.Image) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f"Instruction: {instruction}\n\n"
                        "Based on what you see, choose one action: "
                        "MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, or STOP.\n"
                        "Respond with: Action: <ACTION>"
                    ),
                },
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return inputs


@torch.no_grad()
def run_rollout(
    model: Any,
    processor: Any,
    client: HabitatClient,
    episode: dict[str, Any],
    device: torch.device,
    max_steps: int,
) -> dict[str, Any]:
    instruction = episode.get("instruction", "Navigate to the goal.")
    if isinstance(instruction, dict):
        instruction = instruction.get("instruction_text", "Navigate to the goal.")

    start_pos = episode["start_position"]
    start_rot = episode["start_rotation"]
    ref_path = episode.get("reference_path", [])
    goals = episode.get("goals", [])
    goal_pos = ref_path[-1] if ref_path else (goals[0]["position"] if goals else start_pos)

    obs = client.reset(start_pos=start_pos, start_rot=start_rot, goal_pos=goal_pos)
    initial_geodesic = obs["geodesic"]

    actions: list[str] = []
    geodesics: list[float] = []
    stopped = False

    for step in range(max_steps):
        image = obs_to_pil(obs)
        inputs = build_prompt(processor, instruction, image)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        response_ids = gen_out[0, prompt_len:]
        response_text = processor.tokenizer.decode(response_ids, skip_special_tokens=True)

        if step == 0:
            print(f"[baseline] response[:200]: {response_text[:200]!r}", flush=True)

        action = parse_action(response_text)
        obs = client.step(action)
        geodesics.append(obs["geodesic"])
        actions.append(action)

        if obs["done"] or action == "STOP":
            stopped = True
            break

    final_geodesic = geodesics[-1] if geodesics else float("inf")
    pl = sum(1 for a in actions if a == "MOVE_FORWARD") * FORWARD_STEP_M
    success = stopped and final_geodesic < SUCCESS_THRESHOLD

    spl = 0.0
    if success:
        sp = initial_geodesic
        spl = sp / max(pl, sp, 1e-8)

    return {
        "episode_id": episode.get("episode_id", episode.get("id", "?")),
        "scene_id": episode.get("scene_id", ""),
        "success": success,
        "final_geodesic": final_geodesic,
        "initial_geodesic": initial_geodesic,
        "path_length_m": pl,
        "spl": spl,
        "actions": actions,
        "stopped": stopped,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Vanilla Qwen3-VL baseline eval for R2R")
    p.add_argument("--split", default="val_seen",
                   choices=["val_seen", "val_unseen", "train"])
    p.add_argument("--episode-json", type=str, default=None)
    p.add_argument("--scene-dir", type=str,
                   default="/mnt/home/npant/ceph/datasets/habitat/habitat-sim/data/scene_datasets/mp3d")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--habitat-python", type=str, default=HABITAT_PYTHON)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    episode_json = args.episode_json or str(
        REPO / "data" / "episodes" / "r2r" / f"{args.split}.json.gz"
    )
    out_path = args.out or str(
        REPO / "artifacts" / "eval" / f"baseline_vlm_{args.split}.json"
    )

    episodes = load_r2r_episodes(Path(episode_json))
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"Episodes: {len(episodes)} ({args.split})")

    model, processor = load_model(device, dtype)

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_scene[ep.get("scene_id", "unknown")].append(ep)

    results: list[dict[str, Any]] = []

    for scene_id, scene_eps in by_scene.items():
        scene_path = episode_scene_path({"scene_id": scene_id}, args.scene_dir)
        client = HabitatClient(
            habitat_python=args.habitat_python,
            worker_script=WORKER_SCRIPT,
            log_path=str(REPO / "artifacts" / "eval" / "habitat_worker_baseline.log"),
        )
        try:
            client.load_scene(scene_path)
            for ep in scene_eps:
                t0 = time.time()
                result = run_rollout(model, processor, client, ep, device, args.max_steps)
                result["elapsed_s"] = round(time.time() - t0, 1)
                results.append(result)
                print(
                    f"  ep {result['episode_id']}: "
                    f"{'SUCCESS' if result['success'] else 'fail'} "
                    f"geo={result['final_geodesic']:.2f} "
                    f"actions={result['actions']}",
                    flush=True,
                )
        finally:
            client.close()

    sr = sum(r["success"] for r in results) / max(len(results), 1)
    spl = sum(r["spl"] for r in results) / max(len(results), 1)
    print(f"\nSR={sr:.3f}  SPL={spl:.3f}  (n={len(results)})")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "split": args.split,
            "episodes_evaluated": len(results),
            "sr": sr,
            "spl": spl,
            "episodes": results,
        }, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
