#!/usr/bin/env python3
"""Post-process SFT frames to replace thin targets with richer ones.

Current targets (~44 labeled tokens):
  "Reasoning: {caption} Following the instruction to '{instr[:80]}'. Action: MOVE_FORWARD"

Enriched targets (~100 labeled tokens):
  "Reasoning: {caption} I am at step {N} of {T}, following the instruction:
   '{full_instruction}'. Action: MOVE_FORWARD"

Runs CPU-only. Rewrites target field in-place; all other frame data untouched.

Usage:
  python scripts/enrich_sft_targets.py --frame-dir /path/to/frames/sft
  python scripts/enrich_sft_targets.py --frame-dir /path/to/frames/sft --dry-run
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch


def make_rich_target(
    step_idx: int,
    total_steps: int,
    instruction: str,
    caption: str,
    source: str,
    old_target: str,
) -> str:
    """Build a richer target string from frame metadata.

    Extracts the action from the existing target rather than recomputing it,
    so action labels are guaranteed to stay identical.
    """
    # Extract action token from old target (always at the end after "Action: ")
    action = "MOVE_FORWARD"
    if "Action:" in old_target:
        action = old_target.split("Action:")[-1].strip()
    elif "Answer:" in old_target:
        # SQA3D — enrich the reasoning but keep answer format
        answer = old_target.split("Answer:")[-1].strip()
        scene = caption.rstrip(".") if caption else "Based on the current viewpoint"
        return (
            f"Reasoning: {scene}. "
            f"The question asks: '{instruction}'. "
            f"Answer: {answer}"
        )

    scene_prefix = f"{caption} " if caption else ""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich SFT frame targets")
    parser.add_argument("--frame-dir", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print samples without writing files",
    )
    args = parser.parse_args()

    frame_dir = args.frame_dir
    if not frame_dir.is_dir():
        sys.exit(f"ERROR: {frame_dir} is not a directory")

    files = sorted(frame_dir.glob("*.pt"))
    if not files:
        sys.exit(f"ERROR: no .pt files found in {frame_dir}")
    print(f"Found {len(files)} frames in {frame_dir}")

    # Pass 1: determine total_steps per episode
    print("Pass 1: scanning episode lengths...")
    episode_max_step: dict[str, int] = defaultdict(int)
    for path in files:
        frame = torch.load(path, weights_only=False)
        ep_id = str(frame.get("episode_id", "unknown"))
        step = int(frame.get("step_idx", 0))
        episode_max_step[ep_id] = max(episode_max_step[ep_id], step)
    # total_steps = max_step_idx + 1
    episode_total: dict[str, int] = {ep: s + 1 for ep, s in episode_max_step.items()}
    print(f"  {len(episode_total)} unique episodes")

    # Pass 2: enrich and rewrite
    print("Pass 2: enriching targets...")
    old_lens, new_lens = [], []
    for i, path in enumerate(files):
        frame = torch.load(path, weights_only=False)

        old_target = frame.get("target", "")
        if not old_target:
            continue  # prealign frame or no target — skip

        ep_id = str(frame.get("episode_id", "unknown"))
        step_idx = int(frame.get("step_idx", 0))
        total_steps = episode_total.get(ep_id, step_idx + 1)
        instruction = frame.get("instruction", "")
        caption = frame.get("caption", "")
        source = frame.get("source", "")

        new_target = make_rich_target(
            step_idx, total_steps, instruction, caption, source, old_target
        )

        old_lens.append(len(old_target))
        new_lens.append(len(new_target))

        if args.dry_run and i < 5:
            print(f"\n--- {path.name} ---")
            print(f"  OLD ({len(old_target)} chars): {old_target!r}")
            print(f"  NEW ({len(new_target)} chars): {new_target!r}")
            continue

        frame["target"] = new_target
        torch.save(frame, path)

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(files)} done...")

    if old_lens:
        avg_old = sum(old_lens) / len(old_lens)
        avg_new = sum(new_lens) / len(new_lens)
        print(f"\nDone. Updated {len(old_lens)} frames.")
        print(f"  Avg target length: {avg_old:.0f} chars → {avg_new:.0f} chars "
              f"(+{avg_new - avg_old:.0f}, {avg_new / avg_old:.1f}x)")
        if args.dry_run:
            print("  (dry-run: no files written)")


if __name__ == "__main__":
    main()
