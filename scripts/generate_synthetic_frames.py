#!/usr/bin/env python3
"""Generate synthetic .pt frames for pipeline testing without Habitat.

Creates random RGB + depth tensors with correct shapes and dummy text.
Use these to verify the full training pipeline works end-to-end before
switching to real Habitat-rendered frames.

Usage
-----
  # Generate 100 prealign frames
  python scripts/generate_synthetic_frames.py \
      --output data/frames/synthetic/ --num-frames 100

  # Then test training:
  python scripts/train.py --config configs/train_prealign.yaml \
      --frame-dir data/frames/synthetic/ --device cuda --dtype bf16 \
      --limit 32 --wandb-mode disabled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SAMPLE_INSTRUCTIONS = [
    "Walk forward past the dining table, then turn left at the hallway.",
    "Go through the kitchen and stop near the refrigerator.",
    "Turn right at the doorway and proceed to the living room.",
    "Navigate past the bookshelf to the window on the far side.",
    "Head down the corridor and enter the second door on the right.",
    "Move towards the staircase and stop at the bottom step.",
    "Walk around the couch and approach the fireplace.",
    "Go straight through the bedroom door and wait by the bed.",
    "Turn left after the bathroom and continue to the end of the hall.",
    "Proceed past the plant and stop at the glass sliding door.",
]

SAMPLE_TARGETS = [
    "Reasoning: I see a dining table ahead. The instruction says to walk past it and turn left. Action: MOVE_FORWARD",
    "Reasoning: I'm in the kitchen area. The refrigerator is to my right. Action: TURN_RIGHT",
    "Reasoning: There's a doorway ahead. I need to turn right here. Action: TURN_RIGHT",
    "Reasoning: The bookshelf is to my left. The window is farther ahead. Action: MOVE_FORWARD",
    "Reasoning: I'm in the corridor. The second door should be upcoming on my right. Action: MOVE_FORWARD",
    "Reasoning: I can see the staircase ahead. I need to stop at the bottom. Action: MOVE_FORWARD",
    "Reasoning: The couch is blocking direct path. I'll go around it. Action: TURN_LEFT",
    "Reasoning: The bedroom door is in front of me. I should go through it. Action: MOVE_FORWARD",
    "Reasoning: I just passed the bathroom. Turning left as instructed. Action: TURN_LEFT",
    "Reasoning: The plant is to my right. The glass door is straight ahead. Action: MOVE_FORWARD",
]


def generate_frame(idx: int, include_target: bool = True) -> dict:
    """Create one synthetic frame dict."""
    return {
        "rgb": torch.rand(3, 518, 518),  # [3, H, W] float [0,1]
        "depth": torch.rand(518, 518) * 5.0 + 0.5,  # [H, W] metric metres, 0.5-5.5m
        "intrinsics": {
            "fx": 256.0,
            "fy": 256.0,
            "cx": 259.0,
            "cy": 259.0,
            "width": 518,
            "height": 518,
        },
        "instruction": SAMPLE_INSTRUCTIONS[idx % len(SAMPLE_INSTRUCTIONS)],
        "target": SAMPLE_TARGETS[idx % len(SAMPLE_TARGETS)] if include_target else None,
        "episode_id": f"synthetic_{idx:05d}",
        "source": "synthetic",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Generate synthetic frames for pipeline testing")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--num-frames", type=int, default=100, help="Number of frames to generate")
    p.add_argument("--no-target", action="store_true", help="Omit target text (prealign mode)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_frames} synthetic frames in {out_dir}/")
    for i in range(args.num_frames):
        frame = generate_frame(i, include_target=not args.no_target)
        path = out_dir / f"frame_{i:05d}.pt"
        torch.save(frame, path)

    print(f"Done. {args.num_frames} frames saved to {out_dir}/")
    # Report size
    total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.pt"))
    print(
        f"Total size: {total_bytes / 1e6:.1f} MB ({total_bytes / args.num_frames / 1e6:.2f} MB/frame)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
