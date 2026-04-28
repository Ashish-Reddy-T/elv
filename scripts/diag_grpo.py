#!/usr/bin/env python3
"""Diagnose GRPO generation failure.

Run on the interactive GPU node:
    python scripts/diag_grpo.py --device cuda

Two tests:
  Test 1 — checkpoint key matching: are LoRA weights actually loaded?
  Test 2 — text-only generation: does the backbone produce coherent text?
  Test 3 — spatial generation: does full SpatialVLM generate anything?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

CKPT = "/mnt/home/npant/ceph/elv/checkpoints/sft_epoch2_step668.pt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    from spatialvlm.config.model import SpatialVLMConfig
    from spatialvlm.model import SpatialVLM
    from spatialvlm.data.tokenization import SYSTEM_PROMPT
    from transformers import AutoTokenizer

    print("=" * 60)
    print("TEST 1 — checkpoint key matching")
    print("=" * 60)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    ckpt_keys = set(ckpt["model_state_dict"].keys())
    lora_in_ckpt = [k for k in ckpt_keys if "lora" in k.lower()]
    print(f"Total keys in checkpoint: {len(ckpt_keys)}")
    print(f"LoRA keys in checkpoint:  {len(lora_in_ckpt)}")
    if lora_in_ckpt:
        print("  Sample ckpt LoRA keys:")
        for k in lora_in_ckpt[:4]:
            print(f"    {k}")

    print("\nLoading SpatialVLM model...")
    model_cfg = SpatialVLMConfig()
    model_cfg.backbone.model_id = "Qwen/Qwen3-VL-8B-Instruct"
    model_cfg.backbone.lora_rank = 32
    model_cfg.backbone.lora_alpha = 64
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    model = SpatialVLM(config=model_cfg, device=device, torch_dtype=dtype,
                       lazy_load_backbone=False)
    model.to(dtype)

    model_keys = set(model.state_dict().keys())
    lora_in_model = [k for k in model_keys if "lora" in k.lower()]
    print(f"Total keys in model:      {len(model_keys)}")
    print(f"LoRA keys in model:       {len(lora_in_model)}")

    matched = ckpt_keys & model_keys
    only_ckpt = ckpt_keys - model_keys
    only_model = ckpt_keys - ckpt_keys  # placeholder
    only_model = model_keys - ckpt_keys
    print(f"\nMatched keys:             {len(matched)}")
    print(f"Only in checkpoint:       {len(only_ckpt)}")
    print(f"Only in model:            {len(only_model)}")

    if only_ckpt:
        print("  First 5 keys ONLY in checkpoint (not in model):")
        for k in list(only_ckpt)[:5]:
            print(f"    {k}")
    if only_model:
        print("  First 5 keys ONLY in model (not in checkpoint):")
        for k in list(only_model)[:5]:
            print(f"    {k}")

    lora_matched = [k for k in lora_in_ckpt if k in model_keys]
    lora_dropped = [k for k in lora_in_ckpt if k not in model_keys]
    print(f"\nLoRA keys matched: {len(lora_matched)} / {len(lora_in_ckpt)}")
    if lora_dropped:
        print(f"  !! {len(lora_dropped)} LoRA keys DROPPED (key mismatch):")
        for k in lora_dropped[:5]:
            print(f"    {k}")

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"\nAfter load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 2 — text-only generation (backbone, no spatial injection)")
    print("=" * 60)
    prompt = f"{SYSTEM_PROMPT}\n\nInstruction: Walk forward to the kitchen door.\n\n"
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Prompt: {prompt!r}")
    print(f"Prompt tokens: {ids.shape[1]}")

    model.eval()
    with torch.no_grad():
        out = model.backbone.model.generate(
            ids,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"Generated: {text!r}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 3 — full SpatialVLM generate (with spatial tokens, fake depth)")
    print("=" * 60)
    from spatialvlm.utils.camera import CameraIntrinsics

    intrinsics = CameraIntrinsics(fx=256.0, fy=256.0, cx=259.0, cy=259.0, width=518, height=518)

    # Fake inputs — flat depth (1m), grey image
    siglip_px = torch.ones(1, 3, 384, 384, device=device, dtype=dtype) * 0.5
    dinov2_px = torch.ones(1, 3, 518, 518, device=device, dtype=dtype) * 0.5
    depth = torch.ones(1, 518, 518, device=device, dtype=dtype)

    instruction = "Walk forward to the kitchen door."
    placeholder_id = tokenizer.get_vocab().get("<|image_pad|>", tokenizer.pad_token_id or 0)
    prefix_text = f"{SYSTEM_PROMPT}\n\nInstruction: {instruction}\n\nObservation:"
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=True)
    spatial_start_idx = len(prefix_ids)
    spatial_ids = [placeholder_id] * 1369
    suffix_ids = tokenizer.encode("\n\n", add_special_tokens=False)
    all_ids = prefix_ids + spatial_ids + suffix_ids
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_ids)
    print(f"Prefix text: {prefix_text!r}")
    print(f"Prefix tokens: {len(prefix_ids)}")

    print(f"Prompt length: {input_ids.shape[1]} tokens (incl. {len(spatial_ids)} spatial)")
    with torch.no_grad():
        out = model.generate(
            siglip_pixels=siglip_px,
            dinov2_pixels=dinov2_px,
            depth=depth,
            intrinsics=intrinsics,
            input_ids=input_ids,
            spatial_start_idx=spatial_start_idx,
            attention_mask=attn_mask,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = input_ids.shape[1]
    text3 = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    print(f"Generated: {text3!r}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 3b — same prompt, NO visual injection (placeholder tokens only)")
    print("=" * 60)
    # Bypass visual injection: call backbone.model.generate directly with spatial
    # placeholder token embeddings unchanged. If this generates coherent text, SFT
    # was trained without working DeepStack injection.
    with torch.no_grad():
        out3b = model.backbone.model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text3b = tokenizer.decode(out3b[0, prompt_len:], skip_special_tokens=True)
    print(f"Generated (no injection): {text3b!r}")

    frame_path = "/mnt/home/npant/ceph/datasets/spatialvlm/frames/prealign/ep_10000_step_000.pt"

    print("\n" + "=" * 60)
    print("TEST 4 — Real Frame SpatialVLM Generate")
    print("=" * 60)

    print(f"Loading real frame from: {frame_path}")
    frame = torch.load(frame_path, map_location="cpu", weights_only=False)
    print(f"Frame keys: {list(frame.keys())}")

    import torch.nn.functional as F_
    from spatialvlm.utils.camera import CameraIntrinsics

    # Preprocess RGB: frame["rgb"] is [3, H, W] float32 (may be [0,255] or [0,1])
    rgb = frame["rgb"].float()
    if rgb.max() > 1.0:
        rgb = rgb / 255.0  # normalise to [0,1]
    rgb_4d = rgb.unsqueeze(0)  # [1, 3, H, W]
    siglip_px = F_.interpolate(rgb_4d, size=(384, 384), mode="bilinear", align_corners=False)
    dinov2_px = F_.interpolate(rgb_4d, size=(518, 518), mode="bilinear", align_corners=False)
    siglip_px = siglip_px.to(device, dtype=dtype)
    dinov2_px = dinov2_px.to(device, dtype=dtype)

    depth4 = frame["depth"].unsqueeze(0).to(device, dtype=dtype)  # [1, 518, 518]

    intr_d = frame["intrinsics"]
    intrinsics4 = CameraIntrinsics(
        fx=intr_d["fx"], fy=intr_d["fy"],
        cx=intr_d["cx"], cy=intr_d["cy"],
        width=int(intr_d["width"]), height=int(intr_d["height"]),
    )

    instruction4 = frame["instruction"]
    print(f"Instruction: {instruction4!r}")

    # SFT inference format: SYSTEM + instruction + Observation: [spatial] + \n\n
    # Must NOT use build_input_ids(target=None) — that gives prealign format + EOS.
    placeholder_id4 = tokenizer.get_vocab().get("<|image_pad|>", tokenizer.pad_token_id or 0)
    prefix_text4 = f"{SYSTEM_PROMPT}\n\nInstruction: {instruction4}\n\nObservation:"
    prefix_ids4 = tokenizer.encode(prefix_text4, add_special_tokens=True)
    spatial_start4 = len(prefix_ids4)
    spatial_ids4 = [placeholder_id4] * 1369
    suffix_ids4 = tokenizer.encode("\n\n", add_special_tokens=False)
    all_ids4 = prefix_ids4 + spatial_ids4 + suffix_ids4
    input_ids4 = torch.tensor([all_ids4], dtype=torch.long, device=device)
    attn_mask4 = torch.ones_like(input_ids4)

    print(f"Prompt length: {input_ids4.shape[1]} tokens (spatial_start={spatial_start4})")

    with torch.no_grad():
        out4 = model.generate(
            siglip_pixels=siglip_px,
            dinov2_pixels=dinov2_px,
            depth=depth4,
            intrinsics=intrinsics4,
            input_ids=input_ids4,
            spatial_start_idx=spatial_start4,
            attention_mask=attn_mask4,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len4 = input_ids4.shape[1]
    gen_ids4 = out4[0, prompt_len4:]
    text4 = tokenizer.decode(gen_ids4, skip_special_tokens=True)
    print(f"Generated Raw IDs: {gen_ids4.tolist()}")
    print(f"Generated Text: {text4!r}")

    print("\nDone.")

if __name__ == "__main__":
    main()
