#!/usr/bin/env python3
"""Targeted regression test for Bug 4 (GC disable around model.generate).

Checks three things:
  1. No "use_cache=True is incompatible with gradient checkpointing" warning.
     If that warning fires, HF overrides use_cache=False, the DeepStack hook
     self-clears after prefill, and all decode steps see blank visual tokens.
  2. Gradient checkpointing is re-enabled on the inner model after generate.
  3. Response is not all newlines (visual injection is working).

Usage (single GPU, ~5 min):
    python scripts/test_grpo_generate.py --ckpt /path/to/sft_epoch2_step668.pt
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from spatialvlm.config.model import SpatialVLMConfig
from spatialvlm.model import SpatialVLM
from spatialvlm.utils.camera import CameraIntrinsics
from transformers import AutoTokenizer

# Import helpers from the training script
sys.path.insert(0, str(REPO / "scripts"))
from train_grpo import build_generation_prompt, obs_to_model_inputs


def _gc_enabled(hf_model: torch.nn.Module) -> bool:
    """True if any submodule has gradient_checkpointing=True (matches HF's own check)."""
    return any(getattr(m, "gradient_checkpointing", False) for m in hf_model.modules())


@torch.no_grad()
def _run_spatial_variant(
    *,
    name: str,
    model: SpatialVLM,
    tokenizer: AutoTokenizer,
    siglip_px: torch.Tensor,
    dinov2_px: torch.Tensor,
    depth_t: torch.Tensor,
    intrinsics: CameraIntrinsics,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    spatial_start_idx: int,
    use_spatial_rope: bool,
    use_deepstack: bool,
    deepstack_blend_alpha: float = 1.0,
    max_new_tokens: int = 32,
) -> tuple[str, float]:
    print(f"\n── Ablation: {name} ──", flush=True)
    out = model.generate(
        siglip_pixels=siglip_px,
        dinov2_pixels=dinov2_px,
        depth=depth_t,
        intrinsics=intrinsics,
        input_ids=input_ids,
        spatial_start_idx=spatial_start_idx,
        attention_mask=attn_mask,
        use_spatial_rope=use_spatial_rope,
        use_deepstack=use_deepstack,
        deepstack_blend_alpha=deepstack_blend_alpha,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    seq = out.sequences if hasattr(out, "sequences") else out
    response = tokenizer.decode(seq[0, input_ids.shape[1]:], skip_special_tokens=True)
    newline_frac = response.count("\n") / max(len(response), 1)
    print(
        f"{name} response ({len(response)} chars, {newline_frac:.0%} newlines): "
        f"{response[:160]!r}",
        flush=True,
    )
    return response, newline_frac


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="/mnt/home/npant/ceph/elv/checkpoints/sft_epoch2_step668.pt",
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # ── Load model ───────────────────────────────────────────────────────────
    print("Loading model...", flush=True)
    model_cfg = SpatialVLMConfig()
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.backbone.model_id)
    model = SpatialVLM(config=model_cfg, device=device, torch_dtype=dtype,
                       lazy_load_backbone=False)
    model.to(dtype)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Checkpoint loaded: {args.ckpt}", flush=True)

    # ── Enable gradient checkpointing (mirrors what train_grpo.py does) ──────
    hf = model.backbone.model
    if hasattr(hf, "enable_input_require_grads"):
        hf.enable_input_require_grads()
    hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("Gradient checkpointing enabled.", flush=True)

    assert _gc_enabled(hf), "FAIL: GC should be enabled before generate"
    print("CHECK 0 PASS: GC is enabled on inner model before generate.", flush=True)

    # ── Real rendered Habitat frame (same distribution as SFT training) ──────
    # Random synthetic inputs are too far from the training distribution and
    # cause degenerate generation. Use a real pre-rendered frame instead.
    # Use an actual SFT training frame so we can check whether the model
    # reproduces its memorized SFT target (overfitting hypothesis test).
    frame_path = Path("/mnt/home/npant/ceph/datasets/spatialvlm/frames/sft/ep_1_step_000.pt")
    frame = torch.load(str(frame_path), map_location="cpu", weights_only=False)
    intr = frame["intrinsics"]
    intrinsics = CameraIntrinsics(
        fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"], width=518, height=518
    )
    # frame["rgb"] is [3, 518, 518] float or uint8
    rgb_t = frame["rgb"].float()
    if rgb_t.max() > 1.0:
        rgb_t = rgb_t / 255.0
    siglip_px = torch.nn.functional.interpolate(
        rgb_t.unsqueeze(0), size=(384, 384), mode="bilinear", align_corners=False
    ).to(device)
    dinov2_px = rgb_t.unsqueeze(0).to(device)  # already 518×518
    depth_t = frame["depth"].unsqueeze(0).to(device)  # [1, 518, 518]
    instruction = frame.get("instruction", "Navigate to the goal.")
    input_ids, attn_mask, spatial_start_idx = build_generation_prompt(
        tokenizer, instruction
    )
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    # ── BASELINE: text-only generation (no spatial tokens, no visual pipeline) ──
    # Strips the 1369 spatial placeholder tokens and calls the LM directly.
    # If this also produces all newlines → SFT checkpoint is degenerate.
    # If this produces real text → visual injection path is what's broken.
    print("\n── Baseline: text-only generation (no spatial tokens) ──", flush=True)
    model.eval()
    text_only_ids = torch.cat([
        input_ids[:, :spatial_start_idx],
        input_ids[:, spatial_start_idx + 1369:],
    ], dim=1).to(device)
    with torch.no_grad():
        text_gen = model.backbone.model.generate(
            input_ids=text_only_ids,
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text_response = tokenizer.decode(text_gen[0, text_only_ids.shape[1]:], skip_special_tokens=True)
    text_newline_frac = text_response.count("\n") / max(len(text_response), 1)
    print(f"Text-only response ({len(text_response)} chars, {text_newline_frac:.0%} newlines): "
          f"{text_response[:150]!r}", flush=True)

    # ── BASELINE 2: full sequence with current spatial placeholder, NO injection ──
    # This should stay coherent. If it fails, the placeholder/prompt format itself
    # is disruptive before visual injection even enters the picture.
    print("\n── Baseline 2: full sequence, current placeholder, NO visual injection ──", flush=True)
    _rope_patch_mod_pre = None
    try:
        import spatialvlm.backbone.rope_patch as _rope_patch_mod_pre
        _embed_for_b2 = _rope_patch_mod_pre._find_embed_tokens(model.backbone.model)
        # Make sure no stash is set
        _embed_for_b2._deepstack_visual_embeds = None
        _embed_for_b2._deepstack_spatial_mask = None
    except Exception as _e2:
        print(f"  [WARN] could not clear embeds for baseline2: {_e2}", flush=True)
    with torch.no_grad():
        no_inject_gen = model.backbone.model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attn_mask.to(device),
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    no_inject_resp = tokenizer.decode(no_inject_gen[0, input_ids.shape[1]:], skip_special_tokens=True)
    no_inject_nf = no_inject_resp.count("\n") / max(len(no_inject_resp), 1)
    print(f"Current-placeholder full-seq response ({len(no_inject_resp)} chars, {no_inject_nf:.0%} newlines): "
          f"{no_inject_resp[:150]!r}", flush=True)

    # ── BASELINE 3: force old native <|image_pad|> placeholder, NO injection ──
    # Kept as a regression check for the failure mode we just isolated.
    image_pad_id = tokenizer.get_vocab().get("<|image_pad|>")
    image_pad_input_ids = input_ids.clone()
    if image_pad_id is not None:
        image_pad_input_ids[:, spatial_start_idx : spatial_start_idx + 1369] = image_pad_id
    print("\n── Baseline 3: full sequence with forced <|image_pad|> token ──", flush=True)
    with torch.no_grad():
        alt_no_inject_gen = model.backbone.model.generate(
            input_ids=image_pad_input_ids,
            attention_mask=attn_mask,
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    alt_no_inject_resp = tokenizer.decode(
        alt_no_inject_gen[0, image_pad_input_ids.shape[1]:], skip_special_tokens=True
    )
    alt_no_inject_nf = alt_no_inject_resp.count("\n") / max(len(alt_no_inject_resp), 1)
    print(
        f"Forced-image-pad full-seq response "
        f"({len(alt_no_inject_resp)} chars, {alt_no_inject_nf:.0%} newlines): "
        f"{alt_no_inject_resp[:150]!r}",
        flush=True,
    )

    # ── Snapshot embed module state — confirms stash+cleanup cycle ran ────────
    from spatialvlm.backbone.rope_patch import _find_embed_tokens
    import spatialvlm.backbone.rope_patch as _rope_patch_mod
    embed_module = _find_embed_tokens(model.backbone.model)
    print(f"\n── Embed module identity ──", flush=True)
    print(f"embed_module type: {type(embed_module).__name__}", flush=True)
    print(f"embed_module id:   {id(embed_module)}", flush=True)
    print(f"embed_module forward hooks: {list(embed_module._forward_hooks.keys())}", flush=True)
    pre_stash_has_hook = len(embed_module._forward_hooks) > 0
    print(f"hook registered: {pre_stash_has_hook}", flush=True)

    # Reset hook call counter before generate
    _rope_patch_mod._HOOK_CALL_COUNTER = 0
    print(f"Hook call counter reset to 0", flush=True)

    # ── Injection ablations ────────────────────────────────────────────────────
    # These isolate whether degeneration comes from 3D RoPE, DeepStack embedding
    # replacement, or their combination.
    model.eval()
    rope_only_resp, rope_only_nf = _run_spatial_variant(
        name="RoPE only (coords yes, visual embeds no)",
        model=model,
        tokenizer=tokenizer,
        siglip_px=siglip_px,
        dinov2_px=dinov2_px,
        depth_t=depth_t,
        intrinsics=intrinsics,
        input_ids=input_ids,
        attn_mask=attn_mask,
        spatial_start_idx=spatial_start_idx,
        use_spatial_rope=True,
        use_deepstack=False,
    )
    deepstack_only_resp, deepstack_only_nf = _run_spatial_variant(
        name="DeepStack only (visual embeds yes, coords no)",
        model=model,
        tokenizer=tokenizer,
        siglip_px=siglip_px,
        dinov2_px=dinov2_px,
        depth_t=depth_t,
        intrinsics=intrinsics,
        input_ids=input_ids,
        attn_mask=attn_mask,
        spatial_start_idx=spatial_start_idx,
        use_spatial_rope=False,
        use_deepstack=True,
    )
    regular_token_deepstack_resp, regular_token_deepstack_nf = _run_spatial_variant(
        name="DeepStack with current placeholder token",
        model=model,
        tokenizer=tokenizer,
        siglip_px=siglip_px,
        dinov2_px=dinov2_px,
        depth_t=depth_t,
        intrinsics=intrinsics,
        input_ids=input_ids,
        attn_mask=attn_mask,
        spatial_start_idx=spatial_start_idx,
        use_spatial_rope=False,
        use_deepstack=True,
    )
    regular_token_deepstack_10_resp, regular_token_deepstack_10_nf = _run_spatial_variant(
        name="DeepStack with current placeholder token, alpha=0.10",
        model=model,
        tokenizer=tokenizer,
        siglip_px=siglip_px,
        dinov2_px=dinov2_px,
        depth_t=depth_t,
        intrinsics=intrinsics,
        input_ids=input_ids,
        attn_mask=attn_mask,
        spatial_start_idx=spatial_start_idx,
        use_spatial_rope=False,
        use_deepstack=True,
        deepstack_blend_alpha=0.10,
    )
    regular_token_deepstack_25_resp, regular_token_deepstack_25_nf = _run_spatial_variant(
        name="DeepStack with current placeholder token, alpha=0.25",
        model=model,
        tokenizer=tokenizer,
        siglip_px=siglip_px,
        dinov2_px=dinov2_px,
        depth_t=depth_t,
        intrinsics=intrinsics,
        input_ids=input_ids,
        attn_mask=attn_mask,
        spatial_start_idx=spatial_start_idx,
        use_spatial_rope=False,
        use_deepstack=True,
        deepstack_blend_alpha=0.25,
    )

    print(
        "Ablation summary: "
        f"RoPE-only empty={rope_only_resp.strip() == ''} newlines={rope_only_nf:.0%}; "
        f"DeepStack-only empty={deepstack_only_resp.strip() == ''} newlines={deepstack_only_nf:.0%}; "
        f"Regular-token DeepStack empty={regular_token_deepstack_resp.strip() == ''} "
        f"newlines={regular_token_deepstack_nf:.0%}; "
        f"Regular-token alpha=0.10 empty={regular_token_deepstack_10_resp.strip() == ''} "
        f"newlines={regular_token_deepstack_10_nf:.0%}; "
        f"Regular-token alpha=0.25 empty={regular_token_deepstack_25_resp.strip() == ''} "
        f"newlines={regular_token_deepstack_25_nf:.0%}",
        flush=True,
    )

    # ── Run full spatial generate, capturing warnings ─────────────────────────
    print("\n── Full spatial generate ──", flush=True)
    caught: list[warnings.WarningMessage] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.eval()
        gen_out = model.generate(
            siglip_pixels=siglip_px,
            dinov2_pixels=dinov2_px,
            depth=depth_t,
            intrinsics=intrinsics,
            input_ids=input_ids,
            spatial_start_idx=spatial_start_idx,
            attention_mask=attn_mask,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    print(f"\nHook fired {_rope_patch_mod._HOOK_CALL_COUNTER} times total during generate()", flush=True)

    # Did the stash+cleanup cycle run?  After SpatialVLM.generate() the attr
    # should be deleted (not present), not None.
    stash_ran = not hasattr(embed_module, "_deepstack_visual_embeds") or \
                getattr(embed_module, "_deepstack_visual_embeds", "PRESENT") == "PRESENT"
    print(f"Stash+cleanup ran (attr deleted): "
          f"{not hasattr(embed_module, '_deepstack_visual_embeds')}", flush=True)

    # ── CHECK 1: no use_cache override warning ────────────────────────────────
    use_cache_warns = [w for w in caught if "use_cache" in str(w.message).lower()]
    if use_cache_warns:
        print(f"FAIL CHECK 1: use_cache override warning fired: {use_cache_warns[0].message}")
        sys.exit(1)
    print("CHECK 1 PASS: no 'use_cache=True incompatible' warning.", flush=True)

    # ── CHECK 2: GC re-enabled after generate ────────────────────────────────
    if not _gc_enabled(hf):
        print("FAIL CHECK 2: gradient checkpointing was NOT re-enabled after generate.")
        sys.exit(1)
    print("CHECK 2 PASS: GC re-enabled on inner model after generate.", flush=True)

    # ── CHECK 3: response is not all newlines ─────────────────────────────────
    prompt_len = input_ids.shape[1]
    seq = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
    response_ids = seq[0, prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(f"Full-pipeline response ({len(response_text)} chars): {response_text[:200]!r}", flush=True)

    newline_fraction = response_text.count("\n") / max(len(response_text), 1)
    if newline_fraction > 0.8 or response_text.strip() == "":
        # If text-only ALSO fails, the SFT checkpoint is the problem, not injection
        if text_newline_frac > 0.8 or text_response.strip() == "":
            print("FAIL CHECK 3: BOTH text-only and full-pipeline are degenerate. "
                  "SFT checkpoint is broken/overfit — re-run SFT with lower LR or fewer epochs.")
        else:
            print(f"FAIL CHECK 3: full-pipeline degenerate ({newline_fraction:.0%} newlines) "
                  "but text-only is fine → visual injection is still broken.")
        sys.exit(1)
    print("CHECK 3 PASS: response contains non-newline content.", flush=True)

    print("\n=== ALL CHECKS PASSED ===", flush=True)


if __name__ == "__main__":
    main()
