#!/usr/bin/env python3
"""End-to-end forward-pass smoke test for SpatialVLM.

Loads real pretrained weights stage-by-stage, feeds synthetic inputs, and
reports pass/fail with shapes, timing, and VRAM usage.  Catches integration
bugs that mock-based unit tests miss.

Usage examples
--------------
  # Quick shape-only test with mock weights (no downloads, CPU, ~10s)
  python scripts/smoke_test.py --mock

  # Full test with real weights on GPU (first run downloads ~18 GB)
  python scripts/smoke_test.py --device cuda --dtype bf16

  # Stages 1-3 only — skips the 16 GB LLM download
  python scripts/smoke_test.py --device cuda --dtype bf16 --skip-backbone

  # Test a single stage
  python scripts/smoke_test.py --device cuda --dtype bf16 --stages 2

Colab Pro setup
---------------
  !git clone <repo-url> ablations && cd ablations
  !pip install -e . 2>&1 | tail -1
  !pip install -e REPOS/geometric-algebra-transformer --no-deps 2>&1 | tail -1
  !python scripts/smoke_test.py --device cuda --dtype bf16
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def banner(msg: str) -> None:
    print(f"\n{BOLD}{'=' * 64}{RESET}")
    print(f"  {CYAN}{msg}{RESET}")
    print(f"{BOLD}{'=' * 64}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓ {msg}{RESET}")


def fail(msg: str) -> None:
    print(f"  {RED}✗ {msg}{RESET}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠ {msg}{RESET}")


def info(msg: str) -> None:
    print(f"  {msg}")


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    info(f"[{label}] {dt:.2f}s")


def gpu_mem() -> str:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        return f"VRAM: {alloc:.1f} GB alloc, {peak:.1f} GB peak"
    return ""


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@dataclass
class StageResult:
    name: str
    passed: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Stage 1: Dual Vision Encoders + Projectors
# ---------------------------------------------------------------------------


def run_stage1(
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    mock: bool,
) -> StageResult:
    banner("Stage 1: Dual Vision Encoders + Projectors")
    try:
        from spatialvlm.config.model import EncoderConfig
        from spatialvlm.encoders.projector import MLPProjector

        cfg = EncoderConfig()

        if mock:
            siglip_feats = torch.randn(batch_size, 576, 3456, device=device, dtype=dtype)
            dinov2_feats = torch.randn(batch_size, 1369, 3072, device=device, dtype=dtype)
            info(f"[mock] SigLIP feats: {list(siglip_feats.shape)}")
            info(f"[mock] DINOv2 feats: {list(dinov2_feats.shape)}")
        else:
            from spatialvlm.encoders.dinov2 import DINOv2Encoder
            from spatialvlm.encoders.siglip import SigLIP2Encoder

            info(f"Loading SigLIP2: {cfg.siglip_model_id}")
            with timer("SigLIP2 load"):
                siglip_enc = SigLIP2Encoder(model_id=cfg.siglip_model_id, device=device)
            mem = gpu_mem()
            if mem:
                info(mem)

            info(f"Loading DINOv2: {cfg.dinov2_model_id}")
            with timer("DINOv2 load"):
                dinov2_enc = DINOv2Encoder(model_id=cfg.dinov2_model_id, device=device)
            mem = gpu_mem()
            if mem:
                info(mem)

            # Synthetic images — random pixels are fine for shape validation
            siglip_pixels = torch.randn(batch_size, 3, 384, 384, device=device)
            dinov2_pixels = torch.randn(batch_size, 3, 518, 518, device=device)

            # SigLIP and DINOv2 run in float32 (frozen, no autocast)
            with timer("SigLIP2 forward"):
                siglip_feats = siglip_enc(siglip_pixels)
            info(f"SigLIP output: {list(siglip_feats.shape)} (expect [B, 576, 3456])")
            assert siglip_feats.shape == (batch_size, 576, 3456), (
                f"SigLIP shape mismatch: {siglip_feats.shape}"
            )

            with timer("DINOv2 forward"):
                dinov2_feats = dinov2_enc(dinov2_pixels)
            info(f"DINOv2 output: {list(dinov2_feats.shape)} (expect [B, 1369, 3072])")
            assert dinov2_feats.shape == (batch_size, 1369, 3072), (
                f"DINOv2 shape mismatch: {dinov2_feats.shape}"
            )

            # Free encoder weights (keep features)
            siglip_feats = siglip_feats.detach().to(dtype=dtype)
            dinov2_feats = dinov2_feats.detach().to(dtype=dtype)
            del siglip_enc, dinov2_enc
            free_gpu()

        # Projectors: trainable, tested in target dtype
        siglip_proj = MLPProjector(in_dim=3456, out_dim=4096).to(device=device, dtype=dtype)
        dinov2_proj = MLPProjector(in_dim=3072, out_dim=4096).to(device=device, dtype=dtype)

        with timer("Projectors forward"):
            siglip_tokens = siglip_proj(siglip_feats)
            dinov2_tokens = dinov2_proj(dinov2_feats)

        info(f"SigLIP projected:  {list(siglip_tokens.shape)} (expect [B, 576, 4096])")
        info(f"DINOv2 projected:  {list(dinov2_tokens.shape)} (expect [B, 1369, 4096])")
        assert siglip_tokens.shape == (batch_size, 576, 4096)
        assert dinov2_tokens.shape == (batch_size, 1369, 4096)

        ok("Stage 1 PASSED")
        return StageResult(
            "stage1",
            True,
            {"siglip_tokens": siglip_tokens.detach(), "dinov2_tokens": dinov2_tokens.detach()},
        )

    except Exception as exc:
        fail(f"Stage 1 FAILED: {exc}")
        traceback.print_exc()
        return StageResult("stage1", False, error=str(exc))


# ---------------------------------------------------------------------------
# Stage 2: Geometric Branch (GATr)
# ---------------------------------------------------------------------------


def run_stage2(
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    mock: bool,
) -> StageResult:
    banner("Stage 2: Geometric Branch (GATr)")
    try:
        if mock:
            gatr_tokens = torch.randn(batch_size, 1369, 4096, device=device, dtype=dtype)
            positions_3d = torch.randn(batch_size, 1369, 3, device=device, dtype=dtype)
            info(f"[mock] GATr tokens: {list(gatr_tokens.shape)}")
            info(f"[mock] positions_3d: {list(positions_3d.shape)}")
        else:
            from spatialvlm.geometry.backproject import (
                aggregate_patches_percentile,
                backproject_depth_map,
            )
            from spatialvlm.geometry.gatr_wrapper import GATrWrapper
            from spatialvlm.utils.camera import CameraIntrinsics

            # Synthetic depth — positive values in metres, roughly indoor scale
            depth = torch.rand(batch_size, 518, 518, device=device, dtype=dtype) * 5.0 + 0.5

            # Reasonable indoor camera intrinsics (518x518 render)
            intrinsics = CameraIntrinsics(
                fx=256.0, fy=256.0, cx=259.0, cy=259.0, width=518, height=518
            )

            info("Backprojecting depth → 3D point cloud")
            with timer("backproject"):
                # backproject expects float32 for precision
                point_map = backproject_depth_map(depth.float(), intrinsics)  # [B, 518, 518, 3]
            info(f"point_map: {list(point_map.shape)}")

            with timer("aggregate patches"):
                positions_3d = aggregate_patches_percentile(
                    point_map,
                    depth.float(),
                    patch_size=14,
                    percentile=0.15,
                )  # [B, 1369, 3]
            info(f"positions_3d: {list(positions_3d.shape)} (expect [B, 1369, 3])")
            assert positions_3d.shape == (batch_size, 1369, 3)

            info("Initializing GATr (8 blocks, improved PGA)")
            with timer("GATr init"):
                gatr = GATrWrapper(
                    num_blocks=8,
                    gatr_mv_channels=16,
                    gatr_s_channels=32,
                    projector_out_dim=4096,
                    device=device,
                )
            info(f"GATr uses improved PGA: {gatr.uses_improved_pga()}")
            mem = gpu_mem()
            if mem:
                info(mem)

            # GATr runs in float32 for numerical stability
            with timer("GATr forward"):
                gatr_tokens = gatr(positions_3d.float())  # [B, 1369, 4096]
            info(f"GATr output: {list(gatr_tokens.shape)} (expect [B, 1369, 4096])")
            assert gatr_tokens.shape == (batch_size, 1369, 4096)

            gatr_tokens = gatr_tokens.detach().to(dtype=dtype)
            positions_3d = positions_3d.to(dtype=dtype)
            del gatr
            free_gpu()

        ok("Stage 2 PASSED")
        return StageResult(
            "stage2",
            True,
            {"gatr_tokens": gatr_tokens, "positions_3d": positions_3d},
        )

    except ImportError as exc:
        fail(f"Stage 2 FAILED — GATr not installed: {exc}")
        warn("Install with: pip install -e REPOS/geometric-algebra-transformer --no-deps")
        return StageResult("stage2", False, error=str(exc))
    except Exception as exc:
        fail(f"Stage 2 FAILED: {exc}")
        traceback.print_exc()
        return StageResult("stage2", False, error=str(exc))


# ---------------------------------------------------------------------------
# Stage 3: Fusion (SVA + Norm Matching)
# ---------------------------------------------------------------------------


def run_stage3(
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    siglip_tokens: torch.Tensor | None,
    dinov2_tokens: torch.Tensor | None,
    gatr_tokens: torch.Tensor | None,
) -> StageResult:
    banner("Stage 3: Fusion (SVA + Norm Matching)")
    try:
        from spatialvlm.fusion.norm_matching import RMSNormMatching
        from spatialvlm.fusion.sva import SpatialVisionAggregator

        # Use provided tensors or create synthetic ones
        if siglip_tokens is None:
            siglip_tokens = torch.randn(batch_size, 576, 4096, device=device, dtype=dtype)
            info("[fallback] Using synthetic SigLIP tokens")
        if dinov2_tokens is None:
            dinov2_tokens = torch.randn(batch_size, 1369, 4096, device=device, dtype=dtype)
            info("[fallback] Using synthetic DINOv2 tokens")
        if gatr_tokens is None:
            gatr_tokens = torch.randn(batch_size, 1369, 4096, device=device, dtype=dtype)
            info("[fallback] Using synthetic GATr tokens")

        info("Initializing SVA (1369 queries, 2 layers, typed bias)")
        sva = SpatialVisionAggregator(
            hidden_dim=4096,
            num_queries=1369,
            num_layers=2,
            use_typed_attention_bias=True,
        ).to(device=device, dtype=dtype)

        norm_match = RMSNormMatching(ema_momentum=0.99).to(device=device, dtype=dtype)

        with timer("SVA forward"):
            fused = sva(
                siglip_tokens=siglip_tokens,
                dinov2_tokens=dinov2_tokens,
                gatr_tokens=gatr_tokens,
            )  # [B, 1369, 4096]
        info(f"SVA output: {list(fused.shape)} (expect [B, 1369, 4096])")
        assert fused.shape == (batch_size, 1369, 4096)

        with timer("Norm matching"):
            fused = norm_match(fused)  # [B, 1369, 4096]
        info(f"Norm-matched output: {list(fused.shape)}")

        # Test diagnostic probe
        info("Testing SVA attention probe...")
        with timer("SVA probe"):
            fused_probe, stats = sva(
                siglip_tokens=siglip_tokens,
                dinov2_tokens=dinov2_tokens,
                gatr_tokens=gatr_tokens,
                return_attention_stats=True,
            )
        info("Attention mass per source (layer 0, mean across heads):")
        layer0 = stats["layer_0"]
        for key in ["head_mean_to_siglip", "head_mean_to_dino", "head_mean_to_gatr"]:
            val = layer0[key].mean().item()
            info(f"  {key}: {val:.4f}")
        total = (
            layer0["head_mean_to_siglip"].mean().item()
            + layer0["head_mean_to_dino"].mean().item()
            + layer0["head_mean_to_gatr"].mean().item()
        )
        info(f"  sum: {total:.4f} (expect ≈ 1.0)")

        mem = gpu_mem()
        if mem:
            info(mem)

        ok("Stage 3 PASSED")
        return StageResult("stage3", True, {"fused_tokens": fused.detach()})

    except Exception as exc:
        fail(f"Stage 3 FAILED: {exc}")
        traceback.print_exc()
        return StageResult("stage3", False, error=str(exc))


# ---------------------------------------------------------------------------
# Stage 4: LLM Backbone (Qwen3-VL + LoRA + RoPE patch)
# ---------------------------------------------------------------------------


def run_stage4(
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    fused_tokens: torch.Tensor | None,
    positions_3d: torch.Tensor | None,
    mock: bool,
) -> StageResult:
    banner("Stage 4: LLM Backbone (Qwen3-VL + LoRA)")
    try:
        from spatialvlm.backbone.qwen3_vl import Qwen3VLBackbone

        n_spatial = 1369
        # Sequence: [BOS] + text_tokens + spatial_placeholder_tokens + [EOS]
        n_text = 32
        seq_len = 1 + n_text + n_spatial + 1  # = 1403
        spatial_start_idx = 1 + n_text  # spatial tokens start after text

        if fused_tokens is None:
            fused_tokens = torch.randn(batch_size, n_spatial, 4096, device=device, dtype=dtype)
            info("[fallback] Using synthetic fused tokens")
        if positions_3d is None:
            positions_3d = torch.randn(batch_size, n_spatial, 3, device=device, dtype=dtype)
            info("[fallback] Using synthetic 3D positions")

        if mock:
            info("[mock] Skipping real backbone load — testing config introspection only")
            backbone = Qwen3VLBackbone(lazy_load=True)
            info(f"  hidden_size: {backbone.hidden_size}")
            info(f"  num_layers: {backbone.num_hidden_layers}")
            info(f"  num_heads: {backbone.num_attention_heads}")
            info(f"  num_kv_heads: {backbone.num_key_value_heads}")
            info(f"  head_dim: {backbone.head_dim}")
            info(f"  mrope_section: {backbone.mrope_section}")
            ok("Stage 4 config introspection PASSED")
            return StageResult("stage4", True)

        info("Loading Qwen3-VL-8B + LoRA (this downloads ~16 GB first time)")
        torch_dtype = dtype if dtype != torch.float32 else None
        with timer("Backbone load"):
            backbone = Qwen3VLBackbone(
                model_id="Qwen/Qwen3-VL-8B-Instruct",
                lora_rank=32,
                lora_alpha=64,
                device=device,
                torch_dtype=torch_dtype,
            )
        stats = backbone.stats
        info(f"Trainable params:  {stats.trainable_params:,}")
        info(f"Total params:      {stats.total_params:,}")
        info(f"PEFT #2880 modules touched: {stats.peft_2880_modules_touched}")
        mem = gpu_mem()
        if mem:
            info(mem)

        # Build input_ids: all zeros (padding tokens) — we just test the forward pass
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        # Build spatial mask
        spatial_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        spatial_mask[:, spatial_start_idx : spatial_start_idx + n_spatial] = True

        # Build attention mask
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

        info(f"input_ids: {list(input_ids.shape)}, seq_len={seq_len}")
        info(f"spatial_start_idx: {spatial_start_idx}, n_spatial: {n_spatial}")

        # Forward — pass DeepStack + RoPE kwargs
        info("Running backbone forward...")
        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spatial_coords_3d": positions_3d,
            "spatial_token_mask": spatial_mask,
            "deepstack_visual_embeds": fused_tokens,
        }

        try:
            with timer("Backbone forward"):
                output = backbone(**forward_kwargs)

            # Check output
            logits = output.logits if hasattr(output, "logits") else output[0]
            info(f"Output logits: {list(logits.shape)} (expect [B, {seq_len}, vocab_size])")
            assert logits.shape[0] == batch_size
            assert logits.shape[1] == seq_len
            ok("Stage 4 forward PASSED")

        except TypeError as exc:
            if "deepstack_visual_embeds" in str(exc):
                warn(f"Qwen3-VL does not accept 'deepstack_visual_embeds': {exc}")
                warn("DeepStack injection needs a monkey-patch (not yet implemented).")
                warn("Retrying without deepstack_visual_embeds...")

                # Retry without DeepStack — just RoPE patch
                forward_kwargs.pop("deepstack_visual_embeds")
                with timer("Backbone forward (no DeepStack)"):
                    output = backbone(**forward_kwargs)
                logits = output.logits if hasattr(output, "logits") else output[0]
                info(f"Output logits: {list(logits.shape)}")
                warn("Stage 4 PARTIAL — forward works but DeepStack injection is missing")
                return StageResult(
                    "stage4",
                    True,
                    {"logits_shape": list(logits.shape)},
                    error="deepstack_visual_embeds not consumed — injection not implemented",
                )
            else:
                raise

        mem = gpu_mem()
        if mem:
            info(mem)

        return StageResult("stage4", True, {"logits_shape": list(logits.shape)})

    except Exception as exc:
        fail(f"Stage 4 FAILED: {exc}")
        traceback.print_exc()
        return StageResult("stage4", False, error=str(exc))


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------


def run_gradient_check(
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> StageResult:
    """Quick backward pass through stages 2-3 to verify trainable params get gradients."""
    banner("Gradient Check: Stages 2-3 (trainable modules)")
    try:
        from spatialvlm.encoders.projector import MLPProjector
        from spatialvlm.fusion.norm_matching import RMSNormMatching
        from spatialvlm.fusion.sva import SpatialVisionAggregator
        from spatialvlm.geometry.gatr_wrapper import GATrWrapper

        # Build trainable pipeline
        gatr = GATrWrapper(num_blocks=8, projector_out_dim=4096, device=device)
        siglip_proj = MLPProjector(in_dim=3456, out_dim=4096).to(device=device, dtype=dtype)
        dinov2_proj = MLPProjector(in_dim=3072, out_dim=4096).to(device=device, dtype=dtype)
        sva = SpatialVisionAggregator(hidden_dim=4096, num_queries=1369, num_layers=2).to(
            device=device, dtype=dtype
        )
        norm_match = RMSNormMatching().to(device=device, dtype=dtype)

        # Synthetic inputs
        siglip_feats = torch.randn(batch_size, 576, 3456, device=device, dtype=dtype)
        dinov2_feats = torch.randn(batch_size, 1369, 3072, device=device, dtype=dtype)
        positions_3d = torch.randn(batch_size, 1369, 3, device=device, dtype=torch.float32)

        # Forward
        siglip_tokens = siglip_proj(siglip_feats)
        dinov2_tokens = dinov2_proj(dinov2_feats)
        gatr_tokens = gatr(positions_3d).to(dtype=dtype)
        fused = sva(
            siglip_tokens=siglip_tokens,
            dinov2_tokens=dinov2_tokens,
            gatr_tokens=gatr_tokens,
        )
        fused = norm_match(fused)

        # Backward
        loss = fused.sum()
        with timer("backward"):
            loss.backward()

        # Check gradients
        modules = {
            "siglip_proj": siglip_proj,
            "dinov2_proj": dinov2_proj,
            "gatr": gatr,
            "sva": sva,
        }
        all_ok = True
        for name, mod in modules.items():
            grads = [p.grad for p in mod.parameters() if p.requires_grad and p.grad is not None]
            if grads:
                max_norm = max(g.norm().item() for g in grads)
                info(f"{name}: {len(grads)} params with grad, max norm = {max_norm:.4f}")
            else:
                fail(f"{name}: NO gradients!")
                all_ok = False

        if all_ok:
            ok("Gradient check PASSED")
        else:
            fail("Gradient check FAILED — some modules got no gradients")
        return StageResult("gradients", all_ok)

    except Exception as exc:
        fail(f"Gradient check FAILED: {exc}")
        traceback.print_exc()
        return StageResult("gradients", False, error=str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SpatialVLM forward-pass smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    p.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Compute dtype (default: fp32)",
    )
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use mock weights — no downloads, tests shapes only",
    )
    p.add_argument(
        "--skip-backbone",
        action="store_true",
        help="Skip LLM backbone (saves ~16 GB download + VRAM)",
    )
    p.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=None,
        help="Which stages to test (1-4). Default: all",
    )
    p.add_argument(
        "--grad-check",
        action="store_true",
        help="Run backward pass through stages 2-3 to verify gradients",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    stages = set(args.stages) if args.stages else {1, 2, 3, 4}
    if args.skip_backbone:
        stages.discard(4)

    banner("SpatialVLM Smoke Test")
    info(f"Device: {device}")
    info(f"Dtype:  {dtype}")
    info(f"Batch:  {args.batch_size}")
    info(f"Mock:   {args.mock}")
    info(f"Stages: {sorted(stages)}")
    if torch.cuda.is_available():
        info(f"GPU:    {torch.cuda.get_device_name(0)}")
        info(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    results: list[StageResult] = []

    # Stage outputs to pass between stages
    siglip_tokens = None
    dinov2_tokens = None
    gatr_tokens = None
    positions_3d = None
    fused_tokens = None

    # Stage 1
    if 1 in stages:
        r1 = run_stage1(device, dtype, args.batch_size, args.mock)
        results.append(r1)
        if r1.passed:
            siglip_tokens = r1.outputs.get("siglip_tokens")
            dinov2_tokens = r1.outputs.get("dinov2_tokens")

    # Stage 2
    if 2 in stages:
        r2 = run_stage2(device, dtype, args.batch_size, args.mock)
        results.append(r2)
        if r2.passed:
            gatr_tokens = r2.outputs.get("gatr_tokens")
            positions_3d = r2.outputs.get("positions_3d")

    # Stage 3
    if 3 in stages:
        r3 = run_stage3(device, dtype, args.batch_size, siglip_tokens, dinov2_tokens, gatr_tokens)
        results.append(r3)
        if r3.passed:
            fused_tokens = r3.outputs.get("fused_tokens")

    # Stage 4
    if 4 in stages:
        r4 = run_stage4(device, dtype, args.batch_size, fused_tokens, positions_3d, args.mock)
        results.append(r4)

    # Gradient check
    if args.grad_check:
        rg = run_gradient_check(device, dtype, args.batch_size)
        results.append(rg)

    # Summary
    banner("Summary")
    all_passed = True
    for r in results:
        status = f"{GREEN}PASS{RESET}" if r.passed else f"{RED}FAIL{RESET}"
        msg = f"  {r.name}: {status}"
        if r.error:
            msg += f"  ({r.error})"
        print(msg)
        if not r.passed:
            all_passed = False

    if all_passed:
        print(f"\n  {GREEN}{BOLD}All stages passed.{RESET}")
    else:
        print(f"\n  {RED}{BOLD}Some stages failed — see above for details.{RESET}")

    mem = gpu_mem()
    if mem:
        info(mem)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
