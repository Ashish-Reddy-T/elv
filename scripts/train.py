#!/usr/bin/env python3
"""Main training script for SpatialVLM pre-alignment and SFT.

Loads a YAML config, builds the model + dataset + dataloader, and runs the
training loop with wandb logging, gradient accumulation, cosine LR schedule,
and checkpointing.  Handles SIGTERM for GCP Spot / Slurm --requeue.

Usage
-----
  # Pre-alignment on synthetic frames (pipeline test)
  python scripts/train.py --config configs/train_prealign.yaml \
      --frame-dir data/frames/synthetic/ --device cuda --dtype bf16

  # SFT on real Habitat frames
  python scripts/train.py --config configs/train_sft.yaml \
      --frame-dir data/frames/sft/ --device cuda --dtype bf16

  # Resume from checkpoint
  python scripts/train.py --config configs/train_sft.yaml \
      --frame-dir data/frames/sft/ --resume checkpoints/sft_epoch2.pt

Multi-GPU (DDP)
---------------
  torchrun --nproc_per_node=4 scripts/train.py --config configs/train_sft.yaml \
      --frame-dir data/frames/sft/ --dtype bf16
  # --device is ignored when torchrun sets LOCAL_RANK; each rank uses cuda:<local_rank>.

HPC (NYU Cloud Bursting)
-------------------------
  See scripts/hpc/run_prealign.slurm and docs/RUNBOOK.md.
  Spot instances send SIGTERM 30s before kill — this script catches it and
  saves an emergency checkpoint so --requeue picks up where it left off.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Distributed setup — auto-detected from torchrun env vars
# ---------------------------------------------------------------------------


@dataclass
class DistContext:
    enabled: bool
    rank: int        # global rank
    local_rank: int  # rank on this node (= GPU index)
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _setup_dist() -> DistContext:
    """Init process group when launched with torchrun, no-op otherwise."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return DistContext(enabled=False, rank=0, local_rank=0, world_size=1)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return DistContext(
        enabled=True,
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
    )


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, stripping DDP if present."""
    return getattr(model, "module", model)


# ---------------------------------------------------------------------------
# Lazy imports — keep startup fast for --help
# ---------------------------------------------------------------------------


def _build_model(cfg: Any, device: torch.device, dtype: torch.dtype | None):
    from spatialvlm.config.model import SpatialVLMConfig
    from spatialvlm.model import SpatialVLM

    model_cfg = SpatialVLMConfig()
    if hasattr(cfg, "model"):
        if hasattr(cfg.model, "lora_rank"):
            model_cfg.backbone.lora_rank = cfg.model.lora_rank
        if hasattr(cfg.model, "lora_alpha"):
            model_cfg.backbone.lora_alpha = cfg.model.lora_alpha

    model = SpatialVLM(
        config=model_cfg,
        device=device,
        torch_dtype=dtype,
        lazy_load_encoders=False,
        lazy_load_backbone=False,
        local_files_only=False,
    )
    if dtype is not None and dtype != torch.float32:
        # Cast small trainable modules (projectors, GATr, SVA) that aren't loaded via
        # from_pretrained. The backbone is already in `dtype` from torch_dtype above.
        for name, p in model.named_parameters():
            if p.requires_grad and p.dtype != dtype:
                p.data = p.data.to(dtype=dtype)
    return model


def _maybe_enable_gradient_checkpointing(model: Any, cfg: Any) -> None:
    """Enable HF gradient checkpointing on the PEFT-wrapped backbone when requested."""
    training = getattr(cfg, "training", None)
    if training is None or not training.get("gradient_checkpointing", False):
        return
    backbone = getattr(model, "backbone", None)
    if backbone is None or getattr(backbone, "model", None) is None:
        return
    hf = backbone.model
    try:
        if hasattr(hf, "gradient_checkpointing_enable"):
            hf.gradient_checkpointing_enable()
            print("Gradient checkpointing: enabled on backbone.")
    except (AttributeError, RuntimeError, ValueError) as exc:
        print(f"Warning: could not enable gradient checkpointing: {exc}")


def _build_trainer(cfg: Any, model: torch.nn.Module, stage: str):
    if stage == "prealign":
        from spatialvlm.training.prealign import PrealignConfig, PrealignmentTrainer

        groups = None
        if hasattr(cfg, "training") and hasattr(cfg.training, "trainable_groups"):
            g = cfg.training.trainable_groups
            if g is not None:
                groups = tuple(g)
        pc = PrealignConfig(
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            max_grad_norm=cfg.training.max_grad_norm,
            trainable_groups=groups,
        )
        return PrealignmentTrainer(model, pc)
    else:
        from spatialvlm.training.sft import SFTConfig, SFTTrainer

        groups = None
        if hasattr(cfg, "training") and hasattr(cfg.training, "trainable_groups"):
            g = cfg.training.trainable_groups
            if g is not None:
                groups = tuple(g)
        sc = SFTConfig(
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            max_grad_norm=cfg.training.max_grad_norm,
            label_smoothing=getattr(cfg.training, "label_smoothing", 0.0),
            trainable_groups=groups,
        )
        return SFTTrainer(model, sc)


def _build_dataloader(
    frame_dir: str,
    tokenizer: Any,
    stage: str,
    batch_size: int,
    num_workers: int,
    limit: int | None,
    dist_ctx: DistContext | None = None,
):
    from spatialvlm.data.collation import SpatialVLMCollator
    from spatialvlm.data.datasets import CachedFrameDataset

    dataset = CachedFrameDataset(frame_dir, limit=limit)
    collator = SpatialVLMCollator(tokenizer, stage=stage)

    sampler = None
    shuffle = True
    if dist_ctx is not None and dist_ctx.enabled:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False  # sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in the batch dict to device."""
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# LR Scheduler — cosine with linear warmup
# ---------------------------------------------------------------------------


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    """Cosine schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(current_step / max(warmup_steps, 1), 1e-8)
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.5 * (1.0 + math.cos(math.pi * progress)), 0.0)

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# SIGTERM handler — emergency checkpoint on Spot preemption
# ---------------------------------------------------------------------------

_SIGTERM_RECEIVED = False


def _sigterm_handler(signum: int, frame: Any) -> None:
    global _SIGTERM_RECEIVED
    _SIGTERM_RECEIVED = True
    print("\n[SIGTERM] Preemption signal received — will checkpoint and exit after current step.")


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
    epoch: int,
    global_step: int,
    cfg: Any,
    ckpt_dir: Path,
    stage: str,
    tag: str = "",
) -> Path:
    suffix = f"_{tag}" if tag else ""
    ckpt_path = ckpt_dir / f"{stage}_epoch{epoch + 1}_step{global_step}{suffix}.pt"
    # Always save the unwrapped model so checkpoints load correctly in single-GPU runs too.
    raw_model = _unwrap(model)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpatialVLM training")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--frame-dir", required=True, help="Cached frames directory")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--wandb-project", default="spatialvlm", help="wandb project name")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    p.add_argument("--resume", default=None, help="Resume from checkpoint .pt file")
    p.add_argument("--limit", type=int, default=None, help="Limit dataset size (for testing)")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config training.batch_size (use 1-4 on A100 40GB)",
    )
    p.add_argument("--log-every", type=int, default=10, help="Log every N optimizer steps")
    p.add_argument("--save-every-epoch", action="store_true", help="Save checkpoint each epoch")
    return p.parse_args()


DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    dtype = DTYPE_MAP[args.dtype]
    stage = cfg.get("stage", "sft")

    # ── Distributed setup (no-op when not launched with torchrun) ───────────
    dist_ctx = _setup_dist()
    if dist_ctx.enabled:
        device = torch.device(f"cuda:{dist_ctx.local_rank}")
    else:
        device = torch.device(args.device)

    # SIGTERM handler for Spot preemption
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # wandb — rank-0 only
    import wandb

    os.environ["WANDB_MODE"] = args.wandb_mode
    if dist_ctx.is_main:
        run = wandb.init(
            project=args.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{stage}_{time.strftime('%m%d_%H%M')}",
            tags=[stage, args.dtype],
            reinit="finish_previous",
        )
    else:
        run = None

    # Tokenizer (from backbone model ID)
    from transformers import AutoTokenizer

    model_id = cfg.model.get("backbone_model_id", "Qwen/Qwen3-VL-8B-Instruct")
    if dist_ctx.is_main:
        print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Model
    if dist_ctx.is_main:
        print(
            f"Building model on {device} with dtype={dtype}"
            + (f" [DDP world_size={dist_ctx.world_size}]" if dist_ctx.enabled else "")
        )
    torch_dtype = dtype if dtype != torch.float32 else None
    model = _build_model(cfg, device, torch_dtype)
    _maybe_enable_gradient_checkpointing(model, cfg)

    # Trainer (sets up freeze groups + optimizer) — built on the unwrapped model
    # so requires_grad flags and optimizer param refs are established before DDP.
    trainer = _build_trainer(cfg, model, stage)
    trainable_count = trainer.trainable_parameter_count()
    if dist_ctx.is_main:
        print(f"Trainable parameters: {trainable_count:,}")
        wandb.log({"setup/trainable_params": trainable_count})

    # Wrap with DDP after the trainer is built.  trainer.model is updated so
    # that forward_backward() goes through DDP's gradient sync.  The optimizer
    # already holds refs to the underlying params — DDP doesn't change those.
    if dist_ctx.enabled:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[dist_ctx.local_rank], find_unused_parameters=True)
        trainer.model = model

    # DataLoader
    batch_size = (
        args.batch_size if args.batch_size is not None else cfg.training.get("batch_size", 4)
    )
    if args.batch_size is not None and dist_ctx.is_main:
        print(f"Using --batch-size override: {batch_size}")
    grad_accum = cfg.training.get("grad_accum_steps", 1)
    num_workers = cfg.data.get("num_workers", 0)
    epochs = cfg.training.get("epochs", 1)

    if dist_ctx.is_main:
        print(f"Loading dataset from: {args.frame_dir}")
    loader = _build_dataloader(
        args.frame_dir, tokenizer, stage, batch_size, num_workers, args.limit, dist_ctx
    )
    if dist_ctx.is_main:
        print(f"Dataset size: {len(loader.dataset)}, batches/epoch: {len(loader)}")

    # LR scheduler — cosine with linear warmup
    steps_per_epoch = len(loader) // grad_accum  # optimizer steps per epoch
    total_optim_steps = steps_per_epoch * epochs
    warmup_ratio = cfg.training.get("warmup_ratio", 0.03)
    warmup_steps = cfg.training.get("warmup_steps", int(total_optim_steps * warmup_ratio))
    scheduler = _build_scheduler(trainer.optimizer, total_optim_steps, warmup_steps)
    if dist_ctx.is_main:
        effective_batch = batch_size * grad_accum * dist_ctx.world_size
        print(
            f"LR schedule: {warmup_steps} warmup steps, {total_optim_steps} total optimizer steps "
            f"(grad_accum={grad_accum}, effective_batch={effective_batch})"
        )

    # Checkpoint dir (created by rank-0 only; barrier ensures it exists before others proceed)
    ckpt_dir = Path(args.checkpoint_dir)
    if dist_ctx.is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if dist_ctx.enabled:
        dist.barrier()

    # Resume
    start_epoch = 0
    global_step = 0  # counts optimizer steps (not micro-batch steps)
    if args.resume:
        if dist_ctx.is_main:
            print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        _unwrap(model).load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        if dist_ctx.is_main:
            print(f"Resumed at epoch={start_epoch}, step={global_step}")

    # Training loop
    if dist_ctx.is_main:
        print(f"\n{'=' * 60}")
        print(f"  Starting {stage} training: {epochs} epochs")
        print(f"  batch_size={batch_size}, grad_accum={grad_accum}")
        print(f"  effective_batch={batch_size * grad_accum * dist_ctx.world_size}")
        print(f"{'=' * 60}\n")

    model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    loss_scale = 1.0 / grad_accum

    preempted = False
    for epoch in range(start_epoch, epochs):
        # DistributedSampler must be re-seeded each epoch for correct shuffling.
        if dist_ctx.enabled and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        micro_steps_in_epoch = 0
        optim_steps_in_epoch = 0
        t_epoch = time.perf_counter()

        for micro_idx, batch in enumerate(loader):
            batch = _move_batch(batch, device)

            # Accumulate: forward + scaled backward (no optimizer step)
            loss_val = trainer.forward_backward(batch, loss_scale=loss_scale)
            epoch_loss += loss_val
            micro_steps_in_epoch += 1

            # Optimizer step every grad_accum micro-batches
            is_accum_boundary = micro_steps_in_epoch % grad_accum == 0
            is_last_batch = micro_idx == len(loader) - 1

            if is_accum_boundary or is_last_batch:
                grad_norm_val = trainer.clip_and_step()
                scheduler.step()
                global_step += 1
                optim_steps_in_epoch += 1

                # Logging — rank-0 only
                if dist_ctx.is_main and global_step % args.log_every == 0:
                    log_dict = {
                        "train/loss": loss_val,
                        "train/grad_norm": grad_norm_val,
                        "train/epoch": epoch + micro_steps_in_epoch / len(loader),
                        "train/global_step": global_step,
                        "train/lr": scheduler.get_last_lr()[0],
                    }
                    if torch.cuda.is_available():
                        log_dict["system/vram_gb"] = torch.cuda.memory_allocated() / 1e9
                    wandb.log(log_dict, step=global_step)
                    print(
                        f"  [epoch {epoch + 1}/{epochs}] "
                        f"step {global_step}/{total_optim_steps} | "
                        f"loss={loss_val:.4f} | grad_norm={grad_norm_val:.4f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

            # Spot preemption — save and exit (rank-0 saves, all ranks break)
            if _SIGTERM_RECEIVED:
                if dist_ctx.is_main:
                    print("[SIGTERM] Saving emergency checkpoint...")
                    ckpt_path = _save_checkpoint(
                        model,
                        trainer.optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        cfg,
                        ckpt_dir,
                        stage,
                        tag="preempted",
                    )
                    print(f"  Emergency checkpoint: {ckpt_path}")
                    wandb.save(str(ckpt_path))
                preempted = True
                break

        if preempted:
            break

        dt = time.perf_counter() - t_epoch
        avg_loss = epoch_loss / max(micro_steps_in_epoch, 1)
        if dist_ctx.is_main:
            print(f"\n  Epoch {epoch + 1} done in {dt:.1f}s — avg loss: {avg_loss:.4f}\n")
            wandb.log(
                {"train/epoch_loss": avg_loss, "train/epoch_time_s": dt},
                step=global_step,
            )

            # Save checkpoint — rank-0 only
            if args.save_every_epoch or epoch == epochs - 1:
                ckpt_path = _save_checkpoint(
                    model,
                    trainer.optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    cfg,
                    ckpt_dir,
                    stage,
                )
                print(f"  Checkpoint saved: {ckpt_path}")
                wandb.save(str(ckpt_path))

    if dist_ctx.is_main:
        if preempted:
            print("\nTraining interrupted by preemption. Resume with --resume <checkpoint>.")
        else:
            print("\nTraining complete.")
        run.finish()

    if dist_ctx.enabled:
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
