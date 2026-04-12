#!/usr/bin/env python3
"""Main training script for SpatialVLM pre-alignment and SFT.

Loads a YAML config, builds the model + dataset + dataloader, and runs the
training loop with wandb logging, gradient accumulation, and checkpointing.

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

Colab
-----
  Mount Google Drive, point --frame-dir at your cached frames, and run.
  wandb will prompt for login on first use (or set WANDB_API_KEY).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

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
        lazy_load_encoders=False,
        lazy_load_backbone=False,
        local_files_only=False,
    )
    if dtype is not None and dtype != torch.float32:
        # Move trainable modules to target dtype; frozen encoders stay fp32
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.to(dtype=dtype)
    return model


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
):
    from spatialvlm.data.collation import SpatialVLMCollator
    from spatialvlm.data.datasets import CachedFrameDataset

    dataset = CachedFrameDataset(frame_dir, limit=limit)
    collator = SpatialVLMCollator(tokenizer, stage=stage)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
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
    p.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    p.add_argument("--save-every-epoch", action="store_true", help="Save checkpoint each epoch")
    return p.parse_args()


DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    stage = cfg.get("stage", "sft")

    # wandb
    import wandb

    os.environ["WANDB_MODE"] = args.wandb_mode
    run = wandb.init(
        project=args.wandb_project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"{stage}_{time.strftime('%m%d_%H%M')}",
        tags=[stage, args.dtype],
        reinit="finish_previous",
    )

    # Tokenizer (from backbone model ID)
    from transformers import AutoTokenizer

    model_id = cfg.model.get("backbone_model_id", "Qwen/Qwen3-VL-8B-Instruct")
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Model
    print(f"Building model on {device} with dtype={dtype}")
    torch_dtype = dtype if dtype != torch.float32 else None
    model = _build_model(cfg, device, torch_dtype)

    # Trainer (sets up freeze groups + optimizer)
    trainer = _build_trainer(cfg, model, stage)
    trainable_count = trainer.trainable_parameter_count()
    print(f"Trainable parameters: {trainable_count:,}")
    wandb.log({"setup/trainable_params": trainable_count})

    # DataLoader
    batch_size = cfg.training.get("batch_size", 4)
    grad_accum = cfg.training.get("grad_accum_steps", 1)
    num_workers = cfg.data.get("num_workers", 0)
    epochs = cfg.training.get("epochs", 1)

    print(f"Loading dataset from: {args.frame_dir}")
    loader = _build_dataloader(
        args.frame_dir, tokenizer, stage, batch_size, num_workers, args.limit
    )
    print(f"Dataset size: {len(loader.dataset)}, batches/epoch: {len(loader)}")

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed at epoch={start_epoch}, step={global_step}")

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"  Starting {stage} training: {epochs} epochs")
    print(f"  batch_size={batch_size}, grad_accum={grad_accum}")
    print(f"{'=' * 60}\n")

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch = time.perf_counter()

        for step_idx, batch in enumerate(loader):
            batch = _move_batch(batch, device)

            # Trainer.step() handles forward + backward + optimizer step
            output = trainer.step(batch)
            epoch_loss += output.loss
            epoch_steps += 1
            global_step += 1

            # Logging
            if global_step % args.log_every == 0:
                log_dict = {
                    "train/loss": output.loss,
                    "train/grad_norm": output.grad_norm,
                    "train/epoch": epoch + (step_idx + 1) / len(loader),
                    "train/global_step": global_step,
                    "train/lr": trainer.optimizer.param_groups[0]["lr"],
                }
                if torch.cuda.is_available():
                    log_dict["system/vram_gb"] = torch.cuda.memory_allocated() / 1e9
                wandb.log(log_dict, step=global_step)
                print(
                    f"  [epoch {epoch + 1}/{epochs}] step {step_idx + 1}/{len(loader)} | "
                    f"loss={output.loss:.4f} | grad_norm={output.grad_norm:.4f}"
                )

        dt = time.perf_counter() - t_epoch
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\n  Epoch {epoch + 1} done in {dt:.1f}s — avg loss: {avg_loss:.4f}\n")
        wandb.log(
            {"train/epoch_loss": avg_loss, "train/epoch_time_s": dt},
            step=global_step,
        )

        # Save checkpoint
        if args.save_every_epoch or epoch == epochs - 1:
            ckpt_path = ckpt_dir / f"{stage}_epoch{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved: {ckpt_path}")
            wandb.save(str(ckpt_path))

    run.finish()
    print("\nTraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
