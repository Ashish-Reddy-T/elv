#!/bin/bash
#SBATCH --job-name=svlm-sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=np3006@nyu.edu
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

# =================================================================
# Stage 2: SFT — two phases in one job
#
# Phase 1 (train_sft_phase1.yaml, 1 epoch):
#   Trains gatr + dino_proj + sva + lora; siglip_proj frozen.
#   Lets the geometric pathway stabilise before unlocking SigLIP.
#
# Phase 2 (train_sft.yaml, 2 epochs):
#   Unfreezes siglip_proj; trains all groups together.
#   Resumes automatically from the phase-1 checkpoint.
#
# Usage:
#   # v1 — old prealign checkpoint (bad-loss prealign, SFT self-corrects projectors):
#   sbatch --job-name=svlm-sft-v1 \
#     --export="RESUME_CKPT=/mnt/home/npant/ceph/elv/checkpoints/prealign_epoch3_step1521.pt,CKPT_DIR=/mnt/home/npant/ceph/elv/checkpoints/sft-v1" \
#     scripts/hpc/run_sft.slurm
#
#   # v2 — new prealign checkpoint (correct-loss prealign, submitted by cron after job 6326939):
#   sbatch --job-name=svlm-sft-v2 \
#     --export="RESUME_CKPT=<prealign-v2-ckpt>,CKPT_DIR=/mnt/home/npant/ceph/elv/checkpoints/sft-v2" \
#     scripts/hpc/run_sft.slurm
#
#   # Resume mid-phase-1 after preemption:
#   sbatch --export="RESUME_CKPT=<preempted_ckpt>,CKPT_DIR=<same-dir>,SKIP_PHASE1=1" scripts/hpc/run_sft.slurm
#
#   # Skip phase 1 (already done), go straight to phase 2:
#   sbatch --export="RESUME_CKPT=<phase1_ckpt>,CKPT_DIR=<same-dir>,SKIP_PHASE1=1" scripts/hpc/run_sft.slurm
# =================================================================
set -euo pipefail

cd /mnt/home/npant/nyu_classes/elv
mkdir -p artifacts/slurm /mnt/home/npant/ceph/elv/checkpoints

echo "Job $SLURM_JOB_ID (${SLURM_JOB_NAME}) on $(hostname) — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

source /mnt/home/npant/nyu_classes/elv/elvvenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CKPT_DIR can be overridden at submission time via --export to keep v1/v2 separate.
CKPT_DIR="${CKPT_DIR:-/mnt/home/npant/ceph/elv/checkpoints}"
mkdir -p "${CKPT_DIR}"
echo "Checkpoint dir: ${CKPT_DIR}"

FRAME_DIR="${FRAME_DIR:-/mnt/home/npant/ceph/datasets/spatialvlm/frames/sft}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WANDB_MODE="${WANDB_MODE:-disabled}"

COMMON_ARGS="
    --frame-dir ${FRAME_DIR}
    --dtype bf16
    --batch-size ${BATCH_SIZE}
    --wandb-project spatialvlm
    --wandb-mode ${WANDB_MODE}
    --checkpoint-dir ${CKPT_DIR}
    --save-every-epoch
    --log-every 5
"

# ── Phase 1: geometric pathway (gatr + dino_proj + sva + lora) ────────────
if [ -z "${SKIP_PHASE1:-}" ]; then
    echo "=== Phase 1: gatr+dino_proj+sva+lora (1 epoch) ==="
    RESUME_FLAG=""
    if [ -n "${RESUME_CKPT:-}" ]; then
        RESUME_FLAG="--resume ${RESUME_CKPT}"
        echo "  Starting from: ${RESUME_CKPT}"
    fi

    torchrun --nproc_per_node=4 scripts/train.py \
        --config configs/train_sft_phase1.yaml \
        ${COMMON_ARGS} \
        ${RESUME_FLAG}

    echo "Phase 1 complete — $(date)"
fi

# ── Phase 2: all groups including siglip_proj (2 epochs) ─────────────────
echo "=== Phase 2: all groups (2 epochs) ==="

# Find the most recent sft_epoch* checkpoint (phase 1 output or RESUME_CKPT if SKIP_PHASE1 set)
if [ -n "${SKIP_PHASE1:-}" ] && [ -n "${RESUME_CKPT:-}" ]; then
    PHASE2_CKPT="${RESUME_CKPT}"
else
    PHASE2_CKPT=$(ls -t "${CKPT_DIR}"/sft_epoch*.pt 2>/dev/null | head -1)
    if [ -z "${PHASE2_CKPT}" ]; then
        echo "ERROR: no sft_epoch*.pt checkpoint found after phase 1" >&2
        exit 1
    fi
fi

echo "  Starting from: ${PHASE2_CKPT}"

torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/train_sft.yaml \
    ${COMMON_ARGS} \
    --resume "${PHASE2_CKPT}"

echo "Phase 2 complete — $(date)"
echo "Job $SLURM_JOB_ID finished — $(date)"
