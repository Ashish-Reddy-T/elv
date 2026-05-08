#!/bin/bash
#SBATCH --job-name=svlm-grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=np3006@nyu.edu
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

# =================================================================
# Stage 3: GRPO (~206M trainable, online Habitat rollouts)
#
# Usage:
#   # v1 — degenerate pre-alignment (wrong-loss prealign → SFT v1):
#   sbatch --job-name=svlm-grpo-v1-badpa \
#     --export="RESUME_CKPT=/mnt/home/npant/ceph/elv/checkpoints/sft-v1/sft_epoch2_step668.pt,CKPT_DIR=/mnt/home/npant/ceph/elv/checkpoints/grpo-v1-bad-prealign" \
#     scripts/hpc/run_grpo.slurm
#
#   # v2 — correct pre-alignment (prealign-v2 → SFT v2, preferred):
#   sbatch --job-name=svlm-grpo-v2 \
#     --export="RESUME_CKPT=<sft-v2-ckpt>,CKPT_DIR=/mnt/home/npant/ceph/elv/checkpoints/grpo-v2" \
#     scripts/hpc/run_grpo.slurm
#
#   # Debug run with limited episodes:
#   sbatch --job-name=svlm-grpo-v1-badpa \
#     --export="RESUME_CKPT=...,CKPT_DIR=...,LIMIT=50" \
#     scripts/hpc/run_grpo.slurm
# =================================================================
set -euo pipefail

cd /mnt/home/npant/nyu_classes/elv
mkdir -p artifacts/slurm /mnt/home/npant/ceph/elv/checkpoints

echo "Job $SLURM_JOB_ID (${SLURM_JOB_NAME}) on $(hostname) — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# --- Environment ---
source /mnt/home/npant/nyu_classes/elv/elvvenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200

# Habitat worker runs under its own venv (spawned as subprocess by train_grpo.py)
export HABITAT_PYTHON=/mnt/home/npant/ceph/datasets/habitat/habitat-venv/bin/python

SCENE_DIR=/mnt/home/npant/ceph/datasets/habitat/habitat-sim/data/scene_datasets/mp3d
EPISODE_JSON=data/episodes/r2r/train.json.gz

# CKPT_DIR can be overridden at submission time via --export to keep runs separate.
CKPT_DIR="${CKPT_DIR:-/mnt/home/npant/ceph/elv/checkpoints}"
mkdir -p "${CKPT_DIR}"
echo "Checkpoint dir: ${CKPT_DIR}"

RESUME_FLAG=""
if [ -n "${RESUME_CKPT:-}" ]; then
    RESUME_FLAG="--resume ${RESUME_CKPT}"
    echo "Resuming from: ${RESUME_CKPT}"
fi

LIMIT_FLAG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
    echo "Episode limit: ${LIMIT}"
fi

START_EPOCH_FLAG=""
if [ -n "${START_EPOCH:-}" ]; then
    START_EPOCH_FLAG="--start-epoch ${START_EPOCH}"
    echo "Starting at curriculum epoch: ${START_EPOCH}"
fi

NPROC="${NPROC:-8}"
LAUNCHER="torchrun --nproc_per_node=${NPROC}"
if [ "${NPROC}" = "1" ]; then
    LAUNCHER="python"
fi
${LAUNCHER} scripts/train_grpo.py \
    --config configs/train_grpo.yaml \
    --scene-dir "${SCENE_DIR}" \
    --episode-json "${EPISODE_JSON}" \
    --dtype bf16 \
    --checkpoint-dir "${CKPT_DIR}" \
    --log-every 10 \
    --save-every-steps 200 \
    ${RESUME_FLAG} \
    ${LIMIT_FLAG} \
    ${START_EPOCH_FLAG}

echo "Job $SLURM_JOB_ID finished — $(date)"

