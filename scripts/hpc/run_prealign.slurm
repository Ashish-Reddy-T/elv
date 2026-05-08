#!/bin/bash
#SBATCH --job-name=svlm-prealign
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=np3006@nyu.edu
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

# =================================================================
# Stage 1: Pre-alignment (projectors only, ~77M trainable)
#
# Usage:
#   sbatch scripts/hpc/run_prealign.slurm
#
# Resume after preemption:
#   sbatch --export=RESUME_CKPT=/mnt/home/npant/ceph/elv/checkpoints/prealign_epoch1_step500_preempted.pt \
#       scripts/hpc/run_prealign.slurm
# =================================================================
set -euo pipefail

cd /mnt/home/npant/nyu_classes/elv
mkdir -p artifacts/slurm /mnt/home/npant/ceph/elv/checkpoints

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# --- Environment ---
source /mnt/home/npant/nyu_classes/elv/elvvenv/bin/activate

# --- Build command ---
RESUME_FLAG=""
if [ -n "${RESUME_CKPT:-}" ]; then
    RESUME_FLAG="--resume ${RESUME_CKPT}"
    echo "Resuming from: ${RESUME_CKPT}"
fi

FRAME_DIR="${FRAME_DIR:-/mnt/home/npant/ceph/datasets/spatialvlm/frames/prealign}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train.py \
    --config configs/train_prealign.yaml \
    --frame-dir "${FRAME_DIR}" \
    --device cuda --dtype bf16 \
    --batch-size "${BATCH_SIZE}" \
    --wandb-project spatialvlm \
    --wandb-mode "${WANDB_MODE}" \
    --checkpoint-dir /mnt/home/npant/ceph/elv/checkpoints \
    --save-every-epoch \
    --log-every 5 \
    ${RESUME_FLAG}

echo "Job $SLURM_JOB_ID finished — $(date)"
