#!/bin/bash
#SBATCH --job-name=svlm-prealign
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

# =================================================================
# Stage 1: Pre-alignment (projectors only, ~77M trainable)
#
# Usage:
#   sbatch scripts/hpc/run_prealign.slurm
#
# Resume after preemption:
#   sbatch --export=RESUME_CKPT=checkpoints/prealign_epoch1_step500_preempted.pt \
#       scripts/hpc/run_prealign.slurm
# =================================================================
set -euo pipefail

mkdir -p artifacts/slurm checkpoints

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# --- Container + environment ---
OVERLAY=/scratch/$USER/spatialvlm_env.ext3
IMAGE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

if [ ! -f "$OVERLAY" ]; then
    echo "ERROR: Overlay not found at $OVERLAY. Run setup_env.sh first."
    exit 1
fi

# --- Build command ---
RESUME_FLAG=""
if [ -n "${RESUME_CKPT:-}" ]; then
    RESUME_FLAG="--resume ${RESUME_CKPT}"
    echo "Resuming from: ${RESUME_CKPT}"
fi

FRAME_DIR="${FRAME_DIR:-data/frames/prealign/}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WANDB_MODE="${WANDB_MODE:-online}"

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${IMAGE}" \
    /bin/bash -c "
source /ext3/env.sh
cd $(pwd)

python scripts/train.py \
    --config configs/train_prealign.yaml \
    --frame-dir ${FRAME_DIR} \
    --device cuda --dtype bf16 \
    --batch-size ${BATCH_SIZE} \
    --wandb-project spatialvlm \
    --wandb-mode ${WANDB_MODE} \
    --checkpoint-dir checkpoints \
    --save-every-epoch \
    --log-every 5 \
    ${RESUME_FLAG}
"

echo "Job $SLURM_JOB_ID finished — $(date)"
