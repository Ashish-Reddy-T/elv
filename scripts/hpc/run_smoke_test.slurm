#!/bin/bash
#SBATCH --job-name=svlm-smoke
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

# =================================================================
# Quick GPU smoke test — verify forward pass works on HPC.
# Run this first before submitting real training jobs.
#
# Usage:
#   sbatch scripts/hpc/run_smoke_test.slurm
# =================================================================
set -euo pipefail

mkdir -p artifacts/slurm

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

OVERLAY=/scratch/$USER/spatialvlm_env.ext3
IMAGE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${IMAGE}" \
    /bin/bash -c "
source /ext3/env.sh
cd $(pwd)

echo '=== Stage-by-stage smoke test (mock, CPU) ==='
python scripts/smoke_test.py --mock --grad-check

echo ''
echo '=== Synthetic frame pipeline test (GPU, 8 frames) ==='
python scripts/generate_synthetic_frames.py --output /tmp/svlm_smoke_frames/ --num-frames 8

python scripts/train.py \
    --config configs/train_prealign.yaml \
    --frame-dir /tmp/svlm_smoke_frames/ \
    --device cuda --dtype bf16 \
    --batch-size 2 --limit 4 \
    --log-every 1 \
    --wandb-mode disabled

echo ''
echo '=== All smoke tests passed ==='
"

echo "Job $SLURM_JOB_ID finished — $(date)"
