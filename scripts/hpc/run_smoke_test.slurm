#!/bin/bash
#SBATCH --job-name=svlm-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=smoke.out
#SBATCH --error=smoke.err

# =================================================================
# Quick GPU smoke test — verify forward pass works on HPC.
# Run this first before submitting real training jobs.
#
# Usage:
#   sbatch scripts/hpc/run_smoke_test.slurm
# =================================================================
set -euo pipefail

mkdir -p artifacts/slurm

source /mnt/home/npant/nyu_classes/elv/elvvenv/bin/activate

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

echo '=== Stage-by-stage smoke test (mock, CPU) ==='
python scripts/smoke_test.py --mock --grad-check

echo ''
echo '=== Synthetic frame pipeline test (GPU, 8 frames) ==='
python scripts/generate_synthetic_frames.py --output /tmp/svlm_smoke_frames/ --num-frames 8

torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/train_prealign.yaml \
    --frame-dir /tmp/svlm_smoke_frames/ \
    --device cuda --dtype bf16 \
    --batch-size 1 --limit 4 \
    --log-every 1 \
    --wandb-mode disabled

echo ''
echo '=== All smoke tests passed ==='


echo "Job $SLURM_JOB_ID finished — $(date)"
