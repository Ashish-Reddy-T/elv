#!/bin/bash
#SBATCH --job-name=svlm-grpo-gentest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=np3006@nyu.edu
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

# =================================================================
# Regression test for Bug 4 (GC disable around model.generate).
# Single GPU, ~10 min. Must show "ALL CHECKS PASSED" to proceed.
#
# Usage:
#   sbatch scripts/hpc/run_grpo_generate_test.slurm
# =================================================================
set -euo pipefail

cd /mnt/home/npant/nyu_classes/elv
mkdir -p artifacts/slurm

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

source /mnt/home/npant/nyu_classes/elv/elvvenv/bin/activate

CKPT=${CKPT:-/mnt/home/npant/ceph/elv/checkpoints/sft_epoch2_step668.pt}
echo "Testing with checkpoint: ${CKPT}"

python scripts/test_grpo_generate.py --ckpt "${CKPT}" --device cuda

echo "Job $SLURM_JOB_ID finished — $(date)"
python scripts/test_grpo_generate.py --ckpt /mnt/home/npant/ceph/elv/checkpoints/sft-v1/sft_epoch2_step668.pt --device cuda


sbatch --job-name=svlm-grpo-v1-badpa --export="RESUME_CKPT=/mnt/home/npant/ceph/elv/checkpoints/sft-v1/sft_epoch2_step668.pt,CKPT_DIR=/mnt/home/npant/ceph/elv/checkpoints/grpo-v1-bad-prealign" scripts/hpc/run_grpo.slurm


sbatch --job-name=svlm-render-sft-turns --time=12:00:00 --export="STAGE=sft,FRAMES_PER_EPISODE=0,OUT_DIR=/mnt/home/npant/ceph/datasets/spatialvlm/frames/sft-with-turns" scripts/hpc/render_frames.slurm