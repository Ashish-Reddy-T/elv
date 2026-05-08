#!/bin/bash
#SBATCH --job-name=svlm-render-traj
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=np3006@nyu.edu
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

# =================================================================
# Render ego-centric trajectory video for one episode.
#
# Usage (defaults render ep 588, GRPO-v2 step 400 — a known success):
#   sbatch scripts/hpc/render_trajectory.slurm
#
# Override checkpoint / episode / output:
#   sbatch --export="CKPT=<path>,EPISODE_ID=125,OUT=artifacts/eval/traj_ep125.mp4" \
#       scripts/hpc/render_trajectory.slurm
# =================================================================
set -euo pipefail

cd /mnt/home/npant/nyu_classes/elv
mkdir -p artifacts/slurm artifacts/eval

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

source /mnt/home/npant/nyu_classes/elv/elvvenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HABITAT_PYTHON=/mnt/home/npant/ceph/datasets/habitat/habitat-venv/bin/python

CKPT="${CKPT:-/mnt/home/npant/ceph/elv/checkpoints/grpo-v2/grpo_epoch1_step400.pt}"
EPISODE_ID="${EPISODE_ID:-588}"
OUT="${OUT:-artifacts/eval/traj_ep${EPISODE_ID}.mp4}"
SPLIT="${SPLIT:-val_seen}"
FPS="${FPS:-2}"

echo "Checkpoint:  ${CKPT}"
echo "Episode ID:  ${EPISODE_ID}"
echo "Output:      ${OUT}"

python scripts/render_trajectory.py \
    --checkpoint "${CKPT}" \
    --episode-id "${EPISODE_ID}" \
    --split "${SPLIT}" \
    --out "${OUT}" \
    --fps "${FPS}" \
    --dtype bf16

echo ""
echo "Job $SLURM_JOB_ID finished — $(date)"
