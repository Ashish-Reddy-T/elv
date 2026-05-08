#!/bin/bash
#SBATCH --job-name=svlm-psi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=np3006@nyu.edu
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

# =================================================================
# Spatial grounding permutation test (PSI)
#
# Usage:
#   sbatch --export="CKPT=<path>,LIMIT=100" scripts/hpc/run_permutation.slurm
#   sbatch --export="CKPT=<path>,LIMIT=100,NUM_PERMS=3" scripts/hpc/run_permutation.slurm
# =================================================================
set -euo pipefail

cd /mnt/home/npant/nyu_classes/elv
mkdir -p artifacts/slurm artifacts/eval

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

source /mnt/home/npant/nyu_classes/elv/elvvenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HABITAT_PYTHON=/mnt/home/npant/ceph/datasets/habitat/habitat-venv/bin/python

CKPT="${CKPT:?ERROR: must set CKPT}"
SCENE_DIR=/mnt/home/npant/ceph/datasets/habitat/habitat-sim/data/scene_datasets/mp3d
LIMIT="${LIMIT:-100}"
NUM_PERMS="${NUM_PERMS:-3}"
SPLIT="${SPLIT:-val_seen}"

CKPT_TAG=$(basename "${CKPT}" .pt)
OUT="artifacts/eval/psi_${CKPT_TAG}_${SPLIT}.json"

echo "Checkpoint: ${CKPT}"
echo "Split: ${SPLIT}  Limit: ${LIMIT}  Permutations: ${NUM_PERMS}"
echo "Output: ${OUT}"
echo ""

python scripts/eval_permutation.py \
    --checkpoint "${CKPT}" \
    --split "${SPLIT}" \
    --scene-dir "${SCENE_DIR}" \
    --out "${OUT}" \
    --limit "${LIMIT}" \
    --num-permutations "${NUM_PERMS}" \
    --dtype bf16

echo ""
echo "Job $SLURM_JOB_ID finished — $(date)"
