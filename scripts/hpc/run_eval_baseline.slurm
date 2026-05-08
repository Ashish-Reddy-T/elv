#!/bin/bash
#SBATCH --job-name=svlm-baseline
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
# Vanilla Qwen3-VL-8B baseline: RGB + instruction, no spatial modules
#
# Usage:
#   sbatch scripts/hpc/run_eval_baseline.slurm
#
#   # Single split or episode limit:
#   sbatch --export="SPLIT=val_seen,LIMIT=50" scripts/hpc/run_eval_baseline.slurm
# =================================================================
set -euo pipefail

cd /mnt/home/npant/nyu_classes/elv
mkdir -p artifacts/slurm artifacts/eval

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

source /mnt/home/npant/nyu_classes/elv/elvvenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HABITAT_PYTHON=/mnt/home/npant/ceph/datasets/habitat/habitat-venv/bin/python

SCENE_DIR=/mnt/home/npant/ceph/datasets/habitat/habitat-sim/data/scene_datasets/mp3d

LIMIT_FLAG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
    echo "Episode limit: ${LIMIT}"
fi

SPLIT="${SPLIT:-}"

run_split() {
    local split="$1"
    local out="artifacts/eval/baseline_vlm_${split}.json"
    echo ""
    echo "=== Baseline eval ${split} ==="
    python scripts/eval_baseline_vlm.py \
        --split "${split}" \
        --scene-dir "${SCENE_DIR}" \
        --out "${out}" \
        --dtype bf16 \
        ${LIMIT_FLAG}
    echo "=== ${split} done — $(date) ==="
}

if [ -n "${SPLIT}" ]; then
    run_split "${SPLIT}"
else
    run_split val_seen
    run_split val_unseen
fi

echo ""
echo "Job $SLURM_JOB_ID finished — $(date)"
