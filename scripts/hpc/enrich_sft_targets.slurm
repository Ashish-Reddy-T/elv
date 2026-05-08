#!/bin/bash
#SBATCH --job-name=svlm-enrich
#SBATCH --partition=gen
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=np3006@nyu.edu
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

# =================================================================
# Enrich SFT frame targets in-place.
#
# Replaces thin targets (~44 tokens) with richer ones (~100 tokens)
# that include step progress and the full instruction string.
# CPU-only; rewrites target field in each .pt file, all else untouched.
#
# Usage:
#   sbatch scripts/hpc/enrich_sft_targets.slurm
#
#   # Dry-run (prints 5 samples, writes nothing):
#   sbatch --export=DRY_RUN=1 scripts/hpc/enrich_sft_targets.slurm
# =================================================================
set -euo pipefail

cd /mnt/home/npant/nyu_classes/elv
mkdir -p artifacts/slurm

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"

source /mnt/home/npant/nyu_classes/elv/elvvenv/bin/activate

FRAME_DIR="${FRAME_DIR:-/mnt/home/npant/ceph/datasets/spatialvlm/frames/sft}"

DRY_RUN_FLAG=""
if [ -n "${DRY_RUN:-}" ]; then
    DRY_RUN_FLAG="--dry-run"
    echo "Dry-run mode: no files will be written"
fi

echo "Frame dir: ${FRAME_DIR}"

python scripts/enrich_sft_targets.py \
    --frame-dir "${FRAME_DIR}" \
    ${DRY_RUN_FLAG}

echo "Job $SLURM_JOB_ID finished — $(date)"
