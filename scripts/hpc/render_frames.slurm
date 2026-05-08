#!/bin/bash
#SBATCH --job-name=svlm-render
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

# =================================================================
# Render and cache RGB + depth frames from Habitat for SpatialVLM.
#
# Renders 3 frames/episode from R2R-CE train split → ~50K prealign frames.
# Uses --skip-existing so the job is safely resumable after preemption.
#
# Usage:
#   # Pre-alignment frames:
#   sbatch scripts/hpc/render_frames.slurm
#
#   # SFT frames (all steps, no subsampling):
#   sbatch --export=STAGE=sft,FRAMES_PER_EPISODE=0 scripts/hpc/render_frames.slurm
#
#   # Specific split:
#   sbatch --export=SPLIT=val_seen scripts/hpc/render_frames.slurm
# =================================================================
set -euo pipefail

mkdir -p artifacts/slurm

HABITAT_PYTHON=/mnt/home/npant/ceph/datasets/habitat/habitat-venv/bin/python
REPO_DIR=/mnt/home/npant/nyu_classes/elv
SCENE_DIR=/mnt/home/npant/ceph/datasets/habitat/habitat-sim/data/scene_datasets/mp3d
EPISODE_DIR=${REPO_DIR}/data/episodes/r2r

# Configurable via --export
STAGE=${STAGE:-prealign}
SPLIT=${SPLIT:-train}
FRAMES_PER_EPISODE=${FRAMES_PER_EPISODE:-3}  # 0 = render all steps (for SFT)

# Output goes to ceph (frames are ~150GB for prealign, ~900GB for SFT).
# Override via --export="OUT_DIR=..." to write to a different directory.
OUT_DIR=${OUT_DIR:-/mnt/home/npant/ceph/datasets/spatialvlm/frames/${STAGE}}
mkdir -p "${OUT_DIR}"


echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Stage: ${STAGE} | Split: ${SPLIT} | Frames/episode: ${FRAMES_PER_EPISODE}"
echo "Output: ${OUT_DIR}"
echo ""

# Build optional flags
EXTRA_FLAGS=""
if [ "${FRAMES_PER_EPISODE}" -gt 0 ] 2>/dev/null; then
    EXTRA_FLAGS="--frames-per-episode ${FRAMES_PER_EPISODE}"
fi

EPISODE_JSON="${EPISODE_DIR}/${SPLIT}.json.gz"
if [ ! -f "${EPISODE_JSON}" ]; then
    # val_seen / val_unseen stored under their own names
    EPISODE_JSON="${EPISODE_DIR}/${SPLIT}/${SPLIT}.json.gz"
fi
if [ ! -f "${EPISODE_JSON}" ]; then
    echo "ERROR: Episode JSON not found at ${EPISODE_JSON}"
    exit 1
fi

${HABITAT_PYTHON} "${REPO_DIR}/scripts/render_training_frames.py" \
    --dataset r2r-ce \
    --episode-json "${EPISODE_JSON}" \
    --scene-dir "${SCENE_DIR}" \
    --out-dir "${OUT_DIR}" \
    --stage "${STAGE}" \
    --skip-existing \
    ${EXTRA_FLAGS}

echo ""
echo "Done — $(date)"
echo "Frames written: $(ls ${OUT_DIR}/*.pt 2>/dev/null | wc -l)"
