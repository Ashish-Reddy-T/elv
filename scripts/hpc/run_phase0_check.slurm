#!/bin/bash
#SBATCH --job-name=spatialvlm-phase0-check
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

mkdir -p artifacts/slurm artifacts/phase0

# Expected: run from repo root. Activate your NYU HPC environment before submission if needed.
# Example:
#   sbatch scripts/hpc/run_phase0_check.slurm
# Optional strict mode:
#   sbatch --export=CHECK_STRICT=1 scripts/hpc/run_phase0_check.slurm

STRICT_FLAG=""
if [[ "${CHECK_STRICT:-0}" == "1" ]]; then
  STRICT_FLAG="--strict"
fi

python scripts/check_phase0_env.py \
  --load-models \
  --output artifacts/phase0/env_check_hpc.json \
  ${STRICT_FLAG}
