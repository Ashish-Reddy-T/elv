#!/bin/bash
#SBATCH --job-name=spatialvlm-phase9
#SBATCH --output=artifacts/slurm/%x-%A_%a.out
#SBATCH --error=artifacts/slurm/%x-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-15

set -euo pipefail

mkdir -p artifacts/slurm artifacts/phase9/runs

RUN_IDS=(
  "full-model"
  "no-gridcell-rope3d"
  "no-gatr"
  "siglip-only"
  "dinov2-only"
  "dinov2-pooled-576"
  "concat-fusion"
  "no-rms-norm-matching"
  "no-typed-attn-bias"
  "sft-only"
  "permutation-test"
  "gt-depth-vs-depth-anything-v2"
  "mean-vs-15pct-aggregation"
  "scale-ratio-sweep"
  "grpo-vs-fdpo-vs-sft"
  "dense-vs-sparse-rewards"
)

RUN_ID="${RUN_IDS[$SLURM_ARRAY_TASK_ID]}"
export RUN_ID
echo "Running Phase 9 job for: ${RUN_ID}"

# Replace the command below with your actual training/eval launcher.
# Keep the JSON schema compatible with scripts/run_phase9_ablations.py expectations.
python - <<'PY'
import json
import os
from pathlib import Path

run_id = os.environ["RUN_ID"]
out = Path("artifacts/phase9/runs") / f"{run_id}.json"
payload = {
    "status": "pending_execution",
    "score": None,
    "delta_vs_full_model": None,
    "notes": "Replace placeholder with real benchmark output from NYU HPC job launcher."
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Wrote placeholder: {out}")
PY
