# Next Steps (Execution Note)

This note summarizes what is implemented, what is still pending, and the exact commands to finish.

## Implemented

- Phases 1–8: code + tests complete.
- Phase 9: ablation run-matrix tooling complete (`scripts/run_phase9_ablations.py`).
- Phase 10: paper scaffold + asset generation complete.
- `training/rewards.py` + `tests/test_rewards.py` implemented and validated.
- Lint baseline clean.

## Pending (Real-World Execution)

### Phase 0 runtime verification on NYU HPC
- Habitat install/runtime on cluster node
- Qwen3-VL and DINOv2 full loading on cluster node
- wandb project login/config check
- GPU/VRAM validation for your target jobs

Run:

```bash
sbatch scripts/hpc/run_phase0_check.slurm
```

Artifact:
- `artifacts/phase0/env_check_hpc.json`

### Phase 9 real ablation runs
- Current ablation artifact is structural coverage and placeholders by design.
- Replace placeholders with real benchmark outputs from cluster jobs.

Run matrix artifact:

```bash
python scripts/run_phase9_ablations.py \
  --config configs/eval.yaml \
  --output artifacts/phase9/ablation_results.json
```

Parallel NYU job array template:

```bash
sbatch scripts/hpc/run_phase9_ablation_array.slurm
```

## If You Need wandb Immediately

```bash
python scripts/setup_wandb.py --project spatialvlm --entity <entity>
```

Result:
- `artifacts/phase0/wandb_check.json`

## Regenerate Paper Assets After Real Scores

```bash
python scripts/generate_paper_assets.py \
  --phase9-results artifacts/phase9/ablation_results.json \
  --paper-dir paper
```

## Validation Commands (before any PR merge)

```bash
make lint
make test-unit
```
