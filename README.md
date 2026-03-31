# SpatialVLM (Ablations Repo)

Recovering destroyed spatial intelligence in foundation models for indoor navigation.

This repository is organized as a phase-based implementation of the SpatialVLM research plan.

## Current Status

Implemented and validated:
- Phases 1–8 code modules and tests
- Phase 9 run-matrix infrastructure (no fabricated benchmark scores)
- Phase 10 paper scaffold + table/figure asset pipeline
- Lint baseline cleanup (`make lint` passes)
- Unit test suite green (`make test-unit` passes)

Still pending in practice:
- Phase 0 runtime environment checks on NYU GPU nodes (Habitat + CUDA/VRAM validation)
- Actual Phase 9 benchmark executions and real ablation scores

## Repository Map

Core package:
- `src/spatialvlm/config/`: dataclass configs
- `src/spatialvlm/encoders/`: SigLIP2 / DINOv2 wrappers + projectors
- `src/spatialvlm/geometry/`: backprojection, GridCellRoPE3D, GATr wrapper
- `src/spatialvlm/fusion/`: SVA, gated cross-attention, norm matching
- `src/spatialvlm/backbone/`: Qwen3-VL wrapper + position routing
- `src/spatialvlm/training/`: prealign, SFT, GRPO, fDPO, curriculum, rewards
- `src/spatialvlm/eval/`: metrics, permutation test, benchmarks, ablations, Phase 9 helpers

Infrastructure:
- `configs/`: training/eval YAMLs
- `tests/`: fast unit tests (+ `@slow` marker where relevant)
- `scripts/`: orchestration utilities
- `scripts/hpc/`: Slurm templates for NYU cluster workflows
- `paper/`: LaTeX scaffold and generated tables/figure CSV
- `artifacts/`: generated outputs (phase reports, env checks)

Planning and logs:
- `AGENTS.md`: implementation rules and architecture constraints
- `TODO.md`: progress tracker
- `docs/sessionLogs/`: implementation log history

## Environment and Setup

Install package (editable):

```bash
pip install -e .
```

For GATr compatibility with this pinned stack:

```bash
pip install -e REPOS/geometric-algebra-transformer --no-deps
```

For Habitat (when running data/env pipeline on proper nodes):

```bash
pip install -e .[habitat]
```

## Commands You’ll Actually Use

Lint:

```bash
make lint
```

Fast test suite (default dev loop):

```bash
make test-unit
```

Full tests including slow:

```bash
make test
```

Targeted new modules:

```bash
pytest tests/test_rewards.py -v --tb=short
pytest tests/test_phase9.py tests/test_paper_assets.py -v --tb=short
```

## Interpreting Test/Lint Output

- `make lint` is pass/fail by `ruff` exit code.
- `make test-unit` is pass/fail by pytest exit code.
- Warnings from third-party libraries (for example `deepeval`, `pydantic`, or GATr/Torch deprecations)
  can appear even on successful runs. Treat them as non-blocking unless they become errors.
- Current validated local baseline: `make lint` passes and `make test-unit` reports `164 passed`.

## Phase 0 Checks (Local/HPC)

General environment check:

```bash
python scripts/check_phase0_env.py --output artifacts/phase0/env_check.json
```

Full model-load verification (heavier):

```bash
python scripts/check_phase0_env.py --load-models --strict
```

NYU HPC Slurm wrapper:

```bash
sbatch scripts/hpc/run_phase0_check.slurm
```

## wandb Integration

Quick wandb bootstrap/check:

```bash
python scripts/setup_wandb.py --project spatialvlm --entity <your_entity>
```

Offline mode:

```bash
python scripts/setup_wandb.py --project spatialvlm --mode offline
```

Output report:
- `artifacts/phase0/wandb_check.json`

## Phase 9 Execution (Ablations)

Materialize run matrix artifact:

```bash
python scripts/run_phase9_ablations.py \
  --config configs/eval.yaml \
  --output artifacts/phase9/ablation_results.json
```

NYU HPC array template (16 ablation slots):

```bash
sbatch scripts/hpc/run_phase9_ablation_array.slurm
```

## Paper Assets

Generate paper-ready tables/CSV from Phase 9 artifact:

```bash
python scripts/generate_paper_assets.py \
  --phase9-results artifacts/phase9/ablation_results.json \
  --paper-dir paper
```

Outputs:
- `paper/tables/main_results.tex`
- `paper/tables/ablation_results.tex`
- `paper/figures/permutation_curve.csv`

## What To Do Next

1. Run Phase 0 checks on NYU GPU nodes and update `TODO.md` checkboxes with real results.
2. Replace placeholder Phase 9 run outputs with real benchmark scores.
3. Regenerate paper assets from real Phase 9 results.
4. Expand paper sections from scaffold text to full submission content.

Detailed execution checklist is in:
- `docs/NEXT_STEPS.md`
