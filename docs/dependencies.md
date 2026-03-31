# Dependency Policy (Global Stack)

This repository uses a single pinned Python stack in `pyproject.toml` for reproducibility.

## Core pinned versions

- `torch==2.5.1`
- `transformers==4.57.6`
- `peft==0.15.2`
- `trl==0.18.1`
- `accelerate==1.6.0`
- `einops==0.8.1`
- `xformers==0.0.29.post1`
- `omegaconf==2.3.0`
- `wandb==0.20.1`
- `pytest==8.3.5`
- `numpy==2.3.5`
- `opt_einsum` pinned to commit `1a984b7b75f3e532e7129f6aa13f7ddc3da66e10`

## GATr install rule

`GATr` is used from `REPOS/geometric-algebra-transformer` in editable mode:

```bash
pip install -e REPOS/geometric-algebra-transformer --no-deps
```

Rationale:
- The upstream `GATr` package metadata pins `numpy<1.25`, which conflicts with the global stack.
- The code path used in this repo is validated with the pinned stack above plus:
  - cached einsum planning disabled at runtime (`enable_cached_einsum(False)`).

## Environment isolation

Use a dedicated virtual environment for this repo.
The host machine may have unrelated packages (LLM tooling, TensorFlow, vision stacks) with
conflicting constraints, which can make `pip check` noisy even when SpatialVLM itself is valid.

## Habitat packages

Habitat packages are optional until Stage 6 data pipeline work:
- `habitat-sim>=0.3`
- `habitat-lab>=0.3`
