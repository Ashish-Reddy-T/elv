# SpatialVLM

**Recovering Destroyed Spatial Intelligence in Foundation Models for Indoor Navigation**

## Core Thesis

Foundation models destroy spatial information at five architectural stages. We fix each with a
targeted module and prove each fix works via ablation. The permutation test is the paper's
smoking gun: shuffling all vision tokens barely hurts SOTA VLMs (~2% drop) but is catastrophic
for ours (>15% drop), proving spatial structure is finally being used.

---

## Architecture Overview (5 Stages)

```
Stage 1: Dual Vision Encoding
  SigLIP2-SO400M/16 @ 384px  → [B, 576, 4096]   (semantic)
  DINOv2-L/14 @ 518px        → [B, 1369, 4096]  (structural, NO pooling)

Stage 2: Geometric Branch (parallel to Stage 1)
  GT Depth @ 518x518 → backproject → 15th-percentile aggregation → [B, 1369, 3]
  GATr (8 equivariant blocks, PGA)                               → [B, 1369, 4096]
  GridCellRoPE3D (4 tetra dirs × 8 freqs × sin/cos)              → [B, 1369, 64]  (stored)

Stage 3: Fusion
  SVA cross-attention: 576 queries attend 3314 KV tokens          → [B, 576, 4096]
  RMS norm matching: scale to text-token magnitude                 → [B, 576, 4096]

Stage 4: LLM Backbone
  Qwen3-VL-8B with LoRA rank-32 on all 36 layers
  Gated cross-attention at layers {4,8,12,16,20,24,28,32,36}
  Position routing: text → M-RoPE, spatial → GridCellRoPE3D

Stage 5: Training
  Pre-alignment → SFT → GRPO (with dense spatial rewards + curriculum)
```

## Backbone: Qwen3-VL-8B-Instruct

```
HuggingFace ID:       Qwen/Qwen3-VL-8B-Instruct
hidden_size:          4096
num_hidden_layers:    36
num_attention_heads:  32
num_key_value_heads:  8   (GQA 4:1)
head_dim:             128
intermediate_size:    12288
mrope_section:        [24, 20, 20]  → 64 rotary pairs total
Vision encoder:       SigLIP2-SO400M (1152 hidden, 27 layers, patch 16)
```

## Trainable Parameter Budget

```
Component                               Params      Trainable
SigLIP2-SO400M encoder                  ~543M       Frozen
DINOv2-L encoder                        ~307M       Frozen
GATr (8 equivariant blocks)              ~12M       Yes (Stage 2+)
SigLIP MLP projector (3456→4096)         ~31M       Yes (Stage 1+)
DINOv2 MLP projector (3072→4096)         ~29M       Yes (Stage 1+)
GATr MLP projector (48→4096)             ~17M       Yes (Stage 1+)
SVA queries + cross-attn (2 layers)      ~86M       Yes (Stage 2+)
Gated cross-attn (9 layers)             ~378M       Yes (Stage 2+)
Qwen3-VL-8B backbone                   ~8,000M      Frozen
Qwen3 LoRA rank-32                       ~31M       Yes (Stage 2+)
────────────────────────────────────────────────────────────
Total trainable:                        ~584M   (6.6%)
```

---

## Repository Layout

```
ablations/
├── CLAUDE.md              # this file
├── TODO.md                # implementation progress tracker
├── pyproject.toml         # package config
├── Makefile               # test/lint/format shortcuts
│
├── src/spatialvlm/        # main package
│   ├── config/            # model + training configs (dataclasses)
│   ├── encoders/          # Stage 1: SigLIP2, DINOv2, shared MLP projector
│   ├── geometry/          # Stage 2: backprojection, GATr wrapper, GridCellRoPE3D
│   ├── fusion/            # Stage 3: SVA, RMS norm matching, gated cross-attention
│   ├── backbone/          # Stage 4: Qwen3-VL wrapper, LoRA, position routing
│   ├── training/          # Stage 5: pre-alignment, SFT, GRPO, fDPO, rewards
│   ├── data/              # Habitat env wrapper, dataset loaders
│   ├── eval/              # benchmark runners, metrics, ablation orchestrator
│   └── utils/             # camera intrinsics, logging, tensor helpers
│
├── tests/                 # unit + integration tests
├── scripts/               # training/eval launch scripts
├── configs/               # YAML configs for training runs
│
├── docs/                  # existing: plan.md, architecture.md, critique.md
├── REPOS/                 # cloned reference repos (read-only)
├── notes/                 # research notes
└── papers/                # reference papers (PDFs)
```

## Reference Repos (under REPOS/)

| Repo                            | Purpose                                    | Key Files                                                                                         |
| ------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `EAGLE2`                        | Multi-encoder VLM (dual encoder pattern)   | `eagle/model/eagle_arch.py`, `multimodal_encoder/multi_backbone_channel_concatenation_encoder.py` |
| `cambrian`                      | SVA cross-attention fusion                 | `cambrian/model/vision_sampler.py`, `cambrian/model/cambrian_arch.py`                             |
| `geometric-algebra-transformer` | GATr equivariant transformer               | `gatr/nets/gatr.py`, `gatr/layers/gatr_block.py`, `gatr/primitives/`                              |
| `grid-cell-conformal-isometry`  | Grid cell theory (TF/JAX, for theory only) | `model.py`, `torus.py`                                                                            |
| `gridpe-2d`                     | GridPE 2D reference (extend to 3D)         | `VIT/models_gridpe.py`, `PCT/model_pct_grid.py`                                                   |
| `open_flamingo`                 | Gated cross-attention pattern              | `open_flamingo/src/flamingo.py`, `open_flamingo/src/flamingo_lm.py`                               |

---

## Implementation Guidelines

### Code Style

- Python 3.10+, type hints on all public functions.
- Lint with `ruff`. Format with `ruff format`. Config in pyproject.toml.
- 100-char line limit. Imports sorted by ruff (isort-compatible).
- Tensor shape comments on every forward() method: `# [B, 576, 4096]`.
- No wildcard imports. No mutable default arguments.

### Testing Strategy

- **Unit tests required** for all geometry/math modules (backprojection, GridCellRoPE3D,
  GATr wrapper, reward functions). These are high-stakes and easily breakable.
- Tests use `pytest`. Mark slow tests with `@pytest.mark.slow`.
- Test tensor shapes explicitly: assert output.shape == (B, 1369, 4096).
- Test equivariance properties: rotate input → check output transforms correctly.
- Test numerical edge cases: zero depth, NaN, very large coordinates.
- Run `make test` before any commit.

### Critical Numerical Values (DO NOT HALLUCINATE)

These are verified ground-truth values. Always cross-reference before using:

```python
# Qwen3-VL-8B
HIDDEN_SIZE = 4096
NUM_LAYERS = 36
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
MROPE_SECTION = [24, 20, 20]  # temporal, height, width
ROTARY_PAIRS = 64              # sum of mrope_section

# SigLIP2
SIGLIP_PATCH_SIZE = 16
SIGLIP_INPUT_RES = 384
SIGLIP_PATCHES = 576           # 24 * 24
SIGLIP_HIDDEN = 1152
SIGLIP_DEPTH = 27
SIGLIP_EXTRACT_LAYERS = [9, 18, 27]  # evenly-spaced thirds

# DINOv2-L
DINO_PATCH_SIZE = 14
DINO_INPUT_RES = 518
DINO_PATCHES = 1369            # 37 * 37
DINO_HIDDEN = 1024
DINO_DEPTH = 24
DINO_EXTRACT_LAYERS = [8, 16, 24]

# GridCellRoPE3D
TETRA_DIRS = 4                 # tetrahedral directions
NUM_FREQS = 8                  # golden-ratio spaced
GOLDEN_RATIO = 1.618033988749895
BASE_FREQ = 10.0               # f_k = 10 * phi^k
ROTARY_DIMS = 64               # 4 * 8 * 2, matches Qwen3's 64 pairs

# GATr
GATR_BLOCKS = 8
GATR_MV_CHANNELS = 16          # multivector channels
GATR_SCALAR_CHANNELS = 32
PGA_DIM = 16                   # projective geometric algebra basis dimension
GATR_INVARIANT_DIM = 48        # 32 scalars + 16 mv norms

# SVA
SVA_NUM_QUERIES = 576          # 24 * 24 grid
SVA_KV_TOKENS = 3314           # 576 + 1369 + 1369
SVA_LAYERS = 2

# Gated Cross-Attention
CROSS_ATTN_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36]  # 9 injection points

# Backprojection
DEPTH_PERCENTILE = 0.15        # 15th percentile foreground bias
DEPTH_MAP_RES = 518            # matches DINOv2 input

# Training
LORA_RANK = 32
LORA_ALPHA = 64
GRPO_GROUP_SIZE = 8
GRPO_CLIP_EPS = 0.2
GRPO_KL_BETA = 0.001
```

### Module Dependencies (build order)

```
Independent (build first, in parallel):
  1. src/spatialvlm/geometry/backproject.py      — pure math, no model deps
  2. src/spatialvlm/geometry/gridcell_rope3d.py  — pure math, no model deps
  3. src/spatialvlm/fusion/norm_matching.py       — pure math
  4. src/spatialvlm/training/rewards.py           — pure math + Habitat API
  5. src/spatialvlm/utils/camera.py               — pure math

Depends on reference repos:
  6. src/spatialvlm/encoders/siglip.py            — wraps HF SigLIP2
  7. src/spatialvlm/encoders/dinov2.py             — wraps HF DINOv2-L
  8. src/spatialvlm/geometry/gatr_wrapper.py       — wraps REPOS/geometric-algebra-transformer
  9. src/spatialvlm/fusion/sva.py                  — based on REPOS/cambrian vision_sampler.py
 10. src/spatialvlm/fusion/gated_cross_attn.py     — based on REPOS/open_flamingo flamingo.py

Depends on backbone:
 11. src/spatialvlm/backbone/qwen3_vl.py           — Qwen3-VL-8B + LoRA + monkey-patching
 12. src/spatialvlm/backbone/position_routing.py   — M-RoPE / GridCellRoPE3D dispatch

Integration (depends on all above):
 13. src/spatialvlm/training/prealign.py
 14. src/spatialvlm/training/sft.py
 15. src/spatialvlm/training/grpo.py
 16. src/spatialvlm/data/habitat_env.py
 17. src/spatialvlm/eval/benchmarks.py
```

### Known Risks & Gotchas

1. **GATr convergence**: hPGA-DP (2025) found GATr alone converges very slowly. Our config
   (8 blocks, 1369 tokens) is untested at this scale. Monitor loss curves early.
   Must use **improved PGA** with join bilinear layers (AISTATS 2024), not basic PGA.

2. **LoRA + positional encoding swap**: Evidence for LoRA adapting to new PE comes from
   multilingual MT, not VLMs. Test early with a minimal probe before full training.

3. **PEFT bug #2880**: LoRA gradients on ViT QKV modules are zero unless
   `requires_grad=True` is manually set on the target modules.

4. **Norm explosion**: Vision token norms are 10-100x text norms in LLaVA-style models.
   RMS norm matching MUST be applied before gated cross-attention injection.

5. **Vanishing advantages in GRPO**: As training progresses, more groups yield zero
   advantages. Implement Selective Sample Replay (SSR) as mitigation.

6. **Habitat depth at 518x518**: Must render depth at DINOv2's input resolution for
   pixel-perfect patch alignment. Default Habitat depth resolution is different.

### Commit Conventions

- Prefix: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `exp:` (for experiment results)
- Reference hypotheses: `feat: implement GridCellRoPE3D [H2b]`
- Reference stages: `feat(stage2): GATr wrapper with invariant extraction`

---

## Key Hypotheses (from plan.md)

Track these — every module exists to test one or more:

| ID  | Hypothesis                                           | Module  | Status |
| --- | ---------------------------------------------------- | ------- | ------ |
| H1a | DINOv2+SigLIP > either alone on spatial benchmarks   | Stage 1 | TODO   |
| H1b | Pre-alignment necessary for DINOv2                   | Stage 1 | TODO   |
| H1c | Multi-layer > final-layer extraction                 | Stage 1 | TODO   |
| H1d | Full 1369 DINOv2 tokens > pooled 576                 | Stage 1 | TODO   |
| H2a | GATr complementary to vision encoders                | Stage 2 | TODO   |
| H2b | GridCellRoPE3D > M-RoPE for spatial tokens           | Stage 2 | TODO   |
| H2c | GT depth vs Depth Anything V2 gap                    | Stage 2 | TODO   |
| H2d | Golden ratio > other scale ratios                    | Stage 2 | TODO   |
| H2e | 15th-pct > mean aggregation                          | Stage 2 | TODO   |
| H3a | Gated cross-attn > LLaVA concat by >5%               | Stage 3 | TODO   |
| H3b | RMS norm matching adds measurable benefit            | Stage 3 | TODO   |
| H3c | Permutation test: >15% drop (ours) vs <3% (baseline) | Stage 3 | TODO   |
| H3d | Typed attention bias helps stability                 | Stage 3 | TODO   |
| H3e | 3314-token KV > 1728-token KV                        | Stage 3 | TODO   |
| H4a | GridCellRoPE3D improves navigation SR                | Stage 4 | TODO   |
| H4b | CoT decoding benefits more from GRPO                 | Stage 4 | TODO   |
| H5a | SFT+GRPO > SFT-only                                  | Stage 5 | TODO   |
| H5b | fDPO > GRPO on spatial quality/quantity              | Stage 5 | TODO   |
| H5c | Dense rewards > sparse rewards                       | Stage 5 | TODO   |
| H5d | Consistency reward prevents hacking                  | Stage 5 | TODO   |

---

## Docs Reference

- `docs/plan.md` — full research blueprint (stages, hypotheses, compute budget)
- `docs/architecture.md` — dimensional analysis of every tensor in the pipeline
- `docs/critique.md` — feasibility analysis and competitive landscape
- `notes/points.txt` — research notes (LoRA feasibility concern, typed attention bias idea)
