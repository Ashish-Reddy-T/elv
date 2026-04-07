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
  SigLIP2-SO400M/16 @ 384px  → [B, 576, 4096]   (semantic, KV in SVA)
  DINOv2-L/14 @ 518px        → [B, 1369, 4096]  (structural, SVA QUERY BASE)

Stage 2: Geometric Branch (parallel to Stage 1)
  GT Depth @ 518x518 → backproject → 15th-percentile aggregation → [B, 1369, 3]
  GATr (8 equivariant blocks, PGA)                               → [B, 1369, 4096]
  IcosahedralRoPE3D (6 icosahedral dirs × 8 freqs × sin/cos)    → [B, 1369, 96]
  (NO position pooling — 1369 positions map 1:1 to SVA queries)

Stage 3: Fusion
  SVA cross-attention: 1369 queries attend 3314 KV tokens        → [B, 1369, 4096]
  RMS norm matching: scale to text-token magnitude                → [B, 1369, 4096]
  (Gated cross-attention REMOVED — DeepStack used instead)

Stage 4: LLM Backbone
  Qwen3-VL-8B with LoRA rank-32 on all 36 layers
  DeepStack (native Qwen3-VL): inject encoder features at early LLM layers
  RoPE monkey-patch: text → M-RoPE (64 pairs), spatial → Icosahedral (48+16 identity)

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
Qwen3-VL-8B backbone                   ~8,000M      Frozen
Qwen3 LoRA rank-32                       ~31M       Yes (Stage 2+)
────────────────────────────────────────────────────────────
Total trainable:                        ~206M   (~2.3%)
```

Note: Gated cross-attention (~378M) was removed in the icosahedral redesign. Qwen3-VL's
native DeepStack mechanism replaces it with zero additional trainable parameters.

---

## Repository Layout

```
ablations/
├── AGENTS.md              # this file (mirrors CLAUDE.md)
├── TODO.md                # implementation progress tracker
├── pyproject.toml         # package config
├── Makefile               # test/lint/format shortcuts
│
├── src/spatialvlm/        # main package
│   ├── config/            # model + training configs (dataclasses)
│   ├── encoders/          # Stage 1: SigLIP2, DINOv2, shared MLP projector
│   ├── geometry/          # Stage 2: backprojection, GATr wrapper, IcosahedralRoPE3D
│   ├── fusion/            # Stage 3: SVA, RMS norm matching
│   ├── backbone/          # Stage 4: Qwen3-VL wrapper, LoRA, RoPE monkey-patch
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
| `open_flamingo`                 | Gated cross-attention (DEPRECATED)         | `open_flamingo/src/flamingo.py` — replaced by DeepStack                                           |
| `Qwen3-VL`                      | Backbone finetune code + RoPE internals    | `qwen-vl-finetune/qwenvl/modeling_qwen3_vl.py`, `qwenvl/data/rope2d.py`                           |

---

## Implementation Guidelines

### Device Policy

- **NEVER hardcode `cuda`**. Always use a `device` parameter or resolve it at the top level (could be cuda, mps, cpu etc.):
- Pass `device` through constructors and `.to(device)` calls. This ensures compatibility
  with CPU-only debugging, Docker containers, and multi-GPU setups (where device = `cuda:1` etc).
- Tensors created inside modules (e.g. `torch.zeros(...)`) must inherit device from input
  tensors or an explicit device arg — never default to CPU silently.

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

### Critical Numerical Values — VERIFY AT RUNTIME

**WARNING: Architecture constants for pre-trained models (Qwen3-VL, SigLIP2, DINOv2) MUST be
introspected from the loaded model at runtime, not hardcoded from documentation or memory.**
Documentation can be wrong, model versions can differ, and LLMs (including Claude) hallucinate
numerical details. The values below are our _best current understanding_ but every one marked
with `# ⚠ VERIFY` must be confirmed by loading the model and reading its config.

**Verification pattern** (use this before trusting any constant):

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
print(cfg.to_dict())  # read ALL values from here, not from memory
```

```python
# Qwen3-VL-8B — ⚠ VERIFY all from AutoConfig.from_pretrained()
HIDDEN_SIZE = 4096              # ⚠ VERIFY: cfg.hidden_size
NUM_LAYERS = 36                 # ⚠ VERIFY: cfg.num_hidden_layers
NUM_HEADS = 32                  # ⚠ VERIFY: cfg.num_attention_heads
NUM_KV_HEADS = 8                # ⚠ VERIFY: cfg.num_key_value_heads
HEAD_DIM = 128                  # ⚠ VERIFY: cfg.hidden_size // cfg.num_attention_heads
MROPE_SECTION = [24, 20, 20]    # ⚠ VERIFY: cfg.rope_scaling.mrope_section
ROTARY_PAIRS = 64               # ⚠ VERIFY: sum(MROPE_SECTION)

# SigLIP2 — ⚠ VERIFY from Qwen3-VL's vision_config or standalone model
SIGLIP_PATCH_SIZE = 16          # ⚠ VERIFY: cfg.vision_config.patch_size
SIGLIP_INPUT_RES = 384          # ⚠ VERIFY: cfg.vision_config.image_size
SIGLIP_PATCHES = 576            # derived: (384/16)^2 — but verify image_size & patch_size first
SIGLIP_HIDDEN = 1152            # ⚠ VERIFY: cfg.vision_config.hidden_size
SIGLIP_DEPTH = 27               # ⚠ VERIFY: cfg.vision_config.num_hidden_layers
SIGLIP_EXTRACT_LAYERS = [9, 18, 27]  # ⚠ VERIFY: check deepstack config or model source

# DINOv2-L — ⚠ VERIFY from facebook/dinov2-large config
DINO_PATCH_SIZE = 14            # ⚠ VERIFY
DINO_INPUT_RES = 518            # our choice (518/14 = 37 exact), not model default
DINO_PATCHES = 1369             # derived: (518/14)^2
DINO_HIDDEN = 1024              # ⚠ VERIFY
DINO_DEPTH = 24                 # ⚠ VERIFY
DINO_EXTRACT_LAYERS = [8, 16, 24]  # our design choice (evenly-spaced thirds)

# IcosahedralRoPE3D — from GridPE paper (Li et al. AAAI 2025), extended to 3D
ICOSAHEDRAL_DIRS = 6            # 6 antipodal pairs from regular icosahedron vertices
NUM_FREQS = 8                   # e^(1/3)-spaced (optimal for 3D per economy principle)
FREQ_RATIO = 1.3956             # e^(1/3), optimal scale ratio for p=3 dimensions
BASE_FREQ = 10.0                # f_k = 10 * e^(k/3) (our design)
ROTARY_DIMS = 96                # 6 * 8 * 2, padded to 128 with 16 identity pairs (cos=1, sin=0)

# GATr — ⚠ VERIFY against REPOS/geometric-algebra-transformer defaults
GATR_BLOCKS = 8                 # our config choice
GATR_MV_CHANNELS = 16           # ⚠ VERIFY: check GATr default config
GATR_SCALAR_CHANNELS = 32       # ⚠ VERIFY: check GATr default config
PGA_DIM = 16                    # PGA basis dimension (mathematical constant)
GATR_INVARIANT_DIM = 48         # derived: GATR_SCALAR_CHANNELS + GATR_MV_CHANNELS

# SVA — derived from encoder outputs
SVA_NUM_QUERIES = 1369          # matches DINOv2 patch count (query base)
SVA_KV_TOKENS = 3314            # 576 + 1369 + 1369
SVA_LAYERS = 2

# DeepStack (native Qwen3-VL, replaces gated cross-attention)
# Injects encoder intermediate features at early LLM layers via residual addition
# Uses visual_pos_masks to identify spatial token positions
# Zero additional trainable parameters — uses Qwen3-VL's built-in mechanism

# Backprojection
DEPTH_PERCENTILE = 0.15         # 15th percentile foreground bias (our design)
DEPTH_MAP_RES = 518             # matches DINOv2 input (our choice)

# Training
LORA_RANK = 32
LORA_ALPHA = 64
GRPO_GROUP_SIZE = 8
GRPO_CLIP_EPS = 0.2
GRPO_KL_BETA = 0.001
```

**Rule: When implementing any module that uses a pre-trained model's dimensions, load the
config first and read from it. Never copy-paste numbers from this file into code. Use
`model.config.hidden_size` not `4096`.**

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

Depends on backbone:
 10. src/spatialvlm/backbone/qwen3_vl.py           — Qwen3-VL-8B + LoRA
 11. src/spatialvlm/backbone/rope_patch.py          — RoPE monkey-patch (icosahedral injection)

Integration (depends on all above):
 12. src/spatialvlm/training/prealign.py
 13. src/spatialvlm/training/sft.py
 14. src/spatialvlm/training/grpo.py
 15. src/spatialvlm/data/habitat_env.py
 16. src/spatialvlm/eval/benchmarks.py

Deprecated (kept for reference, not used):
  -  src/spatialvlm/fusion/gated_cross_attn.py     — replaced by DeepStack
  -  src/spatialvlm/backbone/position_routing.py   — replaced by rope_patch.py
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
   RMS norm matching MUST be applied before spatial tokens enter the LLM sequence.

5. **Vanishing advantages in GRPO**: As training progresses, more groups yield zero
   advantages. Implement Selective Sample Replay (SSR) as mitigation.

6. **Habitat depth at 518x518**: Must render depth at DINOv2's input resolution for
   pixel-perfect patch alignment. Default Habitat depth resolution is different.

### Scope: GT Depth Only (No Depth Anything V2)

The primary system uses **Habitat GT depth exclusively**. Depth Anything V2 is NOT part of
the main architecture — it exists only as a single diagnostic ablation row (H2c) to quantify
the sim-to-real transfer gap. This means:

- **No real-image benchmarks in the core evaluation.** Benchmarks like What's Up that use
  real photos (no GT depth available) are out-of-scope for the main results. They can be
  added as supplementary experiments later to see how the model generalizes, but they are
  NOT part of the primary evaluation pipeline.
- **Primary benchmarks are all Habitat/simulation-based**: VLN-CE (R2R, RxR), ObjectNav HM3D,
  SQA3D (ScanNet reconstructions with available depth), VSI-Bench, NavTrust.
- The Depth Anything V2 ablation (H2c) is implemented last, as a nice-to-have comparison.
  It is explicitly deprioritized relative to the core architecture and ablations.

### Commit Conventions

- Prefix: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `exp:` (for experiment results)
- Reference hypotheses: `feat: implement GridCellRoPE3D [H2b]`
- Reference stages: `feat(stage2): GATr wrapper with invariant extraction`

---

## Key Hypotheses (from plan.md)

Track these — every module exists to test one or more:

| ID  | Hypothesis                                              | Module  | Status |
| --- | ------------------------------------------------------- | ------- | ------ |
| H1a | DINOv2+SigLIP > either alone on spatial benchmarks      | Stage 1 | TODO   |
| H1b | Pre-alignment necessary for DINOv2                      | Stage 1 | TODO   |
| H1c | Multi-layer > final-layer extraction                    | Stage 1 | TODO   |
| H1d | 1369 DINOv2 queries (full res) > 576 (pooled)           | Stage 1 | TODO   |
| H2a | GATr complementary to vision encoders                   | Stage 2 | TODO   |
| H2b | IcosahedralRoPE3D > M-RoPE for spatial tokens           | Stage 2 | TODO   |
| H2c | GT depth vs Depth Anything V2 gap                       | Stage 2 | TODO   |
| H2d | e^(1/3) ratio (optimal) > golden ratio (2D approx)      | Stage 2 | TODO   |
| H2e | 15th-pct > mean aggregation                             | Stage 2 | TODO   |
| H2f | 6 icosahedral dirs > 4 tetrahedral dirs                 | Stage 2 | TODO   |
| H3b | RMS norm matching adds measurable benefit               | Stage 3 | TODO   |
| H3c | Permutation test: >15% drop (ours) vs <3% (baseline)    | Stage 3 | TODO   |
| H3e | 3314-token KV > 1728-token KV                           | Stage 3 | TODO   |
| H3f | DeepStack multi-layer injection > no injection          | Stage 3 | TODO   |
| H4a | IcosahedralRoPE3D improves navigation SR                | Stage 4 | TODO   |
| H4b | CoT decoding benefits more from GRPO                    | Stage 4 | TODO   |
| H5a | SFT+GRPO > SFT-only                                     | Stage 5 | TODO   |
| H5b | fDPO > GRPO on spatial quality/quantity                 | Stage 5 | TODO   |
| H5c | Dense rewards > sparse rewards                          | Stage 5 | TODO   |
| H5d | Consistency reward prevents hacking                     | Stage 5 | TODO   |

Note: H3a (gated cross-attn) and H3d (typed attention bias in cross-attn) removed —
gated cross-attention was replaced by DeepStack. H2f added for icosahedral vs tetrahedral.

---

## Docs Reference

- `docs/plan.md` — full research blueprint (stages, hypotheses, compute budget)
- `docs/architecture.md` — dimensional analysis of every tensor in the pipeline
- `docs/critique.md` — feasibility analysis and competitive landscape
- `notes/points.txt` — research notes (LoRA feasibility concern, typed attention bias idea)

Update `TODO.md` each time after completing a phase!
