# SpatialVLM

Recovering destroyed spatial intelligence in foundation models for indoor navigation.

## Architecture (5 Stages)

```
Stage 1: SigLIP2-SO400M/16 @ 384px → [B, 576, 4096] + DINOv2-L/14 @ 518px → [B, 1369, 4096]
Stage 2: GT Depth → backproject+15%ile → [B, 1369, 3] → GATr (8 blocks) → [B, 1369, 4096]
         IcosahedralRoPE3D (6 dirs × 8 freqs × sin/cos) → [B, 1369, 96]
Stage 3: SVA cross-attn: 1369 DINOv2-based queries × 3314 KV → [B, 1369, 4096]
         RMS norm matching → [B, 1369, 4096]. DeepStack replaces gated cross-attn (0 params)
Stage 4: Qwen3-VL-8B + LoRA rank-32. RoPE monkey-patch: spatial→Icosahedral, text→M-RoPE
Stage 5: Pre-alignment (77M) → SFT (206M) → GRPO (dense rewards + curriculum)
```

Backbone: `Qwen/Qwen3-VL-8B-Instruct` — hidden=4096, 36 layers, 32 heads, 8 KV heads, head_dim=128.
Trainable: ~206M (2.3%). Frozen: SigLIP2, DINOv2, Qwen3-VL backbone.

## Critical Rules

### Numerical Values — VERIFY AT RUNTIME

Architecture constants for pre-trained models MUST be introspected from the loaded model,
not hardcoded. Use `model.config.hidden_size` not `4096`.

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
# Read ALL values from cfg, not from memory or this file
```

### Device Policy

**NEVER hardcode `cuda`**. Always use a `device` parameter. Tensors created inside modules
must inherit device from input tensors or an explicit device arg.

### Scope: GT Depth Only

Primary system uses **Habitat GT depth exclusively**. No real-image benchmarks in core eval.
Depth Anything V2 is a single diagnostic ablation (H2c), deprioritized.

## Code Style

- Python 3.10+, type hints on all public functions
- `ruff` for lint + format. 100-char line limit. Config in pyproject.toml
- Tensor shape comments on every `forward()`: `# [B, 576, 4096]`
- Unit tests required for all geometry/math modules. `pytest`, `@pytest.mark.slow` for GPU tests
- `make test` before any commit

## Commit Conventions

Prefix: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `exp:`
Reference hypotheses: `feat: implement GridCellRoPE3D [H2b]`

## Known Risks

1. **GATr convergence**: Must use improved PGA with join bilinears (AISTATS 2024), not basic PGA
2. **PEFT bug #2880**: LoRA gradients zero unless `requires_grad=True` manually set on ViT QKV
3. **Norm explosion**: Vision tokens 10-100x text norms. RMS norm matching MUST be applied first
4. **Vanishing advantages**: Selective Sample Replay (SSR) mitigates in GRPO
5. **Habitat depth**: Must render at 518x518 for DINOv2 pixel-perfect alignment

## Key Modules

| Module | File | Notes |
|--------|------|-------|
| SVA (1369 queries) | `fusion/sva.py` | `return_attention_stats=True` for diagnostic probe |
| Freeze groups | `training/sft.py` | `set_trainable_by_groups()`: siglip_proj, dino_proj, gatr, sva, lora |
| IcosahedralRoPE3D | `geometry/gridcell_rope3d.py` | 6 icosahedral dirs, e^(1/3) freqs, 96 dims → pad to 128 |
| RoPE monkey-patch | `backbone/rope_patch.py` | Replaces Qwen3 RoPE for spatial tokens |
| DeepStack | native Qwen3-VL | Replaces gated cross-attn (0 params) |
| GATr wrapper | `geometry/gatr_wrapper.py` | 8 blocks, PGA, 48-dim invariants → MLP → 4096 |
| Norm matching | `fusion/norm_matching.py` | EMA-tracked RMS scaling, 0 params |

## Deprecated Modules

- `fusion/gated_cross_attn.py` — replaced by DeepStack
- `backbone/position_routing.py` — replaced by `rope_patch.py`

## Docs

- `docs/NEXT_STEPS.md` — dataset plan, implementation gaps, execution order
- `docs/pipeline_complete.md` — exact end-to-end trace with tensor shapes and ablation IDs
- `docs/architecture.md` — dimensional analysis of every tensor
- `docs/final.md` — implementation report (what was built, why, what remains)
- `docs/plan.md` — research blueprint, hypotheses, compute budget
- `docs/critique.md` — feasibility analysis
- `TODO.md` — implementation progress tracker (update after completing each phase)

## Hypotheses

See `docs/plan.md` for full list. Key: H1a-H1d (encoding), H2a-H2f (geometry),
H3b-H3f (fusion), H4a-H4b (backbone), H5a-H5d (training). Gated cross-attn hypotheses
(H3a, H3d) removed — replaced by DeepStack. H2f added for icosahedral vs tetrahedral.
