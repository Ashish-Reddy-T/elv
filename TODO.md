# SpatialVLM — Implementation Progress

> Auto-generated checklist. Check items off as completed.

---

## Phase 0: Environment Setup
- [ ] Install Habitat 3.0 (habitat-sim + habitat-lab)
- [ ] Verify Qwen3-VL-8B loads with HuggingFace Transformers
- [ ] Verify DINOv2-L loads (facebook/dinov2-large)
- [x] Verify GATr repo installs (`pip install -e REPOS/geometric-algebra-transformer`)
- [ ] Set up wandb project for experiment tracking
- [ ] Verify GPU access + VRAM budget (~40GB for full model, ~18GB LoRA)

## Phase 1: Pure Math Modules (no model dependencies)
- [x] `src/spatialvlm/utils/camera.py` — camera intrinsics + projection helpers
- [x] `tests/test_camera.py`
- [x] `src/spatialvlm/geometry/backproject.py` — depth→3D with 15th-percentile aggregation
- [x] `tests/test_backproject.py` — verify shapes, edge cases (zero depth, NaN)
- [x] `src/spatialvlm/geometry/gridcell_rope3d.py` — tetrahedral Fourier basis RoPE
- [x] `tests/test_gridcell_rope3d.py` — verify dims=64, isotropy, frequency spacing
- [x] `src/spatialvlm/fusion/norm_matching.py` — RMS norm matching (zero params)
- [x] `tests/test_norm_matching.py`
- [ ] `src/spatialvlm/training/rewards.py` — R_format, R_progress, R_collision, R_goal, R_consistency
- [ ] `tests/test_rewards.py`

## Phase 2: Encoder Wrappers
- [x] `src/spatialvlm/config/model.py` — EncoderConfig, GeometryConfig, FusionConfig, BackboneConfig, SpatialVLMConfig
- [x] `src/spatialvlm/encoders/projector.py` — shared 2-layer MLP projector
- [x] `src/spatialvlm/encoders/siglip.py` — SigLIP2 multi-layer extraction {9,18,27} → [B, 576, 3456]
- [x] `src/spatialvlm/encoders/dinov2.py` — DINOv2-L multi-layer extraction {8,16,24} → [B, 1369, 3072]
- [x] `tests/test_encoders.py` — shape checks (fast + @slow), frozen param verification
- [x] `tests/conftest.py` — pytest slow marker registration

## Phase 3: Geometry Modules
- [x] `src/spatialvlm/geometry/gatr_wrapper.py` — GATr integration + invariant feature extraction
- [x] `tests/test_gatr_wrapper.py` — shape [B,1369,48]→[B,1369,4096], equivariance test
- [x] Verify improved PGA (join bilinear layers) is used, NOT basic PGA

## Phase 4: Fusion Modules
- [x] `src/spatialvlm/fusion/sva.py` — SVA: 576 queries, 3314 KV, 2 cross-attn layers
- [x] `tests/test_sva.py` — shape checks, typed attention bias (3x3 matrix)
- [x] `src/spatialvlm/fusion/gated_cross_attn.py` — tanh(alpha) gating, zero-init
- [x] `tests/test_gated_cross_attn.py` — verify gate=0 at init (pure passthrough)

## Phase 5: Backbone Integration
- [x] `src/spatialvlm/backbone/qwen3_vl.py` — Qwen3-VL-8B wrapper + LoRA setup
- [x] `src/spatialvlm/backbone/position_routing.py` — M-RoPE vs GridCellRoPE3D dispatch
- [x] `tests/test_qwen3_vl.py` — config introspection, LoRA path, PEFT #2880 workaround
- [x] `tests/test_position_routing.py`
- [x] Verify PEFT bug #2880 workaround (manual requires_grad=True on ViT QKV)

## Phase 6: Data Pipeline
- [x] `src/spatialvlm/data/habitat_env.py` — Habitat 3.0 wrapper (RGB + depth @ 518x518)
- [x] `src/spatialvlm/data/datasets.py` — R2R-CE, RxR-CE, SQA3D loaders
- [x] `src/spatialvlm/data/preprocessing.py` — image resize, depth normalization

## Phase 7: Training Pipeline
- [x] `src/spatialvlm/training/prealign.py` — Stage 1: projectors only (~77M), LLaVA-558K
- [x] `src/spatialvlm/training/sft.py` — Stage 2: full 584M trainable
- [x] `src/spatialvlm/training/grpo.py` — Stage 3: GRPO with curriculum
- [x] `src/spatialvlm/training/fdpo.py` — Stage 3 alt: fDPO comparison
- [x] `src/spatialvlm/training/curriculum.py` — staged reward weighting schedule
- [x] `configs/train_prealign.yaml`
- [x] `configs/train_sft.yaml`
- [x] `configs/train_grpo.yaml`

## Phase 8: Evaluation
- [x] `src/spatialvlm/eval/metrics.py` — SR, SPL, PSI, CMB
- [x] `src/spatialvlm/eval/permutation_test.py` — the smoking-gun diagnostic
- [x] `src/spatialvlm/eval/benchmarks.py` — What's Up, CV-Bench, VSI-Bench, VLN-CE, ObjectNav
- [x] `src/spatialvlm/eval/ablations.py` — ablation orchestrator (one-module-removed variants)
- [x] `configs/eval.yaml`

## Phase 9: Ablation Runs
- [ ] Full model (all modules)
- [ ] No GridCellRoPE3D (standard M-RoPE) [H2b]
- [ ] No GATr [H2a]
- [ ] SigLIP only [H1a]
- [ ] DINOv2 only [H1a]
- [ ] DINOv2 pooled to 576 [H1d, H3e]
- [ ] Concatenation fusion (no SVA/cross-attn) [H3a]
- [ ] No RMS norm matching [H3b]
- [ ] No typed attention bias [H3d]
- [ ] SFT only (no GRPO) [H5a]
- [ ] Permutation test [H3c]
- [ ] GT depth vs Depth Anything V2 [H2c]
- [ ] Mean vs 15th-pct aggregation [H2e]
- [ ] Scale ratio sweep (phi vs sqrt(e) vs pow2) [H2d]
- [ ] GRPO vs fDPO vs SFT-only [H5b]
- [ ] Dense vs sparse rewards [H5c]

## Phase 10: Paper
- [ ] LaTeX template (NeurIPS 2025/ICLR 2026 format)
- [ ] Figures: architecture diagram, ablation tables, attention visualizations
- [ ] Main results table
- [ ] Ablation results table
- [ ] Permutation test figure (the money plot)
- [ ] Writing: intro, related work, method, experiments, conclusion
