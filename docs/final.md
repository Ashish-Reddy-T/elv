# SpatialVLM: Implementation Report

## What Has Been Built, Why It Was Built That Way, and What Remains

**Date**: 2026-03-31
**Test status**: 184 passed, 0 failed, 14 slow-marked deselected
**Codebase**: 25 source modules, 26 test files, 4 config files

---

## 0. The System At a Glance

SpatialVLM is a 5-stage architecture that recovers destroyed spatial intelligence in vision-language models. Every module exists to fix a specific architectural failure point where standard VLMs silently discard 3D spatial information. The implementation is complete through all code modules (Stages 1-5, data pipeline, evaluation infrastructure). What remains is environment setup, training runs, and ablation experiments.

```
RGB frame ──→ SigLIP2-SO400M/16 ──→ [B, 576, 4096]  (semantic)        ─────────┐
         ──→ DINOv2-L/14        ──→ [B, 1369, 4096] (structural)      --───────┤
                                                                               ├──→ SVA ──→ [B, 576, 4096]
GT Depth ──→ Backproject+15%ile ──→ [B, 1369, 3] ──→ GATr ──→ [B, 1369, 4096] ─┘              │
                                                      │                                  RMS Norm Match
                                                pool 37×37 → 24×24                            │
                                                GridCellRoPE3D ──→ [B, 576, 64]          [B, 576, 4096]
                                                                         │                    │
                                                                         └─── Qwen3-VL-8B ←───┘
                                                                            + LoRA rank-32
                                                                            + 9× Gated Cross-Attn (GQA)
                                                                            + Position Routing
                                                                                  │
                                                                            Action + Reasoning
```

Trainable: ~583M parameters (6.6% of ~8,890M total). Frozen: SigLIP2, DINOv2, Qwen3-VL backbone.

---

## 1. Stage 1 -- Dual Vision Encoding

### What was implemented

**SigLIP2 Encoder** (`src/spatialvlm/encoders/siglip.py`)

Wraps `google/siglip2-so400m-patch16-naflex` from HuggingFace. Multi-layer feature extraction at layers {9, 18, 27} via forward hooks on the encoder's transformer blocks. Layer identification uses `_find_encoder_layers()` which walks the model's module tree to find the correct layer objects regardless of HuggingFace's internal naming conventions.

- Input: `[B, 384, 384, 3]` -- 384/16 = 24.0 exactly, zero edge artifacts
- Extracted: 3 layers channel-concatenated to `[B, 576, 3456]`
- Projected via MLP to `[B, 576, 4096]`

**DINOv2 Encoder** (`src/spatialvlm/encoders/dinov2.py`)

Wraps `facebook/dinov2-large`. Same multi-layer hook pattern at layers {8, 16, 24}. CLS token is explicitly stripped (DINOv2 prepends a CLS token; we keep only the 1369 spatial patch tokens). **No spatial pooling** -- the full 37x37 grid is preserved.

- Input: `[B, 518, 518, 3]` -- 518/14 = 37.0 exactly, zero edge artifacts
- Extracted: 3 layers channel-concatenated to `[B, 1369, 3072]`
- Projected via MLP to `[B, 1369, 4096]`

**MLP Projector** (`src/spatialvlm/encoders/projector.py`)

Shared 2-layer MLP pattern: `Linear(in_dim, out_dim) -> GELU -> Linear(out_dim, out_dim)`. Used for all three encoder outputs (SigLIP, DINOv2, GATr). ~31M params for SigLIP, ~29M for DINOv2, ~17M for GATr.

### Why this design

**Two encoders, not one**: SigLIP provides semantic, language-aligned features (trained with text supervision). DINOv2 provides structural, spatially-aware features (self-supervised, no text). El Banani et al. (CVPR 2024) showed CLIP features do NOT encode depth -- DINOv2 dramatically outperforms on depth, surface normals, and correspondence. Eagle (NVIDIA, ICLR 2025) confirmed: dual encoding with pre-alignment outperforms either single encoder on 23/27 benchmarks.

**Multi-layer extraction (3 layers per encoder)**: Following Qwen3-VL's own deepstack design which empirically validated 3 intermediate layers. Three evenly-spaced layers capture the full representational hierarchy (low-level spatial -> mid-level structural -> high-level semantic). Both projectors upsample (3456->4096, 3072->4096), which is an easier learning problem than compression.

**No DINOv2 pooling**: Preserving full 1369-token resolution is a testable hypothesis (H1d). Downsampling to 576 would discard the fine-grained structural detail that is the reason for using DINOv2 alongside SigLIP.

**NaFlex variant of SigLIP2**: The patch16 NaFlex variant yields exactly 576 tokens at 384px. The original patch14 variant produces 384/14 = 27.43 patches/side -- a non-integer requiring cropping or padding, both of which discard or fabricate spatial information at image boundaries.

### Hypotheses tested

- **H1a**: DINOv2 + SigLIP outperforms either alone on spatial benchmarks
- **H1b**: Pre-alignment is necessary for DINOv2
- **H1c**: Multi-layer > final-layer extraction
- **H1d**: Full 1369 DINOv2 tokens > pooled 576

---

## 2. Stage 2 -- Geometric Branch

### What was implemented

**Depth Backprojection** (`src/spatialvlm/geometry/backproject.py`)

Three functions implementing the geometry pipeline:

1. `backproject_depth_map(depth, intrinsics)`: Standard pinhole camera backprojection. For each pixel `(u, v)` with depth `d`: `X = (u - cx) * d / fx`, `Y = (v - cy) * d / fy`, `Z = d`. Produces `[B, H, W, 3]` point map. Invalid pixels (depth=0) produce (0,0,0).

2. `aggregate_patches_percentile(point_map, depth, patch_size=14, percentile=0.15)`: For each 14x14 patch (matching DINOv2's patch grid), ranks all 196 pixels by depth ascending, selects the pixel at the 15th percentile, and returns its 3D coordinates. Produces `[B, 1369, 3]`.

3. `pool_positions_to_sva_grid(positions, source_h=37, source_w=37, target_h=24, target_w=24)`: Pools 1369 positions to 576 using `F.adaptive_avg_pool2d`. This ensures the 3D position assigned to each SVA query corresponds to the **same spatial sub-region** that the SVA query covers in cross-attention. Produces `[B, 576, 3]`.

**Camera Utilities** (`src/spatialvlm/utils/camera.py`)

`CameraIntrinsics` dataclass (fx, fy, cx, cy), `make_pixel_grid()` for generating UV coordinates, `backproject_pixel()` for vectorized unprojection.

**GATr Wrapper** (`src/spatialvlm/geometry/gatr_wrapper.py`)

Wraps the reference `geometric-algebra-transformer` library with a SpatialVLM-oriented interface:

1. Input normalization: scales points by max radius to `[-1, 1]` range (GATr is sensitive to coordinate magnitudes)
2. PGA embedding: each `[x,y,z]` -> trivector in G(3,0,1) via `embed_point()`, producing `[B, N, 1, 16]`
3. 8 equivariant transformer blocks with improved PGA bilinears (join + geometric product)
4. Invariant extraction: 32 scalar channels + 16 multivector L2-norms = 48 rotation-invariant features
5. MLP projection: `[B, N, 48]` -> `[B, N, 4096]`

The constructor verifies at init time that the installed GATr version uses `GeometricBilinear` layers (improved PGA), raising `RuntimeError` if not. Basic PGA without join bilinears is not expressive enough (AISTATS 2024).

Cached einsum is disabled by default (`enable_cached_einsum(False)`) due to compatibility issues with modern numpy stacks.

**GridCellRoPE3D** (`src/spatialvlm/geometry/gridcell_rope3d.py`)

Zero-parameter rotary position encoding for 3D spatial coordinates:

- **4 tetrahedral directions**: Vertices of a regular tetrahedron on the unit sphere, satisfying the isotropy condition `sum(d_i @ d_i^T) = (4/3) I_3`. No direction in 3D is privileged.
- **8 golden-ratio frequencies**: `f_k = 10.0 * phi^k` for `k=0..7`. The golden ratio is maximally irrational -- no two frequencies share a common period, minimizing aliasing. Range: 10cm detail to 2.91m room-scale.
- **Output**: `4 dirs x 8 freqs x 2 (sin/cos) = 64 dimensions`, matching Qwen3-VL's 64 rotary pairs exactly. No projection layer needed.

All constants are stored as registered buffers (no learnable parameters). The encoding is a deterministic function of 3D position.

### Why this design

**15th-percentile aggregation, not mean**: Mean depth within a 14x14 patch is biased toward distant backgrounds (far walls, open doorways, ceilings). A patch spanning a doorframe and open air has mean depth ~3.2m but 15th-percentile ~0.4m (the doorframe itself). GATr processing the correct 0.4m identifies the obstacle; processing 3.2m encodes phantom geometry. Robust to ~29 outlier pixels per 196-pixel patch.

**SVA-aligned position pooling**: This is architecturally critical. After SVA fusion, only 576 tokens enter the LLM. GridCellRoPE3D needs `[B, 576, 3]` positions. These CANNOT be an arbitrary downsampling of the 1369 DINOv2-grid positions. The pooling uses `F.adaptive_avg_pool2d` which partitions the 37x37 source grid into 24x24 non-overlapping bins -- the **same spatial sub-regions** that SVA queries implicitly cover. This ensures each fused token's semantic content (from SVA cross-attention over a region) and its positional encoding (from GridCellRoPE3D) describe the same physical area. Breaking this invariant would mean a token "knows" about one part of the scene but claims to be somewhere else.

**GATr, not a standard transformer**: GATr processes 3D points equivariantly -- rotating the input point cloud rotates the output features correspondingly. This geometric inductive bias means the model doesn't need to learn rotation invariance from data. Equivariant linear layers use 9 learnable parameters per (in, out) pair vs. 256 for standard linear (28x more efficient).

**Golden ratio spacing**: The golden ratio `phi = 1.618...` is maximally irrational, meaning the frequency ratios `f_k/f_j` are never close to simple rational numbers. This delays aliasing onset longer than any alternative (power-of-2, sqrt(e), etc.). Testable via ablation H2d.

### Hypotheses tested

- **H2a**: GATr features are complementary to vision encoder features
- **H2b**: GridCellRoPE3D > M-RoPE for spatial tokens
- **H2c**: GT depth vs. Depth Anything V2 gap (diagnostic only)
- **H2d**: Golden ratio > other scale ratios
- **H2e**: 15th-percentile > mean aggregation

---

## 3. Stage 3 -- Fusion

### What was implemented

**Spatial Vision Aggregator (SVA)** (`src/spatialvlm/fusion/sva.py`)

Cross-attention module where 576 learnable query tokens attend to 3314 concatenated KV tokens:

```
KV = concat([576 SigLIP, 1369 DINOv2, 1369 GATr]) = [B, 3314, 4096]
Q  = siglip_tokens + learnable_query_embed          = [B, 576, 4096]
```

Two cross-attention layers, each with:
- Pre-LayerNorm on Q and KV separately
- Manual Q/K/V/O linear projections (32 heads, 128 head_dim)
- `F.scaled_dot_product_attention` (uses Flash Attention when available)
- Residual connection + post-LayerNorm

**Typed attention bias**: A learned 3x3 scalar matrix over token source types (0=SigLIP, 1=DINOv2, 2=GATr). Added to attention logits as an additive mask. 9 parameters that prevent any single modality from dominating during early training.

**Design choice -- query initialization**: Queries are initialized as SigLIP tokens plus a learnable embedding (`self.query_embed`), rather than purely learnable tokens. This provides a strong starting point (SigLIP's semantic features) while the learnable component adapts during training. Eagle and Cambrian both use encoder features as query base.

**RMS Norm Matching** (`src/spatialvlm/fusion/norm_matching.py`)

Scales vision token norms to match text token magnitude:

- Tracks an EMA (momentum=0.99) of text token RMS norms during training
- At forward time: `output = vision_tokens * (text_rms_ema / vision_rms)`
- Zero learnable parameters
- EMA state stored as `register_buffer` with in-place updates (`mul_` + `add_`) to ensure correct `state_dict` serialization and device movement

**Gated Cross-Attention** (`src/spatialvlm/fusion/gated_cross_attn.py`)

Flamingo-style gated residual injection with GQA:

```
x <- x + tanh(alpha_attn) * CrossAttention(Q=x, KV=vision)
x <- x + tanh(alpha_ff)   * FeedForward(x)
```

- **GQA**: 32 query heads, 8 KV heads (4:1 ratio matching Qwen3-VL). K/V projections are `Linear(4096, 1024)` (8 heads x 128 dim), not full `Linear(4096, 4096)`. KV heads are expanded via `unsqueeze(2).expand().reshape()` to match query head count.
- **Zero-init gates**: `alpha_attn = alpha_ff = 0.0` at initialization. `tanh(0) = 0`, so the block is a pure passthrough at init -- the pretrained Qwen3 behavior is preserved exactly.
- **9 injection points**: At Qwen3 layers {4, 8, 12, 16, 20, 24, 28, 32, 36} (every 4th of 36 layers)
- ~41.9M parameters per layer, ~377M total across 9 layers

### Why this design

**Cross-attention, not concatenation**: Three independent 2025-2026 papers validate this for spatial tasks:
- Spa3R (Feb 2026): Cross-attention outperforms concatenation by +7.5% on VSI-Bench
- SpaceMind (Dec 2025): Camera-Guided Modality Fusion via cross-attention achieves SOTA
- LVLDrive (Dec 2025): Gated cross-attention allows adaptation without drifting from learned manifold

**Norm matching before injection**: "Beyond Semantics" (2024) measured vision token L2 norms at 10-100x text token norms. "Why Is Spatial Reasoning Hard for VLMs?" found that despite comprising ~90% of input, image tokens receive only ~10% of attention. Randomly permuting all 576 vision tokens causes only 0.2-2.74% drops. Our norm matching eliminates the magnitude imbalance so RoPE positional information isn't drowned out.

**GQA in cross-attention**: Qwen3-VL uses 4:1 GQA natively. Matching this ratio in our injected cross-attention layers ensures architectural consistency and reduces KV cache overhead by 4x compared to standard MHA.

### Hypotheses tested

- **H3a**: Gated cross-attention > LLaVA concatenation by >5%
- **H3b**: RMS norm matching adds measurable benefit
- **H3c**: Permutation test: >15% drop (ours) vs. <3% (baseline)
- **H3d**: Typed attention bias improves stability
- **H3e**: 3314-token KV > 1728-token KV

---

## 4. Stage 4 -- LLM Backbone

### What was implemented

**Qwen3-VL Wrapper** (`src/spatialvlm/backbone/qwen3_vl.py`)

Lazy-loading wrapper around `Qwen/Qwen3-VL-8B-Instruct`:

- **Runtime config introspection**: All architecture constants (hidden_size, num_layers, head_dim, mrope_section, etc.) are read from `AutoConfig.from_pretrained()` at load time, never hardcoded
- **LoRA**: Rank-32, alpha-64 (effective scale 2.0) applied to Q/K/V/O projections in all 36 layers via PEFT. ~30.7M trainable parameters
- **PEFT #2880 workaround**: `requires_grad=True` is manually set on ViT QKV modules to fix zero-gradient bug
- **Lazy loading**: Model weights are not loaded until `load_model()` is called or `forward()` is first invoked, keeping memory free during module construction
- **Stats tracking**: `Qwen3BackboneStats` dataclass reports total params, trainable params, LoRA rank, etc.

**Position Routing** (`src/spatialvlm/backbone/position_routing.py`)

`PositionRouter` dispatches positional encodings based on token type:

- **Text tokens**: Standard M-RoPE with `mrope_section=[24, 20, 20]` (temporal, height, width). Sequential position IDs `[B, 3, T]`.
- **Spatial tokens**: All 64 M-RoPE rotary pairs replaced with GridCellRoPE3D's 64-dim output. Physical 3D distance governs attention, not sequence position.
- **Validation**: Constructor asserts `sum(mrope_section) == expected_spatial_rotary_dim == 64`
- **Output**: `RoutedPositionBatch` dataclass containing combined tokens, spatial mask, text position IDs, and spatial RoPE3D encodings

### Why this design

**LoRA, not full fine-tuning**: The 8B backbone has too many parameters for full fine-tuning within our compute budget. LoRA rank-32 provides sufficient adaptation capacity (~31M params) while keeping the pretrained knowledge intact. Rank-32 is chosen as a balance: rank-16 may be insufficient for learning new positional encoding semantics, rank-64 doubles the trainable params for diminishing returns.

**Position routing, not uniform encoding**: Text tokens benefit from sequential M-RoPE (language coherence depends on word order). Spatial tokens benefit from 3D geometric RoPE (spatial reasoning depends on physical distance, not sequence position). Applying M-RoPE to spatial tokens would encode arbitrary sequence order as if it were physical location. Applying GridCellRoPE3D to text would destroy language coherence.

**Exact 64-dim match**: GridCellRoPE3D outputs 64 dimensions (4x8x2) which exactly equals Qwen3's 64 rotary pairs (sum of mrope_section [24,20,20]). This means no adapter layer is needed -- the encoding slots directly into the existing RoPE computation. One fewer learned projection means one fewer source of error.

---

## 5. Stage 5 -- Training Pipeline

### What was implemented

**Pre-alignment** (`src/spatialvlm/training/prealign.py`)

- Freezes all parameters, then unfreezes only MLP projectors (~77M trainable)
- `PrealignmentTrainer` with `masked_lm_loss()` for next-token prediction
- 1 epoch over ~50K image-caption pairs (LLaVA-558K subset)

**Supervised Fine-Tuning** (`src/spatialvlm/training/sft.py`)

- `SFTTrainer` with label smoothing support
- `set_trainable_by_keyword()` selectively unfreezes modules matching name patterns
- All 583M trainable parameters active

**GRPO** (`src/spatialvlm/training/grpo.py`)

Group Relative Policy Optimization:

- `GRPOConfig`: group_size=8, clip_epsilon=0.2, kl_beta=0.001
- `grpo_loss()`: Clipped surrogate objective with KL regularization and optional entropy bonus. Takes pre-computed token logprobs (new, old, reference), advantages, and optional mask.
- `compute_group_advantages()`: Per-group normalization (subtract mean, divide by std within each group of 8)
- `approximate_kl()`: Token-level KL approximation `exp(r) - r - 1` where `r = log(pi/pi_ref)`
- `SelectiveSampleReplay`: Circular buffer (capacity=4096) storing only high-advantage trajectories (`|advantage| >= 0.05`). Mitigates vanishing advantages as training progresses (VL-Rethinker finding).
- `GRPOTrainer`: End-to-end trainer wrapping loss computation, gradient clipping, optimizer step, and replay buffer insertion.

**fDPO** (`src/spatialvlm/training/fdpo.py`)

Fine-grained DPO with segment-specific optimization pressures:

- `FDPOConfig`: `beta_grounding=0.1` (stricter for spatial descriptions), `beta_reasoning=0.05` (gentler for logical inference)
- `fdpo_loss()`: Standard DPO objective with per-sample beta values determined by `segment_mask`. When `segment_mask[i] = True`, sample `i` uses `beta_grounding`; when `False`, uses `beta_reasoning`.
- `FDPOLossBreakdown`: Reports total, grounding, and reasoning losses separately for monitoring
- Falls back to `beta_grounding` for all samples when no segment mask is provided (degrades gracefully to standard DPO)

**Dense Spatial Rewards** (`src/spatialvlm/training/rewards.py`)

Five reward terms:

| Term | Formula | Type |
|------|---------|------|
| `format` | +1.0 if response contains "Reasoning:" and "Action:" markers | Binary, per-step |
| `progress` | `geodesic_prev - geodesic_curr`, clipped to [-2, 2] | Dense, per-step |
| `collision` | -2.0 if `clearance < 0.1m` | Penalty, per-step |
| `goal` | +10.0 if `geodesic < 1.0m` AND agent called STOP | Sparse, terminal |
| `consistency` | -1.0 if predicted action != executed action | Penalty, per-step |

`compute_reward_terms()` computes all five from a batch. `total_reward()` applies curriculum-weighted aggregation.

**Reward Curriculum** (`src/spatialvlm/training/curriculum.py`)

Piecewise-linear interpolation over training progress:

| Epochs | Format | Accuracy | Spatial |
|--------|--------|----------|---------|
| 1-2 | 1.0 | 0.1 | 0.0 |
| 3-4 | 0.3 | 0.5 | 0.2 |
| 5-6 | 0.1 | 0.3 | 0.6 |

`RewardCurriculum.get_weights(progress)` returns interpolated `RewardWeights` for any training progress value in [0, 1]. `aggregate_weighted_rewards()` applies these weights to named reward terms.

### Why this design

**Three-stage training** (pre-alignment -> SFT -> GRPO): Pre-alignment is necessary because DINOv2 has never seen text -- without it, DINOv2 features hurt performance (Eagle finding). SFT teaches the model the task format and basic spatial reasoning. GRPO discovers the full solution distribution that SFT's single-demonstration supervision cannot cover (SpatialReasoner finding: SFT 58.3%, second SFT round drops to 54.7%, RL after SFT improves to 60.3%).

**GRPO over PPO**: GRPO eliminates the value network entirely -- sample G=8 outputs per prompt, normalize advantages within each group. Validated for spatial reasoning (R1-V on CLEVR, MultihopSpatial March 2026). Well-supported in HuggingFace TRL.

**Selective Sample Replay**: As training progresses, more groups yield identical rewards (zero advantages), shrinking effective batch size. SSR maintains a buffer of informative (high-advantage) past experiences that can be replayed to maintain learning signal. Directly addresses the VL-Rethinker finding.

**fDPO with dual betas**: The "fine-grained" aspect of fDPO is applying different optimization pressures to different output segments. Grounding segments ("the door is 2m ahead") need stricter optimization (beta=0.1) because spatial precision is critical. Reasoning segments ("therefore I should turn left") need gentler optimization (beta=0.05) because there are multiple valid reasoning paths. Without segment-specific betas, fDPO degrades to standard DPO, making the GRPO vs. fDPO comparison (H5b) meaningless.

**Consistency reward**: Prevents reward hacking where the model writes plausible CoT ("I see the goal ahead, I should move forward") but takes a random action (TURN_LEFT). From Embodied-R (ACM MM 2025).

### Hypotheses tested

- **H5a**: SFT+GRPO > SFT-only
- **H5b**: fDPO > GRPO on spatial quality/quantity
- **H5c**: Dense rewards > sparse rewards
- **H5d**: Consistency reward prevents reward hacking

---

## 6. Data Pipeline

### What was implemented

**Habitat Environment Wrapper** (`src/spatialvlm/data/habitat_env.py`)

- `HabitatEnvWrapper`: Wraps Habitat 3.0 environments with resolution validation (RGB and depth both at 518x518 for pixel-perfect DINOv2 alignment)
- `HabitatEnvConfig`: Dataclass with `build_overrides()` method for config-driven environment construction
- `extract_rgb_depth()`: Convenience function returning GPU tensors from Habitat observations
- Lazy import: `require_habitat()` checks for `habitat` availability and raises helpful `ImportError`

**Dataset Loaders** (`src/spatialvlm/data/datasets.py`)

- `R2RCEDataset`, `RxRCEDataset`, `SQA3DDataset`: All inherit from `_BaseNavDataset`
- `NavSample` dataclass with episode_id, instruction, scene_id
- `build_dataset(name, path)` factory function
- `iter_instructions()` generator for streaming through large datasets

**Preprocessing** (`src/spatialvlm/data/preprocessing.py`)

- `preprocess_rgb_depth()`: End-to-end function for converting raw Habitat observations to model-ready tensors
- Handles float conversion, resizing, depth normalization (NaN/zero handling)

---

## 7. Evaluation Infrastructure

### What was implemented

**Permutation Test** (`src/spatialvlm/eval/permutation_test.py`)

The paper's smoking gun diagnostic. `run_permutation_test()`:

1. Compute baseline score on unmodified tokens
2. For N=64 permutations: shuffle token order, compute score
3. Report: baseline vs. permuted mean, absolute/relative drop, empirical p-value

`PermutationTestResult.is_spatially_grounded(min_relative_drop=0.15)`: Returns True if shuffling tokens causes >15% drop -- proving the model actually uses spatial token ordering.

`permute_tokens()` supports optional `spatial_mask` for selective permutation (only shuffle within masked positions).

**Metrics** (`src/spatialvlm/eval/metrics.py`)

- `success_rate()`, `spl()`: Standard navigation metrics
- `permutation_sensitivity_index()`: Quantifies how much permutation hurts
- `weighted_composite()`: Configurable weighted combination
- `compute_metric_bundle()`: Computes all metrics from `NavigationEpisodeResult` lists

**Benchmark Runner** (`src/spatialvlm/eval/benchmarks.py`)

- `BenchmarkSpec` dataclass with name, dataset, metric function, primary flag
- `default_benchmark_suite()`: Returns specs for VLN-CE R2R, RxR, ObjectNav HM3D, SQA3D, VSI-Bench, NavTrust
- `validate_primary_suite_is_indoor()`: Asserts no real-image benchmarks in core evaluation (GT-depth-only scope)
- `BenchmarkRunner.run()`: Executes benchmarks and returns `BenchmarkResult` list

**Ablation Orchestrator** (`src/spatialvlm/eval/ablations.py`)

- `AblationSpec`: Defines config overrides for each ablation variant
- `default_ablation_specs()`: All architecture and diagnostic ablations from Section 7 of plan.md
- `AblationOrchestrator.run()`: Applies overrides, runs benchmarks, collects results

**Phase 9 Specs** (`src/spatialvlm/eval/phase9.py`)

- `phase9_run_specs()`: Complete list of 15 ablation run specifications
- Each spec has name, hypothesis ID, config overrides, and expected direction of change

**Paper Assets** (`src/spatialvlm/eval/paper_assets.py`)

- `render_ablation_table_tex()`: Generates LaTeX ablation table
- `render_main_results_table_tex()`: Generates LaTeX main results table
- `write_permutation_csv()`: Exports permutation test data
- `write_paper_assets()`: Orchestrates all asset generation

---

## 8. Configuration

### What was implemented

**Model Config** (`src/spatialvlm/config/model.py`)

Hierarchical dataclass structure:

```python
SpatialVLMConfig
  ├── EncoderConfig      # SigLIP/DINOv2 model IDs, resolutions, extract layers
  ├── GeometryConfig     # GATr blocks/channels, depth percentile, GridCellRoPE3D params
  ├── FusionConfig       # SVA queries/layers, cross-attn layer indices, norm EMA
  └── BackboneConfig     # Qwen3 model ID, hidden dims, LoRA rank/alpha, mrope_section
```

Every pre-trained model constant is marked with `# ⚠ VERIFY` comments. Computed properties (`gatr_invariant_dim`, `rope3d_dims`) derive values from base config rather than hardcoding.

**YAML Configs** (`configs/`)

- `train.yaml`: Training hyperparameters
- `eval.yaml`: Benchmark specifications (primary indoor-only, supplementary disabled)
- `model.yaml`: Model architecture config

---

## 9. Test Coverage

**184 tests across 26 test files**, all passing. Key test categories:

| Category | Tests | What they verify |
|----------|-------|------------------|
| Geometry math | ~35 | Backprojection shapes, percentile aggregation, SVA-aligned pooling, GridCellRoPE3D dimensions and equivariance, camera utils |
| GATr wrapper | ~8 | Output shapes, invariant dimensions, improved PGA verification |
| SVA fusion | ~6 | Output shapes, typed bias matrix, padding mask handling |
| Gated cross-attn | ~7 | GQA 4:1 ratio, KV projection shapes, zero-init passthrough |
| Norm matching | ~5 | EMA updates, scaling behavior, buffer persistence |
| Training losses | ~15 | GRPO clipping, advantage normalization, fDPO segment-specific betas, reward computation |
| Position routing | ~6 | M-RoPE shape, spatial mask, dimension validation |
| Encoders | ~6 | Hook registration, multi-layer extraction (slow: require model download) |
| Data pipeline | ~10 | Dataset loading, preprocessing shapes, Habitat wrapper |
| Eval infra | ~15 | Permutation test, metrics, ablation specs, benchmark validation |

---

## 10. Architectural Invariants

These are properties that must hold across the entire pipeline. Each was verified during code review:

1. **Dimension consistency**: All three encoder outputs project to 4096 (= Qwen3's hidden_size). Verified at runtime via `AutoConfig`.

2. **Spatial co-registration**: DINOv2 tokens and GATr tokens share the same 37x37 spatial grid. Token (i,j) from each describes the same image region.

3. **Content-position consistency**: SVA query (i,j) covers a spatial sub-region of the 37x37 grid. GridCellRoPE3D position for query (i,j) is the mean of source positions in that same sub-region. Content and position describe the same physical area.

4. **Rotary dimension match**: GridCellRoPE3D outputs 64 dims = Qwen3's 64 rotary pairs = sum([24, 20, 20]). No adapter needed.

5. **GQA ratio match**: Gated cross-attention uses 32:8 Q:KV heads, matching Qwen3-VL's native ratio.

6. **Zero-init passthrough**: At initialization, gated cross-attention gates are 0, so the model behaves exactly as pretrained Qwen3.

7. **Device agnosticism**: No module hardcodes `cuda`. All tensors inherit device from inputs or explicit `device` parameters.

8. **GT-depth-only scope**: Primary evaluation uses Habitat GT depth exclusively. No real-image benchmarks in core results.

---

## 11. Issues Found and Fixed During Review

### Critical (3)

| ID | Issue | Fix |
|----|-------|-----|
| C1 | SVA-aligned position pooling missing entirely | Added `pool_positions_to_sva_grid()` using `F.adaptive_avg_pool2d` with same spatial binning as SVA queries |
| C2 | Gated cross-attention used `nn.MultiheadAttention` (no GQA support) | Complete rewrite with manual Q/K/V projections, 32:8 GQA head ratio, `F.scaled_dot_product_attention` |
| C3 | fDPO had single beta (= standard DPO) | Added `beta_grounding`/`beta_reasoning` + `segment_mask` parameter |

### Medium (3)

| ID | Issue | Fix |
|----|-------|-----|
| M2 | GATr improved PGA check only ran on-demand | Moved to `__init__` with `RuntimeError` on failure |
| M3 | Norm matching EMA broke `state_dict()` via tensor reassignment | Changed to in-place `mul_()` + `add_()` |
| M5 | SVA contiguity check was CUDA-only | Removed device check; `.contiguous()` is a no-op on already-contiguous tensors |

### Low (2)

| ID | Issue | Fix |
|----|-------|-----|
| L1 | Missing shape assertions after critical reshapes | Added `assert` statements in backproject.py |
| L4 | Permutation test used string `"cpu"` | Changed to `torch.device("cpu")` |

---

## 12. What Remains

### Must complete before training

- [ ] **Environment setup**: Install Habitat 3.0, download HM3D scenes, configure GPU cluster
- [ ] **Model downloads**: Qwen3-VL-8B-Instruct, DINOv2-L, SigLIP2-SO400M (for slow tests)
- [ ] **Data preparation**: R2R-CE, RxR-CE, SQA3D, LLaVA-558K subset

### Training runs (Phase 9)

All 15 ablation variants specified in `src/spatialvlm/eval/phase9.py`:

- [ ] Full model (baseline)
- [ ] No GridCellRoPE3D (H2b)
- [ ] No GATr (H2a)
- [ ] SigLIP only (H1a)
- [ ] DINOv2 only (H1a)
- [ ] DINOv2 pooled to 576 (H1d, H3e)
- [ ] Concatenation fusion (H3a)
- [ ] No RMS norm matching (H3b)
- [ ] No typed attention bias (H3d)
- [ ] SFT only, no GRPO (H5a)
- [ ] Permutation test (H3c)
- [ ] GT vs. estimated depth (H2c)
- [ ] Mean vs. 15th percentile (H2e)
- [ ] Scale ratio sweep (H2d)
- [ ] GRPO vs. fDPO (H5b)
- [ ] Dense vs. sparse rewards (H5c)

### Paper

- [ ] Run experiments and collect results
- [ ] Generate paper assets (LaTeX tables, permutation CSVs)
- [ ] Write paper text
- [ ] Create figures

---

## 13. File Manifest

```
src/spatialvlm/
├── config/
│   └── model.py                    # SpatialVLMConfig (hierarchical dataclass)
├── encoders/
│   ├── siglip.py                   # SigLIP2-SO400M/16 wrapper (576 tokens)
│   ├── dinov2.py                   # DINOv2-L/14 wrapper (1369 tokens)
│   └── projector.py                # Shared 2-layer MLP projector
├── geometry/
│   ├── backproject.py              # Depth->3D, 15th-pct aggregation, SVA pooling
│   ├── gatr_wrapper.py             # GATr 8-block PGA transformer
│   └── gridcell_rope3d.py          # Tetrahedral Fourier RoPE (zero params)
├── fusion/
│   ├── sva.py                      # 576-query SVA (3314 KV, typed bias)
│   ├── norm_matching.py            # RMS norm EMA matching (zero params)
│   └── gated_cross_attn.py         # Flamingo-style + GQA (32:8 heads)
├── backbone/
│   ├── qwen3_vl.py                 # Qwen3-VL-8B + LoRA + PEFT workaround
│   └── position_routing.py         # M-RoPE / GridCellRoPE3D dispatch
├── training/
│   ├── prealign.py                 # Stage 1: projector-only SFT
│   ├── sft.py                      # Stage 2: full SFT
│   ├── grpo.py                     # Stage 3: GRPO + SSR replay
│   ├── fdpo.py                     # Stage 3 alt: fDPO (segment betas)
│   ├── rewards.py                  # 5 dense spatial reward functions
│   └── curriculum.py               # Piecewise-linear reward weighting
├── data/
│   ├── habitat_env.py              # Habitat 3.0 wrapper (518x518)
│   ├── datasets.py                 # R2R, RxR, SQA3D loaders
│   └── preprocessing.py            # RGB/depth preprocessing
├── eval/
│   ├── permutation_test.py         # Smoking gun diagnostic
│   ├── metrics.py                  # SR, SPL, PSI, composite
│   ├── benchmarks.py               # Benchmark suite runner
│   ├── ablations.py                # Ablation orchestrator
│   ├── phase9.py                   # 15 ablation run specs
│   └── paper_assets.py             # LaTeX table generators
└── utils/
    └── camera.py                   # Pinhole intrinsics + backprojection

tests/  (26 files, 184 passing tests)
configs/  (train.yaml, eval.yaml, model.yaml)
```
