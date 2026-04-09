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

---

## 14. Benchmark Suite: What to Beat and Where

Our system uses **Habitat GT depth exclusively**. This constrains the primary evaluation to benchmarks where ground-truth 3D structure is available — simulation-based navigation benchmarks and ScanNet-based 3D QA benchmarks. Real-image benchmarks (What's Up, CV-Bench) are supplementary only, as they cannot leverage our geometric branch without estimated depth.

### 14.1 Primary Benchmarks (P0) — Must Beat SOTA

These define the paper's headline results. All have GT depth available natively.

#### VSI-Bench (Visual Spatial Intelligence Benchmark)
- **Venue**: CVPR 2025 (NYU — "Thinking in Space")
- **Tests**: 8 spatial subtasks — object counting, relative direction, absolute distance estimation, route planning, object size, room size, relative distance, appearance order
- **Data**: 5,000+ QA pairs from ARKitScenes, ScanNet, ScanNet++ (all have GT depth and 3D reconstructions)
- **Metrics**: Accuracy (numerical estimation + multiple choice), broken down by subtask
- **GT depth**: Yes — all underlying scenes are 3D-scanned environments
- **Current SOTA**:

| Model | Accuracy | Notes |
|-------|----------|-------|
| SpaceMind (Dec 2025) | **69.6%** | Camera-guided fusion, InternVL3-8B |
| VLM-3R-7B | 60.9% | 3D reconstruction tokens from monocular video |
| Spatial-MLLM-4B (NeurIPS 2025) | 48.4% | Dual encoder (2D + 3D geometry) |
| Gemini 1.5 Pro | 48.8% | Proprietary |
| GPT-4o | ~35-40% | Proprietary |
| Human | **79.0%** | — |

- **Why P0**: Largest gap between VLMs and humans (~40 points). Distance estimation and room layout subtasks are near-random for all VLMs. Our architecture (GATr geometric branch + GridCellRoPE3D) is specifically designed to solve these metric spatial tasks. **Beating SpaceMind's 69.6% or closing the gap to human 79% would be the paper's strongest result.**

#### SQA3D (Situated Question Answering in 3D Scenes)
- **Venue**: ICLR 2023
- **Tests**: Situated spatial QA — questions about spatial relations, distances, object properties, all grounded in the agent's position and orientation within a 3D ScanNet scene
- **Data**: 33.4K questions, 6.8K situations, 650+ ScanNet scenes, 20.4K descriptions
- **Metrics**: Exact Match (EM@1), broken down by question type (what, where, how many, spatial relation)
- **GT depth**: Yes — ScanNet provides full 3D reconstructions; depth renderable from any viewpoint at 518x518
- **Current SOTA**:

| Model | EM@1 | Notes |
|-------|------|-------|
| SpaceMind (Dec 2025) | **~55%** | Claims SOTA on EM@1 and EM@R1 |
| GPS4Scene / 3D-aware VLMs | 55-65% | Scene-graph + LLM approaches |
| LEO (NeurIPS 2023) | ~50% | 3D embodied generalist |
| 3D-LLM | ~47% | Point cloud + LLM |
| GPT-4V (zero-shot) | ~45-50% | No 3D input |
| Human | **90.06%** | — |

- **Why P0**: The gold-standard 3D spatial QA benchmark. Published at ICLR, well-established leaderboard. Directly tests whether the model understands "where am I, what's around me, how far is X." Our GATr branch + situated position encoding should dramatically help on spatial relation questions.

#### VLN-CE R2R val-unseen (Vision-Language Navigation — Continuous Environments)
- **Venue**: ECCV 2020 (Krantz et al.), community standard since
- **Tests**: Following natural language navigation instructions ("Walk past the table and turn right at the hallway...") in continuous Habitat/Matterport3D environments with low-level actions
- **Data**: ~2K val-unseen episodes, ~4K train episodes, 90 Matterport3D environments
- **Metrics**: Success Rate (SR), SPL (Success weighted by Path Length), nDTW, SDTW
- **GT depth**: Yes — Habitat simulator provides perfect depth at any resolution
- **Current SOTA**:

| Model | SR | SPL | Notes |
|-------|-----|-----|-------|
| V2-VLNCE (RA-L 2026) | **~72%** | — | +8-15% over Efficient-VLN |
| Efficient-VLN (2024) | 64.2% | — | Previous SOTA |
| NavFoM | 64.9% | — | Cross-embodiment foundation model |
| BEVBert / ETPNav | ~60% | ~50% | BEV representations |
| NaVILA (NeurIPS 2025) | 47%+ | — | VLM-based agent |

- **Why P0**: The community-standard navigation benchmark. Every VLN paper reports R2R numbers. The field has progressed from ~40% to ~72% SR — we need to be competitive with V2-VLNCE or show that our spatial understanding transfers to better path quality (SPL).

#### ObjectNav HM3D
- **Venue**: Habitat Challenge series (NeurIPS), HM3D (Ramakrishnan et al., NeurIPS 2021)
- **Tests**: Navigate to a specified object category ("find a chair") in previously unseen HM3D environments via exploration
- **Data**: 2K+ val episodes, 6 standard categories (expandable to 20+)
- **Metrics**: Success Rate (SR), SPL, SoftSPL
- **GT depth**: Yes — Habitat on HM3D meshes
- **Current SOTA**:

| Model | SR | SPL | Notes |
|-------|-----|-----|-------|
| CogNav (ICCV 2025) | **72.5%** | — | LLM cognitive process modeling |
| PixNav / PIRLNav | ~65% | — | End-to-end pixel navigation |
| VLFM (NeurIPS 2023) | ~55% | — | Vision-language frontier maps |
| ESC / L3MVN / CoW | 40-60% | — | LLM-based planners |

- **Why P0**: Tests exploration + semantic + spatial reasoning. CogNav's 72.5% is a high bar. Our geometric branch (GATr understanding scene layout) should help with efficient exploration — knowing where objects are likely to be based on room geometry.

### 14.2 Strong Secondary Benchmarks (P1) — Expected to Report

These strengthen the paper's claims and test different spatial abilities.

#### VLN-CE RxR val-unseen
- **Venue**: ACL 2020 (Ku et al.), CE variant in Habitat
- **Tests**: Same as R2R but with longer, more detailed multilingual instructions; paths are longer and more complex
- **Data**: ~4K val-unseen (English subset), ~126K instructions total (English, Hindi, Telugu)
- **Metrics**: SR, SPL, nDTW
- **GT depth**: Yes (Habitat)
- **Current SOTA**: SR generally 5-15 points below R2R; BEVBert-style models at 40-50% SR. V2-VLNCE likely ~60%+.
- **Why P1**: Harder variant of R2R. Longer trajectories stress spatial memory — our GridCellRoPE3D should help maintain 3D consistency across many steps.

#### Spartun3D
- **Venue**: ICLR 2025
- **Tests**: Situated captioning and QA requiring reasoning about surrounding objects relative to agent's standing point and orientation in 3D ScanNet scenes
- **Data**: ~133K examples
- **Metrics**: QA accuracy, captioning quality
- **GT depth**: Yes — built on ScanNet 3D scenes with situated agent positions
- **Why P1**: Directly tests situated spatial reasoning with agent orientation. Complements SQA3D. Our position routing (GridCellRoPE3D encodes agent's 3D viewpoint) is built for this.

#### ScanQA
- **Venue**: CVPR 2022
- **Tests**: Free-form 3D QA about spatial scenes requiring understanding of object layouts and relations
- **Data**: ~41K QA pairs across 800 ScanNet scenes
- **Metrics**: CIDEr, BLEU-4, METEOR, ROUGE-L
- **GT depth**: Yes (ScanNet reconstructions)
- **Current SOTA**: CIDEr ~72 (Gen3DQA); Video-3D LLM also claims SOTA
- **Why P1**: Open-ended generation rather than multiple-choice. Tests whether our spatial understanding transfers to fluent spatial descriptions.

#### ScanRefer
- **Venue**: ECCV 2020
- **Tests**: 3D object grounding — given a natural language description, localize the referred object in a ScanNet scene
- **Data**: ~36K descriptions for ~7K objects in 703 ScanNet scenes
- **Metrics**: Acc@0.25, Acc@0.5 (IoU thresholds)
- **GT depth**: Yes (full 3D point clouds)
- **Current SOTA**: 55-65% Acc@0.5 (3D-VisTA, EDA); newer methods pushing toward 70%
- **Why P1**: Tests whether GATr geometric features help with precise 3D localization. Our equivariant representations should encode object positions more accurately than appearance-only features.

#### REVERIE
- **Venue**: CVPR 2020
- **Tests**: Navigation + remote object grounding ("Go to the bedroom and bring me the pillow on the bed")
- **Data**: ~3.5K val-unseen episodes in Matterport3D
- **Metrics**: SR, SPL, RGS (Remote Grounding Success), RGSPL
- **GT depth**: Yes (Habitat/Matterport3D)
- **Current SOTA**: Mid-30s to low-40s SR
- **Why P1**: Combines navigation and grounding. Tests end-to-end spatial understanding — navigate there AND identify the right object.

### 14.3 Diagnostic and Meta-Benchmarks (P2) — Ablation Support

These provide fine-grained analysis of which spatial abilities improve.

#### SITE (Spatial Intelligence Thorough Evaluation)
- **Venue**: ICCV 2025 (Boston Univ + Microsoft Research)
- **Tests**: 30+ datasets unified; covers figural-to-environmental scales, spatial visualization, spatial orientation, intrinsic/extrinsic reference frames, static/dynamic
- **Data**: 8,068 tasks aggregated from 31 datasets
- **Metrics**: Accuracy across multi-choice VQA per spatial cognition dimension
- **GT depth**: Mixed — aggregates from 31 datasets, some include ScanNet/indoor
- **Why P2**: Meta-benchmark showing *breadth* of spatial improvement across cognitive dimensions. Strong evidence that our architecture doesn't just help on one narrow task.

#### 3DSRBench (3D Spatial Reasoning Benchmark)
- **Venue**: ICCV 2025
- **Tests**: Height, location, orientation, multi-object reasoning across 12 question types, including uncommon 6D viewpoints
- **Data**: 2,772 manually annotated VQA pairs; includes HSSD (Habitat Synthetic Scenes Dataset) subset
- **Metrics**: Accuracy per question type
- **GT depth**: Yes for HSSD subset (Habitat-compatible)
- **Why P2**: The HSSD subset runs directly in Habitat with GT depth. Tests 3D awareness from unusual viewpoints — where our equivariant GATr representations should shine.

#### NavTrust
- **Venue**: arXiv March 2026
- **Tests**: Robustness of navigation agents under RGB/depth corruptions (motion blur, depth noise, missing data) and instruction perturbations
- **Data**: Systematic corruption applied to VLN-CE and ObjectNav episodes
- **Metrics**: SR degradation under each corruption type
- **GT depth**: Yes (Habitat-based)
- **Why P2**: Tests robustness, not just peak performance. Our explicit geometric processing should be more robust to visual corruptions than appearance-only models (GATr operates on depth geometry, not pixels).

#### SpatialRGPT-Bench
- **Venue**: NeurIPS 2024
- **Tests**: Region-grounded spatial cognition — distance estimation, size comparison, relative depth, spatial relation classification
- **Data**: Subsets from SUNRGBD and Hypersim (both have GT depth)
- **Metrics**: Accuracy on spatial relation classification
- **GT depth**: Yes for SUNRGBD/Hypersim subsets
- **Current SOTA**: SpatialRGPT >20% accuracy gain over GPT-4V; SpatialReasoner-R1 adds +9.8% via fDPO
- **Why P2**: Our fDPO implementation with segment-specific betas follows SpatialReasoner-R1's exact approach — this benchmark directly validates H5b.

#### RoboSpatial
- **Venue**: CVPR 2025 (NVIDIA, Oral)
- **Tests**: Ego-centric, world-centric, and object-centric spatial relationships; affordance prediction
- **Data**: 1M images, 5K 3D scans, 3M annotated spatial relationships (indoor)
- **Metrics**: Accuracy on spatial relationship prediction
- **GT depth**: Yes — real indoor 3D scans + egocentric images with GT annotations
- **Why P2**: Adopted by Qwen3-VL (our backbone) as a benchmark. Showing improvement over the backbone's own spatial scores is a clean demonstration of our architectural additions.

#### EXPRESS-Bench (Exploration-Aware EQA)
- **Venue**: ICCV 2025
- **Tests**: Whether agents explore correctly before answering spatial questions — measures exploration-answer consistency
- **Data**: 777 trajectories, 2,044 QA pairs in Habitat
- **Metrics**: EAC (Exploration-Answer Consistency)
- **GT depth**: Yes (Habitat-based)
- **Why P2**: Tests a combined exploration + reasoning skill. Our architecture's explicit 3D scene understanding should produce more purposeful exploration.

### 14.4 Supplementary Benchmarks (P3) — Nice-to-Have

These use real images (no GT depth). Include only if we extend to Depth Anything V2 (ablation H2c).

| Benchmark | Venue | Tests | SOTA | Notes |
|-----------|-------|-------|------|-------|
| MultihopSpatial | arXiv Mar 2026 | 1-3 hop compositional spatial reasoning | GPT-5.2: 9.4% on 3-hop grounded | Real images; extremely hard |
| What's Up | EMNLP 2023 | Basic spatial relations (up/down/left/right) | All VLMs ~56% (chance) | Real images; smoking gun |
| CV-Bench (spatial) | 2024 | Depth ordering, distance comparison | GPT-4V ~65-70% | Real images |
| SpatialBench (cognitive) | arXiv 2025 | 15 tasks across 5 cognitive levels | Models fail at symbolic reasoning | Mixed sources |
| ViewSpatial-Bench | arXiv May 2025 | Multi-perspective spatial localization | — | ScanNet subset usable |

### 14.5 The Competitive Landscape: What We Must Beat

The table below summarizes the models we're competing against, organized by approach:

| Model | Approach | VSI-Bench | SQA3D | VLN-CE R2R SR | ObjectNav SR |
|-------|----------|-----------|-------|---------------|--------------|
| **SpaceMind** (Dec 2025) | Camera-guided cross-attn fusion | **69.6%** | SOTA | — | — |
| **VLM-3R** (CVPR 2026) | 3D reconstruction tokens | 60.9% | — | — | — |
| **CogNav** (ICCV 2025) | LLM cognitive process model | — | — | — | **72.5%** |
| **V2-VLNCE** (RA-L 2026) | Improved VLN with views | — | — | **~72%** | — |
| **Spatial-MLLM** (NeurIPS 2025) | Dual encoder (2D + 3D) | 48.4% | — | — | — |
| **SpatialLadder** (ICLR 2026) | GRPO + curriculum training | — | — | — | — |
| **SpatialReasoner-R1** (NeurIPS 2025) | fDPO with segment betas | — | — | — | — |
| **NaVILA** (NeurIPS 2025) | VLM navigation agent | — | — | 47%+ | — |
| Efficient-VLN (2024) | Efficient VLN | — | — | 64.2% | — |
| Human | — | 79.0% | 90.06% | — | — |

**Key insight**: No single model dominates across all benchmarks. SpaceMind leads VSI-Bench, CogNav leads ObjectNav, V2-VLNCE leads VLN-CE. Our architecture's advantage is the **unified geometric backbone** — we compete on ALL spatial tasks with one architecture, while existing SOTA models are specialized per-benchmark.

### 14.6 Where We Should Win and Why

| Benchmark | Our Advantage | Key Module |
|-----------|---------------|------------|
| **VSI-Bench distance/size estimation** | GATr encodes metric 3D geometry; GridCellRoPE3D encodes physical distances | Stage 2 |
| **SQA3D spatial relations** | Situated position encoding via GridCellRoPE3D; GATr understands relative positions | Stage 2 + 4 |
| **VLN-CE R2R/RxR** | 3D-aware attention (nearby obstacles get more weight); norm-balanced fusion prevents token drowning | Stage 2 + 3 |
| **ObjectNav HM3D** | Geometric scene understanding for efficient exploration; GATr encodes room layout | Stage 2 |
| **Spartun3D / ScanRefer** | Equivariant GATr representations for precise 3D localization | Stage 2 |
| **Permutation test (all benchmarks)** | >15% drop (ours) vs. <3% (baselines) — proves spatial structure is used | Stage 3 (H3c) |

### 14.7 Recommended Evaluation Protocol

**Phase 1 — Core Results Table** (required for submission):
1. VSI-Bench (8 subtasks) — headline spatial intelligence result
2. SQA3D (by question type) — headline 3D QA result
3. VLN-CE R2R val-unseen (SR, SPL) — headline navigation result
4. ObjectNav HM3D (SR, SPL) — headline exploration result
5. Permutation test across all 4 — the smoking gun

**Phase 2 — Extended Results** (strengthens the paper):
6. VLN-CE RxR val-unseen
7. Spartun3D
8. ScanQA
9. ScanRefer (Acc@0.5)
10. REVERIE (SR, RGS)

**Phase 3 — Diagnostic Analysis** (ablation support):
11. SITE meta-benchmark (spatial cognition dimensions)
12. 3DSRBench HSSD subset (uncommon viewpoints)
13. NavTrust (robustness under corruption)
14. SpatialRGPT-Bench Hypersim subset (metric spatial tasks)
15. RoboSpatial (Qwen3-VL's own benchmark)
16. EXPRESS-Bench (exploration-answer consistency)

**Phase 4 — Supplementary** (if time permits, requires Depth Anything V2):
17. MultihopSpatial (compositional reasoning)
18. What's Up (basic spatial relations)
19. CV-Bench spatial subset

---

## 15. Gap Analysis: What the Literature Reveals We Must Address

Based on Liu et al. (2025), "Spatial Intelligence in Vision-Language Models: A Comprehensive Survey" (TechRxiv, 47 pages, 37 models evaluated across 9 benchmarks), the following gaps and risks are identified for our architecture.

### 15.1 Critical Warning: Generalization of 3D-Enhanced Models

**Key Finding 5 from the survey**: *"Explicit 2D/3D spatial enhancements demonstrate limited generalization, with general-purpose models often outperforming specialized variants."*

The survey found that models injecting explicit 3D information (our category) **underperform general-purpose VLMs** in broad evaluations:

| Method Category | Perception | Understanding | Extrapolation | All-Three |
|----------------|------------|---------------|---------------|-----------|
| General VLM | 42.6% | 58.9% | 31.6% | 47.0% |
| §5.4 3D info | 37.3% | 51.8% | 23.9% | 34.6% |
| §5.2 Model-centric | 45.1% | 59.0% | 31.7% | 43.7% |

3D-enhanced models score **12.4 points below** general VLMs on the All-Three composite. This is the single biggest risk to our paper. The survey attributes this to: (a) overfitting to narrow spatial tasks at the cost of general spatial reasoning, (b) degraded performance on non-metric spatial tasks (relative relations, scene understanding).

**Mitigation in our design**:
- We freeze the Qwen3-VL backbone and inject via zero-init gated cross-attention — the model starts as a general VLM and gradually incorporates 3D signal. This preserves general capabilities.
- Our SVA fusion aggregates 3D alongside semantic (SigLIP) and structural (DINOv2) features, not replacing them.
- Our ablation design explicitly tests whether spatial improvement comes at a generalization cost.

**Action required**: We must evaluate on **broad spatial benchmarks** (not just navigation/3D QA) to demonstrate we don't suffer this generalization failure. The survey's 9-benchmark suite should be a secondary evaluation.

### 15.2 Cognitive Hierarchy: We Under-Test Extrapolation

The survey defines three cognitive levels:
1. **Perception**: Object attributes, depth estimation, orientation → Our GATr handles this
2. **Understanding**: Spatial relations, grounding → Our SVA + GridCellRoPE3D handles this
3. **Extrapolation**: Mental rotation, occlusion reasoning, spatial planning → **Largely untested in our benchmark suite**

**Key Finding 1**: VLMs perform Understanding > Perception > Extrapolation. The hardest tasks (extrapolation) include:
- **Mental rotation** (MINDCUBE): Even GPT-5 scores only 42.96%
- **Spatial simulation** (SRBench): GPT-5 scores 56.20%, open models ~28-33%
- **Occlusion reasoning**: Inferring objects behind occluders

Our architecture has no explicit mechanism for extrapolation-level tasks. GridCellRoPE3D encodes spatial positions but doesn't model mental transformations. GATr is equivariant (handles rotations of the observed scene) but not predictive (doesn't simulate unobserved states).

**Benchmarks to add**:
- **MINDCUBE** (Yin et al., 2025) — mental rotation of 3D polycubes; tests whether 3D awareness helps with spatial visualization. SOTA: GPT-5 at 42.96%.
- **SRBench** (spatial reasoning beyond visible) — occlusion, spatial simulation. SOTA: GPT-5 at 56.20%.

### 15.3 Missing Benchmarks from the Survey's Evaluation Suite

The survey evaluates 37 models on 9 benchmarks. We should include several of these to enable direct comparison:

| Benchmark | Cognitive Level | Our Coverage | Priority |
|-----------|----------------|--------------|----------|
| **EgoOrientBench** | Perception | MISSING | Should add — tests orientation estimation from egocentric views, directly relevant to navigation |
| **GeoMeter** | Perception | MISSING | Should add — geometric reasoning (angles, lengths, areas) from real-world images |
| **SEED-Bench (spatial)** | Understanding | MISSING | Should add — widely used, enables cross-study comparison |
| **What's Up** | Understanding | In P3 (supplementary) | Promote to P2 — the survey shows Qwen2.5-VL-7B scores 50.92% (near-chance) while GPT-5 scores 99.63% |
| **SRBench** | Extrapolation | MISSING | Should add — tests spatial reasoning beyond the visible |
| **MINDCUBE** | Extrapolation | MISSING | Should add — 3D mental rotation |
| **OmniSpatial** | All Three | MISSING | Should add — comprehensive spatial evaluation |
| **RealWorldQA** | All Three | MISSING | Lower priority — real images only |

### 15.4 Updated What's Up Assessment

Our `plan.md` states: *"All VLMs ~56% (near-chance). Humans 99%."*

**This is outdated.** The survey's Table 1 shows:

| Model | What's Up Score |
|-------|----------------|
| GPT-5 | **99.63%** |
| GPT-4o | **99.50%** |
| Gemini 2.5 Pro | **99.63%** |
| Qwen2.5-VL-72B | 96.72% |
| LLaVA-NeXT-7B | 78.17% |
| **Qwen2.5-VL-7B** | **50.92%** |
| LLaVA-v1.5-7B | 19.02% |

Commercial models have essentially solved What's Up. The near-chance finding applies primarily to older/smaller open-source models. Qwen2.5-VL-7B (similar scale to our Qwen3-VL-8B backbone) still scores near-chance at 50.92%.

**Implication**: What's Up is no longer the smoking gun it was in 2023. However, **for 7-8B scale open-source models**, it's still very relevant. Our paper should frame the What's Up test as: "At the 8B scale, spatial structure remains destroyed — our architecture recovers it."

### 15.5 The Referential Ambiguity Problem

**Survey Challenge #3**: Spatial language is inherently ambiguous between egocentric (viewer-centered) and allocentric (object-centered) perspectives. "The cat is to the left of the dog" depends on whether you mean viewer-left or dog-left.

Our architecture encodes positions in camera coordinates (egocentric) via backprojection. We don't explicitly model allocentric reference frames. This is fine for navigation (egocentric is natural) but may cause failures on:
- Perspective-taking tasks (SQA3D "situated" questions)
- Multi-agent spatial reasoning
- Instructions given from a third-person viewpoint

**Mitigation**: Our GridCellRoPE3D uses tetrahedral directions (rotationally symmetric), so the encoding doesn't privilege any particular viewpoint orientation. However, the backprojection is always in camera frame. For allocentric tasks, the model must learn the frame transformation through LoRA + GRPO, which is possible but not guaranteed.

### 15.6 Training Data Considerations from the Survey

The survey identifies 21 spatially-oriented training datasets. Notable ones we should consider:

| Dataset | Scale | Type | Relevance |
|---------|-------|------|-----------|
| **SpatialVLM-data** (Chen et al., CVPR 2024) | 2M | Spatial QA from internet images + depth | Pre-alignment data |
| **SpatialReasoner-data** | 50K | Spatial reasoning with grounding | GRPO training data |
| **Sparkle** (2025) | Synthetic | Direction, localization, distance QA | Curriculum stage 1 data |
| **Open3DVQA** | Simulator-based | Urban 3D QA | Could adapt to indoor |

Our current training data plan (LLaVA-558K pre-alignment, R2R/RxR/SQA3D SFT, Habitat rollouts GRPO) may benefit from supplementing with dedicated spatial reasoning training data during SFT.

### 15.7 Benchmark Design Bias Warning

**Key Finding 2**: *"Current spatial reasoning benchmarks disproportionately emphasize vision-based relational tasks while underrepresenting metric and object-centric reasoning."*

Our architecture is specifically strong on **metric reasoning** (distances, sizes via GATr) which is exactly what current benchmarks underrepresent. This means:
- Standard benchmarks may not fully reveal our architecture's advantages
- We should prioritize benchmarks that test metric spatial reasoning (VSI-Bench distance subtask, SpatialRGPT-Bench, GeoMeter)
- We should report subtask-level breakdowns, not just aggregate scores

### 15.8 Competitive Models We Weren't Tracking

The survey reveals several models with approaches similar to ours:

| Model | Approach | Similarity to Ours |
|-------|----------|-------------------|
| **SD-VLM** (2025) | Depth Positional Encoding — encodes depth maps into depth-aware position embeddings, fused via element-wise addition | Closest analog to GridCellRoPE3D; direct comparison needed |
| **VCoder** (2024) | Versatile vision encoder with depth as additional input | Similar dual-encoder concept |
| **SpatialBot** (2024) | Customized depth module alongside frozen visual backbone | Similar to our GATr branch |
| **LLaVA-3D** (2024) | Empowering LMMs with 3D-awareness via 3D patch features | 3D-aware VLM at similar scale |
| **SSR** (2025) | Intermediate latent rationale tokens from depth maps guide generation | Similar depth-guided reasoning |
| **Cambrian-1** (2024) | SVA connector — our SVA is directly based on this | We must cite and compare |
| **VLM-3R** (CVPR 2026) | Multi-view → 3D reconstruction tokens | Strong VSI-Bench competitor |
| **Spatial-MLLM** (NeurIPS 2025) | Dual encoder (2D + 3D geometry) via CUT3R/VGGT | Almost identical dual-encoder strategy |

**Spatial-MLLM is our closest competitor** — it uses a 2D encoder + 3D spatial encoder with a 2D-3D fusion module, exactly our SigLIP + GATr + SVA pattern. Key differences: they use CUT3R/VGGT for 3D (from multi-view), we use GATr on GT depth. They report 48.4% on VSI-Bench.

### 15.9 Summary: Required Changes

| # | Action | Priority | Impact |
|---|--------|----------|--------|
| 1 | Add generalization safeguards: evaluate on survey's 9-benchmark suite to prove no degradation | **CRITICAL** | Addresses Key Finding 5 |
| 2 | Add MINDCUBE + SRBench to benchmark suite (extrapolation level) | HIGH | Addresses cognitive hierarchy gap |
| 3 | Add EgoOrientBench + OmniSpatial + SEED-Bench spatial to benchmark suite | HIGH | Enables cross-study comparison |
| 4 | Update What's Up framing in paper narrative (not universal near-chance anymore) | MEDIUM | Factual accuracy |
| 5 | Compare against SD-VLM, Spatial-MLLM, VLM-3R as key baselines | HIGH | Competitive positioning |
| 6 | Report subtask-level breakdowns on VSI-Bench and SQA3D | MEDIUM | Shows metric reasoning advantage |
| 7 | Consider supplementing SFT data with dedicated spatial reasoning data | MEDIUM | Addresses training data gap |
| 8 | Acknowledge allocentric limitation in paper's "Limitations" section | LOW | Intellectual honesty |
