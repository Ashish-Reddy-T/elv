# SpatialVLM: Recovering Destroyed Spatial Intelligence in Foundation Models for Indoor Navigation

## A Comprehensive Research Blueprint (Revised v3)

**Target venue**: NeurIPS / ICLR (pure representation learning and foundational architecture paper)
**Advisor context**: Mengye Ren's research prompt on spatial/physical intelligence
**Team**: 3 researchers, ~3,000 H100-hours
**Environment**: Indoor navigation — Habitat 3.0 / HM3D / Matterport3D

---

## 0. The Core Thesis

Foundation models destroy spatial information at five distinct pipeline stages — not because they lack spatial data, but because their architectural choices silently discard metric 3D structure before learning even begins. We demonstrate that fixing each stage with a targeted module — grid cell-inspired 3D positional encoding, geometric algebra spatial reasoning, dual vision encoders preserving both semantic and spatial features, norm-balanced cross-attention fusion, and RL post-training with dense spatial rewards — recovers spatial intelligence that no amount of additional training data can provide to standard architectures. The evidence is stark: randomly permuting all vision tokens in a state-of-the-art VLM causes only 0.2–2.74% performance drops, revealing that these models treat visual input as an unordered bag of features. Our architecture makes that same permutation catastrophic — proving that spatial structure is finally being used.

---

## 1. Stage 1 — Dual Vision Encoding

### 1.1 What happens to the image, precisely

The agent captures an egocentric RGB frame from Habitat's camera. This single image is resized to two resolutions and fed through two parallel frozen vision encoders.

**SigLIP 2 (SO400M/16-NaFlex at 384px)**: The image is divided into a grid of 16×16 pixel patches. Because 384 / 16 = 24.0 exactly, this produces a clean 24×24 = 576 patches with zero edge truncation or padding. Each patch is linearly projected into a 1152-dimensional vector through 27 transformer layers. We extract features from layers {9, 18, 27} — evenly-spaced thirds of the encoder — channel-concatenating to [B, 576, 3456], then projecting via a 2-layer MLP to [B, 576, 4096].

The NaFlex variant is used because it supports 16×16 patches at 384px, yielding exactly 576 tokens aligned to the SVA query grid. The original patch14 variant produces 384/14 = 27.43 patches per side — a non-integer requiring center-cropping to 378px or edge padding, both of which discard or fabricate spatial information at image boundaries.

These tokens encode *semantic, language-aligned* features. SigLIP 2 adds self-distillation and masked prediction objectives improving spatial localization over original SigLIP: +5 mIoU on segmentation, 70.10→86.21 on RefCOCO testA.

**DINOv2-L/14 at 518px**: Because 518 / 14 = 37.0 exactly, this produces a clean 37×37 = 1369 patches with zero edge artifacts. Features extracted from layers {8, 16, 24} are channel-concatenated to [B, 1369, 3072], then projected via a 2-layer MLP to [B, 1369, 4096]. **No spatial pooling is applied** — DINOv2's full 1369-token resolution is preserved. Downsampling to 576 would discard the fine-grained structural detail that is the entire reason for using DINOv2 alongside SigLIP.

These tokens encode *spatial, structural* features. El Banani et al. (CVPR 2024, "Probing 3D Awareness"): CLIP features do NOT encode depth — DINOv2 dramatically outperforms CLIP on depth, surface normals, and correspondence. Lexicon3D (NeurIPS 2024) confirmed DINOv2 outperforms CLIP even on language-related 3D QA despite having no language training.

### 1.2 Multi-layer feature extraction: why 3 layers

We extract 3 evenly-spaced layers per encoder, following Qwen3-VL's own deepstack which uses exactly 3 intermediate layers out of 27 as its empirically validated choice. 3 layers covers the full representational hierarchy (low-level spatial → mid-level structural → high-level semantic) while keeping MLP projector inputs tractable:

```
SigLIP  3 layers: [B, 576, 3×1152]  = [B, 576, 3456]  →  [B, 576, 4096]   (upsampling)
DINOv2  3 layers: [B, 1369, 3×1024] = [B, 1369, 3072]  →  [B, 1369, 4096]  (upsampling)
```

Both projectors upsample in the channel dimension — an easier learning problem than the compression that 5 layers would require (5760→4096, 5120→4096).

### 1.3 Pre-alignment

DINOv2 has never seen text. Pre-alignment trains a 2-layer MLP projector per encoder (1 epoch, ~50K image-caption pairs from LLaVA-558K) to map features into a space where they can predict the next text token. Encoders stay frozen; only projectors learn. Eagle (NVIDIA, ICLR 2025): DINOv2+SigLIP with pre-alignment outperforms either single encoder on 23/27 benchmarks. Without pre-alignment, DINOv2 features hurt performance because the LLM treats them as noise.

### 1.4 Stage 1 output — exact tensor shapes

```
SigLIP2-SO400M/16-NaFlex @ 384px:
  Input:                [B, 384, 384, 3]
  Patch grid:           384 / 16 = 24.0  →  24×24 = 576  ✓ exact integer, no edge discard
  Encoder hidden dim:   1152
  Encoder depth:        27 layers
  Extracted layers:     {9, 18, 27}
  Channel concat:       [B, 576, 3×1152] = [B, 576, 3456]
  MLP projector:        Linear(3456→4096) + GELU + Linear(4096→4096)  ~31M params
  Output:               [B, 576, 4096]

DINOv2-L/14 @ 518px:
  Input:                [B, 518, 518, 3]
  Patch grid:           518 / 14 = 37.0  →  37×37 = 1369  ✓ exact integer, no edge discard
  Encoder hidden dim:   1024
  Encoder depth:        24 layers
  Extracted layers:     {8, 16, 24}
  Channel concat:       [B, 1369, 3×1024] = [B, 1369, 3072]
  Spatial pooling:      NONE — full 1369 tokens preserved
  MLP projector:        Linear(3072→4096) + GELU + Linear(4096→4096)  ~29M params
  Output:               [B, 1369, 4096]
```

### 1.5 Key hypotheses to test

**H1a**: DINOv2 + SigLIP 2 outperforms either single encoder on all spatial benchmarks.
**H1b**: Pre-alignment is necessary for DINOv2 — without it, dual encoding performs worse than SigLIP alone.
**H1c**: Multi-layer extraction (3 layers) outperforms final-layer-only extraction on spatial tasks.
**H1d**: Preserving DINOv2 at full 1369 tokens outperforms pooling to 576 on spatial benchmarks.

---

## 2. Stage 2 — Geometric Branch: GT Depth → 3D Positions → GATr + GridCellRoPE3D

### 2.1 The parallel branch

This runs simultaneously with Stage 1. Habitat's depth sensor renders a metric depth map in the same simulation step as the RGB frame — zero additional compute. The three branches (SigLIP, DINOv2, Geometry) take their respective inputs from the same timestep independently.

### 2.2 Why Habitat GT depth, not a monocular estimator

We use Habitat's native metric depth sensor rather than Depth Anything V2. The paper's thesis is that spatial reasoning failure is an *architectural* problem, not a data or sensor problem. Injecting monocular depth estimation error (typically 5–15% RMSE indoors) would conflate architectural representation loss with sensor estimation error. GT depth isolates the variable cleanly: ablations attribute performance differences to the geometry modules rather than depth quality.

The real-world transfer question — how much does switching to estimated depth cost? — is answered by a single diagnostic ablation row (H2c) comparing GT vs. Depth Anything V2. This is a secondary finding, not a core system dependency. Depth Anything V2 is removed from the main architecture entirely.

The depth map is rendered at 518×518 to match DINOv2's input resolution, ensuring pixel-perfect spatial registration between the depth map and DINOv2's 37×37 patch grid.

### 2.3 Backprojection with foreground-biased aggregation

For each pixel (u, v) with depth d, using Habitat's camera intrinsics K = (fx, fy, cx, cy):

```
X = (u - cx) × d / fx
Y = (v - cy) × d / fy
Z = d
```

Full 3D point map: [B, 518×518, 3] = [B, 268324, 3] in camera coordinates.

**Why the DINOv2 patch grid for aggregation**: We aggregate to one 3D position per 14×14 pixel region — exactly the DINOv2 patch grid. This means GATr token (i,j) and DINOv2 token (i,j) describe the same image region. The SVA can trivially learn to fuse geometric and structural descriptions for the same location, rather than having to learn a soft interpolation between misaligned grids.

**15th-percentile foreground-biased aggregation**: Rather than averaging pixel depths within each 14×14 patch — which biases the centroid toward background surfaces (far walls, open doorways, ceiling) — we take the **15th percentile** depth within each region.

- Pure mean: biased toward distant background. A patch spanning a doorframe and open air behind it has mean depth ~3.2m.
- Pure minimum (0th percentile): too sensitive to single noisy/specular pixels out of 196.
- 15th percentile: captures the nearest foreground surface reliably, robust to up to ~29 outlier pixels per patch. The same patch has 15th-percentile depth ~0.4m (the doorframe itself).

GATr processing 0.4m correctly identifies the obstacle and its distance. GATr processing 3.2m encodes a phantom surface in empty space.

The 15th-percentile depth d_patch is backprojected using patch center coordinates (u_c, v_c):

```
X_patch = (u_c - cx) × d_patch / fx
Y_patch = (v_c - cy) × d_patch / fy
Z_patch = d_patch

Output: [B, 1369, 3]  — one foreground-biased 3D position per DINOv2 patch
```

### 2.4 What GATr features encode vs. Stage 1 features

Stage 1 features encode **appearance**: texture, color, semantic category, object identity. They are 4096-dim vectors in the LLM's learned embedding space.

GATr features encode **geometry**: distances between objects, angles of surfaces, relative positions, coplanarity. They operate in Projective Geometric Algebra G(3,0,1) with 16 basis components: 1 scalar, 4 planes, 6 lines, 4 points, 1 pseudoscalar.

Concrete example: a patch showing part of a doorframe has Stage 1 features "wooden surface, brown, likely a door" and GATr features "at (2.3, 0.1, 1.8)m, oriented vertically, 0.4m from adjacent wall, coplanar with patches above and below." Semantics tell the model what the door is; geometry tells it exactly where it is and whether the agent can pass through.

**How GATr processes**: Each [x,y,z] is embedded as a PGA trivector (e₀₁₂=x, e₀₁₃=y, e₀₂₃=z, e₁₂₃=1). Eight equivariant transformer blocks process these — rotation of the entire point cloud rotates output features correspondingly. Equivariant linear layers use 9 learnable parameters per (in,out) channel pair vs. 256 for standard linear. After 8 blocks, invariant features are extracted: scalar quantities (distances, angles, norms) that are unchanged under rotation.

Code: `github.com/Qualcomm-AI-research/geometric-algebra-transformer` (BSD-3, PyTorch). Config: 8 blocks, 16 multivector channels, 32 scalar channels.

### 2.5 GATr dimensional trace

```
Input:                    [B, 1369, 3]

PGA embedding:
  Each [x,y,z] → trivector: e₀₁₂=x, e₀₁₃=y, e₀₂₃=z, e₁₂₃=1, rest=0
  Multivector stream:     [B, 1369, 16, 16]  (16 mv channels × 16 PGA basis dims)
  Scalar stream:          [B, 1369, 32]

8 equivariant transformer blocks:
  Equivariant self-attention + equivariant FFN per block
  Rotation equivariance preserved throughout ✓
  Params per equivariant linear: 9 vs 256 for standard (28× more efficient)

Extract invariant features (rotation-independent scalars):
  32 scalar channel values:   [B, 1369, 32]
  16 multivector L2-norms:    [B, 1369, 16]
  Total invariant:            [B, 1369, 48]

MLP projection:
  Linear(48→4096) + GELU + Linear(4096→4096)  ~17M params

Output:                       [B, 1369, 4096]
  Spatially co-registered with DINOv2 37×37 token grid ✓
```

### 2.6 GridCellRoPE3D

GridCellRoPE3D does NOT produce features — it produces rotation matrices applied to Q and K inside the LLM's attention layers (Stage 4). The 3D positions from backprojection are stored and used later.

Standard RoPE: token at sequence position m gets rotation R(m). Two tokens at positions m,n have attention governed by (m-n). GridCellRoPE3D: spatial token at 3D position **p**=(x,y,z) gets rotation from multi-scale periodic functions of **p**. Two tokens at **p₁**, **p₂** have attention governed by ‖**p₁**-**p₂**‖ — physical 3D distance. Spatially nearby tokens attend more to each other, encoding genuine 3D adjacency rather than arbitrary sequence order.

**Dimensional trace**:

```
Input:                    [B, 1369, 3]

4 tetrahedral directions (regular tetrahedron vertices on unit sphere):
  d₁ = [+1,+1,+1]/√3
  d₂ = [+1,−1,−1]/√3
  d₃ = [−1,+1,−1]/√3
  d₄ = [−1,−1,+1]/√3

Per position p, scalar projection:  sᵢ = dᵢ · p  →  [B, 1369, 4]

8 frequency modules, golden ratio spacing (φ=1.618):
  fₖ = 10 × φᵏ  for k=0..7
  Periods (m): [0.10, 0.16, 0.26, 0.42, 0.69, 1.11, 1.80, 2.91]
  Covers sub-object detail (10cm) through room-scale (2.91m)

Encoding per token:
  [sin(2π fₖ sᵢ), cos(2π fₖ sᵢ)]  for all (i,k)
  4 directions × 8 frequencies × 2 = 64 rotary dims

Qwen3-VL M-RoPE: head_dim=128  →  64 rotary pairs
  mrope_section = [24 time, 20 height, 20 width]  (sums to 64 ✓)

Routing:
  Text tokens:    standard M-RoPE [24t, 20h, 20w] — unchanged
  Spatial tokens: all 64 rotary pairs replaced with GridCellRoPE3D
  Mapping:        4×8×2 = 64 = Qwen3's 64 rotary pairs ✓  no projection needed

Stored output:    [B, 1369, 64]  applied as Q,K rotations in Stage 4
Parameters:       0  (pure computation on GT positions)
```

φ=1.618 (golden ratio): maximally incommensurate — no two frequencies share a common period, delaying aliasing longer than any rational ratio. The 8 periods span the range from individual object detail through full indoor room scale.

### 2.7 Stage 2 output

```
GATr invariant features:     [B, 1369, 4096]
GridCellRoPE3D rotations:    [B, 1369, 64]   (stored, applied in Stage 4)
```

### 2.8 Key hypotheses to test

**H2a**: GATr features are complementary to vision encoder features — removing GATr degrades spatial benchmarks but not semantic benchmarks.
**H2b**: GridCellRoPE3D outperforms standard M-RoPE for all tokens and also outperforms 3D sinusoidal positional encoding.
**H2c** (diagnostic ablation): GT depth (Habitat) vs. Depth Anything V2 — quantifies real-world transfer gap. Primary results use GT depth; this ablation measures the cost of removing the simulation advantage.
**H2d**: Golden ratio scale ratio (φ=1.618) outperforms √e≈1.649 and power-of-2 spacing used in NeRF.
**H2e**: 15th-percentile foreground-biased aggregation outperforms mean aggregation on spatial benchmarks.

---

## 3. Stage 3 — Norm-Balanced Cross-Attention Fusion

### 3.1 The problem, precisely quantified

"Beyond Semantics" (arXiv 2503.17349) measured vision token L₂ norms at 10–100× larger than text token norms in LLaVA 1.5 7B after MLP projection. Despite this, "Why Is Spatial Reasoning Hard for VLMs?" (arXiv 2503.01773, ICML 2025) found that although image tokens comprise ~90% of the input sequence, they receive only ~10% of attention. Randomly permuting all 576 vision tokens causes only 0.2–2.74% drops across VQAv2, POPE, GQA, and CV-Bench. The model cannot tell if you scramble all spatial ordering of visual information.

### 3.2 Cross-attention validated for spatial fusion

Three independent 2025–2026 papers confirm cross-attention is better than concatenation for spatial tasks:

**Spa3R** (February 2026): Their Residual Cross-Attention Adapter outperforms naive token appending by +7.5% on VSI-Bench. The pre-trained VLM suffers "modality collapse" — it preferentially attends to familiar visual tokens and ignores alien spatial latents. Cross-attention forces the VLM to actively query relevant spatial information.

**SpaceMind** (December 2025): Camera-Guided Modality Fusion uses cross-attention between spatial tokens and visual tokens. Ablation shows each cross-attention component adds cumulative benefit. SOTA on VSI-Bench, SQA3D, and SPBench.

**LVLDrive** (December 2025): "Gradual Fusion Q-Former" incrementally injects spatial embeddings via gated attention, allowing the model to adapt without drifting from its learned visual-linguistic manifold. Directly validates our zero-init gated cross-attention approach.

### 3.3 The fusion architecture, step by step

**Inputs**:
```
SigLIP tokens:    [B, 576, 4096]    semantic,   24×24 grid
DINOv2 tokens:    [B, 1369, 4096]   structural, 37×37 grid, full resolution
GATr tokens:      [B, 1369, 4096]   geometric,  37×37 grid, co-registered with DINOv2
```

**Step 1 — Spatial Vision Aggregator (SVA)**

Following Cambrian-1, 576 learnable query tokens are organized in a 24×24 spatial grid. Each query is explicitly associated with a spatial sub-region of the feature maps. The K/V source is all three feature streams concatenated along the token dimension:

```
K/V source:  concat([576 SigLIP, 1369 DINOv2, 1369 GATr])
           = [B, 3314, 4096]

Learnable queries:  [B, 576, 4096]  (24×24 grid, nn.Parameter)

Cross-attention layer 1:
  Q:  [B, 576, 4096]   → 32 heads × 128 dims
  K:  [B, 3314, 4096]  → 32 heads × 128 dims
  V:  [B, 3314, 4096]
  Attention map:  [B, 32, 576, 3314]
  Output:         [B, 576, 4096]   (queries updated)

Cross-attention layer 2:
  Q:  [B, 576, 4096]   (output of layer 1)
  K/V: [B, 3314, 4096] (original source, unchanged)
  Output: [B, 576, 4096]
```

Attention cost comparison:
```
SVA:           O(576  × 3314) = ~1.9M ops per head  →  5.7× cheaper
Self-attention: O(3314 × 3314) = ~11.0M ops per head
```

**Typed attention bias (9 parameters)**: Added to SVA cross-attention logits to prevent any single modality from dominating during early training:

```
type ∈ {semantic (SigLIP), structural (DINOv2), dense_geometric (GATr)}
Attn(Q,K) += bias[type_Q, type_K]   ← learned 3×3 scalar matrix
Parameters: 9
```

This stabilizes multi-encoder fusion without adding architectural complexity. The bias matrix is learned during Stage 2 SFT and updated through GRPO.

**Step 2 — RMS Norm Matching**

After SVA, the fused vision tokens are scaled so their RMS norm matches the running exponential moving average of text token norms in the LLM. Zero parameters, negligible compute.

```
σ_text   = EMA of RMS norms over text token embeddings (updated each training step)
σ_vision = RMS norm of current [B, 576, 4096] fused tokens
output   = input × (σ_text / σ_vision)
Output:  [B, 576, 4096]  norms now match text token magnitude distribution
```

Vision tokens are 10–100× larger than text tokens in raw magnitude. Without this step, visual data "screams" at the LLM, drowning out text and destabilizing attention patterns even if the gated cross-attention gates start at zero.

**Step 3 — Gated Cross-Attention**

Inserted at every 4th block of Qwen3-VL-7B's 36 transformer layers: {4, 8, 12, 16, 20, 24, 28, 32, 36} = 9 injection points. At each injection layer, the LLM's current hidden states query the 576 fused spatial tokens:

```
h_new = h_old + tanh(α) × CrossAttn(Q=h_old, K=spatial_576, V=spatial_576)

α = nn.Parameter(scalar, init=0.0)

At init:       tanh(0) = 0    →  h_new = h_old  (pure pretrained Qwen3 ✓)
Converged:     tanh(large) → ±1  →  full cross-attention signal blends in
```

The spatial tokens [B, 576, 4096] are **static** across all 9 injection layers — they are the output of SVA computed once before the LLM forward pass. Each injection layer reads from the same fixed spatial memory. This is the Text-attending-to-Vision pathway; SVA was the Vision-attending-to-Vision pathway.

**Step 3 output**: LLM hidden states progressively enriched with spatial information through 9 injection points. The LLM's own self-attention operates normally on all tokens; cross-attention provides an additional spatial signal channel at each injection depth.

### 3.4 Key hypotheses

**H3a**: Gated cross-attention fusion outperforms LLaVA-style concatenation by >5% absolute on spatial benchmarks.
**H3b**: RMS norm matching provides additional measurable benefit on top of cross-attention.
**H3c**: Permutation test — shuffling vision token order causes >15% performance drop with our architecture vs. <3% for baseline VLMs.
**H3d**: Typed attention bias (9 params) improves training stability and final performance vs. no bias.
**H3e**: Preserving DINOv2+GATr at 1369 tokens (3314-token KV pool) outperforms pooling all to 576 (1728-token KV pool).

---

## 4. Stage 4 — LLM Backbone with GridCellRoPE3D + Action Decoding

### 4.1 Backbone: Qwen3-VL-7B

```
hidden_size:            4096   ← D_proj = 4096 matches exactly ✓
num_hidden_layers:      36
num_attention_heads:    32
num_key_value_heads:    8      (GQA, 4:1 ratio)
head_dim:               128
intermediate_size:      12288  (FFN)
positional encoding:    M-RoPE, mrope_section=[24t, 20h, 20w], 64 rotary pairs total
```

### 4.2 Position routing inside the LLM

The LLM processes a mixed sequence:

```
[<system_tokens> | <instruction_tokens> | <spatial_vision_tokens> | <history_tokens>]
     ~128               ~64                       576                     ~32
Total T ≈ 800 tokens at inference
```

Each token carries a type flag {TEXT, SPATIAL}:

**Text tokens**: Standard M-RoPE with mrope_section=[24t, 20h, 20w] applied at all attention layers. Sequential position encodes language coherence.

**Spatial tokens**: GridCellRoPE3D replaces all 64 M-RoPE rotary pairs for spatial tokens. The 4×8×2=64 encoding dimensions map exactly to Qwen3's 64 rotary pairs — no projection layer needed. Physical 3D distance governs spatial token attention, not sequence position. The M-RoPE temporal section (24 dims) and spatial sections (20h+20w=40 dims) are entirely overridden for spatial tokens.

Cross-modal attention (text hidden states querying spatial tokens via gated cross-attention) is handled by Stage 3 modules — not by the positional encoding system.

### 4.3 Gated cross-attention: per-layer parameter detail

```
Injection layers: {4,8,12,16,20,24,28,32,36} in 36-layer Qwen3 → 9 layers total
  Note: layer 36 is the final transformer block — ablate whether to include.
  Ablation variant: inject at {4,8,12,16,20,24,28,32} = 8 layers, skip final.

Per injection layer:
  Q: LLM hidden states        [B, T, 4096]
  K/V: SVA spatial tokens     [B, 576, 4096]  (static, same across all 9 layers)

  GQA: 32 query heads, 8 KV heads
  W_Q:  4096 × 4096 = 16,777,216 params
  W_K:  4096 × (8×128) = 4096 × 1024 = 4,194,304 params
  W_V:  4096 × 1024 = 4,194,304 params
  W_O:  4096 × 4096 = 16,777,216 params
  α:    1 scalar param
  Total per layer: ~41.9M params

9 layers: ~377M params
```

### 4.4 LoRA on Qwen3

```
Applied to Q, K, V, O projections in all 36 transformer blocks.
rank=32, alpha=64  →  effective LR scale = alpha/rank = 2.0

Per attention layer:
  W_Q LoRA: (4096×32) + (32×4096) = 262,144 params
  W_K LoRA: (4096×32) + (32×1024) = 163,840 params  (GQA: K dim = 1024)
  W_V LoRA: (4096×32) + (32×1024) = 163,840 params
  W_O LoRA: (4096×32) + (32×4096) = 262,144 params
  Per layer: 852,032 params  (~852K)

36 layers: ~30.7M LoRA params total
```

### 4.5 Full parameter table

```
Component                          Params       Trainable
─────────────────────────────────────────────────────────
SigLIP2-SO400M encoder             400M         Frozen
DINOv2-L encoder                   307M         Frozen
Habitat GT depth sensor              0          N/A (simulator)
GATr (8 equivariant blocks)         ~12M        Trainable Stage 2+
SigLIP MLP projector                ~31M        Trainable Stage 1+
DINOv2 MLP projector                ~29M        Trainable Stage 1+
GATr MLP projector (48→4096)        ~17M        Trainable Stage 1+
SVA learnable queries [576×4096]     2.4M       Trainable Stage 3+
SVA cross-attn (2 layers, KV=3314)  ~84M        Trainable Stage 3+
SVA typed attention bias (3×3)       0.000M     Trainable Stage 3+
Gated cross-attn (9 layers)         ~377M       Trainable Stage 3+
Qwen3-VL-7B backbone              7,600M        Frozen
Qwen3 LoRA rank-32                   ~31M       Trainable Stage 2+
─────────────────────────────────────────────────────────
Total:                            ~8,890M
Trainable (full training):          ~583M       (6.6% of total)
Frozen:                           ~8,307M
```

### 4.6 Action decoding options

**Option A — Mid-level language commands (NaVILA-style)**: Output strings like "turn right 30 degrees" or "move forward 75cm." A separate RL-trained locomotion controller translates to motor commands. Most natural fit for a VLM. NaVILA: 47%+ SR on R2R-CE. No special vocabulary needed.

**Option B — Discrete multi-step prediction (Uni-NaVid-style)**: Output "FORWARD FORWARD TURN_LEFT FORWARD" — 4 future actions simultaneously. Compatible with Habitat's discrete action space: {FORWARD 25cm, TURN_LEFT 30°, TURN_RIGHT 30°, STOP}. Simplest to implement and evaluate.

**Option C — Chain-of-thought + action**: "I see the kitchen counter ahead and to the left. The target mug is likely on the counter. I should turn left. Action: TURN_LEFT." Most compatible with GRPO — reward both reasoning quality and action correctness separately.

Recommended order: B first (fastest iteration), then C (enables GRPO with reasoning rewards), then A (requires separate locomotion controller).

### 4.7 Single-frame inference cost

```
SigLIP2 encode:        ~8ms   (400M frozen, 576 tokens)
DINOv2 encode:        ~18ms   (307M frozen, 1369 tokens — no pooling)
GT depth + backproject: ~1ms  (free from simulator + fast percentile arithmetic)
GATr (1369 tokens):    ~5ms   (12M, equivariant ops)
MLP projectors ×3:     ~2ms
SVA (KV=3314):         ~6ms   (576 queries attend 3314 KV tokens)
RMS norm match:        ~0ms
Qwen3 decode:         ~25ms   (7.6B, BF16, 9 cross-attn injections)
────────────────────────────
Total per step:       ~65ms   →  ~15 steps/second on one H100
```

### 4.8 Key hypotheses

**H4a**: GridCellRoPE3D for spatial tokens improves navigation SR vs. standard M-RoPE for all tokens.
**H4b**: CoT decoding (Option C) benefits more from GRPO than direct decoding (Option B).
**H4c**: 4-step multi-step prediction outperforms single-step in SPL due to reduced VLM inference calls.

---

## 5. Stage 5 — RL Post-Training with Dense Spatial Rewards

### 5.1 Why RL

RL substantially outperforms SFT for spatial reasoning, especially out-of-distribution. SpatialReasoner (NeurIPS 2025): SFT achieves 58.3% but a second SFT round drops to 54.7%, while RL after SFT improves to 60.3%. Spatial tasks have multiple valid solutions (many paths reach the goal), so SFT overfits to the single demonstrated solution while RL discovers the full solution distribution.

### 5.2 GRPO vs. fDPO

**GRPO** (Group Relative Policy Optimization): eliminates the value network — sample G=8 outputs per prompt, compute rewards, normalize advantages within the group, optimize clipped surrogate objective. Validated for spatial reasoning on CLEVR (R1-V) and shown by MultihopSpatial (March 2026) to transfer to embodied VLA tasks. Well-supported in HuggingFace TRL GRPOTrainer.

**Known issue — Vanishing Advantages**: VL-Rethinker (2025) found that as training progresses, more groups yield zero advantages, shrinking effective batch size. Fix: Selective Sample Replay (SSR) — maintain a replay buffer of high-advantage experiences. Directly relevant for navigation where trajectory rewards can be very similar across rollouts.

**fDPO** (fine-grained DPO, SpatialReasoner-R1, NeurIPS 2025): applies different optimization pressures to different output segments — stricter updates for descriptive grounding ("where is the object?"), different updates for logical reasoning ("therefore I should turn left"). +4.1% over standard DPO on spatial quality tasks, +9.0% on spatial quantity tasks. Requires segmenting outputs into grounding vs. reasoning phases.

**Recommendation**: Primary = GRPO (best tooling, most comparable baselines). Novel comparison = fDPO. Control = SFT-only. This three-way comparison is itself a contribution.

### 5.3 Training stages

```
STAGE 1 — Pre-alignment (frozen everything except projectors)
  Frozen:       SigLIP, DINOv2, GATr encoders, all of Qwen3
  Trainable:    SigLIP MLP projector (~31M)
                DINOv2 MLP projector (~29M)
                GATr MLP projector (~17M)
  Total:        ~77M trainable params
  Data:         LLaVA-558K subset (~50K pairs), 1 epoch
  Objective:    next-token prediction via small frozen LM head

STAGE 2 — SFT
  Frozen:       all encoders, Qwen3 backbone weights
  Trainable:    all Stage 1 params
              + GATr 8-block encoder (~12M)
              + SVA queries + cross-attn (~86M)
              + Typed attention bias (9 params)
              + Gated cross-attn layers (~377M)
              + Qwen3 LoRA rank-32 (~31M)
  Total:        ~583M trainable params
  Data:         R2R-CE, RxR-CE, ScaleVLN, SQA3D, spatial reasoning data

STAGE 3 — GRPO RL
  Trainable:    same 583M as SFT
  G=8 rollouts per prompt
  Clip ε=0.2,  KL β=0.001
  LR=5e-7,  batch=32 prompts per step
  Max response length=2048 tokens
  BF16 precision (FP8 on H100 for ~3× speedup)
  Framework: HuggingFace TRL GRPOTrainer or verl

  Per training step:
    32 prompts × 8 rollouts = 256 Habitat episodes executed
    Each episode: up to ~20 steps of [encode → decode → act → observe]
    Rewards computed per trajectory
    Advantages normalized within each group of 8
    Clipped surrogate loss backward through 583M params
```

### 5.4 Dense spatial reward function

```
R_total = R_format + R_progress + R_collision + R_goal + R_consistency

R_format      =  +1.0   if output follows structured template, else 0.0  [per step]
R_progress    =  geodesic_dist(prev→goal) − geodesic_dist(curr→goal)     [dense, per step]
R_collision   =  −2.0   if min_clearance < 0.1m                          [per step]
R_goal        =  +10.0  if geodesic_dist < 1.0m AND agent calls STOP     [sparse, terminal]
R_consistency =  −1.0   if reasoning trace contradicts final action       [from Embodied-R]
```

Geodesic distance (shortest navigable path, not Euclidean) computed by Habitat's built-in shortest-path module. The consistency reward (Embodied-R, ACM MM 2025) prevents reward hacking where the model writes plausible-sounding CoT but takes random actions.

### 5.5 Staged curriculum

Epochs 1–2: format=1.0, accuracy=0.1, spatial=0.0 — learn structured output format first.
Epochs 3–4: format=0.3, accuracy=0.5, spatial=0.2 — learn correct actions.
Epochs 5–6: format=0.1, accuracy=0.3, spatial=0.6 — learn precise spatial reasoning.

Follows SpatialThinker (1.8× RL gain with dense vs. sparse rewards) and Embodied-R (o1-level spatial performance with only 5K samples using staged RL).

### 5.6 Key hypotheses

**H5a**: SFT+GRPO outperforms SFT-only on all navigation and spatial benchmarks.
**H5b**: fDPO outperforms standard GRPO on spatial quality and quantity tasks.
**H5c**: Dense spatial rewards (R_progress) provide larger gains than sparse rewards (R_goal only).
**H5d**: Consistency reward (R_consistency) prevents reward hacking — without it, models learn to write plausible CoT that contradicts their actions.

---

## 6. How This Competes Against JEPA — Honestly

### 6.1 What JEPA/V-JEPA actually does

V-JEPA learns to predict representations of future video frames in latent space, without pixel reconstruction. V-JEPA 2.1 (March 2026) includes navigation trajectory planning in latent space with strong results on depth estimation, navigation, and grasping. Core principle: learn an internal world model by predicting what happens next in a learned representation space.

### 6.2 Where JEPA is strong and we are weak

JEPA's strength is **learned dynamics** — predicting how scenes change without explicit 3D reconstruction. It operates purely in latent space, scales well, and doesn't need camera intrinsics or explicit geometry. V-JEPA 2.1 plans navigation trajectories directly in latent space.

Our weaknesses: we require explicit geometric processing (backprojection, GATr), adding complexity. We rely on a pretrained LLM backbone not designed for spatial reasoning. We currently operate frame-by-frame without temporal memory across steps.

### 6.3 Where we are strong and JEPA is weak

JEPA's weakness is **interpretability and compositional precision**. Its latent representations are opaque — you cannot ask "how far is the door?" and get a metric answer. It struggles with compositional spatial reasoning ("the mug is on the table which is to the left of the couch") — flat latent spaces don't naturally represent structured compositional relationships. V-JEPA has no mechanism for geometric primitives (points, lines, planes) or their algebraic relationships.

Our strengths: (a) metric 3D understanding via GT depth + GATr, (b) compositional spatial reasoning via language-based CoT, (c) interpretable intermediate representations (you can visualize the point cloud, GATr attention patterns, SVA attention maps), (d) natural language instruction following and explanation.

### 6.4 The honest framing: complementary, not competing

V-JEPA provides the temporal dynamics model — predicting how the world changes as the agent moves. Our spatial modules provide the structured spatial state representation that dynamics models operate over. V-JEPA answers "what will I see next if I move forward?" Our architecture answers "where exactly is the door, how far is it, and can I fit through it?" A complete navigation system needs both.

For AMI specifically: V-JEPA's latent space could be structured using grid cell-inspired encodings (our GridCellRoPE3D) to give it metric spatial awareness. Our RL post-training methodology could be applied to V-JEPA for embodied navigation. These are different layers of the same stack, not competitors.

### 6.5 Where we realistically win and lose

We win on: benchmarks requiring precise metric spatial reasoning (distance estimation, size comparison, spatial relation classification), the What's Up benchmark where ALL models score near-chance, VLN-CE where language understanding meets spatial navigation.

We lose on: pure visual navigation without language (PointNav — V-JEPA doesn't need language processing overhead), long-horizon planning requiring temporal prediction across many steps.

---

## 7. Ablation Study Design

**Full model**: All modules active (GridCellRoPE3D + GATr + DINOv2+SigLIP + SVA + gated cross-attention + GRPO).

**Architecture ablations** (one module removed at a time):
- *No GridCellRoPE3D*: Standard M-RoPE for all tokens. Tests H2b.
- *No GATr*: Skip geometric branch entirely. Tests H2a.
- *SigLIP only*: Remove DINOv2 encoder. Tests H1a.
- *DINOv2 only*: Remove SigLIP encoder. Tests H1a.
- *DINOv2 pooled to 576*: Apply 37×37→24×24 adaptive spatial pooling before projector, revert KV pool to 1728 tokens. Tests H1d and H3e.
- *Concatenation fusion*: Replace SVA + gated cross-attention with LLaVA-style concatenation + MLP. Tests H3a.
- *No RMS norm matching*: Remove norm balancing step. Tests H3b.
- *No typed attention bias*: Remove the 3×3 type bias matrix from SVA. Tests H3d.
- *SFT only*: No GRPO post-training. Tests H5a.

**Diagnostic ablations** (deeper investigation):
- *Permutation test*: Randomly shuffle vision token order. Should be catastrophic (>15% drop) with our architecture, negligible (<3%) for baseline VLMs. Tests H3c.
- *GT depth vs. estimated depth*: Swap Habitat GT depth for Depth Anything V2. Quantifies real-world transfer gap. Tests H2c.
- *Mean vs. 15th percentile backprojection*: Tests H2e.
- *Scale ratio sweep*: φ=1.618 vs. √e≈1.649 vs. power-of-2 for GridCellRoPE3D frequencies. Tests H2d.
- *GRPO vs. fDPO vs. SFT*: Three training paradigms. Tests H5b.
- *Dense vs. sparse rewards*: Full R_total vs. R_goal only. Tests H5c.
- *Injection frequency*: Every 4th block (9 points) vs. every 6th (6 points) for gated cross-attention. Compute vs. performance tradeoff.

**Baseline comparisons**:
- Vanilla Qwen3-VL-7B + Habitat wrapper (no spatial modules).
- NaVid (video-based VLM navigation).
- Efficient-VLN (VLN-CE R2R SOTA, 64.2% SR).
- NavFoM (cross-embodiment navigation foundation model, 64.9% SR).

---

## 8. Benchmarks, Datasets, and Code

### 8.1 Benchmarks

**Spatial reasoning diagnostics** (proving the thesis):
- **What's Up** (EMNLP 2023): 820 questions on basic spatial relations. All VLMs ~56% (near-chance). Humans 99%. Smoking-gun benchmark — significant improvement here makes the paper.
- **CV-Bench**: Spatial reasoning + counting in 2D images.
- **SpatialBench**: Metric spatial tasks (distance estimation, size comparison).
- **MultihopSpatial** (March 2026): Multi-hop compositional spatial reasoning + grounding. GPT-5.2-Thinking drops to 9.4% on 3-hop grounded spatial reasoning. Brand new, high impact.
- **VSI-Bench**: Visual spatial intelligence from video.

**Navigation benchmarks** (proving embodied value):
- **VLN-CE R2R val-unseen**: SR, SPL. SOTA: 64.2% SR (Efficient-VLN).
- **VLN-CE RxR val-unseen**: Longer multilingual instructions. SOTA: ~67% SR.
- **ObjectNav HM3D**: Find a specified object category. SOTA: ~42% SR.
- **NavTrust** (March 2026): Robustness under corruptions (motion blur, depth noise). Current models: up to 25% SR degradation.

**World model alignment** (for AMI framing):
- **SQA3D**: Situated 3D QA from reconstructed indoor scenes (ScanNet).
- **SPBench**: Spatial perception benchmark.

### 8.2 Datasets

**Navigation**: R2R-CE (21K human instructions, Matterport3D), RxR-CE (126K multilingual instructions), ScaleVLN (4.9M synthetic augmented instruction-trajectory pairs).

**Spatial reasoning**: SQA3D (33K questions on ScanNet), ScanQA, SpatialReasoner training set, MultihopSpatial-Train.

**General VLM** (pre-alignment + SFT): LLaVA-558K (pre-alignment), LLaVA-Video-178K subset, VLN-R1 egocentric dataset (180K+ step-wise samples).

### 8.3 Code repositories

**Simulator**: `github.com/facebookresearch/habitat-sim`, `github.com/facebookresearch/habitat-lab`, `github.com/facebookresearch/habitat-challenge`.

**Architecture**: GATr at `github.com/Qualcomm-AI-research/geometric-algebra-transformer` (BSD-3, PyTorch, xformers compatible), GridPE at `github.com/BoyangL1/Gridpe` (2D reference — extend to 3D), conformal isometry theory at `github.com/DehongXu/grid-cell-conformal-isometry`, Eagle dual encoder at `github.com/zhiqi-li/EAGLE2`.

**Training**: TRL GRPOTrainer at `github.com/huggingface/trl`, RL-for-VLM survey at `github.com/Sun-Haoyuan23/Awesome-RL-based-Reasoning-MLLMs`.

**Baselines**: VLN-CE at `github.com/jacobkrantz/VLN-CE`.

Note: Depth Anything V2 (`github.com/DepthAnything/Depth-Anything-V2`) retained for the GT vs. estimated depth diagnostic ablation only, not used in the main architecture.

---

## 9. Compute Budget (3,000 H100-hours)

| Phase | Weeks | H100-hours | Who |
|-------|-------|------------|-----|
| Setup: Habitat 3.0, baseline reproduction | 1–2 | 50 | All 3 |
| Data generation (expert trajectories, ScaleVLN processing) | 2–4 | 150 | P1 |
| Module implementation (GridCellRoPE3D, GATr integration, dual encoders, SVA) | 2–6 | 100 | P2: Modules 1–2, P3: Modules 3–4 |
| Pre-alignment Stage 1 (projectors only, ~77M params) | 3–4 | 50 | P1 |
| SFT Stage 2 (full 583M trainable params) | 4–7 | 400 | P1 |
| SFT Stage 2 extended (DAgger-collected data, deeper fine-tune) | 5–8 | 350 | P2+P1 |
| GRPO RL Stage 3 (staged curriculum, 6 epochs) | 7–12 | 1,200 | P3: rewards, P1: training |
| fDPO comparison run | 9–12 | 200 | P3 |
| Evaluation, ablations (10+ ablation variants), error analysis | 10–14 | 500 | All 3 |
| Buffer (debugging, retraining, unexpected experiments) | Ongoing | 200 | — |
| **Total** | **14 weeks** | **3,200** | |

Note: budget is slightly over 3,000 due to the fDPO comparison run being explicitly added. Recommend deprioritizing one SFT phase or reducing DAgger collection if compute is tight.

---

## 10. Summary

This paper's structure tells a clean story: diagnose the five architectural stages at which foundation models destroy spatial information → fix each with a targeted, principled module → prove each fix works in isolation via ablation → show the combined system achieves SOTA on spatial benchmarks where all current models score near-chance. The theoretical backbone — grid cells = Fourier features = positional encodings, all instances of the same optimal spatial code — provides the "why" behind the engineering choices, not just empirical tuning.

The framing is deliberately for NeurIPS/ICLR, not CVPR/ICCV or ICRA/CoRL. This is a representation learning paper. The question being answered is not "how do we build a robot?" but "do standard neural network architectures represent 3D space at all, and if not, what is the minimal set of architectural changes required to recover this?" The permutation test finding — that scrambling all vision tokens barely affects SOTA VLMs — is this paper's core empirical shock, analogous to the lottery ticket or double descent findings. Once pointed out it seems obvious. The contribution is pointing it out systematically and fixing it.

The complementarity framing with V-JEPA positions the work within AMI's mission rather than against it: V-JEPA learns temporal dynamics of how the world changes; our architecture learns the structured spatial state the world is in. Both are necessary. Neither is sufficient alone.

---

## Appendix: Complete Dimensional Reference

```
STAGE 1 — DUAL VISION ENCODING
───────────────────────────────────────────────────────────────
Input (SigLIP):             [B, 384, 384, 3]
Patch grid:                 384/16 = 24.0  →  24×24 = 576  (exact, no edge artifacts)
Encoder hidden:             1152-dim
Encoder depth:              27 layers
Extracted layers:           {9, 18, 27}
Channel concat:             [B, 576, 3456]
MLP projector:              Linear(3456→4096) + GELU + Linear(4096→4096)
Projector params:           ~31M
Output (SigLIP):            [B, 576, 4096]
Spatial pooling applied:    NONE

Input (DINOv2):             [B, 518, 518, 3]
Patch grid:                 518/14 = 37.0  →  37×37 = 1369  (exact, no edge artifacts)
Encoder hidden:             1024-dim
Encoder depth:              24 layers
Extracted layers:           {8, 16, 24}
Channel concat:             [B, 1369, 3072]
MLP projector:              Linear(3072→4096) + GELU + Linear(4096→4096)
Projector params:           ~29M
Output (DINOv2):            [B, 1369, 4096]
Spatial pooling applied:    NONE  ✓ full resolution preserved

STAGE 2 — GEOMETRIC BRANCH
───────────────────────────────────────────────────────────────
Depth source:               Habitat GT sensor, rendered at 518×518
Backprojection:             [B, 268324, 3]  (all pixels)
Aggregation:                15th-percentile depth per 14×14 patch region
3D token grid:              [B, 1369, 3]  co-registered with DINOv2 ✓

GATr input (PGA embed):     [B, 1369, 16, 16]  mv  +  [B, 1369, 32]  scalar
GATr blocks:                8 equivariant transformer blocks
GATr invariant extracted:   [B, 1369, 48]  (32 scalar + 16 mv-norms)
GATr MLP projector:         Linear(48→4096) + GELU + Linear(4096→4096)  ~17M params
Output (GATr):              [B, 1369, 4096]

GridCellRoPE3D:
  Directions:               4 tetrahedral  [4×3]
  Frequencies:              8 golden-ratio spaced  [0.10m → 2.91m]
  Encoding:                 4 × 8 × 2 = 64 rotary dims
  Qwen3 rotary pairs:       64  →  exact match, no projection needed ✓
  Stored output:            [B, 1369, 64]
  Parameters:               0

STAGE 3 — FUSION
───────────────────────────────────────────────────────────────
SVA K/V source:             [B, 576+1369+1369, 4096] = [B, 3314, 4096]
SVA learnable queries:      [B, 576, 4096]  (24×24 spatial grid)
Typed attention bias:       3×3 scalar matrix  =  9 parameters
SVA cross-attn layers:      2
SVA output:                 [B, 576, 4096]
RMS norm match output:      [B, 576, 4096]  (zero parameters)
SVA attention savings:      O(576×3314) vs O(3314²)  →  5.7× cheaper

STAGE 4 — LLM BACKBONE
───────────────────────────────────────────────────────────────
Backbone:                   Qwen3-VL-7B
  hidden_size:              4096  ← D_proj = 4096 exact match ✓
  num_hidden_layers:        36
  num_attention_heads:      32
  num_key_value_heads:      8  (GQA, 4:1)
  head_dim:                 128
  M-RoPE section:           [24t, 20h, 20w]  =  64 rotary pairs

Gated cross-attn:
  Injection layers:         {4,8,12,16,20,24,28,32,36}  =  9 points
  K/V:                      [B, 576, 4096]  static across all 9 layers
  Gate init:                α=0.0  →  tanh(0)=0  →  pure pretrained behavior at init
  Params per layer:         ~41.9M  (W_Q + W_K + W_V + W_O + α)
  9 layers total:           ~377M params

LoRA:
  Applied to:               Q, K, V, O in all 36 layers
  rank=32, alpha=64
  Params per layer:         ~852K
  36 layers total:          ~30.7M params

PARAMETER SUMMARY
───────────────────────────────────────────────────────────────
Trainable (full training):  ~583M  (6.6% of total)
Frozen:                     ~8,307M
Total:                      ~8,890M

INFERENCE (single H100, BF16)
───────────────────────────────────────────────────────────────
SigLIP2 encode:             ~8ms
DINOv2 encode:              ~18ms
GT depth + percentile:      ~1ms
GATr:                       ~5ms
MLP projectors ×3:          ~2ms
SVA (KV=3314):              ~6ms
RMS norm:                   ~0ms
Qwen3 decode:               ~25ms
Total per step:             ~65ms  →  ~15 steps/second
```
