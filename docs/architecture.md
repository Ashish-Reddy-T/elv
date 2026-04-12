## Dimensional Analysis

---

### Stage 1 — Dual Vision Encoding

**1A: SigLIP2**

```
Input:                  [B, 384, 384, 3]
Patch size 16px:        384 / 16 = 24.0  →  24×24 = 576 patches  ✓ exact
Encoder hidden dim:     1152
Encoder depth:          27 layers

Extract layers {9, 18, 27}:
  Per layer:            [B, 576, 1152]
  Channel concat:       [B, 576, 3×1152] = [B, 576, 3456]

MLP projector:
  Linear(3456 → 4096) + GELU + Linear(4096 → 4096)
  Params: (3456×4096 + 4096) + (4096×4096 + 4096) ≈ 31M

Output:                 [B, 576, 4096]
```

**1B: DINOv2-L — No Spatial Pooling**

```
Input:                  [B, 518, 518, 3]
Patch size 14px:        518 / 14 = 37.0  →  37×37 = 1369 patches  ✓ exact
Encoder hidden dim:     1024
Encoder depth:          24 layers

Extract layers {8, 16, 24}:
  Per layer:            [B, 1369, 1024]
  Channel concat:       [B, 1369, 3×1024] = [B, 1369, 3072]

NO spatial pooling — tokens passed directly to MLP projector.
  The 1369 token count is preserved completely.

MLP projector:
  Linear(3072 → 4096) + GELU + Linear(4096 → 4096)
  Params: (3072×4096 + 4096) + (4096×4096 + 4096) ≈ 29M

Output:                 [B, 1369, 4096]
```

---

### Stage 2 — Geometric Branch

**2A: GT Depth → Backprojection with Lower Quantile**

```
Habitat depth sensor:   [B, 518, 518]   metric depth, zero error
  (depth image rendered at 518×518 to match DINOv2's input resolution exactly)

Backprojection per pixel (u, v) with depth d:
  X = (u - cx) × d / fx
  Y = (v - cy) × d / fy
  Z = d
  Full point map:       [B, 518×518, 3] = [B, 268324, 3]

Aggregate per 14×14 patch region (37×37 grid):
  Each patch covers 196 pixels of the 518×518 depth map

  Instead of mean: take the 15th percentile depth within each patch

    d_patch = percentile(d_pixels_in_region, q=0.15)

    Why 15th percentile:
      - Captures nearest foreground surface reliably
      - Robust to 1-14 noisy/specular pixels out of 196
      - Avoids background bias from open doorways or far walls
      - Not as sensitive as pure minimum (0th) to single outliers

    Apply percentile to depth scalars, then backproject the
    resulting d_patch using patch center (u_c, v_c):
      X_patch = (u_c - cx) × d_patch / fx
      Y_patch = (v_c - cy) × d_patch / fy
      Z_patch = d_patch

Output:                 [B, 1369, 3]   (one 3D foreground-nearest position per patch)
  Spatially registered to DINOv2's 37×37 = 1369 token grid  ✓
```

**2B: GATr — Now 1369 Tokens** (Refer `REPOS/geometric-algebra-transformer`)

```
Input:                    [B, 1369, 3]

PGA embedding:
  Each [x,y,z] → PGA trivector
  e₀₁₂=x, e₀₁₃=y, e₀₂₃=z, e₁₂₃=1, all others=0
  16-dim multivector per point

GATr internal streams:
  Multivector channels:   [B, 1369, 16, 16]  (16 channels × 16 PGA basis dims)
  Scalar channels:        [B, 1369, 32]

8 equivariant transformer blocks.
  Rotation equivariance preserved throughout  ✓

Extract invariant features:
  32 scalar channels:     [B, 1369, 32]
  16 multivector norms:   [B, 1369, 16]
  Total invariant:        [B, 1369, 48]

MLP projection:
  Linear(48 → 4096) + GELU + Linear(4096 → 4096)
  Params: ≈ 17M

Output:                   [B, 1369, 4096]
  Spatially co-registered with DINOv2 tokens  ✓
```

**2C: IcosahedralRoPE3D** (Refer `REPOS/gridpe-2d` for 2D inspiration; extended to 3D icosahedral)

```
Input:                    [B, 1369, 3]  (from backprojection, 37×37 DINOv2 grid)

NO position pooling — 1369 positions map 1:1 to SVA's 1369 DINOv2-based queries.
  Each SVA query (i,j) is one DINOv2 patch token. Its 3D position is the
  backprojected 15th-percentile depth of the same 14×14 pixel region.
  Content and position are geometrically consistent by construction.

6 icosahedral directions (antipodal pairs from regular icosahedron vertices):
  d₁ = [0, +1, +φ]/√(1+φ²)
  d₂ = [0, +1, −φ]/√(1+φ²)
  d₃ = [+1, +φ, 0]/√(1+φ²)
  d₄ = [+1, −φ, 0]/√(1+φ²)
  d₅ = [+φ, 0, +1]/√(1+φ²)
  d₆ = [+φ, 0, −1]/√(1+φ²)
  Isotropy: Σ dᵢdᵢᵀ = 2·I₃  (uniform coverage, no privileged direction)

For each position p:   sᵢ = dᵢ · p  →  [B, 1369, 6]

8 frequencies (e^(1/3) ≈ 1.3956 spacing, optimal for p=3 dimensions):
  fₖ = 10 × e^(k/3)  for k=0..7
  Range: ~10cm detail to ~14m room-scale

Encoding:  6 × 8 × 2 = 96 rotary dims  →  [B, 1369, 96]

Qwen3 M-RoPE has 64 rotary pairs per head (head_dim=128).
Spatial tokens: 48 pairs from IcosahedralRoPE3D + 16 identity pairs (cos=1, sin=0)
Text tokens:    keep standard M-RoPE [24t, 20h, 20w]
Mapping:        48 + 16 = 64 rotary pairs  ✓  padded to match

Stored:                   [B, 1369, 96]  applied at attention in Stage 4
Parameters:               0
```

---

### Stage 3 — Fusion

**3A: SVA** (Refer `REPOS/cambrian`) — **1369 DINOv2-based queries**

```
Inputs:
  SigLIP:   [B, 576, 4096]    semantic, 24×24 grid
  DINOv2:   [B, 1369, 4096]   structural, 37×37 grid, full resolution
  GATr:     [B, 1369, 4096]   geometric, 37×37 grid, co-registered with DINOv2

K/V source (concat along token dim):
  [B, 576+1369+1369, 4096] = [B, 3314, 4096]

Queries (DINOv2 as structural anchor + learnable embedding):
  base = DINOv2 tokens  [B, 1369, 4096]
  queries = base + query_embed  [B, 1369, 4096]

  DINOv2 as query base:
    The structural encoder owns the spatial layout. Its 37×37 tokens
    form the skeleton onto which SigLIP semantics and GATr geometry
    are fused via KV cross-attention. 1:1 position-to-query mapping
    eliminates the 37×37 → 24×24 compression bottleneck.

Cross-attention layer 1:
  Q:  [B, 1369, 4096]   →  32 heads × 128 dims
  K:  [B, 3314, 4096]   →  32 heads × 128 dims
  V:  [B, 3314, 4096]
  Attention map:  [B, 32, 1369, 3314]
  Output:         [B, 1369, 4096]

Cross-attention layer 2:
  Q:  updated [B, 1369, 4096]
  K/V: same [B, 3314, 4096]
  Output: [B, 1369, 4096]

Typed attention bias (learned 3×3 matrix):
  Token source types: 0=SigLIP, 1=DINOv2, 2=GATr
  bias[query_type, kv_type] → additive logit bias per head
  9 params per layer, prevents single-modality domination

Diagnostic probe (return_attention_stats=True):
  Per-layer, per-head attention mass on each source type
  Answers "is SVA actually using GATr?" without architecture changes
  Uses unfused attention path — diagnostic only, zero overhead on default path

FLOPs per SVA layer:
  1369 × 3314 × 128 × 2 × 32 heads = 37.1 GFLOPs/sample

vs self-attention on full 3314-token pool:
  3314² × 128 × 2 × 32 = 89.4 GFLOPs/sample

SVA savings:  2.4× cheaper than naive self-attention on merged tokens
```

**3B: RMS Norm Matching**

```
Input:   [B, 1369, 4096]
σ_text   = running EMA of text token RMS norms (updated during training)
σ_vision = RMS norm of fused spatial tokens
Scale:   output = input × (σ_text / σ_vision)
Output:  [B, 1369, 4096]
Params:  0
```

---

### Stage 4 — LLM Backbone

**4A: Sequence**

```
[<system_tokens> | <instruction_tokens> | <spatial_tokens(1369)> | <history_tokens>]
     ~128               ~64                      1369                    ~32

T ≈ 1593 total tokens at inference

Type routing:
  ~224 text tokens         → M-RoPE (64 pairs: [24t, 20h, 20w])
  1369 spatial tokens      → IcosahedralRoPE3D (48 pairs + 16 identity)
```

**4B: DeepStack** (native Qwen3-VL mechanism, replaces gated cross-attention)

```
Qwen3-VL's built-in DeepStack injects encoder intermediate features at
early LLM layers via residual addition. Uses visual_pos_masks to identify
spatial token positions within the sequence.

Params:  0  (uses backbone's native mechanism, no additional weights)

Gated cross-attention (~378M) was REMOVED in the icosahedral redesign.
DeepStack achieves the same injection goal with zero trainable parameters
by leveraging Qwen3-VL's existing multi-layer feature routing.
```

**4C: LoRA**

```
Applied to all 36 Qwen3 transformer blocks, Q/K/V/O projections
rank=32, alpha=64 (effective scale = 2.0)

Per layer:
  W_Q LoRA: 4096×32 + 32×4096 = 262,144
  W_K LoRA: 4096×32 + 32×1024 = 163,840  (GQA)
  W_V LoRA: same as K
  W_O LoRA: 4096×32 + 32×4096 = 262,144
  Per layer: ≈ 852K

36 layers: ≈ 30.7M params
```

---

### Full Parameter Table

```
Component                               Params     Trainable
──────────────────────────────────────────────────────────────
SigLIP2-SO400M encoder                  ~543M      Frozen
DINOv2-L encoder                        ~307M      Frozen
Habitat GT depth                           0       N/A
GATr (8 blocks, 1369 tokens)             ~12M      Trainable
SigLIP MLP projector (3456→4096)         ~31M      Trainable
DINOv2 MLP projector (3072→4096)         ~29M      Trainable
GATr MLP projector (48→4096)             ~17M      Trainable
SVA queries [1369×4096] + cross-attn     ~86M      Trainable
DeepStack (native Qwen3-VL)                0       N/A
Qwen3-VL-8B backbone                  ~8,000M      Frozen
Qwen3 LoRA rank-32                       ~31M      Trainable
──────────────────────────────────────────────────────────────
Total                                 ~9,056M
Trainable                               ~206M     (2.3%)
Frozen                               ~8,850M

Note: Gated cross-attention (~378M) removed in icosahedral redesign.
DeepStack replaces it with zero additional trainable parameters.
```

---

### Stage 5 — Training Pipeline

```
STAGE 1 — Pre-alignment
  Frozen:    SigLIP, DINOv2, GATr encoders, all of Qwen3
  Trainable: 3 MLP projectors (~77M)
  Data:      LLaVA-558K subset, 1 epoch
  Objective: next-token prediction via small LM head

STAGE 2 — SFT
  Frozen:    all encoders, Qwen3 weights
  Trainable: all projectors + GATr + SVA + LoRA (~206M)
  Data:      R2R-CE, RxR-CE, ScaleVLN, SQA3D, spatial reasoning
  Optional:  per-stream freeze groups for staged training
             (e.g. gatr+sva first → add dino_proj → all combined)

STAGE 3 — GRPO
  Trainable: same ~206M as SFT
  G=8 rollouts per prompt, clip ε=0.2, KL β=0.001
  LR=5e-7, batch=32 prompts, max_response=2048

  R_total = R_format + R_progress + R_collision + R_goal + R_consistency
    R_format      =  1.0   (binary template match)
    R_progress    =  Δgeodesic  (dense, per step)
    R_collision   = −2.0   (clearance < 0.1m)
    R_goal        = +10.0  (terminal, geodesic < 1.0m + STOP)
    R_consistency = −1.0   (reasoning contradicts action)

  Curriculum:
    Epochs 1-2:  format×1.0, accuracy×0.1, spatial×0.0
    Epochs 3-4:  format×0.3, accuracy×0.5, spatial×0.2
    Epochs 5-6:  format×0.1, accuracy×0.3, spatial×0.6
```
