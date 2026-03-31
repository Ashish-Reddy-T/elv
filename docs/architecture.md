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

**2C: GridCellRoPE3D** (Refer `REPOS/gridpe-2d` for 2D implementation inspiration)

```
Input:                    [B, 1369, 3]  (from backprojection, 37×37 DINOv2 grid)

SVA-aligned position pool: [B, 576, 3]  (24×24 grid)
  Each SVA query (i,j) owns a spatial sub-region of the 37×37 grid.
  The 3D position for query (i,j) is aggregated from the backprojected
  positions of DINOv2 patches within that same sub-region.
  CRITICAL: must use the same spatial grouping as SVA's query-region
  assignment — content and position must be geometrically consistent.

4 tetrahedral directions:
  d₁ = [+1,+1,+1]/√3
  d₂ = [+1,−1,−1]/√3
  d₃ = [−1,+1,−1]/√3
  d₄ = [−1,−1,+1]/√3

For each position p:   sᵢ = dᵢ · p  →  [B, 576, 4]

8 frequencies (φ=1.618):   fₖ = 10 × φᵏ
  Periods (m): [0.10, 0.16, 0.26, 0.42, 0.69, 1.11, 1.80, 2.91]

Encoding:  4 × 8 × 2 = 64 rotary dims  →  [B, 576, 64]

Qwen3 M-RoPE has 64 rotary pairs per head (head_dim=128).
Spatial tokens: replace all 64 pairs with GridCellRoPE3D
Text tokens:    keep standard M-RoPE [24t, 20h, 20w]
Mapping:        64 = 64  ✓  no projection needed

Stored:                   [B, 576, 64]  applied at attention in Stage 4
Parameters:               0
```

---

### Stage 3 — Fusion

**3A: SVA** (Refer `REPOS/cambrian`)

```
Inputs:
  SigLIP:   [B, 576, 4096]    semantic, 24×24 grid
  DINOv2:   [B, 1369, 4096]   structural, 37×37 grid, full resolution
  GATr:     [B, 1369, 4096]   geometric, 37×37 grid, co-registered with DINOv2

K/V source (concat along token dim):
  [B, 576+1369+1369, 4096] = [B, 3314, 4096]

Learnable queries (24×24 spatial grid):
  [B, 576, 4096]

Cross-attention layer 1:
  Q:  [B, 576, 4096]    →  32 heads × 128 dims
  K:  [B, 3314, 4096]   →  32 heads × 128 dims
  V:  [B, 3314, 4096]
  Attention map:  [B, 32, 576, 3314]
  Output:         [B, 576, 4096]

Cross-attention layer 2:
  Q:  updated [B, 576, 4096]
  K/V: same [B, 3314, 4096]
  Output: [B, 576, 4096]

FLOPs per SVA layer:
  576 × 3314 × 128 × 2 × 32 heads = 15.7 GFLOPs/sample

vs self-attention on full 3314-token pool:
  3314² × 128 × 2 × 32 = 89.4 GFLOPs/sample

SVA savings:  5.7× cheaper than naive self-attention on merged tokens
  (was 3.0× before, now larger because DINOv2 + GATr both preserved at 1369)
```

**3B: RMS Norm Matching**

```
Input:   [B, 576, 4096]
σ_text   = running EMA of text token RMS norms (updated during training)
σ_vision = RMS norm of fused spatial tokens
Scale:   output = input × (σ_text / σ_vision)
Output:  [B, 576, 4096]
Params:  0
```

---

### Stage 4 — LLM Backbone

**4A: Sequence**

```
[<system_tokens> | <instruction_tokens> | <spatial_tokens(576)> | <history_tokens>]
     ~128               ~64                      576                    ~32

T ≈ 800 total tokens at inference

Type routing:
  T=800 text tokens     → M-RoPE
  576 spatial tokens    → GridCellRoPE3D
```

**4B: Gated Cross-Attention** (Refer `REPOS/open_flamingo`)

```
Qwen3: 36 layers.  Injection at {4,8,12,16,20,24,28,32,36} = 9 points.

Per injection:
  Q: LLM hidden state    [B, T, 4096]
  K/V: SVA output        [B, 576, 4096]   ← static memory, same for all 9 layers

  GQA: 32 query heads, 8 KV heads (4:1 ratio)
  K projection:  4096 → 8×128 = 1024 dims  (expanded to 32 heads at attention time)
  V projection:  same

  Gate: h_new = h_old + tanh(α) × CrossAttn(...)
    α = scalar Parameter, init=0

Params per injection layer:
  W_Q: 4096×4096 = 16.8M
  W_K: 4096×1024 =  4.2M   (GQA, 8 KV heads)
  W_V: 4096×1024 =  4.2M
  W_O: 4096×4096 = 16.8M
  α:              =  0.000M (1 scalar)
  Total:          ≈ 42M per layer

9 layers:         ≈ 378M params
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
Component                          Params     Trainable
──────────────────────────────────────────────────────
SigLIP2-SO400M encoder             400M       Frozen
DINOv2-L encoder                   307M       Frozen
Habitat GT depth                     0        N/A
GATr (8 blocks, 1369 tokens)        ~12M      Trainable
SigLIP MLP projector                ~31M      Trainable
DINOv2 MLP projector                ~29M      Trainable
GATr MLP projector                  ~17M      Trainable
SVA queries [576×4096]               2.4M     Trainable
SVA cross-attn (2 layers, KV=3314)  ~84M      Trainable
Gated cross-attn (9 layers)        ~378M      Trainable
Qwen3-VL-8B backbone              8,000M      Frozen
Qwen3 LoRA rank-32                  ~31M      Trainable
──────────────────────────────────────────────────────
Total                             ~8,891M
Trainable                           ~584M     (6.6%)
Frozen                            ~8,307M
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
  Trainable: all projectors + GATr + SVA + cross-attn + LoRA (~584M)
  Data:      R2R-CE, RxR-CE, ScaleVLN, SQA3D, spatial reasoning

STAGE 3 — GRPO
  Trainable: same 584M as SFT
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
