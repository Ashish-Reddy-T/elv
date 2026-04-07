# SpatialVLM: Complete Dimensional Trace

## Every Tensor Shape from Input to Output, with Information Loss Analysis

This document traces every tensor through the full pipeline, noting exact shapes,
where information is lost, and what design alternatives exist at each stage.

---

## Overview: The Full Data Flow

```
                        ┌─────────────────────────────────────────────────┐
                        │            HABITAT SIMULATOR                    │
                        │   RGB: [B, H_orig, W_orig, 3]                   │
                        │   Depth: [B, 518, 518]  (GT metric, metres)     │
                        │   Intrinsics: (fx, fy, cx, cy)                  │
                        └───────────┬────────────────┬────────────────────┘
                                    │                │
                    ┌───────────────┘                └────────────────────┐
                    ▼                                                     ▼
          ┌─────────────────┐                                   ┌──────────────────┐
          │  IMAGE RESIZE   │                                   │  DEPTH PIPELINE  │
          │  ┌────────────┐ │                                   │                  │
          │  │ 384×384    │ │                                   │  518×518 native  │
          │  │ (SigLIP)   │ │                                   │  (no resize)     │
          │  └────────────┘ │                                   │                  │
          │  ┌────────────┐ │                                   └────────┬─────────┘
          │  │ 518×518    │ │                                            │
          │  │ (DINOv2)   │ │                                            ▼
          │  └────────────┘ │                             BACKPROJECTION + AGGREGATION
          └───────┬─────────┘                             + GATR + GRIDCELLROPE3D
                  │
                  ▼
         DUAL VISION ENCODING
         + MLP PROJECTORS
                  │
                  ▼
          ┌──────────────────────────────┐
          │   SVA CROSS-ATTENTION        │  ← KV from all 3 streams
          │   576 queries × 3314 KV      │  ← Positions from depth pipeline
          └──────────────┬───────────────┘
                         │
                         ▼
               RMS NORM MATCHING
                         │
                         ▼
          ┌──────────────────────────────┐
          │   QWEN3-VL-8B BACKBONE       │  ← GridCellRoPE3D on spatial tokens
          │   + 9× Gated Cross-Attention │  ← M-RoPE on text tokens
          │   + LoRA rank-32             │
          └──────────────┬───────────────┘
                         │
                         ▼
                  ACTION + REASONING
```

---

## STEP 0: Habitat Simulator Output

```
RGB frame:        [B, H_orig, W_orig, 3]     float32, range [0, 255] or [0, 1]
                  H_orig, W_orig = simulator render resolution (e.g., 640×480 or 1024×1024)

Depth map:        [B, 518, 518]               float32, metric depth in metres
                  Rendered at 518×518 to match DINOv2 input exactly
                  Zero values = invalid/missing pixels

Camera intrinsics: (fx, fy, cx, cy)            scalars, shared per batch
                  cx = 518/2 = 259, cy = 518/2 = 259  (for 518×518 depth)
```

**Design choice**: Depth is rendered natively at 518×518 (DINOv2's input resolution) so that
each DINOv2 patch's 14×14 pixel region maps to exactly the same 14×14 depth region.
No resize needed for depth. RGB requires resize (see Step 1).

---

## STEP 1: Image Resize

### 1a. Resize for SigLIP2

```
Input:   [B, H_orig, W_orig, 3]     original Habitat frame
Resize:  bilinear interpolation
Output:  [B, 384, 384, 3]           → [B, 3, 384, 384] after channel-first transpose
```

**Information lost**:

- If H_orig > 384: spatial detail lost by (H_orig/384)² factor
- Example: 1024×1024 → 384×384 = 7.1× pixel reduction
- Fine objects (small mug at 5m → ~3px) may become sub-pixel
- Anti-aliasing in bilinear helps but cannot recover sub-pixel detail

**Why 384**: SigLIP2-SO400M-patch16-NaFlex was trained at 384px. 384/16 = 24.0 (exact
integer), giving exactly 576 patches with zero edge truncation or padding. The original
SigLIP patch14 variant would give 384/14 = 27.43 (non-integer → requires cropping).

### 1b. Resize for DINOv2

```
Input:   [B, H_orig, W_orig, 3]     same original Habitat frame
Resize:  bilinear interpolation
Output:  [B, 518, 518, 3]           → [B, 3, 518, 518] after channel-first transpose
```

**Information lost**:

- If H_orig > 518: spatial detail lost by (H_orig/518)² factor
- Less loss than SigLIP since 518 > 384 (1.82× more pixels)
- Example: 1024×1024 → 518×518 = 3.9× pixel reduction

**Why 518**: 518/14 = 37.0 (exact integer), giving exactly 1369 patches. DINOv2's default
is 224px (16×16 = 256 patches) — we use 518 for much higher spatial resolution.

**Alternative to consider**: Render Habitat RGB at 518×518 natively (matching depth).
This eliminates the resize for DINOv2 entirely. SigLIP still needs 384×384 resize.
Cost: negligible (simulator render resolution is configurable).

---

## STEP 2: SigLIP2 Encoding

```
Input:           [B, 3, 384, 384]

Patch embedding: each 16×16×3 pixel block → 1152-dim vector via learned linear
                 384 / 16 = 24 patches per side
                 24 × 24 = 576 patches total

                 [B, 3, 384, 384] → [B, 576, 768] → [B, 576, 1152]

27 transformer layers (frozen - meaning just no backprop, forward pass still is run):
  Each layer: [B, 576, 1152] → [B, 576, 1152]
  Self-attention: 576 × 576 attention map per head (Q . K^T)

Multi-layer extraction at layers {9, 18, 27}:
  Layer  9 output: [B, 576, 1152]    (low-level spatial)
  Layer 18 output: [B, 576, 1152]    (mid-level structural)
  Layer 27 output: [B, 576, 1152]    (high-level semantic)

Channel concatenation:
  [B, 576, 1152] × 3 → [B, 576, 3456]
```

**Information lost**:

- Patch embedding is a learned 768→1152 linear projection (768 = 16×16×3 input pixels).
Pixel-level detail within each 16×16 block is compressed to 1152 dims.
- Self-attention across 576 tokens can mix information between patches, but token count
is preserved (576 in, 576 out at every layer).

**No CLS token**: SigLIP2 does not use a CLS token (unlike CLIP). All 576 tokens are spatial.

---

## STEP 3: SigLIP2 MLP Projector

```
Input:   [B, 576, 3456]    (3 layers × 1152 hidden)

MLP:     Linear(3456, 4096) → GELU → Linear(4096, 4096)
         ~31M parameters

Output:  [B, 576, 4096]
```
> Note: 
> - This is not MLP in the traditional sense, it is just for upscaling (so that's why we don't use the typical format of d -> 4d -> GELU -> d instead just d -> h_d -> GELU -> h_d).
> - You use the last step (GELU -> h_d) because it is standard and we like to have **two** learned layers.
> - SigLIP does Pre-LN during forward pass, so x_next = x_prev + SubLayer(LN(x_prev)) compared agains Post-LN which does x_next = LN(x_prev + SubLayer(x_prev)) [Here SubLayer could be Attn or MLP block (LN happens during both blocks)]

**Information lost**: Upsampling from 3456 → 4096 is relatively lossless (expanding, not
compressing). The linear layers can represent the full 3456-dim input space in 4096 dims.
GELU nonlinearity introduces mild irreversibility (irreversibility basically means you can't get those high negative values back because they have been changed from an unknown number to all zeros)

**This is the final SigLIP output**: [B, 576, 4096] — semantic, language-aligned features
on a 24×24 spatial grid.

---

## STEP 4: DINOv2 Encoding

```
Input:           [B, 3, 518, 518]

Patch embedding: each 14×14×3 pixel block → 1024-dim vector via learned linear
                 518 / 14 = 37 patches per side
                 37 × 37 = 1369 patches total

                 [B, 3, 518, 518] → [B, 1370, 588] → [B, 1370, 1024]    (1369 patches + 1 CLS token)

24 transformer layers (frozen):
  Each layer: [B, 1370, 1024] → [B, 1370, 1024]

CLS token stripping:
  [B, 1370, 1024] → [B, 1369, 1024]    (remove token at index 0)

Multi-layer extraction at layers {8, 16, 24}:
  Layer  8 output: [B, 1369, 1024]    (low-level spatial)
  Layer 16 output: [B, 1369, 1024]    (mid-level structural)
  Layer 24 output: [B, 1369, 1024]    (high-level semantic)

Channel concatenation:
  [B, 1369, 1024] × 3 → [B, 1369, 3072]
```

**Information lost**: Same patch-embedding compression as SigLIP but with 14×14 patches
(588 pixels → 1024 dims per patch, vs SigLIP's 768 pixels → 1152 dims).

**CLS token**: DINOv2 prepends a CLS token. We strip it because it's a global summary
(no spatial position) - very common practice. This is 1 token out of 1370 — negligible.

---

## STEP 5: DINOv2 MLP Projector

```
Input:   [B, 1369, 3072]    (3 layers × 1024 hidden)

MLP:     Linear(3072, 4096) → GELU → Linear(4096, 4096)
         ~29M parameters

Output:  [B, 1369, 4096]
```

**This is the final DINOv2 output**: [B, 1369, 4096] — structural, spatially-aware features
on a 37×37 grid. No pooling applied — full resolution preserved.

---

## STEP 6: Depth Backprojection

```
Input:   depth [B, 518, 518]           metric depth in metres
         intrinsics (fx, fy, cx, cy)    camera parameters

Pixel grid: make_pixel_grid(518, 518) → [518, 518, 2]    (u, v coordinates)

Per-pixel backprojection:
  X = (u - cx) × depth / fx
  Y = (v - cy) × depth / fy
  Z = depth

Output:  [B, 518, 518, 3] = [B, 268324, 3] when flattened
         Full 3D point cloud in camera coordinates
```

**Information lost**: None — this is a lossless, invertible transformation (given intrinsics).
Invalid pixels (depth=0) produce (0,0,0), which are handled in the next step.

---

## STEP 7: Patch-Level Percentile Aggregation (Visualize a 518p x 518p image, you'll understand)

```
Input:   point_map [B, 518, 518, 3]     full 3D point cloud
         depth     [B, 518, 518]        for sorting

Reshape to patches (matching DINOv2's 14×14 grid):
  [B, 518, 518, 3]
  → [B, 37, 14, 37, 14, 3]           (split into 37×37 patches of 14×14 pixels)
  → [B, 37, 37, 14, 14, 3]           (permute)
  → [B, 1369, 196, 3]                (flatten patches)

Per-patch depth sorting (ascending = nearest first):
  Sort 196 pixels by depth within each patch
  Select pixel at index k = floor(0.15 × 196) = 29 (Here use only 196 and not 588 because depth doesn't vary across different color channels, it stays the same whatever channel you choose!)
  (The 30th nearest pixel — 15th percentile)

> Note: You choose one point per patch, specifically we are choosing 15% closest to us in the patch (asc_list[29] -> 30th nearest pixel). You don't want the nearest/min, because it could be too fragile - one bad too-near reading (noise bloom, wrong hit) could dominate the whole patch; or mean which pulls toward far stuff when backgroudn comes into picture (like door, ceiling/sky, floor/road, window) - maybe bad for geometry; or median or other percentiles for similar reasons. 15th percentile chooses the nearest but not the absolute nearest for us to still have idea on the dominant object. YOU CHOOSE 1 POINT SO IT ALIGNS WITH 1369 TOKENS OF DINO IN PLACE OF HAVING 268,324 (196*1369) TOKENS WHICH REQUIRE COMPUTE, DOWNSCALING ETC. DURING FUSION.

Gather 3D coordinates at selected index:
  [B, 1369, 196, 3] → gather at index 29 → [B, 1369, 3] (one depth pixel per patch)

Output:  [B, 1369, 3]    one 3D point per DINOv2 patch
```

**Information lost**: 196:1 compression per patch. 196 pixel-level 3D points → 1 representative
point. The selected point IS a real pixel (not interpolated), but 195 other points are discarded.

**What survives**: The foreground surface position for each patch. Inter-patch geometry
(distances, angles between any two of the 1369 patches) is fully preserved.

**What's destroyed**: Sub-patch geometry. Two objects within the same 14×14 pixel region
(~1.5cm × 1.5cm at 1m depth on a 90° FOV camera) are collapsed to one point.

**Hypothesis H2e tests this**: Replace 15th-percentile with mean aggregation. If 15th-pct
wins, the foreground bias is empirically validated.

---

## STEP 8: GATr Geometric Processing

```
Input:   [B, 1369, 3]    patch-level 3D positions

8a. Coordinate normalization:
    radius = torch.linalg.vector_norm(coords, dim=-1)  # [B, 1369, 3] -> [B, 1369]
    max_radius = radius.amax(dim=-1, keepdim=True).clamp_min(self._eps)  # [B, 1369] -> [B, 1] 
    coords = coords / max_radius.unsqueeze(-1)  # [B, N=1369, 3] (Max across all 1369 patches, so global normalization)

    Output: [B, 1369, 3]    (normalized to [-1, 1] range)


8b. PGA embedding (`gatr.interface.embed_point`):

    Input coordinates `(x, y, z)` are `points_3d[..., 0/1/2]` (after optional batch-wise
    isotropic scaling in `GATrWrapper`).

    Build a 16-D multivector per point; all components start at 0, then four trivector
    slots (indices `11–14` in GATr’s PGA layout; see `gatr/primitives/attention.py`
    `_TRIVECTOR_IDX`) are set as:

      `mv[..., 11] = -z`    
      `mv[..., 12] =  y`
      `mv[..., 13] = -x`
      `mv[..., 14] =  1.0`   (homogeneous / embedding scale; Dorst PGA point embedding)

    Multivector stream: `[B, 1369, 1, 16]` — one MV channel × 16 PGA basis coefficients.

    Scalar stream: `[B, 1369, 1]` — `torch.linalg.vector_norm(coords, dim=-1, keepdim=True)`
    on the **same** `(x,y,z)` passed to `embed_point` (so after normalization it lies in
    roughly `[0, 1]` per point).

Note: The standard PGA formula for embedding a 3D Euclidean point (x, y, z) is:
        P = e_{123} + x * e_{032} + y * e_{013} + z * e_{021}     but GATr internal
        16-D tensor stores its trivector bases in standard lexicographical order:
          e_{012}, e_{013}, e_{023}, e_{123}  

        z * e_{021} becomes -z * e_{012}     -> mv[..., 11] = -z
        y * e_{013} stays    y * e_{013}     -> mv[..., 12] =  y
        x * e_{032} becomes -x * e_{023}     -> mv[..., 13] = -x
        e_{123} is the homogenous coordinate -> mv[..., 14] =  1

      You can have more channels than the current 1. More channels would represent more
      hidden layer dimensions for a transformer block where each layer/channel may learn
      a different thing from the other (it can store multiple independent geometric objects
      like scalars, bivectors etc.)


8c. 8 equivariant transformer blocks:
    Each block: GATrBlock (equivariant self-attention + equivariant FFN)
    MV channels expand:  1 → 16 (gatr_mv_channels)
    Scalar channels:     1 → 32 (gatr_s_channels)

    After all 8 blocks:
      Multivector out: [B, 1369, 16, 16]    (16 MV channels × 16 PGA basis dims)
      Scalar out:      [B, 1369, 32]        (32 scalar channels)


8d. Invariant extraction:
    MV norms: torch.linalg.vector_norm(mv_out, dim=-1) → [B, 1369, 16]
    Concatenate: [scalar_out, mv_norms] → [B, 1369, 48]

    These 48 values are rotation-INVARIANT: rotating the input point cloud
    does NOT change these outputs. They encode distances, angles, coplanarity —
    geometric relationships, not absolute positions.


8e. MLP projection:
    Linear(48, 4096) → GELU → Linear(4096, 4096)
    ~17M parameters

Output:  [B, 1369, 4096]    geometric features, co-registered with DINOv2 grid
```

**Information lost**:

- Coordinate normalization divides by max_radius (recoverable if stored, but not stored)
- PGA embedding is lossless (3D → 16D trivector, with zeros in unused basis elements)
- Invariant extraction is INTENTIONALLY lossy: discards absolute orientation information.
This is a feature, not a bug — the model should be orientation-agnostic for the same scene.
- 48 → 4096 MLP upsampling is lossless (expanding)

**What survives**: Pairwise distances, relative angles, surface normals, coplanarity — all the
geometric relationships between patches. These are exactly what the LLM needs for spatial reasoning.

**What's destroyed**: Absolute orientation of the point cloud. If you rotate the entire scene,
the invariant features are identical. This means the model cannot distinguish "door on the left"
from "door on the right" using GATr alone — it needs the position channel (GridCellRoPE3D)
for that.

**Hypothesis H2a tests this**: Remove GATr entirely. If performance drops on metric tasks
(distance estimation, size comparison) but not on semantic tasks, GATr's geometric features
are confirmed complementary.

---

## STEP 9: Position Pooling to SVA Grid — DEPRECATED

> **DEPRECATED (2026-04-05)**: Position pooling was eliminated by the icosahedral redesign.
> With 1369 DINOv2-based SVA queries, 3D positions map 1:1 to queries — no pooling needed.

```
Input:   [B, 1369, 3]    patch-level 3D positions (same as GATr input, BEFORE normalization)

Reshape to spatial grid:
  [B, 1369, 3] → permute → [B, 3, 37, 37]

Adaptive average pooling:
  F.adaptive_avg_pool2d([B, 3, 37, 37], output_size=(24, 24))
  → [B, 3, 24, 24]

  Bin assignment: 37 → 24 means each output cell covers either 1 or 2 source cells
    floor(37/24) = 1, ceil(37/24) = 2
    13 cells cover 2 source cells, 11 cells cover 1 source cell
    Average compression: 37/24 = 1.54 source cells per side, ~2.38 per 2D cell

Reshape back:
  [B, 3, 24, 24] → [B, 576, 3]

Output:  [B, 576, 3]    one 3D position per SVA query cell
```

**Information lost**: ~2.38:1 averaging within each SVA cell. Very mild — neighbouring
DINOv2 patches (separated by ~1.5cm at 1m depth) are averaged, producing a
representative centroid for each SVA query region.

**Why this specific pooling**: The SVA organizes 576 queries on a 24×24 grid. Each query's
cross-attention covers a spatial sub-region. The position pooling uses the SAME spatial
partitioning (adaptive_avg_pool2d with the same bin boundaries), ensuring the 3D position
assigned to SVA query (i,j) is the centroid of the same region that query (i,j) attends to.

**Alternative: Use 1369 positions directly** (if SVA queries were [B, 1369, 4096]):

- Skip this pooling entirely
- GridCellRoPE3D on raw [B, 1369, 3] → [B, 1369, 64]
- No positional information lost
- Cost: LLM processes 1369 spatial tokens instead of 576 (see trade-offs in Section "Alternatives")

---

## STEP 10: GridCellRoPE3D

```
Input:   [B, 1369, 3]    Unpooled 3D patch positions in metres (matches DINOv2 grid)

10a. Scalar projections onto 6 Icosahedral directions:
     Using the golden ratio φ ≈ 1.618. 
     Normalized axes (1/√(1+φ²), φ/√(1+φ²)) ≈ (0.526, 0.851):

     icosa_dirs: [6, 3] (+ve only as according to paper, so just 6 instead of 12 dirs)
       d₁ = [ 0.000, +0.526, +0.851]
       d₂ = [ 0.000, +0.526, -0.851]
       d₃ = [+0.526, +0.851,  0.000]
       d₄ = [-0.526, +0.851,  0.000]
       d₅ = [+0.851,  0.000, +0.526]
       d₆ = [+0.851,  0.000, -0.526]

     einsum("bnd,fd->bnf", positions, icosa_dirs)
     [B, 1369, 3] × [6, 3] → [B, 1369, 6]

10b. Scale by 8 optimal 3D frequencies (GridPE Theorem):
     Optimal 3D scale ratio ρ = e^(1/3) ≈ 1.3956.
     freqs: [8]
       f₀ = 10.00, f₁ = 13.96, f₂ = 19.48, f₃ = 27.18
       f₄ = 37.94, f₅ = 52.95, f₆ = 73.89, f₇ = 103.12    (fₖ = 10 × ρᵏ)

     [B, 1369, 6].unsqueeze(-1) × [8]
     → [B, 1369, 6, 8]    (scaled projections)

10c. Sin/cos encoding:
     sin([B, 1369, 6, 8]) → [B, 1369, 6, 8]
     cos([B, 1369, 6, 8]) → [B, 1369, 6, 8]

     stack → [B, 1369, 6, 8, 2]
     reshape → [B, 1369, 96]    (6 × 8 × 2 = 96)

Output:  [B, 1369, 96]    rotary position encoding
         Layout: [sin(d₁·p·f₀), cos(d₁·p·f₀), sin(d₁·p·f₁), cos(d₁·p·f₁), ...]
         
```
**Information lost**: The sin/cos encoding is periodic. However, the optimal e^1/3 spacing 
minimizes this by ensuring no two frequencies share a common rational period. Aliasing distance 
for the lowest frequency: λ₀ = 2π/10.0 ≈ 0.63m. For the highest: λ₇ = 2π/103.12 ≈ 0.06m

Multi-frequency encoding means catastrophic aliasing only occurs when ALL 8 frequencies align — 
effectively never within room-scale distances.

**0 learnable parameters**: Pure deterministic function of 3D position. All state is in registered 
buffers (moved with .to(device) but not trained).

**Attention Head Dimension Isolation (CRITICAL)**: GridCellRoPE3D [B, 1369, 96] is stored and 
applied later in the LLM's attention computation (Step 14). Because Qwen3-VL has a head_dim 
of 128, this RoPE must be strictly applied ONLY to dimensions 32-127 for spatial tokens, isolating 
it from the 1D sequential M-RoPE applied to text tokens in dimensions 0-31.


(DEPRECATED)

<!-- ```
Input:   [B, 576, 3]     SVA-aligned 3D positions in metres

10a. Scalar projections onto 4 tetrahedral directions:
     tetra_dirs: [4, 3]
       d₁ = [+1/√3, +1/√3, +1/√3]    ≈ [+0.577, +0.577, +0.577]
       d₂ = [+1/√3, -1/√3, -1/√3]
       d₃ = [-1/√3, +1/√3, -1/√3]
       d₄ = [-1/√3, -1/√3, +1/√3]

     einsum("bnd,fd->bnf", positions, tetra_dirs)
     [B, 576, 3] × [4, 3] → [B, 576, 4]

10b. Scale by 8 golden-ratio frequencies:
     freqs: [8]
       f₀ = 10.0, f₁ = 16.18, f₂ = 26.18, f₃ = 42.33
       f₄ = 68.48, f₅ = 110.77, f₆ = 179.16, f₇ = 289.79    (fₖ = 10 × φᵏ)

     [B, 576, 4].unsqueeze(-1) × [8]
     → [B, 576, 4, 8]    (scaled projections)

10c. Sin/cos encoding:
     sin([B, 576, 4, 8]) → [B, 576, 4, 8]
     cos([B, 576, 4, 8]) → [B, 576, 4, 8]

     stack → [B, 576, 4, 8, 2]
     reshape → [B, 576, 64]    (4 × 8 × 2 = 64)

Output:  [B, 576, 64]    rotary position encoding
         Layout: [sin(d₁·p·f₀), cos(d₁·p·f₀), sin(d₁·p·f₁), cos(d₁·p·f₁), ...] 
```

**Information lost**: The sin/cos encoding is periodic — two positions separated by exactly
one wavelength produce identical encodings (aliasing). The golden-ratio spacing minimizes
this by ensuring no two frequencies share a common period. Aliasing distance for the
lowest frequency: λ₀ = 2π/10.0 ≈ 0.63m. For the highest: λ₇ = 2π/289.79 ≈ 0.022m.

Multi-frequency encoding means aliasing only occurs when ALL 8 frequencies align —
effectively never within room-scale (<10m) distances.

**0 learnable parameters**: Pure deterministic function of 3D position. All state is
in registered buffers (moved with .to(device) but not trained).

**This output bypasses SVA entirely**: GridCellRoPE3D [B, 576, 64] is stored and applied
later in the LLM's attention computation (Step 14). It does NOT go through SVA or
gated cross-attention. This is the explicit position channel.  -->

---

## STEP 11: SVA Cross-Attention (Aggregation — Reduced Compression)

```
Input tokens:
  SigLIP:  [B, 576, 4096]     (semantic, 24×24 grid)
  DINOv2:  [B, 1369, 4096]    (structural, 37×37 grid)
  GATr:    [B, 1369, 4096]    (geometric, 37×37 grid, co-registered with DINOv2)

KV concatenation:
  torch.cat([SigLIP, DINOv2, GATr], dim=1)
  → [B, 3314, 4096]

KV type IDs:
  [0]*576 + [1]*1369 + [2]*1369 = [3314]    (0=SigLIP, 1=DINOv2, 2=GATr)

Query initialization (DINOv2 as structural anchor):
  queries = DINOv2_tokens + query_embed
  DINOv2_tokens: [B, 1369, 4096]    (structural features as starting point)
  query_embed:   [1369, 4096]        (learnable, nn.Parameter)
  queries:       [B, 1369, 4096]

SVA Layer 1:
  Pre-norm:  q_norm(queries) → [B, 1369, 4096]
             kv_norm(kv_tokens) → [B, 3314, 4096]

  Q proj:    Linear(4096, 4096, bias=False) → [B, 1369, 4096]
  K proj:    Linear(4096, 4096, bias=False) → [B, 3314, 4096]
  V proj:    Linear(4096, 4096, bias=False) → [B, 3314, 4096]

  Reshape to heads (32 heads, 128 head_dim):
    Q: [B, 1369, 4096]  → [B, 1369, 32, 128]  → [B, 32, 1369, 128]
    K: [B, 3314, 4096] → [B, 3314, 32, 128] → [B, 32, 3314, 128]
    V: [B, 3314, 4096] → [B, 3314, 32, 128] → [B, 32, 3314, 128]

  Typed attention bias (if enabled):
    bias_matrix: [3, 3]    (9 learnable scalars)
    typed_mask:  [1, 1, 1369, 3314]    (bias[query_type[i], kv_type[j]]) [Here 1, 1 because for each layer (not each head within layer)]
    Added to attention logits before softmax

  THINK OF THIS AS A MATRIX WITH q(row indices of 0,1,2) and k(column indices of 0,1,2) [0->SigLIP, 1->DINO, 2->GATr]

  Scaled dot-product attention:
    attn_weights = softmax(Q @ K^T / √128 + typed_mask)    [B, 32, 1369, 3314] (addition holds as same size))
    attn_out = attn_weights @ V                              [B, 32, 1369, 128]

  Output projection:
    attn_out = [B, 32, 1369, 128] → [B, 1369, 4096] → Linear(4096, 4096) → [B, 1369, 4096]

  Residual + LayerNorm:
    queries = LayerNorm(queries_in + attn_out)    [B, 1369, 4096]

    q2 = LN(q + XAttn(LN(q), LN(kv)))

SVA Layer 2:
  Same architecture, same KV pool (NOT updated — original 3314 tokens reused).
  Queries from Layer 1 output are refined by re-attending to original KV.

  queries = LayerNorm(queries_layer1 + attn_out_layer2)    [B, 1369, 4096]

  q3 = LN(q2 + XAttn(LN(q2), LN(kv)))

Output:  q3 = [B, 1369, 4096]    fused spatial tokens
```

**Information lost — compression is now much milder**:

- 3314 KV tokens → 1369 output tokens = **2.42:1 compression** (was 5.75:1 with 576)
- Each output token is a weighted average of all 3314 KV tokens (softmax-weighted)
- Full 37×37 DINOv2/GATr spatial resolution is preserved through fusion

**What survives**:

- Each query can attend to ALL 3314 tokens — no information is structurally blocked
- The typed attention bias lets the model upweight GATr tokens if geometry is useful
- Two-layer design: layer 2 re-attends to the SAME original KV (not blurred outputs)
- Query initialization from DINOv2 preserves structural/spatial base
- 1:1 correspondence between DINOv2 patches and SVA queries (no spatial pooling needed)

**What's at risk**:

- Whether geometry survives depends on what the attention LEARNS to attend to
- If training signal doesn't reward geometric precision, SVA may ignore GATr tokens
- Compute cost: O(1369 × 3314) = 4.5M ops/head vs. O(576 × 3314) = 1.9M with old design

**Hypothesis H3e tests this**: Compare 3314-token KV pool vs. pooling all to 576 first
(1728-token KV). If 3314 wins, the full resolution KV matters.

---

## STEP 12: RMS Norm Matching

```
Input:   vision_tokens [B, 1369, 4096]    (from SVA)
         text_tokens   [B, T, 4096]       (from Qwen3 tokenizer, during training only)

Compute vision RMS norm:
  vision_rms = sqrt(mean(vision_tokens², dim=-1)).mean()    scalar

Update text RMS EMA (training only):
  text_rms_per_token = sqrt(mean(text_tokens², dim=-1))    [B, T]
  batch_text_rms = text_rms_per_token.mean()               scalar
  text_rms_ema = 0.99 × text_rms_ema + 0.01 × batch_text_rms

Scale:
  scale = text_rms_ema / (vision_rms + 1e-6)              scalar
  output = vision_tokens × scale                           [B, 1369, 4096]

Output:  [B, 1369, 4096]    (same shape, rescaled magnitudes)
```

**Information lost**: None structurally — this is a global scalar multiplication. The relative
relationships between tokens are perfectly preserved. Only the absolute magnitude changes.

**Why necessary**: Without this, vision token norms are 10-100× text norms. The LLM's
attention softmax is dominated by vision token magnitudes, drowning out the RoPE positional
signal. After scaling, vision and text tokens have comparable norms, so positional encoding
(including IcosahedralRoPE3D) can actually influence attention patterns.

---

## STEP 13: RoPE Monkey-Patch (replaces Position Routing)

```
Input:
  text_tokens:    [B, T, 4096]      T ≈ 128-256 (system + instruction + history)
  spatial_tokens: [B, 1369, 4096]   (from RMS norm matching)
  positions_3d:   [B, 1369, 3]      (from Step 7, 3D patch positions in metres)

Concatenation:
  combined = cat([text_tokens, spatial_tokens], dim=1)
  → [B, T+1369, 4096]

Spatial token mask:
  is_spatial = [False]*T + [True]*1369
  → [B, T+1369] boolean

RoPE injection via monkey-patch (rope_patch.py):
  The Catcher: model.forward() intercepts spatial_coords_3d and spatial_token_mask
               from kwargs, stashes them on the rotary_emb module.

  The Math: rotary_emb.forward() computes:
    1. Standard M-RoPE cos/sin for ALL tokens     → [B, T+1369, 128]
    2. IcosahedralRoPE3D for spatial positions     → [B, 1369, 96]
    3. Pad to 128 with identity (cos=1, sin=0)     → [B, 1369, 128]
    4. Overwrite cos/sin at spatial mask positions

  Result:
    Text tokens:    M-RoPE [24T, 20H, 20W] = 64 pairs (128 dims)
    Spatial tokens: Icosahedral 48 pairs (96 dims) + 16 identity pairs (32 dims)

  During autoregressive generation:
    Spatial tokens are in KV cache from prefill.
    Only new text tokens processed → patch is a no-op.

Output: combined_tokens [B, T+1369, 4096] with position-aware cos/sin ready
```

**Information lost**: None — this is position encoding injection, no content modification.

---

## STEP 14: Qwen3-VL-8B LLM Forward Pass

```
Input:
  tokens:     [B, T+1369, 4096]   mixed text + spatial sequence (T ≈ 200, total ≈ 1569)
  cos/sin:    [B, T+1369, 128]    from RoPE monkey-patch (Step 13)
              Text positions: standard M-RoPE
              Spatial positions: IcosahedralRoPE3D (48 pairs) + identity (16 pairs)

Architecture (frozen backbone + LoRA):
  36 transformer layers, each:
    hidden_size:        4096
    num_attention_heads: 32 (query heads)
    num_key_value_heads: 8  (GQA 4:1)
    head_dim:           128
    intermediate_size:  12288 (FFN)

Per layer (all 36):
  RoPE application (inside attention):
    For text tokens at position m:
      Standard M-RoPE: [24 temporal, 20 height, 20 width] = 64 rotary pairs
      Apply rotation R(m) to all 128 dims of each head's Q and K

    For spatial tokens at position p=[x,y,z]:
      IcosahedralRoPE3D: 48 pairs (96 dims) encode 3D position
      16 identity pairs (32 dims) with cos=1, sin=0 → position-agnostic
      Two spatial tokens at positions p₁, p₂ have attention modulated by
      the periodic distance between p₁ and p₂ — not sequence position

  Self-attention:
    Q: [B, T+1369, 4096] → [B, T+1369, 32, 128] → [B, 32, T+1369, 128]
    K: [B, T+1369, 4096] → [B, T+1369, 8, 128]  → [B, 8, T+1369, 128]
    V: [B, T+1369, 4096] → [B, T+1369, 8, 128]  → [B, 8, T+1369, 128]

    GQA expansion: K, V each [B, 8, T+1369, 128] → repeat 4× → [B, 32, T+1369, 128]
    Attention: [B, 32, T+1369, T+1369]    full self-attention map
    Output: [B, 32, T+1369, 128] → [B, T+1369, 4096]

  FFN: [B, T+1369, 4096] → [B, T+1369, 12288] → [B, T+1369, 4096]

  LoRA (on Q, K, V, O projections):
    W_Q LoRA: (4096 × 32) + (32 × 4096) = 262,144 params
    W_K LoRA: (4096 × 32) + (32 × 1024) = 163,840 params    (K dim = 8×128 = 1024)
    W_V LoRA: (4096 × 32) + (32 × 1024) = 163,840 params
    W_O LoRA: (4096 × 32) + (32 × 4096) = 262,144 params
    Per layer total: ~852K params
    36 layers: ~30.7M params

DeepStack injection (native Qwen3-VL mechanism, replaces gated cross-attention):
  Qwen3-VL's built-in mechanism for injecting encoder intermediate features.
  At select early LLM layers, visual embeddings are added as residuals:
    hidden_states[visual_pos_masks, :] += deepstack_visual_embeds

  deepstack_visual_embeds: extracted from multi-layer SigLIP/DINOv2 features
  visual_pos_masks: [B, T+1369] boolean marking spatial token positions

  Zero additional trainable parameters — uses Qwen3-VL's native mechanism.
  Replaces gated cross-attention (which had 9 layers × ~42M = ~378M params).

Final LLM output:  [B, T+1369, 4096]

Decoding (autoregressive):
  LM head: Linear(4096, vocab_size) → logits
  Sample next token from logits
  Repeat until STOP token or max length
```

**Information flow through the LLM**:

- Text tokens get spatial information via self-attention (they attend to spatial tokens in the same sequence)
- DeepStack injects multi-layer encoder features at early layers via residual addition
- Spatial tokens get language context via self-attention (they attend to text tokens)
- IcosahedralRoPE3D modulates which spatial tokens attend to each other based on 3D distance
- M-RoPE modulates text token attention based on sequential position
- The two position encoding systems coexist in the same attention computation

---

## COMPLETE TENSOR FLOW SUMMARY TABLE


| Step | Operation          | Input Shape       | Output Shape                  | Compression             | Info Lost                |
| ---- | ------------------ | ----------------- | ----------------------------- | ----------------------- | ------------------------ |
| 0    | Habitat render     | —                 | [B, H, W, 3] + [B, 518, 518]  | —                       | —                        |
| 1a   | Resize for SigLIP  | [B, H, W, 3]      | [B, 3, 384, 384]              | (H/384)² pixels         | Sub-pixel detail         |
| 1b   | Resize for DINOv2  | [B, H, W, 3]      | [B, 3, 518, 518]              | (H/518)² pixels         | Sub-pixel detail         |
| 2    | SigLIP encoding    | [B, 3, 384, 384]  | [B, 576, 3456]                | 256px → 1152d per patch | Within-patch pixels      |
| 3    | SigLIP projector   | [B, 576, 3456]    | [B, 576, 4096]                | None (upsampling)       | ~None                    |
| 4    | DINOv2 encoding    | [B, 3, 518, 518]  | [B, 1369, 3072]               | 196px → 1024d per patch | Within-patch pixels      |
| 5    | DINOv2 projector   | [B, 1369, 3072]   | [B, 1369, 4096]               | None (upsampling)       | ~None                    |
| 6    | Backprojection     | [B, 518, 518]     | [B, 518, 518, 3]              | None (invertible)       | None                     |
| 7    | Percentile agg.    | [B, 518, 518, 3]  | [B, 1369, 3]                  | **196:1** per patch     | Sub-patch 3D detail      |
| 8    | GATr processing    | [B, 1369, 3]      | [B, 1369, 4096]               | 3 → 48 → 4096 (expand)  | Absolute orientation     |
| 9    | ~~Pos pooling~~    | ~~DEPRECATED~~    | ~~DEPRECATED~~                | ~~removed~~             | ~~N/A~~                  |
| 10   | IcosahedralRoPE3D  | [B, 1369, 3]      | [B, 1369, 96]                 | 3 → 96 (expand)         | Aliasing (>0.63m period) |
| 11   | SVA fusion         | [B, 3314, 4096]   | [B, 1369, 4096]               | **2.42:1** (attention)  | Mild spatial blending    |
| 12   | RMS norm match     | [B, 1369, 4096]   | [B, 1369, 4096]               | None (scalar multiply)  | None                     |
| 13   | RoPE monkey-patch  | Various           | [B, T+1369, 4096]             | None (PE injection)     | None                     |
| 14   | LLM + DeepStack    | [B, T+1369, 4096] | [B, T+1369, 4096]             | None                    | —                        |


**Two major information bottlenecks** (ordered by severity):

1. **Percentile aggregation**: 268K pixels → 1369 points (196:1 per patch)
2. **SVA fusion**: 3314 → 1369 tokens (2.42:1), soft attention blending

Note: Position pooling (formerly #3 at 2.4:1) was eliminated — 1369 positions map 1:1 to SVA queries.

---
 
## ALTERNATIVE DESIGNS AND THEIR HYPOTHESIS TESTS
 
### Alt-A: 576-Query SVA with SigLIP base (ablation baseline)
 
The previous design used 576 queries with SigLIP as the query base.
This is now an ablation (H1d) to validate the 1369-query choice.
 
```
Current (1369):                             Ablation (576):
Q:  [B, 1369, 4096]  (DINOv2 base)         Q:  [B, 576, 4096]  (SigLIP base)
KV: [B, 3314, 4096]                        KV: [B, 3314, 4096]
Out: [B, 1369, 4096]                        Out: [B, 576, 4096]
 
SVA cost:  O(1369 × 3314) = 4.5M           SVA cost:  O(576 × 3314)  = 1.9M  (2.4× cheaper)
LLM cost:  O(1569²)       = 2.5M           LLM cost:  O(776²)        = 0.6M  (4× cheaper)
Position:  direct 1369→RoPE (no pooling)    Position:  pool 1369→576, then RoPE
```
 
### Alt-B: No SVA, Direct Concatenation (LLaVA-style)
 
Replace SVA with simple concatenation + MLP.
 
```
Current:                                    Alternative:
SVA: [B, 3314, 4096] → [B, 1369, 4096]    Concat: [B, 3314, 4096] → MLP → [B, 3314, 4096]
                                            All 3314 tokens enter LLM directly
 
LLM cost: O(1569²) = 2.5M                  LLM cost: O(3514²) = 12.3M  (+5×)
```

### Alt-C: Mean Aggregation Instead of 15th-Percentile

```
Current:  15th-percentile depth pixel within each 14×14 patch
Alt:      mean of all 196 pixel positions within each patch

Both produce: [B, 1369, 3]
```

**Hypothesis H2e tests this**: If 15th-pct wins, foreground bias is validated.
If mean wins, the percentile selection is adding noise not signal.

### Alt-D: Standard M-RoPE for All Tokens (No IcosahedralRoPE3D)
 
```
Current:  Text → M-RoPE [24t, 20h, 20w] = 64 pairs
          Spatial → IcosahedralRoPE3D [48 pairs from 3D positions] + 16 identity
 
Alt:      All tokens → M-RoPE [24t, 20h, 20w]
          Spatial tokens get sequential positions (position 201, 202, ... 1569)
```
**Hypothesis H2b tests this**: If IcosahedralRoPE3D wins, 3D-aware positional encoding
is confirmed essential. If M-RoPE matches, the explicit geometry in the feature channel
(GATr → SVA) is sufficient without positional encoding.

### Alt-E: Single Encoder (SigLIP Only or DINOv2 Only)

```
SigLIP only:                                DINOv2 only:
  Encoders: [B, 576, 4096]                   Encoders: [B, 1369, 4096]
  KV pool:  576 + 1369 GATr = 1945           KV pool:  1369 + 1369 GATr = 2738
  No semantic features from SigLIP            No language-aligned features

  Hypothesis H1a tests this                   Hypothesis H1a tests this
```

### Alt-F: No GATr Branch

```
Current:  KV = SigLIP[576] + DINOv2[1369] + GATr[1369] = 3314
Alt:      KV = SigLIP[576] + DINOv2[1369] = 1945

SVA cost: O(1369 × 1945) = 2.7M  (vs current 4.5M — 40% cheaper)
```

**Hypothesis H2a tests this**: If removing GATr drops metric tasks but not semantic
tasks, GATr is confirmed complementary. If no degradation, GATr adds complexity
without benefit (and our architecture is over-engineered).

### Alt-G: Intermediate Query Count (729 = 27×27)

```
Q: [B, 729, 4096]    (27×27 grid — between 24×24 and 37×37)
Position pooling: 37→27 (1.37:1 ratio, milder than 37→24)

SVA cost: O(729 × 3314) = 2.4M    (26% more than 576)
LLM cost: O(929²) = 0.86M          (43% more than 576)
```

A compromise between geometric preservation and compute cost. Not explicitly
hypothesized, but could be added as a sweep alongside H1d/H3e.

---

## INFORMATION FLOW: TWO CHANNELS THROUGH THE LLM

```
FEATURE CHANNEL (learned, through SVA):
  "What is here" — semantic + structural + geometric features
  ┌──────────────────────────────────────────────────────────────┐
  │ SigLIP[576,4096] ─┐                                          │
  │ DINOv2[1369,4096] ─┼─→ SVA[3314→1369,4096] → NormMatch → DeepStack │
  │ GATr[1369,4096] ──┘     ↑                                    │
  │                    COMPRESSION HERE                          │
  │                    2.42:1 via attention                      │
  │                    Geometry better preserved                 │
  └──────────────────────────────────────────────────────────────┘

POSITION CHANNEL (deterministic, bypasses SVA):
  "Where is here" — raw 3D coordinates as rotary encoding
  ┌──────────────────────────────────────────────────────────────┐
  │ Depth[518,518] → Backproject[268K,3] → 15th%ile[1369,3]      │
  │   → IcosahedralRoPE3D[1369,96] (no pooling — 1:1 mapping)    │
  │   → Padded to [1369,128] with identity pairs                 │
  │   → Applied as Q,K rotations in LLM attention via monkey-patch│
  │                                                              │
  │ No compression: 1369 positions map 1:1 to 1369 SVA queries   │
  │ Geometry fully preserved: explicit 3D coordinates            │
  └──────────────────────────────────────────────────────────────┘

The feature channel can lose geometric detail through SVA compression (2.42:1).
The position channel preserves explicit 3D coordinates with zero compression.
Together, they provide both "what" and "where" to the LLM.
```

---

## TRAINABLE PARAMETER BREAKDOWN

```
Component                              Params      Shape / Notes
─────────────────────────────────────────────────────────────────
SigLIP2 encoder (frozen)               ~400M       27 layers × 1152 hidden
DINOv2-L encoder (frozen)              ~307M       24 layers × 1024 hidden
Qwen3-VL-8B backbone (frozen)          ~8,000M     36 layers × 4096 hidden

SigLIP MLP projector                   ~31M        Linear(3456→4096) + Linear(4096→4096)
DINOv2 MLP projector                   ~29M        Linear(3072→4096) + Linear(4096→4096)
GATr MLP projector                     ~17M        Linear(48→4096) + Linear(4096→4096)
GATr 8 equivariant blocks              ~12M        8 × GATrBlock(mv=16, s=32)

SVA query embed                        5.6M        [1369, 4096]
SVA cross-attn (2 layers)              ~84M        2 × (Q+K+V+O proj + LayerNorms)
SVA typed attention bias               9           [3, 3]
 
Qwen3 LoRA rank-32                     ~31M        36 × (Q+K+V+O LoRA adapters)
DeepStack (native Qwen3-VL)            0           Uses built-in mechanism (no new params)
─────────────────────────────────────────────────────────────────
Total trainable:                       ~206M       ~2.3% of ~8,890M total
 
Note: Gated cross-attention (~377M, 9 layers) was removed in the icosahedral redesign.
Qwen3-VL's native DeepStack mechanism replaces it with zero additional trainable parameters.
```

