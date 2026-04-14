# SpatialVLM: Complete Training Pipeline — Input to Output

Every tensor shape, every code path, every ablation tag, from a raw Habitat
frame to a GRPO-trained navigation model.

---

## 0. Raw Habitat Simulator Output

```
RGB frame:       [B, H_orig, W_orig, 3]   uint8 or float, e.g. 640×480
Depth map:       [B, 518, 518]             float32, metric metres (GT)
Intrinsics:      CameraIntrinsics(fx, fy, cx=259, cy=259, width=518, height=518)
Text prompt:     str — e.g. "Navigate to the red couch. Reasoning: ... Action: ..."
```

> Code: `src/spatialvlm/data/habitat_env.py`
> Config: `src/spatialvlm/config/model.py:GeometryConfig.depth_image_size = 518`

Depth is rendered natively at 518×518 so each DINOv2 14×14 pixel patch
maps exactly to a 14×14 depth region. Zero resize artefacts on the
geometry branch.

---

## 1. Image Preprocessing

> Code: `src/spatialvlm/data/preprocessing.py`

### 1a. Convert to float [0,1] channel-first

```
to_float_rgb01(rgb)                                       → [B, H, W, 3] float
.permute(0, 3, 1, 2)                                      → [B, 3, H, W]
```

### 1b. Resize for SigLIP2 — `resize_rgb_bchw(rgb, (384, 384))`

```
[B, 3, H_orig, W_orig]  → bilinear → [B, 3, 384, 384]
```

384 / 16 = 24 patches/side → 576 patches exact.

### 1c. Resize for DINOv2 — `resize_rgb_bchw(rgb, (518, 518))`

```
[B, 3, H_orig, W_orig]  → bilinear → [B, 3, 518, 518]
```

518 / 14 = 37 patches/side → 1369 patches exact.

> **Ablation H1d**: compares 1369-query (full DINOv2 res) vs 576-query
> (pooled to SigLIP grid). This resize is the reason full-res is possible.

---

## 2. Stage 1 — Dual Vision Encoding (FROZEN)

> Code: `model.py:140-167 → encode_vision()`
> Encoder code: `encoders/siglip.py`, `encoders/dinov2.py`
> Projector code: `encoders/projector.py`

### 2a. SigLIP2-SO400M/16-NaFlex (frozen, 27 layers)

```
Input:            [B, 3, 384, 384]
Patch embed:      16×16×3 pixels → 1152-dim vector
                  24×24 = 576 patches, NO CLS token
                  [B, 576, 1152]

27 transformer layers (frozen forward pass):
  Each: [B, 576, 1152] → [B, 576, 1152]
  Self-attention: 576×576 attention map per head

Multi-layer extraction at {9, 18, 27}:         ← Ablation H1c: multi-layer vs final-only
  Layer  9: [B, 576, 1152]  (low-level)
  Layer 18: [B, 576, 1152]  (mid-level)
  Layer 27: [B, 576, 1152]  (high-level)

Channel concatenation:
  [B, 576, 1152] × 3 → [B, 576, 3456]
```

### 2b. SigLIP MLP projector (TRAINABLE, ~31M)

```
MLPProjector(in_dim=3456, out_dim=4096)
  Linear(3456, 4096) → GELU → Linear(4096, 4096)

[B, 576, 3456] → [B, 576, 4096]               siglip_tokens
```

> Code: `model.py:98-101` → `self.siglip_projector`

### 2c. DINOv2-L/14 (frozen, 24 layers)

```
Input:            [B, 3, 518, 518]
Patch embed:      14×14×3 pixels → 1024-dim vector
                  37×37 = 1369 patches + 1 CLS token
                  [B, 1370, 1024]

24 transformer layers (frozen):
  Each: [B, 1370, 1024] → [B, 1370, 1024]

CLS stripping:
  [B, 1370, 1024] → [B, 1369, 1024]          (remove index 0)

Multi-layer extraction at {8, 16, 24}:         ← Ablation H1c
  Layer  8: [B, 1369, 1024]
  Layer 16: [B, 1369, 1024]
  Layer 24: [B, 1369, 1024]

Channel concatenation:
  [B, 1369, 1024] × 3 → [B, 1369, 3072]
```

### 2d. DINOv2 MLP projector (TRAINABLE, ~29M)

```
MLPProjector(in_dim=3072, out_dim=4096)
  Linear(3072, 4096) → GELU → Linear(4096, 4096)

[B, 1369, 3072] → [B, 1369, 4096]             dinov2_tokens
```

> Code: `model.py:103-106` → `self.dinov2_projector`
>
> **Ablation H1a**: SigLIP-only vs DINOv2-only vs both.
> **Ablation H1b**: is pre-alignment needed specifically for DINOv2?

---

## 3. Stage 2 — Geometric Branch (parallel to Stage 1)

> Code: `model.py:169-200 → encode_geometry()`

### 3a. Depth backprojection — `backproject_depth_map()`

> Code: `geometry/backproject.py:18-42`

```
Input:   depth [B, 518, 518]          metric metres
         intrinsics (fx, fy, cx=259, cy=259)

Per-pixel:
  X = (u - 259) × depth[u,v] / fx
  Y = (v - 259) × depth[u,v] / fy
  Z = depth[u,v]

Output:  [B, 518, 518, 3]             full 3D point cloud, camera coords
```

Lossless, invertible. Invalid pixels (depth=0) → (0, 0, 0).

### 3b. Patch-level 15th-percentile aggregation — `aggregate_patches_percentile()`

> Code: `geometry/backproject.py:45-117`

```
Input:   point_map [B, 518, 518, 3]

Reshape to DINOv2 patch grid:
  [B, 518, 518, 3]
  → [B, 37, 14, 37, 14, 3]      split into 37×37 patches of 14×14 pixels
  → [B, 37, 37, 14, 14, 3]      permute
  → [B, 1369, 196, 3]           flatten

Per-patch depth sort (ascending = nearest first):
  196 pixels ranked by depth
  k = floor(0.15 × 196) = 29    (30th nearest pixel)

Gather 3D point at rank k:
  [B, 1369, 196, 3] → gather index 29 → [B, 1369, 3]

Output:  [B, 1369, 3]             one 3D point per DINOv2 patch
```

196:1 compression. The selected point IS a real pixel (not interpolated).

> **Ablation H2e**: replace 15th-percentile with mean aggregation.

### 3c. GATr geometric processing — `GATrWrapper.forward()`

> Code: `geometry/gatr_wrapper.py:110-157`

```
Input:   [B, 1369, 3]    patch-level 3D positions

Normalize:
  radius = norm(coords, dim=-1)                    → [B, 1369]
  max_radius = radius.amax(dim=-1).clamp_min(eps)  → [B, 1]
  coords = coords / max_radius.unsqueeze(-1)       → [B, 1369, 3] in [-1, 1]

PGA embedding (embed_point):
  (x,y,z) → 16-D multivector trivector:
    mv[11] = -z, mv[12] = +y, mv[13] = -x, mv[14] = 1.0
  MV stream:     [B, 1369, 1, 16]
  Scalar stream: [B, 1369, 1]     (vector norm of coords)

8 equivariant GATr blocks (TRAINABLE, ~12M):
  Each: GATrBlock(equivariant self-attention + equivariant FFN)
  MV channels: 1 → 16
  Scalar channels: 1 → 32
  After 8 blocks:
    MV out:     [B, 1369, 16, 16]
    Scalar out: [B, 1369, 32]

Invariant extraction:                              ← THIS IS WHERE INVARIANCE HAPPENS
  MV norms:    norm(mv_out, dim=-1)  → [B, 1369, 16]
  Concatenate: [scalar_out, mv_norms] → [B, 1369, 48]
  These 48 values are ROTATION-INVARIANT by design.

GATr MLP projector (TRAINABLE, ~17M):
  MLPProjector(48, 4096)
  Linear(48, 4096) → GELU → Linear(4096, 4096)

Output:  [B, 1369, 4096]                          gatr_tokens
Also:    [B, 1369, 3]                              positions_3d (pre-normalization, for RoPE)
```

> **Ablation H2a**: remove GATr entirely (KV pool = SigLIP + DINOv2 = 1945).
> **Ablation H2f**: 6 icosahedral dirs vs 4 tetrahedral dirs.

### 3d. IcosahedralRoPE3D (computed now, INJECTED in Stage 4)

> Code: `geometry/gridcell_rope3d.py:82-148`

```
Input:   [B, 1369, 3]    positions_3d in metres (BEFORE GATr normalization)

6 icosahedral directions:
  Golden ratio φ ≈ 1.618, normalized (0.526, 0.851):
  d₁ = [0, +0.526, +0.851]  ...  d₆ = [+0.851, 0, -0.526]

Scalar projections:
  einsum("bnd,fd->bnf", positions, icosa_dirs)
  [B, 1369, 3] × [6, 3] → [B, 1369, 6]

8 frequencies: f_k = 10 × e^(k/3), k=0..7
  f₀=10.0, f₁=13.96, f₂=19.48, f₃=27.18, f₄=37.94, f₅=52.95, f₆=73.89, f₇=103.12

Scale:
  [B, 1369, 6] × [8] → [B, 1369, 6, 8]

Sin/cos:
  sin([B, 1369, 6, 8])  cos([B, 1369, 6, 8])
  stack → [B, 1369, 6, 8, 2]
  reshape → [B, 1369, 96]

Output:  [B, 1369, 96]   rotary angles (stored for Stage 4 injection)
```

Zero learnable params. Pure function of 3D position.

> **Ablation H2b**: IcosahedralRoPE3D vs standard M-RoPE for spatial tokens.
> **Ablation H2d**: e^(1/3) ratio vs golden ratio vs power-of-2.

---

## 4. Stage 3 — Fusion

> Code: `model.py:202-234 → fuse()`

### 4a. SVA Cross-Attention — `SpatialVisionAggregator.forward()`

> Code: `fusion/sva.py:180-290`

```
Inputs (all [*, 4096]):
  siglip_tokens:  [B, 576, 4096]    (type 0 — semantic)
  dinov2_tokens:  [B, 1369, 4096]   (type 1 — structural)
  gatr_tokens:    [B, 1369, 4096]   (type 2 — geometric)

KV concatenation:
  kv = cat([siglip, dinov2, gatr], dim=1) → [B, 3314, 4096]
  kv_type_ids = [0]*576 + [1]*1369 + [2]*1369     → [3314]

Query initialization (DINOv2 as structural anchor):
  queries = dinov2_tokens + query_embed            → [B, 1369, 4096]
  query_embed: learnable [1369, 4096], scaled by D^{-0.5}
  query_type_ids = [1]*1369                        (all DINOv2 type)

SVA Layer 1 (cross-attention):
  Pre-norm: q_norm(queries), kv_norm(kv_tokens)
  Q: Linear(4096→4096) → [B, 32 heads, 1369, 128]
  K: Linear(4096→4096) → [B, 32 heads, 3314, 128]
  V: Linear(4096→4096) → [B, 32 heads, 3314, 128]

  Typed attention bias (if enabled):
    3×3 learned matrix, indexed by (query_type, kv_type)
    → additive mask [1, 1, 1369, 3314]

  Attention: softmax(Q @ K^T / √128 + typed_mask) → [B, 32, 1369, 3314]
  Output:    attn @ V → [B, 32, 1369, 128] → out_proj → residual + LayerNorm
  → [B, 1369, 4096]

SVA Layer 2: same architecture, same original KV pool (NOT blurred).
  → [B, 1369, 4096]

Output:  [B, 1369, 4096]     fused spatial tokens
```

Trainable: ~86M (query_embed + 2 layers of Q/K/V/O projections + 2 LayerNorms +
2 typed bias [3,3] matrices).

> **Ablation H3e**: 3314-token KV vs pooled 1728-token KV.
> **Ablation H3c**: PERMUTATION TEST — shuffle all 3314 KV tokens randomly.
>   If ours drops >15% and SOTA VLM drops <3%, spatial structure is being used.
>   THIS IS THE SMOKING GUN.
> **Diagnostic**: `return_attention_stats=True` reports per-layer per-head
>   attention mass on SigLIP / DINOv2 / GATr. Answers "is geometry being used?"
>   (`fusion/sva.py:81-117`, `_attention_with_stats`)

### 4b. RMS Norm Matching — `RMSNormMatching.forward()`

> Code: `fusion/norm_matching.py:46-86`

```
Input:   vision_tokens [B, 1369, 4096]  (from SVA)
         text_tokens   [B, T, 4096]     (from tokenizer, training only)

vision_rms = sqrt(mean(vision² per dim)).mean()   → scalar
text_rms_ema updated: 0.99 × old + 0.01 × batch_text_rms
scale = text_rms_ema / (vision_rms + 1e-6)

Output:  [B, 1369, 4096] × scale                   fused_tokens
```

Zero learnable params. Scalar multiply preserves all relative structure.
Only adjusts magnitude so RoPE positional signal isn't drowned.

> **Ablation H3b**: remove norm matching → spatial reasoning should degrade.

---

## 5. Stage 4 — LLM Backbone (Qwen3-VL-8B)

> Code: `model.py:236-357`, `backbone/qwen3_vl.py`, `backbone/rope_patch.py`

### 5a. Sequence construction

```
text_tokens:    tokenize("system + instruction + history") → input_ids [B, T]
spatial_start:  index in the sequence where spatial tokens are injected

Combined:       [B, T + 1369]    text tokens + 1369 placeholder spatial tokens
```

### 5b. DeepStack injection (native Qwen3-VL, zero new params)

> Code: `model.py:236-280 → build_deepstack_inputs()`

```
spatial_token_mask: [B, T+1369] bool — True at spatial positions
deepstack_visual_embeds: fused_tokens [B, 1369, 4096]

At early LLM layers:
  hidden_states[spatial_mask, :] += fused_visual_embeds
```

Replaces the old gated cross-attention (~378M params) with zero-cost
residual addition using Qwen3-VL's built-in mechanism.

> **Ablation H3f**: DeepStack injection vs no injection (just concatenation).

### 5c. RoPE monkey-patch — `patch_rope_forward()`

> Code: `backbone/rope_patch.py:102-162`

```
Standard M-RoPE:
  cos, sin = original_forward(x, position_ids)    → [B, T+1369, 128]

For spatial token positions (mask from build_deepstack_inputs):
  1. IcosahedralRoPE3D(positions_3d)               → [B, 1369, 96]
  2. Split sin/cos pairs                            → [B, 1369, 48] each
  3. Pad to 64 pairs with identity (cos=1, sin=0)   → [B, 1369, 64] each
  4. Duplicate to head_dim: cat([x, x])              → [B, 1369, 128] each
  5. Overwrite cos/sin at spatial positions

Result:
  Text tokens:     M-RoPE [24t, 20h, 20w] = 64 pairs = 128 dims
  Spatial tokens:  Icosahedral 48 pairs + 16 identity = 64 pairs = 128 dims
```

> **Ablation H4a**: IcosahedralRoPE3D improves navigation success rate.

### 5d. Qwen3-VL-8B forward (frozen backbone + LoRA rank-32)

```
36 transformer layers:
  hidden_size: 4096
  32 query heads, 8 KV heads (GQA 4:1)
  head_dim: 128
  FFN intermediate: 12288

Per layer:
  RoPE applied to Q, K (text: M-RoPE, spatial: icosahedral)
  Self-attention: [B, 32, T+1369, 128] full self-attention
  FFN: [B, T+1369, 4096] → 12288 → 4096

  LoRA rank-32 on Q, K, V, O projections (TRAINABLE, ~31M total):
    W_Q LoRA: (4096×32) + (32×4096) per layer
    W_K LoRA: (4096×32) + (32×1024) per layer
    W_V LoRA: (4096×32) + (32×1024) per layer
    W_O LoRA: (4096×32) + (32×4096) per layer
    ~852K per layer × 36 = ~30.7M

LM head: Linear(4096, vocab_size) → logits
Output:  [B, T+1369, vocab_size]   → autoregressive next-token prediction
```

---

## 6. Trainable Parameter Summary

```
Component                          Params    Freeze group name
────────────────────────────────────────────────────────────────
SigLIP2 encoder (frozen)           ~400M     — (always frozen)
DINOv2-L encoder (frozen)          ~307M     — (always frozen)
Qwen3-VL-8B backbone (frozen)     ~8,000M    — (always frozen)

SigLIP MLP projector                ~31M     "siglip_proj"
DINOv2 MLP projector                ~29M     "dino_proj"
GATr (8 blocks + MLP projector)     ~29M     "gatr"
SVA (query_embed + 2 xattn layers)  ~86M     "sva"
Qwen3 LoRA rank-32                  ~31M     "lora"
────────────────────────────────────────────────────────────────
Total trainable:                   ~206M     (~2.3% of ~8,890M)
```

> Code: `training/sft.py:17-26` → `FREEZE_GROUP_PATTERNS`

---

## 7. Training Stage 1 — Pre-alignment (Habitat-rendered frames)

> Code: `training/prealign.py`, config: `configs/train_prealign.yaml`
> Config class: `training/prealign.py:PrealignConfig`

**Goal**: Align vision projectors to the LLM's embedding space.
Data: Habitat-rendered frames from R2R-CE + RxR-CE episodes with GT depth.
(NOT LLaVA-558K — model.forward() requires depth + intrinsics for GATr branch.)

```
Trainable:       "projector" keyword → siglip_projector + dinov2_projector + gatr.projector
                 ≈ 77M params
Frozen:          everything else (encoders, GATr blocks, SVA, LoRA, backbone)
Loss:            cross-entropy on next-token prediction (masked_lm_loss)
LR:              1e-4
```

**What happens**:
1. Habitat image → dual resize → encoders (frozen forward) → projectors (trainable)
2. Depth → backproject → aggregate → GATr (frozen blocks) → projector (trainable)
3. Projected tokens enter SVA (frozen) → norm matching → backbone (frozen)
4. Cross-entropy loss → backprop ONLY through projectors

> **Staged training option**: `PrealignConfig(trainable_groups=("gatr",))`
> trains only GATr's projector first, then switch to `("dino_proj",)` etc.
> See `training/sft.py:set_trainable_by_groups()` and `tests/test_freeze_groups.py`.
>
> **Ablation H1b**: skip pre-alignment for DINOv2 → test if it's necessary.

---

## 8. Training Stage 2 — Supervised Fine-Tuning (SFT)

> Code: `training/sft.py`, config: `configs/train_sft.yaml`
> Config class: `training/sft.py:SFTConfig`

**Goal**: Full spatial instruction-following. The model now learns to
reason about 3D scenes, plan navigation, answer spatial questions.

```
Trainable:       ("projector", "gatr", "sva", "cross_attn", "lora") keywords
                 → all projectors + GATr blocks + SVA + LoRA
                 ≈ 206M params
Frozen:          encoders (SigLIP, DINOv2) + Qwen3-VL base weights
Loss:            cross-entropy on next-token prediction (supervised_loss)
                 with optional label_smoothing
LR:              5e-5
Grad clip:       1.0
```

**What happens**:
1. Full forward pass through all 5 stages (all trainable modules backprop)
2. Text targets are navigation instructions with format: `Reasoning: ... Action: ...`
3. SVA learns to weight GATr tokens (typed bias), GATr blocks learn invariants,
   LoRA adapts the LLM to spatial reasoning
4. Norm matching EMA tracks text-to-vision magnitude ratio

**Data**: R2R-CE, RxR-CE, SQA3D instruction-following pairs from Habitat.

> **Staged training lever**: `SFTConfig(trainable_groups=("gatr", "sva"))`
> trains only geometry+fusion first (professor's suggestion).
> Then `SFTConfig(trainable_groups=("gatr", "sva", "dino_proj"))` etc.
>
> **Ablation H5a**: SFT-only vs SFT+GRPO.

---

## 9. Training Stage 3 — GRPO (Group Relative Policy Optimization)

> Code: `training/grpo.py`, `training/rewards.py`, `training/curriculum.py`
> Config classes: `GRPOConfig`, `RewardConfig`, `RewardCurriculum`
> Config file: `configs/train_grpo.yaml`

**Goal**: Improve spatial actions via RL with dense rewards.

### 9a. Reward computation

> Code: `training/rewards.py:153-209 → compute_reward_terms()`

For each trajectory in a group of K=8 rollouts:

```
Reward terms (all [B]):
  R_format:      +1.0 if response contains "Reasoning:" and "Action:"
  R_progress:    previous_geodesic - current_geodesic (clipped [-2, +2])
  R_collision:   -2.0 if clearance < 0.1m (else 0)
  R_goal:        +10.0 if stopped within 1m of goal (else 0)
  R_consistency: -1.0 if predicted action ≠ executed action (else 0)
```

> **Ablation H5c**: dense rewards (all 5 terms) vs sparse (goal only).
> **Ablation H5d**: consistency reward prevents reward hacking.

### 9b. Curriculum weighting

> Code: `training/curriculum.py:38-117 → RewardCurriculum`

Default 6-epoch schedule:

```
Epoch 1:  format=1.0  progress=0.1  collision=0.1  goal=0.1  consistency=0.0
          (train format compliance first)
Epoch 3:  format=0.4  progress=0.4  collision=0.3  goal=0.5  consistency=0.2
          (ramp up spatial signals)
Epoch 5:  format=0.1  progress=0.7  collision=0.6  goal=1.0  consistency=0.6
          (spatial dominates, format is mastered)
```

Intermediate epochs interpolate linearly between anchor points.

```
R_total = Σ w_i × R_i       (curriculum-weighted sum)
```

### 9c. GRPO optimization loop

> Code: `training/grpo.py:137-188 → grpo_loss()`, `training/grpo.py:250-300 → GRPOTrainer.step()`

```
Per optimization step:

1. Generate K=8 rollouts per prompt (group)
   Each rollout: full forward pass → decode action → execute in Habitat
   → collect (response, geodesic, clearance, stopped)

2. Compute rewards:
   reward_terms = compute_reward_terms(responses, ...)   → 5 × [B]
   weights = curriculum.get_weights(epoch)
   rewards = total_reward(reward_terms, weights)          → [B]

3. Group advantages:
   rewards reshaped to [G, 8]   (G groups of K=8)
   advantages = (rewards - mean) / std  per group         → [G, 8] → [N]

4. Token-level logprobs:
   new_logprobs:  current policy    [N, T]
   old_logprobs:  pre-step policy   [N, T]
   ref_logprobs:  frozen reference  [N, T]

5. GRPO loss:
   ratio = exp(new - old)                                 [N, T]
   clipped = clamp(ratio, 1-ε, 1+ε)       ε = 0.2
   policy_loss = -mean(min(ratio×adv, clipped×adv))
   kl_loss = approximate_kl(new, ref)      β = 0.001
   total = policy_loss + β × kl_loss

6. Selective Sample Replay (SSR):
   If |advantage| > 0.05, store in replay buffer (capacity 4096)
   → mitigates vanishing advantages as training progresses

7. Backward + clip_grad_norm(1.0) + optimizer.step()
```

```
Trainable params:  same ~206M as SFT
LR:                5e-7 (10× lower than SFT)
Clip ε:            0.2
KL β:              0.001
```

> **Ablation H5a**: SFT+GRPO vs SFT-only.
> **Ablation H5b**: GRPO vs fDPO (alternative RL: `training/fdpo.py`).
> **Ablation H4b**: CoT decoding benefits more from GRPO than direct.

---

## 10. What the Model Learns at Each Stage (Information Flow)

```
FEATURE CHANNEL (through SVA → DeepStack):
  "What is here" — semantic + structural + geometric

  SigLIP: 576 tokens [4096-d] — language-aligned object recognition
  DINOv2: 1369 tokens [4096-d] — fine spatial structure, self-supervised
  GATr:   1369 tokens [4096-d] — pairwise distances, angles, coplanarity
                                  (rotation-INVARIANT by Step 3c extraction)

  → SVA compresses 3314→1369 via cross-attention (2.42:1)
  → RMS norm matching scales to text-token magnitude
  → DeepStack adds as residual at early LLM layers

POSITION CHANNEL (bypasses SVA, injected at RoPE):
  "Where is here" — raw 3D coordinates as rotary encoding

  Depth → Backproject → 15th%ile → [B, 1369, 3]
  → IcosahedralRoPE3D → [B, 1369, 96] padded to [B, 1369, 128]
  → Overwrites cos/sin at spatial positions in Q, K rotary
  → Two spatial tokens close in 3D have high rotary dot product
     = naturally attend to each other = geometry in attention pattern

  No compression. 1:1 mapping from DINOv2 patches.
```

---

## 11. Complete Ablation Map (reference IDs to code)

| ID   | What's ablated                      | Code change needed                                  | Module          |
|------|-------------------------------------|-----------------------------------------------------|-----------------|
| H1a  | Single encoder (SigLIP or DINOv2)   | Skip one encoder in `encode_vision`, adjust SVA KV  | Stage 1         |
| H1b  | Skip pre-alignment for DINOv2       | Don't run `prealign.py` for DINOv2 projector         | Stage 1 train   |
| H1c  | Final-layer only (no multi-layer)   | `extract_layers=[27]` / `[24]` in EncoderConfig      | Stage 1         |
| H1d  | Pool to 576 queries (SigLIP grid)   | `sva_num_queries=576`, use `pool_positions_to_sva_grid` | Stage 1/3    |
| H2a  | Remove GATr                         | Skip GATr in `encode_geometry`, KV = siglip+dino=1945 | Stage 2        |
| H2b  | Standard M-RoPE for spatial         | Skip RoPE monkey-patch, spatial gets seq positions    | Stage 2/4       |
| H2c  | Depth Anything V2 vs GT depth       | Swap depth source in `encode_geometry` (low priority) | Stage 2         |
| H2d  | Frequency ratio sweep               | Change `freq_ratio` in GeometryConfig                 | Stage 2         |
| H2e  | Mean vs 15th-percentile             | Change `depth_percentile` in GeometryConfig           | Stage 2         |
| H2f  | 6 icosa dirs vs 4 tetra dirs        | Change `_build_icosahedral_directions` / config       | Stage 2         |
| H3b  | Remove RMS norm matching             | Skip `norm_matching` in `fuse()`                      | Stage 3         |
| H3c  | Permutation test (THE SMOKING GUN)  | Shuffle KV token order before SVA → measure drop      | Stage 3         |
| H3e  | 3314 KV vs 1728 KV (pooled)         | Pool all streams to 576 before concat                 | Stage 3         |
| H3f  | No DeepStack injection               | Remove `deepstack_visual_embeds` from kwargs          | Stage 3/4       |
| H4a  | IcosahedralRoPE3D vs no spatial PE  | Disable RoPE monkey-patch for spatial tokens          | Stage 4         |
| H4b  | CoT vs direct decoding with GRPO     | Prompt format change (Reasoning: ... vs direct)       | Stage 4/5       |
| H5a  | SFT-only (no GRPO)                  | Stop after Stage 2 training                           | Stage 5         |
| H5b  | fDPO vs GRPO                         | Use `fdpo.py` trainer instead of `grpo.py`            | Stage 5         |
| H5c  | Dense vs sparse rewards              | RewardWeights with only `goal_weight > 0`             | Stage 5         |
| H5d  | No consistency reward                | Set `consistency_weight=0` in curriculum              | Stage 5         |

---

## 12. One Forward Pass End-to-End: Shape Trace

```
Habitat RGB [B, 640, 480, 3]
  ├── to_float_rgb01 → permute → [B, 3, 640, 480]
  ├── resize_rgb_bchw(384) → [B, 3, 384, 384]
  │     └── SigLIP2 (frozen) → [B, 576, 3456]
  │           └── MLPProjector(3456→4096) → [B, 576, 4096]           siglip_tokens
  └── resize_rgb_bchw(518) → [B, 3, 518, 518]
        └── DINOv2 (frozen) → [B, 1369, 3072]
              └── MLPProjector(3072→4096) → [B, 1369, 4096]          dinov2_tokens

Habitat Depth [B, 518, 518]
  ├── backproject_depth_map → [B, 518, 518, 3]
  │     └── aggregate_patches_percentile → [B, 1369, 3]              positions_3d
  │           └── GATrWrapper → [B, 1369, 4096]                      gatr_tokens
  └── (positions_3d saved for RoPE injection in Stage 4)

SVA([B,576,4096], [B,1369,4096], [B,1369,4096])
  └── KV: [B, 3314, 4096]  Q: [B, 1369, 4096]
        └── 2-layer cross-attn → [B, 1369, 4096]                    fused

RMSNormMatching(fused, text_tokens) → [B, 1369, 4096]               fused_tokens

Tokenize text → input_ids [B, T+1369]
Build spatial_mask, deepstack_visual_embeds

Qwen3-VL-8B + LoRA + RoPE patch:
  Text:    M-RoPE [24, 20, 20]
  Spatial: Icosahedral 48 pairs + 16 identity
  DeepStack: hidden[spatial_mask] += fused_tokens
  36 layers self-attention + FFN → LM head

  → logits [B, T+1369, vocab_size]
  → cross_entropy(logits, labels)   [SFT]
  → grpo_loss(logprobs, ...)        [GRPO]
```
