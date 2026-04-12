# SpatialVLM: Next Steps

What to do, in what order, with what data.

---

## The Data Problem

Every forward pass through SpatialVLM requires **RGB + GT depth + camera intrinsics**.
The geometric branch (backprojection + GATr) cannot run without depth. This means:

- **LLaVA-558K cannot be used for pre-alignment.** It has no depth maps.
- **All training data must come from environments with GT depth** — in practice,
  Habitat-rendered frames from HM3D or Matterport3D scenes.
- Real-image datasets (What's Up, CV-Bench) are evaluation-only and require
  estimated depth (Depth Anything V2), which is out of scope for the core pipeline.

---

## Dataset Plan by Training Stage

### Stage 1: Pre-alignment (~77M trainable, projectors only)

**Goal**: Align the three MLP projectors (SigLIP, DINOv2, GATr) to the Qwen3-VL
embedding space. The model learns to map visual+geometric features into the LLM's
token manifold via next-token prediction on simple image-text pairs.

**Data source**: Habitat-rendered frames from R2R-CE and RxR-CE episodes.

| Field | Source | Details |
|-------|--------|---------|
| RGB | Habitat render | Native resolution, resized to 384x384 (SigLIP) and 518x518 (DINOv2) |
| GT depth | Habitat render | 518x518, metric metres, zero error |
| Intrinsics | Habitat config | fx, fy, cx, cy from the sensor spec |
| Text | Episode instruction | R2R/RxR navigation instruction for the trajectory |

**How to build it**:

1. Load R2R-CE and RxR-CE episode JSONs (~21K + ~126K episodes).
2. For each episode, load the scene in Habitat, teleport the agent to the start
   position (and optionally 2-4 waypoints along the trajectory).
3. Render RGB + depth at 518x518. Store as pre-cached `.pt` files on disk:
   ```
   data/prealign_frames/
   ├── r2r-ce/
   │   ├── ep_0001_step_000.pt   # {"rgb": [3,518,518], "depth": [518,518], "intrinsics": {...}}
   │   ├── ep_0001_step_005.pt
   │   └── ...
   └── rxr-ce/
       └── ...
   ```
4. Each `.pt` also stores the episode instruction as text. At training time, tokenize
   the instruction and build the full batch (see Collation section below).

**Target size**: ~50-100K frames (3-5 frames per episode from R2R-CE + a subset of RxR-CE).

**Why not LLaVA-558K**: No depth, no intrinsics. The GATr projector would receive
zero-tensors or random noise, learning garbage alignment. All three projectors must
see real geometric data from day one.

**Why not SQA3D here**: SQA3D uses ScanNet scenes (mesh reconstructions). The
question format (situated QA) is harder than simple captioning. Better saved for SFT.

### Stage 2: SFT (~206M trainable)

**Goal**: Full spatial instruction-following. The model learns to produce
`Reasoning: <thought> Action: <nav_action>` responses.

**Data sources** (combined):

| Dataset | Episodes | Scene Source | What It Teaches |
|---------|----------|-------------|-----------------|
| R2R-CE | ~21K | Matterport3D | Step-by-step navigation from detailed English instructions |
| RxR-CE | ~126K | Matterport3D | Multilingual navigation, longer paths, richer spatial language |
| SQA3D | ~33K | ScanNet | Situated spatial QA ("What is to my left?", "How far is the table?") |

**Per-sample format** (for R2R-CE / RxR-CE):
```
Input:  [system] [instruction: "Walk past the bed and turn right..."]
        [spatial_tokens: 1369 fused visual tokens from current viewpoint]
Target: "Reasoning: I see a bed ahead and a doorway to the right.
         The instruction says to walk past the bed. Action: MOVE_FORWARD"
```

**Per-sample format** (for SQA3D):
```
Input:  [system] [question: "What is the object closest to me on the left?"]
        [spatial_tokens: 1369 fused visual tokens from the specified viewpoint]
Target: "Reasoning: Looking to my left, the nearest object is a wooden chair
         approximately 1.2m away. Answer: chair"
```

**How to build it**:

1. **R2R-CE / RxR-CE**: For each episode, load the full trajectory in Habitat.
   At each step: render RGB + depth, record the ground-truth action, build the
   reasoning+action target string. Each trajectory step is one training sample.
   ~21K episodes x ~6 avg steps = ~126K samples from R2R alone.

2. **SQA3D**: For each QA pair, load the ScanNet scene in Habitat, render from the
   specified viewpoint, build the reasoning+answer target. ~33K samples.

3. Pre-render and cache all frames to disk (same `.pt` format as pre-alignment,
   but also include the target text and action label).

**Total SFT samples**: ~300-400K (R2R steps + RxR steps + SQA3D).

### Stage 3: GRPO (~206M trainable, online RL)

**Goal**: Discover better solutions beyond supervised demonstrations via
group-relative policy optimization with dense spatial rewards.

**Data source**: Online Habitat rollouts using R2R-CE and RxR-CE episodes.

**How it works** (no pre-rendering — runs live):

1. Sample a batch of 32 episode prompts (start position + instruction).
2. For each prompt, generate G=8 rollouts by running the model in Habitat:
   - At each step: render RGB+depth, run forward pass, sample action from logits
   - Record the full trajectory (positions, actions, observations)
3. Score each rollout with 5 reward functions:
   - `R_format`: template match on "Reasoning: ... Action: ..."
   - `R_progress`: geodesic distance reduction per step
   - `R_collision`: penalty for clearance < 0.1m
   - `R_goal`: +10 if geodesic < 1.0m and STOP
   - `R_consistency`: penalty if reasoning contradicts action
4. Compute group advantages, GRPO loss, update.

**Curriculum** (from `configs/train_grpo.yaml`):
- Epochs 1-2: mostly format reward (learn the output structure)
- Epochs 3-4: shift to progress + goal (learn to navigate)
- Epochs 5-6: full spatial reward mix (refine collision avoidance, consistency)

**Episodes needed**: Same R2R-CE / RxR-CE episodes. No new data — the "data"
is generated by the model's own rollouts.

---

## What Needs to Be Implemented

### 1. Frame Renderer Script (HIGH PRIORITY)

**File**: `scripts/render_training_frames.py`

Render and cache RGB + depth frames from Habitat for pre-alignment and SFT.

```python
# Pseudocode
for episode in load_episodes("r2r-ce"):
    env = HabitatEnvWrapper.from_config(config)
    env.reset(episode)
    for step_idx, action in enumerate(episode.trajectory):
        rgb, depth = extract_rgb_depth(env.get_observation())
        intrinsics = get_camera_intrinsics(env)
        save_frame(rgb, depth, intrinsics, episode.instruction, step_idx)
        env.step(action)
```

Output: `data/frames/{dataset}/{split}/ep_{id}_step_{idx}.pt`

### 2. Training Collation (HIGH PRIORITY)

**File**: `src/spatialvlm/data/collation.py`

Converts cached frames + text into model-ready batches.

Must produce:
```python
{
    "siglip_pixels":     Tensor[B, 3, 384, 384],   # resized from 518
    "dinov2_pixels":     Tensor[B, 3, 518, 518],
    "depth":             Tensor[B, 518, 518],       # raw metric depth, NOT normalized
    "intrinsics":        CameraIntrinsics,          # shared per batch (same Habitat sensor)
    "input_ids":         Tensor[B, seq_len],        # tokenized instruction + spatial placeholders
    "spatial_start_idx": int,                       # where the 1369 spatial tokens begin
    "attention_mask":    Tensor[B, seq_len],
    "labels":            Tensor[B, seq_len],        # -100 for non-target tokens
}
```

Key details:
- `depth` must be raw metric metres (backprojection needs real-world units),
  NOT the normalized [0,1] output from `preprocess_rgb_depth()`. The
  `normalize_depth_bhw()` function is for visualization only — the model's
  `encode_geometry()` calls `backproject_depth_map()` which needs raw depth.
- `siglip_pixels` is resized from 518 to 384 (SigLIP's input resolution).
  Apply SigLIP's own normalization (mean/std from the model's preprocessor).
- `dinov2_pixels` stays at 518x518. Apply DINOv2's normalization.
- `input_ids` must contain 1369 placeholder token IDs at `spatial_start_idx`
  where the fused visual tokens will be injected via DeepStack.
- `labels` masks everything except the target response with `ignore_index=-100`.

### 3. Tokenization Utilities (HIGH PRIORITY)

**File**: `src/spatialvlm/data/tokenization.py`

Build `input_ids` from instruction text + spatial token placeholders.

```python
def build_input_ids(
    tokenizer,
    instruction: str,
    target: str | None = None,      # for SFT
    num_spatial_tokens: int = 1369,
    spatial_placeholder_id: int = ...,  # from tokenizer config
) -> dict:
    # Returns: input_ids, labels, attention_mask, spatial_start_idx
```

Must handle:
- System prompt + instruction text tokenization
- Insertion of 1369 spatial placeholder tokens at the correct position
- For SFT: target response tokenization with -100 masking on input tokens
- For pre-alignment: simple captioning format

### 4. Training Entry Point (MEDIUM PRIORITY)

**File**: `scripts/train.py`

Main training loop with DataLoader, optimizer, logging, checkpointing.

```python
# Pseudocode
dataset = CachedFrameDataset("data/frames/r2r-ce/train/")
collator = SpatialVLMCollator(tokenizer, stage="sft")
loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
trainer = SFTTrainer(model, config)

for epoch in range(config.epochs):
    for batch in loader:
        output = trainer.step(batch)
        log(output.loss, output.grad_norm)
    save_checkpoint(model, epoch)
```

### 5. GRPO Rollout Loop (LOWER PRIORITY — after SFT works)

**File**: `scripts/train_grpo.py`

Online RL loop that generates rollouts in Habitat and trains with GRPO.

Requires a working Habitat installation and GPU rendering.

---

## Config Fixes Required Now

### `configs/train_prealign.yaml`
- Change `data.dataset: llava-558k` to `data.dataset: habitat-frames`
- Add `data.frame_dir: data/frames/` path
- Add `data.datasets: [r2r-ce, rxr-ce]`

### `configs/train_sft.yaml`
- Remove `cross_attn` from `training.trainable_keywords` (gated cross-attn removed)
- Add `data.frame_dir: data/frames/` path

---

## Data Acquisition Checklist

### Habitat + Scenes (required for ALL stages)

- [ ] Install Habitat-sim 3.0 + Habitat-lab
- [ ] Download HM3D scenes (~140 train, ~36 val) via Habitat download tool
- [ ] Download Matterport3D scenes (61 train + 11 val, used by R2R/RxR)
  - Requires academic license from Matterport3D website
  - Then convert to Habitat format with `habitat-sim` tools

### Navigation Datasets (JSON episode files)

- [ ] **R2R-CE**: Download from [habitat-lab data](https://github.com/facebookresearch/habitat-lab/tree/main/habitat-baselines/habitat_baselines/rl/ddppo)
  - Files: `R2R_VLNCE_v1-3_{train,val_seen,val_unseen}.json.gz`
  - Contains: episode_id, instruction, start/goal positions, trajectory reference
  - ~21,567 instructions across 90 Matterport3D scenes

- [ ] **RxR-CE**: Download from VLN-CE repository
  - Files: `RxR_VLNCE_{en,hi,te}_{train,val_seen,val_unseen}.json.gz`
  - Contains: same as R2R but multilingual, longer instructions
  - ~126,069 instructions (English subset: ~42K)
  - Use English only for our experiments

- [ ] **SQA3D**: Download from [SQA3D GitHub](https://github.com/SilongYong/SQA3D)
  - Files: `SQA3D_{train,val,test}.json`
  - Contains: question, answer, scene_id, agent position/heading
  - ~33,429 QA pairs across 650 ScanNet scenes
  - ScanNet scenes loadable in Habitat via `habitat-scannet` integration

### Pre-rendering (before training)

- [ ] Run `scripts/render_training_frames.py` for pre-alignment frames (~50K)
- [ ] Run `scripts/render_training_frames.py` for SFT frames (~300K)
- [ ] Verify frame integrity: check shapes, depth range, intrinsics consistency

**Estimated disk space for cached frames**:
- Per frame: ~3MB (RGB float16 518x518x3 + depth float32 518x518 + metadata)
- Pre-alignment: ~50K frames x 3MB = ~150GB
- SFT: ~300K frames x 3MB = ~900GB
- **Total: ~1TB** (or ~400GB with lossy compression/float16 depth)

---

## Execution Order

```
Phase A: Environment Setup
  1. Install Habitat-sim 3.0
  2. Download MP3D scenes (R2R/RxR) + HM3D scenes (ObjectNav)
  3. Download episode JSONs (R2R-CE, RxR-CE, SQA3D)
  4. Verify: load a scene, render RGB+depth at 518x518, check intrinsics

Phase B: Data Pipeline Implementation
  5. Implement scripts/render_training_frames.py
  6. Implement src/spatialvlm/data/collation.py
  7. Implement src/spatialvlm/data/tokenization.py
  8. Test: build one batch, run model.forward(), verify shapes

Phase C: Pre-alignment
  9. Render ~50K pre-alignment frames (R2R-CE + RxR-CE subsets)
  10. Run pre-alignment: 1 epoch, ~77M trainable params
  11. Checkpoint projector weights

Phase D: SFT
  12. Render ~300K SFT frames (all R2R-CE steps + RxR-CE steps + SQA3D)
  13. Run SFT: 3 epochs, ~206M trainable params
  14. Optional: try staged training (gatr+sva first → add dino_proj → all)
  15. Run SVA attention probe on eval batch — check GATr attention mass

Phase E: GRPO
  16. Run GRPO: 6 epochs with curriculum, online Habitat rollouts
  17. Monitor reward curves, check SSR buffer utilization

Phase F: Evaluation + Ablations
  18. Run primary benchmarks (VLN-CE R2R, RxR, SQA3D, VSI-Bench)
  19. Run permutation test (the smoking gun)
  20. Run Phase 9 ablation matrix (15 variants)
```

---

## Compute Budget Estimate

| Stage | GPU Hours (A100-80GB) | Notes |
|-------|-----------------------|-------|
| Frame rendering | ~24h | Parallelizable across scenes |
| Pre-alignment | ~8h | 50K samples, 1 epoch, small trainable set |
| SFT | ~48h | 300K samples, 3 epochs, 206M trainable |
| GRPO | ~120h | 6 epochs, 8 rollouts/prompt, online rendering |
| Ablations (15x) | ~720h | Each ablation = ~48h SFT (some share pre-alignment) |
| Evaluation | ~24h | Benchmark runs on checkpoints |
| **Total** | **~950h** | ~40 A100-days |

---

## Immediate Code Tasks (Before Any Training)

1. **Fix `configs/train_prealign.yaml`** — remove LLaVA-558K reference
2. **Fix `configs/train_sft.yaml`** — remove stale `cross_attn` keyword
3. **Implement `src/spatialvlm/data/collation.py`** — batch builder
4. **Implement `src/spatialvlm/data/tokenization.py`** — input_ids + labels
5. **Implement `scripts/render_training_frames.py`** — Habitat frame caching
6. **Implement `scripts/train.py`** — main training loop
7. **Add `CachedFrameDataset`** to `src/spatialvlm/data/datasets.py`
