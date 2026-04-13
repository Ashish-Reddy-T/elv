# SpatialVLM Runbook

How to go from a fresh clone to training on Colab Pro.

---

## 1. Setup (Colab or local)

```bash
# Clone
git clone <repo-url> ablations && cd ablations

# Install (takes ~3 min on Colab)
pip install -e .
pip install -e REPOS/geometric-algebra-transformer --no-deps

# Verify
python scripts/smoke_test.py --mock --grad-check
# All stages should PASS
```

**Colab GPU check** — you need A100 40GB or better:
```python
python -c "import torch; print(torch.cuda.get_device_name(0), f\"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")"
```

**wandb login** (first time only):
```bash
pip install wandb
wandb login   # paste your API key
```

---

## 2. Quick pipeline test (synthetic data, ~5 min)

This verifies the entire training pipeline works without downloading
real data or Habitat scenes.

```bash
# Generate 64 synthetic frames (~200 MB)
python scripts/generate_synthetic_frames.py --output data/frames/synthetic/ --num-frames 64

# Test pre-alignment (4 steps with mock-sized batch)
python scripts/train.py \
    --config configs/train_prealign.yaml \
    --frame-dir data/frames/synthetic/ \
    --device cuda --dtype bf16 \
    --limit 32 --log-every 1 \
    --wandb-mode disabled

# Test SFT
python scripts/train.py \
    --config configs/train_sft.yaml \
    --frame-dir data/frames/synthetic/ \
    --device cuda --dtype bf16 \
    --limit 32 --log-every 1 \
    --wandb-mode disabled
```

If these run without errors, the pipeline is working.

---

## 3. Real data setup

### 3a. Get Habitat + scenes

Habitat-sim doesn't install easily on Colab. Options:
- **Recommended**: Pre-render frames on a workstation or HPC cluster,
  upload `.pt` files to Google Drive
- **Alternative**: Use a Colab instance with conda for Habitat install

```bash
# On a workstation with Habitat installed:
conda install habitat-sim -c conda-forge -c aihabitat

# Download MP3D scenes (requires academic license)
python -m habitat_sim.utils.datasets_download --uids mp3d --data-path data/scene_datasets/

# Download episode JSONs
# R2R-CE: github.com/jacobkrantz/VLN-CE
# RxR-CE: github.com/jacobkrantz/VLN-CE (multilingual)
# SQA3D: github.com/SilongYong/SQA3D
```

### 3b. Render training frames

```bash
# Pre-alignment: ~50K frames from R2R-CE + RxR-CE start positions
python scripts/render_training_frames.py \
    --dataset r2r-ce --split train \
    --output data/frames/prealign/ \
    --max-frames-per-episode 3

# SFT: ~300K frames from full trajectories
python scripts/render_training_frames.py \
    --dataset r2r-ce rxr-ce --split train \
    --output data/frames/sft/ \
    --all-steps
```

> **Note**: `scripts/render_training_frames.py` is not yet implemented.
> Until then, use the Habitat env wrapper (`src/spatialvlm/data/habitat_env.py`)
> to write a custom rendering script. Each frame must be saved as a `.pt` dict
> with keys: `rgb` [3,518,518], `depth` [518,518], `intrinsics` dict,
> `instruction` str, `target` str (for SFT).

### 3c. Upload to Google Drive

```bash
# Zip and upload
tar -czf prealign_frames.tar.gz data/frames/prealign/
# Upload to Google Drive, then on Colab:
# from google.colab import drive; drive.mount('/content/drive')
# !tar -xzf /content/drive/MyDrive/prealign_frames.tar.gz
```

---

## 4. Training stages

### Stage 1: Pre-alignment (projectors only, ~77M trainable)

**Goal**: Align SigLIP/DINOv2/GATr projectors to LLM embedding space.

```bash
python scripts/train.py \
    --config configs/train_prealign.yaml \
    --frame-dir data/frames/prealign/ \
    --device cuda --dtype bf16 \
    --wandb-project spatialvlm \
    --save-every-epoch
```

**Config** (`configs/train_prealign.yaml`):
- 1 epoch, LR=1e-4, batch_size=64 (reduce to 8 on 40GB GPU)
- Only `projector` params are trainable
- ~8 A100-hours for 50K frames

**What to watch in wandb**:
- `train/loss` should decrease steadily
- `train/grad_norm` should be <10 (if >100, something is wrong)

### Stage 2: SFT (~206M trainable)

**Goal**: Full spatial instruction-following.

```bash
python scripts/train.py \
    --config configs/train_sft.yaml \
    --frame-dir data/frames/sft/ \
    --device cuda --dtype bf16 \
    --wandb-project spatialvlm \
    --save-every-epoch \
    --resume checkpoints/prealign_epoch1.pt
```

**Config** (`configs/train_sft.yaml`):
- 3 epochs, LR=5e-5, batch_size=32 (reduce to 4 on 40GB GPU)
- Trainable: projectors + GATr blocks + SVA + LoRA
- ~48 A100-hours for 300K frames

### Staged training schedule (optional)

The professor suggested training geometry modules first, then adding
others. This prevents SigLIP (native to Qwen3-VL) from dominating.

Edit `configs/train_sft.yaml` to uncomment `trainable_groups`:

```yaml
# Phase 1: geometry + fusion only (2 epochs)
training:
  trainable_groups: [gatr, sva]

# Phase 2: add DINOv2 projector (1 epoch)
training:
  trainable_groups: [gatr, sva, dino_proj]

# Phase 3: everything (1 epoch)
training:
  trainable_groups: null  # all keywords active
```

Run each phase as a separate `train.py` invocation with `--resume`.

### Stage 3: GRPO (online RL, requires Habitat)

```bash
# Not yet runnable on Colab — needs live Habitat environment.
# See configs/train_grpo.yaml for the config.
# Implementation: scripts/train_grpo.py (TODO)
```

---

## 5. Monitoring with wandb

Key metrics to track:

| Metric | Healthy range | Red flag |
|--------|--------------|----------|
| `train/loss` | Decreasing | Flat or NaN |
| `train/grad_norm` | 0.1 — 10 | >100 (exploding) |
| `system/vram_gb` | <38 (A100-40) | OOM crash |
| `train/lr` | Per config | Unexpected value |

wandb dashboard: `https://wandb.ai/<your-entity>/spatialvlm`

---

## 6. Evaluation (after SFT)

```bash
# Run SVA attention probe — check if GATr is being used
python scripts/smoke_test.py --device cuda --dtype bf16 --stages 3

# Full eval on benchmarks (requires Habitat for VLN-CE)
# See configs/eval.yaml
```

---

## 7. File reference

| File | Purpose |
|------|---------|
| `scripts/smoke_test.py` | Forward-pass verification (mock or real) |
| `scripts/generate_synthetic_frames.py` | Fake frames for pipeline testing |
| `scripts/train.py` | Main training loop (prealign + SFT) |
| `scripts/setup_wandb.py` | wandb project initialization |
| `src/spatialvlm/data/tokenization.py` | input_ids + labels builder |
| `src/spatialvlm/data/collation.py` | Batch collation for DataLoader |
| `src/spatialvlm/data/datasets.py` | CachedFrameDataset loader |
| `configs/train_prealign.yaml` | Pre-alignment config |
| `configs/train_sft.yaml` | SFT config |
| `configs/train_grpo.yaml` | GRPO config (online RL) |

---

## 8. Remaining TODOs

- [ ] `scripts/render_training_frames.py` — Habitat frame caching
- [ ] `scripts/train_grpo.py` — GRPO online RL loop
- [ ] Test with real Habitat-rendered frames
- [ ] Tune batch_size / grad_accum for target GPU
- [ ] Add LR scheduler (cosine with warmup)
