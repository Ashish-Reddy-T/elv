# SpatialVLM Runbook

How to go from a fresh clone to training on NYU HPC Cloud Bursting.

---

## 1. HPC Access

- **OOD Portal**: https://ood-burst-001.hpc.nyu.edu/
- **VPN**: Must be on [NYU VPN](https://www.nyu.edu/it/vpn) if off-campus
- **Slurm Account**: `ds_ga_3001_003-2026sp`
- **Budget**: 300 GPU hours (18,000 minutes) per student

### Partitions

| Partition | GPU | Use |
|-----------|-----|-----|
| `c12m85-a100-1` | 1x A100 40GB | Training (prealign, SFT) |
| `g2-standard-12` | 1x L4 | Light testing, eval |
| `n1s8-t4-1` | 1x T4 | Smoke tests |

**Use A100 for training.** The full model needs ~38GB VRAM at bf16 with batch_size=4.

---

## 2. One-time Setup

### Step 1: Create overlay filesystem

Open a terminal via OOD (Interactive Apps > Jupyter > Terminal):

```bash
cd /scratch/$USER
cp /share/apps/overlay-fs-ext3/overlay-50G-10M.ext3.gz .
gunzip overlay-50G-10M.ext3.gz
mv overlay-50G-10M.ext3 spatialvlm_env.ext3
```

### Step 2: Clone repo

```bash
cd /scratch/$USER
git clone <repo-url> ablations
cd ablations
```

### Step 3: Install environment

Launch Singularity in **read-write** mode:

```bash
singularity exec --nv \
  --overlay /scratch/$USER/spatialvlm_env.ext3:rw \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash
```

Inside the container:

```bash
cd /scratch/$USER/ablations
bash scripts/hpc/setup_env.sh
```

This installs Miniconda, PyTorch (CUDA 12.1), SpatialVLM, and GATr. Takes ~5 min.

### Step 4: wandb login

Still inside the container:

```bash
source /ext3/env.sh
wandb login   # paste your API key
```

---

## 3. Quick Smoke Test

Verify everything works on GPU before spending budget on real training:

```bash
sbatch scripts/hpc/run_smoke_test.slurm
```

Check output:
```bash
cat artifacts/slurm/svlm-smoke-*.out
```

This runs the mock forward pass + a 2-step training run on synthetic data.
Should complete in <10 min and costs ~0.15 GPU-hours.

---

## 4. Data Setup

### Option A: Synthetic frames (pipeline testing)

Generate fake frames directly on HPC:

```bash
singularity exec --nv \
  --overlay /scratch/$USER/spatialvlm_env.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/env.sh
    cd /scratch/$USER/ablations
    python scripts/generate_synthetic_frames.py --output data/frames/synthetic/ --num-frames 256
  "
```

### Option B: Real Habitat frames

Pre-render on a workstation with Habitat installed, then transfer:

```bash
# On workstation:
tar -czf prealign_frames.tar.gz data/frames/prealign/

# Transfer to HPC:
scp prealign_frames.tar.gz <netid>@dtn.torch.hpc.nyu.edu:/scratch/<netid>/ablations/
# (requires NYU MFA / Duo push)

# On HPC, unpack:
cd /scratch/$USER/ablations
tar -xzf prealign_frames.tar.gz
```

Each frame is a `.pt` dict: `rgb` [3,518,518], `depth` [518,518], `intrinsics` dict,
`instruction` str, `target` str.

---

## 5. Training

### Stage 1: Pre-alignment (projectors only, ~77M trainable)

```bash
sbatch scripts/hpc/run_prealign.slurm
```

Default: A100, batch_size=8, bf16, ~4h time limit.
Override batch size or frame dir:

```bash
sbatch --export=BATCH_SIZE=4,FRAME_DIR=data/frames/synthetic/ \
    scripts/hpc/run_prealign.slurm
```

### Stage 2: SFT (~206M trainable)

Start from the prealign checkpoint:

```bash
sbatch --export=RESUME_CKPT=checkpoints/prealign_epoch1_step500.pt \
    scripts/hpc/run_sft.slurm
```

Default: A100, batch_size=4, grad_accum=2 (effective=8), bf16, ~8h.

### Staged training schedule (optional)

Train geometry modules first to prevent SigLIP dominance:

```yaml
# configs/train_sft.yaml -- uncomment trainable_groups:
training:
  trainable_groups: [gatr, sva]        # Phase 1: geometry + fusion only
```

Run each phase as a separate job with `--resume`:

```bash
# Phase 1: GATr + SVA (2 epochs)
sbatch --export="RESUME_CKPT=checkpoints/prealign_epoch1_step500.pt" \
    scripts/hpc/run_sft.slurm

# Phase 2: add DINOv2 projector (1 epoch)
# Edit trainable_groups: [gatr, sva, dino_proj] in config, then:
sbatch --export="RESUME_CKPT=checkpoints/sft_epoch2_step1000.pt" \
    scripts/hpc/run_sft.slurm

# Phase 3: everything (1 epoch)
# Set trainable_groups: null in config, then:
sbatch --export="RESUME_CKPT=checkpoints/sft_epoch3_step1500.pt" \
    scripts/hpc/run_sft.slurm
```

### Stage 3: GRPO (online RL, requires Habitat)

```bash
# Not yet implemented -- needs live Habitat environment.
# See configs/train_grpo.yaml for the config.
# Implementation: scripts/train_grpo.py (TODO)
```

---

## 6. Spot Preemption & Resuming

All cloud bursting nodes are **GCP Spot instances** -- they can be killed anytime.
The training script handles this:

1. **`--requeue`** in Slurm: job is re-queued automatically on preemption
2. **SIGTERM handler**: saves an emergency checkpoint when the 30s warning arrives
3. **`--resume`**: picks up from the last checkpoint

After preemption, find your latest checkpoint:
```bash
ls -lt checkpoints/ | head -5
# Look for *_preempted.pt files
```

Resume:
```bash
sbatch --export=RESUME_CKPT=checkpoints/sft_epoch2_step800_preempted.pt \
    scripts/hpc/run_sft.slurm
```

---

## 7. Monitoring with wandb

Dashboard: `https://wandb.ai/<your-entity>/spatialvlm`

Key metrics:

| Metric | Healthy range | Red flag |
|--------|--------------|----------|
| `train/loss` | Decreasing | Flat or NaN |
| `train/grad_norm` | 0.1 -- 10 | >100 (exploding) |
| `train/lr` | Warmup then cosine decay | Flat (scheduler broken) |
| `system/vram_gb` | <38 (A100-40GB) | OOM crash |

---

## 8. Job Management

```bash
# Check job status
squeue -u $USER

# Cancel a job
scancel <job_id>

# View output of running job
tail -f artifacts/slurm/svlm-sft-<jobid>.out

# Check GPU budget usage
# (ask TAs for the command or check OOD portal)
```

---

## 9. GPU Budget Estimation

| Stage | Batch | Frames | Est. A100-hours |
|-------|-------|--------|-----------------|
| Smoke test | 2 | 8 | 0.15 |
| Prealign (synthetic, testing) | 8 | 256 | ~1 |
| Prealign (real, 50K frames) | 8 | 50K | ~8 |
| SFT (real, 300K frames, 3 epochs) | 4 | 300K | ~48 |

**Budget is 300 hours.** Prealign + SFT = ~56 hours. Leaves ~244 hours
for ablations, eval, and debugging.

---

## 10. File Reference

| File | Purpose |
|------|---------|
| `scripts/hpc/setup_env.sh` | One-time environment bootstrap |
| `scripts/hpc/run_smoke_test.slurm` | GPU smoke test (<10 min) |
| `scripts/hpc/run_prealign.slurm` | Stage 1 training job |
| `scripts/hpc/run_sft.slurm` | Stage 2 training job |
| `scripts/train.py` | Main training loop |
| `scripts/generate_synthetic_frames.py` | Fake frames for testing |
| `scripts/smoke_test.py` | Forward-pass verification |
| `configs/train_prealign.yaml` | Pre-alignment config |
| `configs/train_sft.yaml` | SFT config |

---

## 11. Remaining TODOs

- [ ] `scripts/render_training_frames.py` -- Habitat frame caching
- [ ] `scripts/train_grpo.py` -- GRPO online RL loop (Stage 3)
- [ ] Test with real Habitat-rendered frames
- [ ] Tune batch_size / grad_accum for actual VRAM usage on A100
