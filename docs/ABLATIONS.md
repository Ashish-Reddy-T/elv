# Ablation Plan

Prerequisite: GRPO production run complete (`grpo_epoch3_*.pt` checkpoint).

---

## Tier 1 — Free (inference only)

### H5a: SFT-only baseline
- **Claim:** GRPO improves over SFT-only
- **How:** Run R2R val_seen eval on `sft_epoch2_step668.pt` (no new training)
- **Command:** Same eval script as final GRPO checkpoint, just swap the checkpoint path

### H3c: Permutation test (spatial arrangement matters)
- **Claim:** SVA attends to spatial structure, not just token content
- **How:** Shuffle all 3314 KV tokens randomly before SVA, run eval on final GRPO checkpoint
- **Code change:** In `fusion/sva.py` `forward()`, add `kv = kv[:, torch.randperm(kv.size(1))]` before cross-attn
- **Expected:** Navigation success rate drops significantly → proves spatial layout is load-bearing

### H3b: Remove RMS norm matching
- **Claim:** Norm matching prevents vision/text token norm mismatch
- **How:** Disable `norm_matching` in `fuse()`, eval final GRPO checkpoint weights
- **Code change:** Skip norm matching call in `fusion/sva.py`
- **Note:** 0 params affected — just an inference-time change

---

## Tier 2 — Cheap (~24h each, resume from SFT)

All three resume from `sft_epoch2_step668.pt` and run GRPO with modified config only.

### H5c: Sparse rewards (goal-only)
- **Claim:** Dense curriculum rewards outperform sparse goal-only reward
- **Config change:** In `configs/train_grpo.yaml`, set all curriculum weights to 0 except `goal: 1.0`
- **Command:**
  ```bash
  sbatch --export=RESUME_CKPT=/mnt/home/npant/ceph/elv/checkpoints/sft_epoch2_step668.pt \
      scripts/hpc/run_grpo.slurm
  ```
  (with modified config)

### H5d: No consistency reward
- **Claim:** Consistency reward prevents reward hacking / repetitive actions
- **Config change:** Set `consistency: 0.0` across all curriculum points in `train_grpo.yaml`
- **Command:** Same as H5c pattern

### H5b: fDPO vs GRPO
- **Claim:** GRPO outperforms offline DPO alternative
- **How:** Use `training/fdpo.py` trainer instead of GRPO, resume from same SFT checkpoint
- **Status:** `training/fdpo.py` may need to be implemented first — check TODO.md

---

## Tier 3 — Full retrain (~48h each from scratch)

Each requires a fresh pre-alignment + SFT run with modified architecture.

### H2a: Remove GATr (no 3D geometry)
- **Claim:** GATr geometric encoding contributes to spatial reasoning
- **How:** Skip GATr in `encode_geometry()`; KV pool = SigLIP + DINOv2 = 1945 tokens
- **Code change:** `model.py` `encode_geometry()` — return zeros or skip; update SVA KV dim
- **Compute:** ~48h pre-align + ~48h SFT + ~24h GRPO = ~5 days

### H1a: Single encoder (SigLIP-only or DINOv2-only)
- **Claim:** Dual encoder fusion beats either encoder alone
- **How:** Skip one encoder in `encode_vision()`, adjust SVA KV accordingly
- **Variants:** Run both SigLIP-only and DINOv2-only for a complete table
- **Compute:** ~96h per variant (two variants = ~8 days total)

### H4a: No spatial position encoding
- **Claim:** IcosahedralRoPE3D improves navigation by encoding 3D direction
- **How:** Disable RoPE monkey-patch for spatial tokens (spatial tokens get sequential positions)
- **Code change:** Skip `stash_spatial_forward_kwargs` / RoPE patch in `backbone/rope_patch.py`
- **Compute:** ~5 days

### H2b: IcosahedralRoPE3D vs standard M-RoPE
- **Claim:** Icosahedral directions outperform standard multi-resolution RoPE for spatial tokens
- **How:** Replace icosahedral directions with standard RoPE, keep patch active
- **Note:** Less fundamental than H4a (no PE at all) — do H4a first

---

## Priority Order

If time is limited, run in this order:

1. **H5a, H3c, H3b** — free, run immediately after GRPO finishes
2. **H5c** — 24h, highest training ROI (validates reward design)
3. **H5d** — 24h, pairs naturally with H5c
4. **H2a** — 5 days, most central architectural claim
5. **H1a** (SigLIP-only only, skip DINOv2-only) — 5 days, justifies dual-encoder complexity
6. **H4a** — 5 days, validates geometric RoPE contribution
7. **H5b, H2b** — lowest priority

---

## Evaluation

All ablations evaluated on **R2R val_seen** split.
Metrics: Success Rate (SR), Oracle SR, SPL, Navigation Error (NE).
Baseline comparison: Qwen3-VL-8B zero-shot (inference only, no training).
