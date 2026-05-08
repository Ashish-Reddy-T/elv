#!/usr/bin/env python3
"""Stage-3 GRPO training with online Habitat rollouts.

Runs under elvvenv (Python 3.12). Spawns habitat_worker.py as a subprocess
under habitat-venv Python for environment interaction.

Usage:
    python scripts/train_grpo.py \\
        --config configs/train_grpo.yaml \\
        --resume checkpoints/sft_epoch2_step668.pt \\
        --scene-dir /mnt/home/npant/ceph/datasets/habitat/habitat-sim/data/scene_datasets/mp3d \\
        --episode-json data/episodes/r2r/train.json.gz \\
        --device cuda
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
import pickle
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from spatialvlm.data.tokenization import SYSTEM_PROMPT
from spatialvlm.data.tokenization import DEFAULT_SPATIAL_PLACEHOLDER

HABITAT_PYTHON = os.environ.get(
    "HABITAT_PYTHON",
    "/mnt/home/npant/ceph/datasets/habitat/habitat-venv/bin/python",
)
WORKER_SCRIPT = str(Path(__file__).parent / "habitat_worker.py")
SCENE_DIR_DEFAULT = "/mnt/home/npant/ceph/datasets/habitat/habitat-sim/data/scene_datasets/mp3d"
MAX_STEPS_PER_ROLLOUT = 25
# 6 min per rollout: ~5× expected (10 steps × 7s/step); leaves 4 min before NCCL barrier times out
ROLLOUT_TIMEOUT_SECS = 360
ACTION_RE = re.compile(r"action\s*:\s*([a-zA-Z_]+)", re.IGNORECASE)


class _RolloutTimeout(Exception):
    pass


def _sigalrm(signum: int, frame: object) -> None:
    raise _RolloutTimeout()
VALID_ACTIONS = {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"}


# ---------------------------------------------------------------------------
# Habitat client (communicates with the worker subprocess)
# ---------------------------------------------------------------------------

class HabitatClient:
    """Thin wrapper around the habitat_worker.py subprocess."""

    def __init__(
        self,
        habitat_python: str = HABITAT_PYTHON,
        worker_script: str = WORKER_SCRIPT,
        log_path: str = "artifacts/slurm/habitat_worker.log",
    ) -> None:
        self._habitat_python = habitat_python
        self._worker_script = worker_script
        self._log_path = log_path
        os.makedirs(Path(log_path).parent, exist_ok=True)
        self._start_worker()

    def _start_worker(self) -> None:
        self._proc = subprocess.Popen(
            [self._habitat_python, self._worker_script, "--log", self._log_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _restart_worker(self) -> None:
        """Kill a stuck worker and start a fresh one to restore IPC sync."""
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
                self._proc.wait(timeout=3)
            except Exception:
                pass
        self._start_worker()

    def _check_alive(self) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"Habitat worker exited unexpectedly (returncode={self._proc.returncode}). "
                f"Check {self._log_path} for details."
            )

    def _send(self, obj: Any) -> None:
        self._check_alive()
        assert self._proc.stdin is not None
        pickle.dump(obj, self._proc.stdin, protocol=4)
        self._proc.stdin.flush()

    def _recv(self, timeout: float = 120.0) -> dict[str, Any]:
        """Read one pickle response with a deadline.

        On timeout the worker is in an unknown IPC state (mid-command), so we
        restart it before raising — subsequent calls will then be in sync.
        """
        import select
        assert self._proc.stdout is not None
        fd = self._proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            self._check_alive()
            self._restart_worker()  # restore IPC sync before raising
            raise TimeoutError(
                f"Habitat worker did not respond within {timeout}s — worker restarted."
            )
        return pickle.load(self._proc.stdout)

    def load_scene(self, scene_path: str) -> None:
        # Large MP3D scenes with navmesh can take 2–5 min to initialise
        self._send({"cmd": "load_scene", "scene_path": scene_path})
        resp = self._recv(timeout=300.0)
        if not resp.get("ok"):
            raise RuntimeError(f"load_scene failed: {resp.get('error')}")

    def reset(
        self,
        start_pos: list[float],
        start_rot: list[float],
        goal_pos: list[float],
    ) -> dict[str, Any]:
        self._send({"cmd": "reset", "start_pos": start_pos,
                    "start_rot": start_rot, "goal_pos": goal_pos})
        resp = self._recv()
        if not resp.get("ok"):
            raise RuntimeError(f"reset failed: {resp.get('error')}")
        return resp

    def step(self, action: str) -> dict[str, Any]:
        self._send({"cmd": "step", "action": action})
        resp = self._recv()
        if not resp.get("ok"):
            raise RuntimeError(f"step failed: {resp.get('error')}")
        return resp

    def close(self) -> None:
        try:
            self._send({"cmd": "quit"})
        except Exception:
            pass
        if self._proc.stdin:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=15)
        except Exception:
            self._proc.terminate()
            self._proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------

def load_r2r_episodes(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open(path, "rt") if path.suffix == ".gz" else open(path)
    with opener as f:
        data = json.load(f)
    episodes = data if isinstance(data, list) else data.get("episodes", data)
    return episodes


def episode_scene_path(episode: dict[str, Any], scene_dir: str) -> str:
    """Resolve R2R-CE scene_id to an absolute .glb path.

    scene_id may be "mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb" (prefix included)
    or a bare name "1LXtFkjw3qL". scene_dir already points inside the dataset
    root (e.g. .../scene_datasets/mp3d), so we strip any leading prefix that
    duplicates it.
    """
    scene_id = episode.get("scene_id", "")
    root = Path(scene_dir)

    # Direct join — works when scene_id has no duplicate prefix
    candidate = root / scene_id
    if candidate.is_file():
        return str(candidate)

    # Strip leading path components one by one until we find a match
    for part in Path(scene_id).parts:
        candidate = root / part
        if candidate.is_dir():
            glbs = sorted(candidate.glob("**/*.glb"))
            if glbs:
                return str(glbs[0])
        if candidate.with_suffix(".glb").exists():
            return str(candidate.with_suffix(".glb"))

    # Last resort: recursive glob by stem name
    stem = Path(scene_id).stem
    matches = sorted(root.glob(f"**/{stem}.glb"))
    if matches:
        return str(matches[0])

    raise FileNotFoundError(
        f"Scene not found for scene_id={scene_id!r} under {scene_dir}. "
        "Check SCENE_DIR in run_grpo.slurm."
    )


# ---------------------------------------------------------------------------
# Tokenization for generation prompts
# ---------------------------------------------------------------------------

def build_generation_prompt(
    tokenizer: Any,
    instruction: str,
    num_spatial_tokens: int = 1369,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build prompt tensor for model.generate().

    Uses the same SYSTEM_PROMPT as SFT training to avoid distribution shift.
    Ends with "\n\nReasoning:" to force the model into the trained format —
    the generated tokens start after this prefix, so response_text must have
    "Reasoning: " prepended before reward/parse evaluation.

    Returns (input_ids, attention_mask, spatial_start_idx).
    """
    placeholder_id: int = tokenizer.get_vocab().get(
        DEFAULT_SPATIAL_PLACEHOLDER,
        tokenizer.unk_token_id or tokenizer.pad_token_id or 0,
    )

    prefix_text = f"{SYSTEM_PROMPT}\n\nInstruction: {instruction}\n\nObservation:"
    prefix_ids: list[int] = tokenizer.encode(prefix_text, add_special_tokens=True)
    spatial_start_idx = len(prefix_ids)

    spatial_ids = [placeholder_id] * num_spatial_tokens
    suffix_ids: list[int] = tokenizer.encode("\n\n", add_special_tokens=False)

    all_ids = prefix_ids + spatial_ids + suffix_ids
    input_ids = torch.tensor([all_ids], dtype=torch.long)
    attn_mask = torch.ones_like(input_ids)
    return input_ids, attn_mask, spatial_start_idx


# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------

def obs_to_model_inputs(
    obs: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert raw numpy observation to model-ready tensors.

    Returns (siglip_pixels, dinov2_pixels, depth) on device.
    """
    import numpy as np

    rgb_np = obs["rgb"].astype(np.float32) / 255.0  # [H, W, 3]
    rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    siglip_px = F.interpolate(rgb, size=(384, 384), mode="bilinear", align_corners=False)
    dinov2_px = F.interpolate(rgb, size=(518, 518), mode="bilinear", align_corners=False)

    depth_np = obs["depth"].astype(np.float32)  # [H, W]
    depth = torch.from_numpy(depth_np).unsqueeze(0)  # [1, H, W]

    return siglip_px.to(device), dinov2_px.to(device), depth.to(device)


# ---------------------------------------------------------------------------
# Action extraction
# ---------------------------------------------------------------------------

_NL_TO_ACTION: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(move|go|walk|step|proceed|continue)\s+(forward|straight|ahead)\b", re.IGNORECASE), "MOVE_FORWARD"),
    (re.compile(r"\bturn\s+left\b|rotate\s+left\b", re.IGNORECASE), "TURN_LEFT"),
    (re.compile(r"\bturn\s+right\b|rotate\s+right\b", re.IGNORECASE), "TURN_RIGHT"),
    (re.compile(r"\bstop\b|\bhalt\b", re.IGNORECASE), "STOP"),
]


def parse_action(response_text: str) -> str:
    # Handle bare token output (e.g. "MOVE_FORWARD" with no prefix)
    bare = response_text.strip().upper()
    if bare in VALID_ACTIONS:
        return bare

    # First try exact discrete token match (e.g. "Action: MOVE_FORWARD")
    m = ACTION_RE.search(response_text)
    if m:
        action = m.group(1).strip().upper()
        if action in VALID_ACTIONS:
            return action

    # Fall back to natural-language mapping (e.g. "Action: Move forward")
    # Search only in the portion after "action:" to avoid false positives in
    # reasoning text that describes past movements.
    action_start = re.search(r"action\s*:", response_text, re.IGNORECASE)
    search_text = response_text[action_start.end():] if action_start else response_text
    for pattern, discrete_action in _NL_TO_ACTION:
        if pattern.search(search_text):
            return discrete_action

    return "STOP"


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_rollout(
    model: Any,
    tokenizer: Any,
    client: HabitatClient,
    episode: dict[str, Any],
    device: torch.device,
    max_steps: int = MAX_STEPS_PER_ROLLOUT,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Run one rollout and return trajectory data for GRPO.

    Returns dict with keys:
        actions: list[str]
        responses: list[str]
        response_ids: list[Tensor]   # token IDs of each response
        old_logprobs: list[Tensor]   # per-token log-probs at generation time
        geodesics: list[float]       # geodesic distance after each step
        clearances: list[float]      # clearance after each step
        stopped: bool
        initial_geodesic: float
    """
    # eval mode so gradient checkpointing is inactive and use_cache=True works
    model.eval()
    from spatialvlm.utils.camera import CameraIntrinsics
    intrinsics = CameraIntrinsics(fx=256.0, fy=256.0, cx=259.0, cy=259.0, width=518, height=518)

    start_pos = episode["start_position"]
    start_rot = episode["start_rotation"]
    ref_path = episode.get("reference_path", [])
    goals = episode.get("goals", [])
    goal_pos = ref_path[-1] if ref_path else (goals[0]["position"] if goals else start_pos)
    instruction = episode.get("instruction", "Navigate to the goal.")
    if isinstance(instruction, dict):
        instruction = instruction.get("instruction_text", "Navigate to the goal.")

    obs = client.reset(start_pos=start_pos, start_rot=start_rot, goal_pos=goal_pos)
    initial_geodesic = obs["geodesic"]

    trajectory: dict[str, Any] = {
        "actions": [],
        "responses": [],
        "response_ids": [],
        "old_logprobs": [],
        "inputs": [],        # (siglip_px, dinov2_px, depth, input_ids, attn_mask, spatial_start_idx) on CPU
        "geodesics": [],
        "clearances": [],
        "stopped": False,
        "initial_geodesic": initial_geodesic,
    }

    for _step in range(max_steps):
        # Build prompt tokens
        siglip_px, dinov2_px, depth = obs_to_model_inputs(obs, device)
        input_ids, attn_mask, spatial_start_idx = build_generation_prompt(
            tokenizer, instruction
        )
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        # Generate response — disable gradient checkpointing so HF honours
        # use_cache=True; without the KV cache the DeepStack embedding hook
        # self-clears after prefill and all decode steps see placeholder tokens.
        # NOTE: PeftModel does not inherit PreTrainedModel, so
        # getattr(peft_model, "is_gradient_checkpointing", False) always returns
        # False even when GC is enabled on the inner model. Always disable
        # unconditionally to avoid that false-negative guard.
        hf_model = model.backbone.model
        hf_model.gradient_checkpointing_disable()
        gen_out = model.generate(
            siglip_pixels=siglip_px,
            dinov2_pixels=dinov2_px,
            depth=depth,
            intrinsics=intrinsics,
            input_ids=input_ids,
            spatial_start_idx=spatial_start_idx,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        hf_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        prompt_len = input_ids.shape[1]
        response_ids = gen_out.sequences[0, prompt_len:]  # [T_resp]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # Compute old_logprobs from stacked scores [T_resp, vocab_size]
        if gen_out.scores:
            scores = torch.stack(gen_out.scores, dim=0)  # [T_resp, 1, vocab_size]
            scores = scores.squeeze(1)  # [T_resp, vocab_size]
            log_probs_all = torch.log_softmax(scores.float(), dim=-1)
            old_logprobs = log_probs_all[
                torch.arange(response_ids.shape[0], device=device), response_ids
            ]  # [T_resp]
        else:
            old_logprobs = torch.zeros(response_ids.shape[0], device=device)

        action = parse_action(response_text)

        # Step environment
        obs = client.step(action)
        geodesic = obs["geodesic"]
        clearance = obs["clearance"]
        done = obs["done"] or action == "STOP"

        trajectory["actions"].append(action)
        trajectory["responses"].append(response_text)
        trajectory["response_ids"].append(response_ids.cpu())
        trajectory["old_logprobs"].append(old_logprobs.cpu())
        trajectory["inputs"].append((
            siglip_px.cpu(), dinov2_px.cpu(), depth.cpu(),
            input_ids.cpu(), attn_mask.cpu(), spatial_start_idx,
        ))
        trajectory["geodesics"].append(geodesic)
        trajectory["clearances"].append(clearance)

        if done:
            trajectory["stopped"] = (action == "STOP")
            break

    return trajectory


# ---------------------------------------------------------------------------
# Reward computation for a full trajectory
# ---------------------------------------------------------------------------

def compute_trajectory_reward(
    trajectory: dict[str, Any],
    reward_config: Any,
    curriculum_weights: Any,
    device: torch.device,
) -> float:
    from spatialvlm.training.rewards import compute_reward_terms, total_reward

    responses = trajectory["responses"]
    actions = trajectory["actions"]
    geodesics = [trajectory["initial_geodesic"]] + trajectory["geodesics"]
    clearances = trajectory["clearances"]
    n = len(responses)
    if n == 0:
        return 0.0

    prev_geo = torch.tensor(geodesics[:n], dtype=torch.float32, device=device)
    curr_geo = torch.tensor(geodesics[1:n+1], dtype=torch.float32, device=device)
    clearance_t = torch.tensor(clearances[:n], dtype=torch.float32, device=device)
    final_geo = torch.full((n,), float(geodesics[-1]), dtype=torch.float32, device=device)
    stopped_t = torch.zeros(n, dtype=torch.float32, device=device)
    if trajectory["stopped"]:
        stopped_t[-1] = 1.0

    terms = compute_reward_terms(
        responses=responses,
        executed_actions=actions,
        previous_geodesic=prev_geo,
        current_geodesic=curr_geo,
        clearance=clearance_t,
        final_geodesic=final_geo,
        stopped=stopped_t,
        config=reward_config,
    )
    step_rewards = total_reward(terms, curriculum_weights)
    return float(step_rewards.sum().item())


# ---------------------------------------------------------------------------
# GRPO update step
# ---------------------------------------------------------------------------

def grpo_update_step(
    model: Any,
    tokenizer: Any,
    grpo_trainer: Any,
    rollouts: list[dict[str, Any]],
    rewards: list[float],
    device: torch.device,
) -> dict[str, float]:
    """Re-forward all rollout steps with gradients, compute GRPO loss, update.

    Uses per-step forward+backward to avoid accumulating all activation graphs
    simultaneously. Also passes logits_to_keep=T_resp+1 so the lm_head only
    computes logits at response positions (~39 MB) instead of the full sequence
    (~516 MB), allowing the forward pass to fit in 40 GB A100 memory.
    """
    from spatialvlm.training.grpo import compute_group_advantages, grpo_loss
    from spatialvlm.utils.camera import CameraIntrinsics
    from torch.nn.utils import clip_grad_norm_
    intrinsics = CameraIntrinsics(fx=256.0, fy=256.0, cx=259.0, cy=259.0, width=518, height=518)

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    advantages = compute_group_advantages(
        reward_tensor.unsqueeze(0),
        normalize=True,
    ).squeeze(0)  # [G]

    # Flatten all (rollout, step, advantage) into a list so we know total_steps
    # upfront for per-step loss normalization.
    all_steps: list[tuple[int, int, float]] = []
    for rollout_idx, rollout in enumerate(rollouts):
        adv_val = float(advantages[rollout_idx].item())
        for step_idx in range(len(rollout["response_ids"])):
            all_steps.append((rollout_idx, step_idx, adv_val))

    if not all_steps:
        return {"total_loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0,
                "clip_fraction": 0.0, "grad_norm": 0.0, "replay_inserted": 0}

    total_steps = len(all_steps)
    model.train()
    grpo_trainer.optimizer.zero_grad(set_to_none=True)

    # Scalar accumulators (detached — no computation graph needed)
    total_loss_acc = 0.0
    policy_loss_acc = 0.0
    kl_loss_acc = 0.0
    clip_fraction_acc = 0.0
    adv_vals: list[float] = []

    for rollout_idx, step_idx, adv_val in all_steps:
        rollout = rollouts[rollout_idx]
        resp_ids = rollout["response_ids"][step_idx].to(device)     # [T_resp]
        old_lp = rollout["old_logprobs"][step_idx].to(device)       # [T_resp]
        siglip_px, dinov2_px, depth_t, input_ids, attn_mask, sp_idx = rollout["inputs"][step_idx]

        # Full sequence: prompt tokens + response tokens
        full_ids = torch.cat(
            [input_ids.to(device), resp_ids.unsqueeze(0)], dim=1
        )  # [1, prompt_len + T_resp]
        full_attn = torch.ones(full_ids.shape, dtype=torch.long, device=device)

        T_resp = resp_ids.shape[0]

        # Re-forward with gradient. logits_to_keep=T_resp+1 tells Qwen3-VL to
        # only compute lm_head for the last T_resp+1 positions (≈39 MB) instead
        # of the full 1700-token sequence (≈516 MB), avoiding OOM on A100-40GB.
        out = model(
            siglip_pixels=siglip_px.to(device),
            dinov2_pixels=dinov2_px.to(device),
            depth=depth_t.to(device),
            intrinsics=intrinsics,
            input_ids=full_ids,
            spatial_start_idx=sp_idx,
            attention_mask=full_attn,
            logits_to_keep=T_resp + 1,
        )
        # With logits_to_keep=T_resp+1, out.logits is [1, T_resp+1, V].
        # Position 0 corresponds to hidden state position prompt_len-1, which
        # predicts resp_ids[0]. Positions 0..T_resp-1 predict resp_ids[0..T_resp-1].
        response_logits = out.logits[0, :T_resp]  # [T_resp, V]
        new_lp = torch.log_softmax(response_logits.float(), dim=-1)[
            torch.arange(T_resp, device=device), resp_ids
        ]  # [T_resp]

        # Per-step GRPO loss using grpo_loss's [N, T] convention with N=1
        adv_t = torch.tensor([adv_val], device=device, dtype=torch.float32)
        loss_out = grpo_loss(
            new_logprobs=new_lp.unsqueeze(0),
            old_logprobs=old_lp.unsqueeze(0),
            ref_logprobs=old_lp.unsqueeze(0),
            advantages=adv_t,
            clip_epsilon=grpo_trainer.config.clip_epsilon,
            kl_beta=grpo_trainer.config.kl_beta,
            entropy_beta=grpo_trainer.config.entropy_beta,
        )

        # Divide by total_steps so accumulated gradients equal the batched mean
        (loss_out.total_loss / total_steps).backward()

        total_loss_acc += float(loss_out.total_loss.detach())
        policy_loss_acc += float(loss_out.policy_loss.detach())
        kl_loss_acc += float(loss_out.kl_loss.detach())
        clip_fraction_acc += float(loss_out.clip_fraction.detach())
        adv_vals.append(adv_val)

        # Explicitly free GPU tensors before the next forward pass
        del out, response_logits, new_lp, loss_out, adv_t

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = clip_grad_norm_(trainable_params, max_norm=grpo_trainer.config.max_grad_norm)
    grpo_trainer.optimizer.step()

    adv_tensor = torch.tensor(adv_vals, dtype=torch.float32)
    replay_inserted = grpo_trainer.replay.add_batch(
        advantages=adv_tensor,
        payloads=[{"idx": i} for i in range(total_steps)],
    )

    return {
        "total_loss": total_loss_acc / total_steps,
        "policy_loss": policy_loss_acc / total_steps,
        "kl_loss": kl_loss_acc / total_steps,
        "clip_fraction": clip_fraction_acc / total_steps,
        "grad_norm": float(grad_norm.detach()),
        "replay_inserted": replay_inserted,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_grpo_checkpoint(
    model: Any,
    optimizer: Any,
    epoch: int,
    global_step: int,
    ckpt_dir: Path,
) -> Path:
    from torch.nn.parallel import DistributedDataParallel as DDP
    raw = model.module if isinstance(model, DDP) else model
    ckpt_path = ckpt_dir / f"grpo_epoch{epoch + 1}_step{global_step}.pt"
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _setup_distributed() -> tuple[int, int, int]:
    """Init NCCL process group if torchrun set LOCAL_RANK; return (rank, world_size, local_rank)."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
        timeout=datetime.timedelta(hours=2),
    )
    return dist.get_rank(), dist.get_world_size(), local_rank


def _sync_model_weights(model: torch.nn.Module) -> None:
    """Average model parameters across all ranks (model averaging at epoch end).

    Per-update gradient allreduce would deadlock when ranks have unequal episode
    counts due to failed scene loads. Epoch-level model averaging is robust,
    gives the same 1/N speedup, and is simpler to reason about.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    for p in model.parameters():
        dist.all_reduce(p.data, op=dist.ReduceOp.AVG)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training with online Habitat rollouts")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default="")
    p.add_argument("--scene-dir", default=SCENE_DIR_DEFAULT)
    p.add_argument("--episode-json", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every-steps", type=int, default=200,
                   help="Save a checkpoint every N global steps (0 = epoch-end only)")
    p.add_argument("--start-epoch", type=int, default=1,
                   help="Curriculum epoch to start from (1-indexed). Use >1 to skip early stages.")
    p.add_argument("--limit", type=int, default=None, help="Limit number of episodes (debug)")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_ROLLOUT)
    p.add_argument("--habitat-python", default=HABITAT_PYTHON)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Distributed setup ──────────────────────────────────────────────────
    rank, world_size, local_rank = _setup_distributed()
    is_main = (rank == 0)

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    # Each rank owns its own GPU
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    ckpt_dir = Path(args.checkpoint_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"[rank {rank}/{world_size}] Starting GRPO training", flush=True)

    # ── Model ──────────────────────────────────────────────────────────────
    if is_main:
        print("Loading model...", flush=True)
    from spatialvlm.config.model import SpatialVLMConfig
    from spatialvlm.model import SpatialVLM
    from transformers import AutoTokenizer

    model_cfg = SpatialVLMConfig()
    model_cfg.backbone.model_id = cfg.model.backbone_model_id
    model_cfg.backbone.lora_rank = cfg.model.lora_rank
    model_cfg.backbone.lora_alpha = cfg.model.lora_alpha

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_model_id)
    model = SpatialVLM(config=model_cfg, device=device, torch_dtype=dtype,
                       lazy_load_backbone=False)
    model.to(dtype)

    hf = model.backbone.model
    try:
        if hasattr(hf, "enable_input_require_grads"):
            hf.enable_input_require_grads()
        if hasattr(hf, "gradient_checkpointing_enable"):
            hf.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if is_main:
                print("Gradient checkpointing enabled.", flush=True)
    except Exception as exc:
        if is_main:
            print(f"Warning: gradient checkpointing not enabled: {exc}", flush=True)

    # ── Resume checkpoint ──────────────────────────────────────────────────
    # All ranks load the same checkpoint — weights are identical at start
    resumed_global_step = 0
    if args.resume:
        if is_main:
            print(f"Loading checkpoint: {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        resumed_global_step = int(ckpt.get("global_step", 0))
        if is_main:
            print(f"Checkpoint loaded (global_step={resumed_global_step}).", flush=True)

    # ── GRPO trainer ───────────────────────────────────────────────────────
    from spatialvlm.training.grpo import GRPOConfig, GRPOTrainer

    grpo_cfg = GRPOConfig(
        group_size=cfg.training.group_size,
        clip_epsilon=cfg.training.clip_epsilon,
        kl_beta=cfg.training.kl_beta,
        entropy_beta=cfg.training.entropy_beta,
        normalize_advantages=cfg.training.normalize_advantages,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        replay_capacity=cfg.training.replay_capacity,
        replay_advantage_threshold=cfg.training.replay_advantage_threshold,
    )
    grpo_trainer = GRPOTrainer(model, grpo_cfg)

    # ── Curriculum ─────────────────────────────────────────────────────────
    from spatialvlm.training.curriculum import CurriculumPoint, RewardCurriculum, RewardWeights
    from spatialvlm.training.rewards import RewardConfig

    curriculum_points = []
    for pt in cfg.curriculum.points:
        w = pt.weights
        curriculum_points.append(CurriculumPoint(
            epoch=pt.epoch,
            weights=RewardWeights(
                format_weight=w.format,
                progress_weight=w.progress,
                collision_weight=w.collision,
                goal_weight=w.goal,
                consistency_weight=w.consistency,
                wrong_stop_weight=getattr(w, "wrong_stop", 0.0),
                proximity_weight=getattr(w, "proximity", 0.0),
            ),
        ))
    curriculum = RewardCurriculum(curriculum_points)
    reward_config = RewardConfig()

    # ── Episodes — split by rank (interleaved so scene variety is even) ────
    all_episodes = load_r2r_episodes(Path(args.episode_json))
    if args.limit:
        all_episodes = all_episodes[:args.limit]
    if is_main:
        print(f"Total episodes loaded: {len(all_episodes)}", flush=True)

    # ── Habitat client — each rank has its own worker process ──────────────
    client = HabitatClient(
        habitat_python=args.habitat_python,
        worker_script=WORKER_SCRIPT,
        log_path=f"artifacts/slurm/habitat_worker_rank{rank}.log",
    )

    # ── Training loop ──────────────────────────────────────────────────────
    group_size = cfg.training.group_size
    epochs = cfg.training.epochs
    batch_size_prompts = cfg.training.batch_size_prompts
    episodes_per_epoch = cfg.training.get("episodes_per_epoch", None)
    max_steps = cfg.training.get("max_steps_per_rollout", args.max_steps)
    start_epoch = max(1, args.start_epoch) - 1  # convert to 0-indexed
    global_step = resumed_global_step

    import random

    try:
        for epoch in range(start_epoch, epochs):
            # Sample a fresh random subset each epoch if episodes_per_epoch is set.
            # Split the subset by rank after sampling so every rank sees different scenes.
            epoch_pool = list(all_episodes)
            random.shuffle(epoch_pool)
            if episodes_per_epoch:
                epoch_pool = epoch_pool[:episodes_per_epoch]
            episodes = epoch_pool[rank::world_size]

            # steps_per_epoch: number of gradient steps one rank will take this epoch
            steps_per_epoch = max(1, len(episodes) // batch_size_prompts)
            epoch_step = 0  # steps taken within this epoch (for fractional interpolation)

            weights = curriculum.get_weights(epoch + 1)
            if is_main:
                print(f"\n── Epoch {epoch + 1}/{epochs} | "
                      f"episodes this epoch: {len(epoch_pool)} "
                      f"(~{len(episodes)} per rank) | "
                      f"format={weights.format_weight:.2f}, "
                      f"progress={weights.progress_weight:.2f}, "
                      f"goal={weights.goal_weight:.2f})", flush=True)

            ep_iter = iter(episodes)
            while True:
                batch_episodes = []
                for _ in range(batch_size_prompts):
                    try:
                        batch_episodes.append(next(ep_iter))
                    except StopIteration:
                        break
                if not batch_episodes:
                    break

                # Interpolate reward weights smoothly across the epoch boundary.
                # fractional_epoch goes from epoch+1 at the start to epoch+2 at the end.
                frac = min(epoch_step / steps_per_epoch, 1.0)
                fractional_epoch = (epoch + 1) + frac
                weights = curriculum.get_weights(fractional_epoch)

                batch_rollouts: list[list[dict]] = []
                batch_rewards: list[list[float]] = []

                for ep in batch_episodes:
                    scene_path = episode_scene_path(ep, args.scene_dir)
                    try:
                        client.load_scene(scene_path)
                    except Exception as exc:
                        print(f"  [rank {rank}][warn] skip episode, scene load failed: {exc}",
                              flush=True)
                        continue

                    ep_rollouts: list[dict] = []
                    ep_rewards: list[float] = []
                    for _ in range(group_size):
                        _old_handler = signal.signal(signal.SIGALRM, _sigalrm)
                        signal.alarm(ROLLOUT_TIMEOUT_SECS)
                        try:
                            rollout = run_rollout(
                                model, tokenizer, client, ep, device,
                                max_steps=max_steps,
                                temperature=cfg.training.rollout_temperature,
                            )
                        except _RolloutTimeout:
                            print(
                                f"  [rank {rank}][warn] rollout timed out after "
                                f"{ROLLOUT_TIMEOUT_SECS}s — skipping, restarting habitat worker",
                                flush=True,
                            )
                            client._restart_worker()
                            continue
                        except Exception as exc:
                            print(f"  [rank {rank}][warn] rollout failed: {exc}", flush=True)
                            continue
                        finally:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, _old_handler)

                        reward = compute_trajectory_reward(
                            rollout, reward_config, weights, device
                        )
                        ep_rollouts.append(rollout)
                        ep_rewards.append(reward)

                    if len(ep_rollouts) >= 2:
                        batch_rollouts.append(ep_rollouts)
                        batch_rewards.append(ep_rewards)

                torch.cuda.empty_cache()

                # Keep NCCL watchdog alive — ranks may spend minutes on rollouts
                # with no collective ops, causing the heartbeat watchdog to fire.
                if dist.is_initialized():
                    _keepalive = torch.zeros(1, device=device)
                    dist.all_reduce(_keepalive, op=dist.ReduceOp.SUM)

                model.train()

                for ep_rollouts, ep_rewards in zip(batch_rollouts, batch_rewards):
                    metrics = grpo_update_step(
                        model, tokenizer, grpo_trainer,
                        ep_rollouts, ep_rewards, device,
                    )
                    global_step += 1
                    epoch_step += 1

                    if is_main and global_step % args.log_every == 0:
                        avg_reward = sum(ep_rewards) / len(ep_rewards)
                        sample_actions = [r["actions"] for r in ep_rollouts]
                        print(
                            f"  step {global_step} | "
                            f"loss={metrics['total_loss']:.4f} | "
                            f"policy={metrics['policy_loss']:.4f} | "
                            f"rewards={[round(r,3) for r in ep_rewards]} | "
                            f"avg={avg_reward:.3f} | "
                            f"actions={sample_actions} | "
                            f"grad_norm={metrics['grad_norm']:.4f}",
                            flush=True,
                        )

                    if (is_main and args.save_every_steps > 0
                            and global_step % args.save_every_steps == 0):
                        ckpt_path = save_grpo_checkpoint(
                            model, grpo_trainer.optimizer, epoch, global_step, ckpt_dir
                        )
                        print(f"  [step {global_step}] Checkpoint saved: {ckpt_path}", flush=True)

            # ── Epoch end: sync model weights across all ranks, then checkpoint
            if dist.is_initialized():
                dist.barrier()          # wait for all ranks to finish epoch
            _sync_model_weights(model)  # average parameters across ranks
            if dist.is_initialized():
                dist.barrier()          # wait for sync to complete on all ranks

            if is_main:
                ckpt_path = save_grpo_checkpoint(
                    model, grpo_trainer.optimizer, epoch, global_step, ckpt_dir
                )
                print(f"  Checkpoint saved: {ckpt_path}", flush=True)

    finally:
        client.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        if is_main:
            print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
