#!/usr/bin/env python3
"""Visualization script for SpatialVLM presentation.

Produces figures for slides. Frame selection is persisted to figures/selected_frames.json
so every figure uses the exact same 4 frames.

Usage:
  python scripts/visualize_frames.py
  python scripts/visualize_frames.py --frame-dir /path/to/frames --out figures/
  python scripts/visualize_frames.py --strip-only   # just the RGB/depth strip
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from spatialvlm.geometry.backproject import aggregate_patches_percentile, backproject_depth_map
from spatialvlm.utils.camera import CameraIntrinsics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frame(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def to_rgb_numpy(rgb: torch.Tensor) -> np.ndarray:
    """[3, H, W] float32 → [H, W, 3] uint8"""
    return (rgb.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def depth_to_rgba(depth: torch.Tensor) -> np.ndarray:
    """[H, W] float32 → [H, W, 4] RGBA using 'plasma' colormap. Invalid=transparent."""
    d = depth.numpy()
    valid = d > 0
    normed = np.zeros_like(d)
    if valid.any():
        dmin, dmax = d[valid].min(), d[valid].max()
        normed[valid] = (d[valid] - dmin) / max(dmax - dmin, 1e-6)
    rgba = plt.cm.plasma(normed)
    rgba[~valid, 3] = 0.0
    return (rgba * 255).astype(np.uint8)


def _depth_to_rgba_scaled(depth: torch.Tensor, dmin: float, dmax: float) -> np.ndarray:
    """Like depth_to_rgba but uses a provided shared [dmin, dmax] range."""
    d = depth.numpy()
    valid = d > 0
    normed = np.zeros_like(d)
    normed[valid] = np.clip((d[valid] - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    rgba = plt.cm.plasma(normed)
    rgba[~valid, 3] = 0.0
    return (rgba * 255).astype(np.uint8)


def intrinsics_from_dict(d: dict) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=float(d["fx"]), fy=float(d["fy"]),
        cx=float(d["cx"]), cy=float(d["cy"]),
        width=int(d["width"]), height=int(d["height"]),
    )


# ---------------------------------------------------------------------------
# Figure 1: RGB + Depth grid
# ---------------------------------------------------------------------------

def fig_rgb_depth_grid(frames: list[dict], out: Path | None = None) -> None:
    n = len(frames)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    fig.patch.set_facecolor("#0f0f0f")

    if n == 1:
        axes = [axes]

    for row, frame in enumerate(frames):
        rgb_ax, depth_ax = axes[row]

        rgb_ax.imshow(to_rgb_numpy(frame["rgb"]))
        rgb_ax.set_title("RGB", color="white", fontsize=11)
        rgb_ax.axis("off")

        depth_ax.imshow(depth_to_rgba(frame["depth"]))
        depth_ax.set_title("GT Depth", color="white", fontsize=11)
        depth_ax.axis("off")

        # Instruction label on the left
        instr = frame.get("instruction", "")
        if len(instr) > 80:
            instr = instr[:77] + "…"
        fig.text(
            0.01, 1 - (row + 0.5) / n,
            f"ep {frame.get('episode_id', '?')} step {frame.get('step_idx', '?')}\n{instr}",
            va="center", ha="left", color="#aaaaaa", fontsize=7,
            wrap=True,
        )

    plt.tight_layout(rect=[0.18, 0, 1, 1])
    _save_or_show(fig, out, "rgb_depth_grid.png")


# ---------------------------------------------------------------------------
# Figure 2: 3D point cloud
# ---------------------------------------------------------------------------

def fig_point_cloud(frame: dict, out: Path | None = None, subsample: int = 8) -> None:
    intr = intrinsics_from_dict(frame["intrinsics"])
    depth = frame["depth"].unsqueeze(0)          # [1, H, W]
    point_map = backproject_depth_map(depth, intr)  # [1, H, W, 3]

    pts = point_map[0].reshape(-1, 3).numpy()       # [H*W, 3]
    rgb_flat = frame["rgb"].permute(1, 2, 0).reshape(-1, 3).numpy()  # [H*W, 3]

    # Remove invalid (Z=0) and subsample for speed
    valid = pts[:, 2] > 0
    pts, rgb_flat = pts[valid][::subsample], rgb_flat[valid][::subsample]

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("#0f0f0f")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0f0f0f")

    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1],
               c=rgb_flat.clip(0, 1), s=0.3, linewidths=0)

    ax.set_xlabel("X", color="white"); ax.set_ylabel("Z (depth)", color="white")
    ax.set_zlabel("Y", color="white")
    ax.tick_params(colors="white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    ax.set_title(
        f"Backprojected Point Cloud — ep {frame.get('episode_id', '?')}",
        color="white", fontsize=13,
    )

    instr = frame.get("instruction", "")[:100]
    fig.text(0.5, 0.02, instr, ha="center", color="#888888", fontsize=8)

    _save_or_show(fig, out, "point_cloud.png")


# ---------------------------------------------------------------------------
# Figure 3: 1369 patch centroids (15th-percentile 3D positions)
# ---------------------------------------------------------------------------

def fig_patch_centroids(frame: dict, out: Path | None = None) -> None:
    intr = intrinsics_from_dict(frame["intrinsics"])
    depth = frame["depth"].unsqueeze(0)
    point_map = backproject_depth_map(depth, intr)
    patch_pts = aggregate_patches_percentile(point_map, depth, patch_size=14, percentile=0.15)
    pts = patch_pts[0].numpy()  # [1369, 3]

    # Colour by depth
    z = pts[:, 2]
    z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    colours = plt.cm.plasma(z_norm)

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("#0f0f0f")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0f0f0f")

    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=colours, s=10, linewidths=0)

    ax.set_xlabel("X", color="white"); ax.set_ylabel("Z (depth)", color="white")
    ax.set_zlabel("Y", color="white")
    ax.tick_params(colors="white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    ax.set_title(
        "1369 DINOv2 Patch Centroids (15th-percentile depth)\nInputs to GATr geometry encoder",
        color="white", fontsize=13,
    )

    _save_or_show(fig, out, "patch_centroids.png")


# ---------------------------------------------------------------------------
# Figure 4: Combined anchor frame — RGB, depth, point cloud, patch centroids
# ---------------------------------------------------------------------------

def fig_anchor_combined(
    frame: dict,
    out: Path | None = None,
    subsample: int = 8,
    white_bg: bool = False,
) -> None:
    bg = "white" if white_bg else "#0f0f0f"
    fg = "black" if white_bg else "white"
    muted = "#555555" if white_bg else "#888888"

    intr = intrinsics_from_dict(frame["intrinsics"])
    depth = frame["depth"].unsqueeze(0)
    point_map = backproject_depth_map(depth, intr)
    patch_pts = aggregate_patches_percentile(point_map, depth, patch_size=14, percentile=0.15)

    pts_full = point_map[0].reshape(-1, 3).numpy()
    rgb_flat = frame["rgb"].permute(1, 2, 0).reshape(-1, 3).numpy()
    valid = pts_full[:, 2] > 0
    pts_full, rgb_flat = pts_full[valid][::subsample], rgb_flat[valid][::subsample]

    pts_patch = patch_pts[0].numpy()

    # Shared depth range — derived from the full depth map so both panels are comparable
    d = frame["depth"].numpy()
    dmin = float(d[d > 0].min())
    dmax = float(d[d > 0].max())

    def norm(z: np.ndarray) -> np.ndarray:
        return np.clip((z - dmin) / max(dmax - dmin, 1e-6), 0, 1)

    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor(bg)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.15)

    # Top-left: RGB
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_rgb.imshow(to_rgb_numpy(frame["rgb"]))
    ax_rgb.set_title("RGB", color=fg, fontsize=12)
    ax_rgb.axis("off")

    # Top-right: depth (shared scale)
    ax_depth = fig.add_subplot(gs[0, 1])
    depth_rgba = _depth_to_rgba_scaled(frame["depth"], dmin, dmax)
    im = ax_depth.imshow(depth_rgba)
    ax_depth.set_title("GT Depth", color=fg, fontsize=12)
    ax_depth.axis("off")

    # Bottom-left: point cloud (coloured by depth on shared scale)
    ax_pc = fig.add_subplot(gs[1, 0], projection="3d")
    ax_pc.set_facecolor(bg)
    ax_pc.scatter(pts_full[:, 0], pts_full[:, 2], -pts_full[:, 1],
                  c=rgb_flat.clip(0, 1), s=0.4, linewidths=0)
    ax_pc.set_title("Point Cloud", color=fg, fontsize=12)
    _style_3d_ax(ax_pc, text_color=fg)

    # Bottom-right: patch centroids (shared scale)
    ax_patch = fig.add_subplot(gs[1, 1], projection="3d")
    ax_patch.set_facecolor(bg)
    ax_patch.scatter(pts_patch[:, 0], pts_patch[:, 2], -pts_patch[:, 1],
                     c=plt.cm.plasma(norm(pts_patch[:, 2])), s=12, linewidths=0)
    ax_patch.set_title("1369 Patch Centroids (GATr input)", color=fg, fontsize=12)
    _style_3d_ax(ax_patch, text_color=fg)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap="plasma",
                               norm=plt.Normalize(vmin=dmin, vmax=dmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_depth, ax_patch], orientation="vertical",
                        fraction=0.02, pad=0.02)
    cbar.set_label("Depth (m)", color=fg, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=fg)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=fg, fontsize=7)

    instr = frame.get("instruction", "")[:110]
    fig.text(0.5, 0.01, instr, ha="center", color=muted, fontsize=8)
    fig.suptitle(f"Episode {frame.get('episode_id','?')} · Step {frame.get('step_idx','?')}",
                 color=fg, fontsize=13)

    fname = "anchor_combined_white.png" if white_bg else "anchor_combined.png"
    _save_or_show(fig, out, fname)


def fig_icosa_sphere(out: Path | None = None) -> None:
    """Standalone icosahedral directions sphere (top-left panel of rope_encoding)."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    norm_val = math.sqrt(1.0 + phi**2)
    dirs = np.array([
        [0., +1., +phi],
        [0., +1., -phi],
        [+1., +phi, 0.],
        [+1., -phi, 0.],
        [+phi, 0., +1.],
        [+phi, 0., -1.],
    ]) / norm_val
    DIR_COLORS = np.array([plt.cm.tab10(i / 9.0) for i in range(6)])

    fig = plt.figure(figsize=(7, 7))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    ax.plot_wireframe(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        alpha=0.12, color="#6666aa", linewidth=0.7,
    )

    for i, (d, c) in enumerate(zip(dirs, DIR_COLORS)):
        ax.quiver(0, 0, 0, d[0], d[1], d[2],
                  color=c, length=1.0, arrow_length_ratio=0.18, linewidth=4.0)
        ax.quiver(0, 0, 0, -d[0], -d[1], -d[2],
                  color=c, length=1.0, arrow_length_ratio=0.18, linewidth=1.8, alpha=0.30)
        ax.text(d[0] * 1.28, d[1] * 1.28, d[2] * 1.28,
                f"$\\hat{{d}}_{i+1}$", fontsize=16, color=c[:3], ha="center", va="center",
                fontweight="bold")

    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_zlim(-1.35, 1.35)
    ax.set_xlabel("X", color="black", fontsize=14, labelpad=8)
    ax.set_ylabel("Z", color="black", fontsize=14, labelpad=8)
    ax.set_zlabel("Y", color="black", fontsize=14, labelpad=8)
    ax.tick_params(colors="black", labelsize=10)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#dddddd")
    ax.set_title(
        "6 Icosahedral Directions on $S^2$\n"
        r"$\sum_i \hat{d}_i \hat{d}_i^{\top} = 2I_3$ (isotropic coverage)",
        fontsize=15, color="black",
    )

    _save_or_show(fig, out, "icosa_sphere.png")


def fig_rope_encoding(out: Path | None = None) -> None:
    """Visualise IcosahedralRoPE3D — no frame data required."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    norm_val = math.sqrt(1.0 + phi**2)
    dirs = np.array([
        [0., +1., +phi],
        [0., +1., -phi],
        [+1., +phi, 0.],
        [+1., -phi, 0.],
        [+phi, 0., +1.],
        [+phi, 0., -1.],
    ]) / norm_val  # [6, 3]

    freq_ratio = math.exp(1.0 / 3.0)  # e^(1/3) ≈ 1.3956
    freqs = np.array([10.0 * (freq_ratio**k) for k in range(8)])
    DIR_COLORS = np.array([plt.cm.tab10(i / 9.0) for i in range(6)])

    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.35)

    # ── Panel 1: 3D sphere + icosahedral direction arrows ────────────────────
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.set_facecolor("white")

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    ax1.plot_wireframe(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        alpha=0.10, color="#6666aa", linewidth=0.35,
    )

    for i, (d, c) in enumerate(zip(dirs, DIR_COLORS)):
        ax1.quiver(0, 0, 0, d[0], d[1], d[2],
                   color=c, length=1.0, arrow_length_ratio=0.18, linewidth=2.2)
        ax1.quiver(0, 0, 0, -d[0], -d[1], -d[2],
                   color=c, length=1.0, arrow_length_ratio=0.18, linewidth=0.9, alpha=0.28)
        ax1.text(d[0] * 1.25, d[1] * 1.25, d[2] * 1.25,
                 f"$\\hat{{d}}_{i+1}$", fontsize=8.5, color=c[:3], ha="center", va="center")

    ax1.set_xlim(-1.35, 1.35); ax1.set_ylim(-1.35, 1.35); ax1.set_zlim(-1.35, 1.35)
    _style_3d_ax(ax1, text_color="black")
    for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
        pane.set_edgecolor("#dddddd")
    ax1.set_title(
        "6 Icosahedral Directions on $S^2$\n"
        r"$\sum_i \hat{d}_i \hat{d}_i^{\top} = 2I_3$ (isotropic coverage)",
        fontsize=10, color="black",
    )

    # ── Panel 2: 2D spatial wave pattern ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    x_grid = np.linspace(-3.5, 3.5, 280)
    z_grid = np.linspace(0.3, 6.5, 280)
    X, Z = np.meshgrid(x_grid, z_grid)
    # xz-plane cross-section at y=0; use direction d_0 at frequency f_4
    pts_xz = np.stack([X.ravel(), np.zeros_like(X.ravel()), Z.ravel()], axis=1)
    proj = pts_xz @ dirs[0]  # scalar projection along d_1
    wave = np.sin(proj * freqs[4]).reshape(X.shape)

    im2 = ax2.imshow(wave, extent=[-3.5, 3.5, 0.3, 6.5], origin="lower",
                     cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r"$\sin(\hat{d}_1 \cdot p \cdot f_4)$", fontsize=8)
    cbar2.ax.tick_params(labelsize=7, colors="black")

    # Overlay direction arrow projected into xz-plane
    d0xz = dirs[0][[0, 2]]  # [x, z] components
    d0xz_n = d0xz / (np.linalg.norm(d0xz) + 1e-9) * 1.2
    ax2.annotate("", xy=(d0xz_n[0], d0xz_n[1] + 0.3 + 0.5),
                 xytext=(0, 0.3 + 0.5),
                 arrowprops=dict(arrowstyle="-|>", color=DIR_COLORS[0][:3],
                                 lw=2.0, mutation_scale=14))
    ax2.text(d0xz_n[0] + 0.15, d0xz_n[1] + 0.3 + 0.5 + 0.2,
             r"$\hat{d}_1$ (xz)", fontsize=8, color=DIR_COLORS[0][:3])

    ax2.set_xlabel("$x$ (m)", fontsize=9)
    ax2.set_ylabel("$z$ depth (m)", fontsize=9)
    ax2.set_title(
        f"Spatial wave: $\\sin(\\hat{{d}}_1 \\cdot p \\cdot f_4)$,  "
        f"$f_4 = {freqs[4]:.1f}$ rad/m\n"
        "$xz$-plane cross-section ($y=0$) — stripes ⊥ $\\hat{d}_1$",
        fontsize=10, color="black",
    )
    ax2.tick_params(colors="black")

    # ── Panel 3: e^(1/3) frequency bank ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    bar_colors = plt.cm.plasma(np.linspace(0.15, 0.88, 8))
    ax3.bar(range(8), freqs, color=bar_colors, edgecolor="white", linewidth=0.6, zorder=2)

    # Annotate e^(1/3) ratio between consecutive bars
    for k in range(7):
        y_mid = math.sqrt(freqs[k] * freqs[k + 1])
        ax3.annotate(
            "", xy=(k + 1, freqs[k + 1] * 0.82), xytext=(k, freqs[k] * 1.22),
            arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.9, mutation_scale=8),
        )
    ax3.text(3.5, freqs[5] * 1.05, r"$\times\,e^{1/3} \approx 1.396$",
             ha="center", fontsize=8.5, color="#555555",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.8))

    for k in range(8):
        ax3.text(k, freqs[k] * 1.18, f"{freqs[k]:.0f}",
                 ha="center", va="bottom", fontsize=7, color="#333333")

    ax3.set_yscale("log")
    ax3.set_xticks(range(8))
    ax3.set_xticklabels([f"$f_{k}$" for k in range(8)], fontsize=9)
    ax3.set_xlabel("Frequency index $k$", fontsize=9)
    ax3.set_ylabel("Frequency (rad/m)", fontsize=9)
    ax3.set_title(
        r"Frequency bank: $f_k = 10 \cdot e^{k/3}$"
        f"\nratio $r = e^{{1/3}} \\approx {freq_ratio:.3f}$ — optimal for 3D (economy principle)",
        fontsize=10,
    )
    ax3.tick_params(colors="black")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.yaxis.grid(True, which="both", alpha=0.2, linewidth=0.5)
    ax3.set_axisbelow(True)

    # ── Panel 4: Encoding vector layout (96 active + 32 identity → 128) ─────
    ax4 = fig.add_subplot(gs[1, 1])

    # Draw 96 active dims (6 dirs × 16 each: 8 freqs × sin/cos)
    for d_idx in range(6):
        c = DIR_COLORS[d_idx]
        x_start = d_idx * 16
        for f_idx in range(8):
            # sin dim
            ax4.barh(0, 1, left=x_start + f_idx * 2,
                     color=c[:3], alpha=0.55, height=0.55, linewidth=0)
            # cos dim
            ax4.barh(0, 1, left=x_start + f_idx * 2 + 1,
                     color=c[:3], alpha=0.88, height=0.55, linewidth=0)

    # 32 padding dims
    ax4.barh(0, 32, left=96, color="#bbbbbb", alpha=0.45, height=0.55, linewidth=0,
             label="identity pad (cos=1, sin=0)")

    # Direction group labels
    for d_idx in range(6):
        ax4.text(d_idx * 16 + 8, 0.42, f"$\\hat{{d}}_{d_idx+1}$",
                 ha="center", va="bottom", fontsize=8, color=DIR_COLORS[d_idx][:3],
                 fontweight="bold")

    # Bracket annotations
    ax4.annotate("", xy=(95, -0.45), xytext=(0, -0.45),
                 arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.2))
    ax4.text(48, -0.62, "96 active dims  (6 dirs × 8 freqs × 2)",
             ha="center", va="top", fontsize=8, color="#333333")

    ax4.annotate("", xy=(127, -0.45), xytext=(96, -0.45),
                 arrowprops=dict(arrowstyle="<->", color="#777777", lw=1.2))
    ax4.text(111, -0.62, "+32 pad", ha="center", va="top", fontsize=7.5, color="#666666")

    # Sin/cos legend within first block
    ax4.text(0.3, -0.12, "sin", fontsize=6.5, color="#555555", ha="left", va="top")
    ax4.text(1.3, -0.12, "cos", fontsize=6.5, color="#222222", ha="left", va="top")

    ax4.axvline(96, color="#555555", lw=1.4, ls="--", alpha=0.7)
    ax4.text(96.5, 0.55, "→ identity\n   padding", fontsize=7, color="#666666", va="bottom")

    ax4.set_xlim(0, 128)
    ax4.set_ylim(-0.85, 0.85)
    ax4.set_xticks([0, 16, 32, 48, 64, 80, 96, 112, 128])
    ax4.set_xticklabels(["0", "16", "32", "48", "64", "80", "96", "112", "128"], fontsize=7.5)
    ax4.set_yticks([])
    ax4.set_xlabel("Dimension index", fontsize=9)
    ax4.set_title(
        "Output encoding vector: 96 → 128 dims\n"
        "Injected into Qwen3-VL head\\_dim=128 per spatial token",
        fontsize=10,
    )
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["left"].set_visible(False)
    ax4.tick_params(colors="black")

    fig.suptitle(
        "IcosahedralRoPE3D — 3D Rotary Position Encoding\n"
        r"$p \in \mathbb{R}^3$"
        r"  $\to$  6 dot products  $\to$  $\times$8 freqs  $\to$  sin/cos  $\to$  "
        r"$\mathbb{R}^{96}$  $\to$  +pad  $\to$  $\mathbb{R}^{128}$",
        fontsize=12, fontweight="bold", color="black", y=0.995,
    )

    _save_or_show(fig, out, "rope_encoding.png")


def _style_3d_ax(ax, text_color: str = "white") -> None:
    ax.set_xlabel("X", color=text_color, fontsize=8)
    ax.set_ylabel("Z", color=text_color, fontsize=8)
    ax.set_zlabel("Y", color=text_color, fontsize=8)
    ax.tick_params(colors=text_color, labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False


# ---------------------------------------------------------------------------
# Figure 5: Architecture diagram
# ---------------------------------------------------------------------------

def fig_architecture(out: Path | None = None) -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ── colour palette (white-background pastels) ────────────────────────────
    C = {
        "input":    "#e8eaf6",
        "encoder":  "#e8f5e9",
        "geom":     "#fff3e0",
        "fusion":   "#f3e5f5",
        "backbone": "#e3f2fd",
        "output":   "#ffebee",
        "border":   "#9999bb",
        "arrow":    "#445566",
        "text":     "#111111",
    }

    def box(x, y, w, h, label, color="input", border=None):
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=C[color], edgecolor=border or C["border"], linewidth=1.2,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", color=C["text"],
                fontsize=9, fontweight="bold", zorder=3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                                   lw=1.3, mutation_scale=12),
                    zorder=1)

    # ── inputs ──────────────────────────────────────────────────────────────
    box(2.2, 8.5, 2.6, 0.8, "RGB Frame", "input")
    box(5.8, 8.5, 2.6, 0.8, "GT Depth", "input")
    box(9.4, 8.5, 2.6, 0.8, "Navigation Instruction", "input")

    # ── visual encoders ─────────────────────────────────────────────────────
    box(1.1, 6.8, 2.2, 0.75, "SigLIP2-SO400M", "encoder")
    box(3.3, 6.8, 2.2, 0.75, "DINOv2-L/14",    "encoder")
    box(5.8, 6.8, 2.2, 0.75, "Backproject",    "geom")

    arrow(2.2, 8.1, 1.7, 7.18)
    arrow(2.2, 8.1, 3.1, 7.18)
    arrow(5.8, 8.1, 5.8, 7.18)

    # ── projectors ──────────────────────────────────────────────────────────
    box(1.1, 5.5, 2.2, 0.65, "MLP Projector", "encoder")
    box(3.3, 5.5, 2.2, 0.65, "MLP Projector", "encoder")

    arrow(1.1, 6.42, 1.1, 5.83)
    arrow(3.3, 6.42, 3.3, 5.83)

    # ── geometry ────────────────────────────────────────────────────────────
    box(5.8, 5.5, 2.2, 0.65, "GATr (8 blocks)",    "geom")
    box(8.3, 5.5, 2.2, 0.65, "IcosahedralRoPE3D",  "geom")

    arrow(5.8, 6.42, 5.8, 5.83)
    arrow(6.9, 5.5,  7.2, 5.5)

    # ── fusion (SVA) ────────────────────────────────────────────────────────
    box(4.5, 4.1, 4.8, 0.8, "SVA Cross-Attention", "fusion")
    box(9.8, 4.1, 2.4, 0.8, "RMS Norm Matching",   "fusion")

    arrow(1.1, 5.17, 3.2, 4.5)
    arrow(3.3, 5.17, 3.8, 4.5)
    arrow(5.8, 5.17, 5.2, 4.5)
    arrow(6.9, 4.1,  8.6, 4.1)

    # ── backbone ────────────────────────────────────────────────────────────
    box(5.5, 2.7, 5.8, 0.85, "Qwen3-VL-8B  +  LoRA rank-32", "backbone")

    arrow(4.5, 3.7, 4.8, 3.13)
    arrow(9.8, 3.7, 7.2, 3.13)
    arrow(9.4, 8.1, 8.1, 3.13)

    # ── output ──────────────────────────────────────────────────────────────
    box(5.5, 1.4, 3.6, 0.75, "Navigation Action", "output")

    arrow(5.5, 2.27, 5.5, 1.78)

    # ── training stage badges ───────────────────────────────────────────────
    for bx, by, label, fc, tc in [
        (12.8, 6.0, "Pre-align\n77M params", "#c8e6c9", "#1b5e20"),
        (12.8, 4.8, "SFT\n206M params",      "#bbdefb", "#0d47a1"),
        (12.8, 3.6, "GRPO\ndense rewards",   "#f8bbd0", "#880e4f"),
    ]:
        patch = FancyBboxPatch((bx - 1.0, by - 0.42), 2.0, 0.84,
                               boxstyle="round,pad=0.05",
                               facecolor=fc, edgecolor="#aaaaaa", lw=0.8, zorder=2)
        ax.add_patch(patch)
        ax.text(bx, by, label, ha="center", va="center",
                color=tc, fontsize=7.5, fontweight="bold", zorder=3)

    ax.text(12.8, 7.0, "Training\nStages", ha="center", va="center",
            color="#777788", fontsize=8, style="italic")

    # ── title ────────────────────────────────────────────────────────────────
    ax.text(8.0, 9.6, "SpatialVLM Architecture",
            ha="center", va="center", color="#111111", fontsize=15, fontweight="bold")
    ax.text(8.0, 9.2, "Recovering spatial intelligence in foundation models for indoor navigation",
            ha="center", va="center", color="#555566", fontsize=9)

    _save_or_show(fig, out, "architecture.png")


# ---------------------------------------------------------------------------
# Figure 6: Live loss curve (parses prealign slurm log)
# ---------------------------------------------------------------------------

def fig_loss_curve(log_glob: str = "artifacts/slurm/svlm-prealign-*.out",
                   out: Path | None = None) -> None:
    import glob, re

    logs = sorted(glob.glob(str(REPO / log_glob)))
    if not logs:
        print(f"No logs found matching {log_glob}")
        return

    steps, losses, lrs = [], [], []
    pattern = re.compile(r"step\s+(\d+)/\d+.*loss=([\d.]+).*lr=([\deE.+-]+)")
    for log in logs:
        with open(log) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    steps.append(int(m.group(1)))
                    losses.append(float(m.group(2)))
                    lrs.append(float(m.group(3)))

    if not steps:
        print("No loss entries found in logs.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444444")

    ax1.plot(steps, losses, color="#e05c5c", linewidth=1.2)
    ax1.set_ylabel("Loss", color="white")
    ax1.set_title("Pre-alignment Training (projectors only, ~77M params)", color="white")

    ax2.plot(steps, lrs, color="#5c9de0", linewidth=1.2)
    ax2.set_ylabel("LR", color="white")
    ax2.set_xlabel("Optimizer step", color="white")

    plt.tight_layout()
    _save_or_show(fig, out, "loss_curve.png")


# ---------------------------------------------------------------------------
# Frame selection persistence  (figures/selected_frames.json)
# ---------------------------------------------------------------------------

def save_selected_frames(paths: list[Path], out: Path) -> None:
    import json
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "selected_frames.json", "w") as f:
        json.dump([str(p) for p in paths], f, indent=2)
    print(f"  Frame selection saved: {out / 'selected_frames.json'}")


def load_selected_frames(out: Path) -> list[Path] | None:
    import json
    p = out / "selected_frames.json"
    if not p.exists():
        return None
    with open(p) as f:
        return [Path(s) for s in json.load(f)]


# ---------------------------------------------------------------------------
# Figure: frame strip — 4 RGB frames + depth maps for slides
# ---------------------------------------------------------------------------

def fig_frame_strip(frames: list[dict], out: Path | None = None,
                    white_bg: bool = False) -> None:
    """4 columns × 2 rows: RGB on top, GT Depth below."""
    bg = "white" if white_bg else "#0f0f0f"
    fg = "#333333" if white_bg else "#cccccc"

    n = len(frames)
    fig = plt.figure(figsize=(3.8 * n, 7))
    fig.patch.set_facecolor(bg)
    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.03, wspace=0.04,
                           left=0.06, right=0.99, top=0.97, bottom=0.03)

    for col, frame in enumerate(frames):
        ax_rgb = fig.add_subplot(gs[0, col])
        ax_rgb.imshow(to_rgb_numpy(frame["rgb"]))
        ax_rgb.axis("off")

        ax_dep = fig.add_subplot(gs[1, col])
        ax_dep.imshow(depth_to_rgba(frame["depth"]))
        ax_dep.axis("off")

    fig.text(0.005, 0.74, "RGB", va="center", ha="left",
             fontsize=11, color=fg, fontweight="bold", rotation=90)
    fig.text(0.005, 0.26, "GT Depth", va="center", ha="left",
             fontsize=11, color=fg, fontweight="bold", rotation=90)

    fname = "frame_strip_white.png" if white_bg else "frame_strip.png"
    _save_or_show(fig, out, fname)


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, out: Path | None, name: str) -> None:
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        path = out / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--frame-dir", default="/mnt/home/npant/ceph/datasets/spatialvlm/frames/prealign")
    p.add_argument("--out", default="figures/", help="Output directory (omit to display interactively)")
    p.add_argument("--n-grid", type=int, default=4, help="Frames to show in RGB/depth grid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--white-bg", action="store_true", help="Use white background for anchor_combined")
    p.add_argument("--rope-only", action="store_true", help="Only render rope_encoding.png (no frame data needed)")
    p.add_argument("--sphere-only", action="store_true", help="Only render icosa_sphere.png (no frame data needed)")
    p.add_argument("--strip-only", action="store_true", help="Only render frame_strip.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out) if args.out else None

    if args.rope_only:
        print("Rendering RoPE encoding diagram…")
        fig_rope_encoding(out)
        print("Done.")
        return

    if args.sphere_only:
        print("Rendering icosahedral sphere…")
        fig_icosa_sphere(out)
        print("Done.")
        return

    frame_dir = Path(args.frame_dir)
    all_paths = sorted(frame_dir.rglob("*.pt"))
    if not all_paths:
        sys.exit(f"No .pt files found in {frame_dir}")
    print(f"Found {len(all_paths)} frames in {frame_dir}")

    # Reuse a saved selection so every figure uses the identical 4 frames.
    saved = load_selected_frames(out) if out else None
    if saved and all(p.exists() for p in saved):
        sample_paths = saved
        print(f"  Reusing saved frame selection ({len(sample_paths)} frames)")
    else:
        random.seed(args.seed)
        sample_paths = random.sample(all_paths, min(args.n_grid, len(all_paths)))
        if out:
            save_selected_frames(sample_paths, out)

    frames = [load_frame(p) for p in sample_paths]

    print("Rendering frame strip…")
    fig_frame_strip(frames, out, white_bg=False)
    fig_frame_strip(frames, out, white_bg=True)

    if args.strip_only:
        print("Done.")
        return

    print("Rendering RGB + depth grid…")
    fig_rgb_depth_grid(frames, out)

    anchor = frames[0]
    print("Rendering 3D point cloud…")
    fig_point_cloud(anchor, out)

    print("Rendering patch centroids…")
    fig_patch_centroids(anchor, out)

    print("Rendering combined anchor figure…")
    fig_anchor_combined(anchor, out)
    if args.white_bg:
        print("Rendering combined anchor figure (white background)…")
        fig_anchor_combined(anchor, out, white_bg=True)

    print("Rendering architecture diagram…")
    fig_architecture(out)

    print("Rendering loss curve…")
    fig_loss_curve(out=out)

    print("Rendering RoPE encoding diagram…")
    fig_rope_encoding(out)

    print("Done.")


if __name__ == "__main__":
    main()
