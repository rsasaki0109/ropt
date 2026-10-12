#!/usr/bin/env python3
"""Render a README GIF from actual Rust BA checkpoints, with visual interpolation."""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ropt-font-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ropt-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

BG = "#080f1c"
PANEL = "#101b2e"
GRID = "#253247"
TEXT = "#eaf1fa"
MUTED = "#8295ad"
TEAL = "#55e2c4"
CORAL = "#ff8b77"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": TEXT,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 9, "axes.edgecolor": GRID, "axes.facecolor": PANEL,
    "figure.facecolor": BG, "savefig.facecolor": BG,
})


def world(points):
    """Display world X/Z/Y, placing optical depth across the ground plane."""
    values = np.asarray(points)
    return values[..., [0, 2, 1]]


def interpolate(a, b, t):
    return (1 - t) * np.asarray(a) + t * np.asarray(b)


def panel(fig, bounds):
    fig.patches.append(FancyBboxPatch(
        (bounds[0], bounds[1]), bounds[2], bounds[3],
        boxstyle="round,pad=0.008,rounding_size=0.015", linewidth=0.8,
        edgecolor=GRID, facecolor=PANEL, transform=fig.transFigure, zorder=-1,
    ))


def render(data, phase):
    states = data["states"]
    total = len(states) - 1
    step = min(int(phase), total)
    following = min(step + 1, total)
    t = phase - step if step < total else 0.0
    eased = t * t * (3 - 2 * t)
    current, upcoming = states[step], states[following]
    points = world(interpolate(current["points"], upcoming["points"], eased))
    truth = world(data["truth_points"])
    observed = np.asarray(data["observations"])
    projected = interpolate(current["projected"], upcoming["projected"], eased)
    fig = plt.figure(figsize=(11, 7), dpi=100)
    fig.text(0.04, 0.918, "ropt", fontsize=42, fontweight="bold", color=TEXT)
    fig.text(0.18, 0.943, "Rust-native nonlinear optimization", fontsize=17, fontweight="bold")
    fig.text(0.18, 0.912, "A solver foundation for robotics.", fontsize=11, color=MUTED)
    fig.text(0.955, 0.943, "PRE-ALPHA", ha="right", fontsize=9,
             color=TEAL, fontfamily="DejaVu Sans Mono")
    fig.text(0.04, 0.85, "BUNDLE ADJUSTMENT", color=TEAL, fontsize=10,
             fontweight="bold", fontfamily="DejaVu Sans Mono")
    fig.text(0.265, 0.85, "4 poses  /  12 landmarks  /  48 observations", color=MUTED,
             fontsize=9, fontfamily="DejaVu Sans Mono")
    fig.text(0.955, 0.85, "LM + BLOCK SCHUR", ha="right", fontsize=9,
             color=TEXT, fontfamily="DejaVu Sans Mono")
    panel(fig, (0.04, 0.29, 0.555, 0.51))
    panel(fig, (0.625, 0.29, 0.33, 0.51))
    panel(fig, (0.04, 0.07, 0.915, 0.18))
    fig.text(0.06, 0.768, "01   WORLD GEOMETRY", fontsize=9, color=MUTED,
             fontfamily="DejaVu Sans Mono")
    fig.text(0.645, 0.768, "02   REPROJECTION / CAM 03", fontsize=9, color=MUTED,
             fontfamily="DejaVu Sans Mono")
    ax = fig.add_axes((0.07, 0.335, 0.49, 0.405), projection="3d")
    ax.view_init(elev=22, azim=-64)
    ax.set_xlim(-0.9, 2.3)
    ax.set_ylim(-0.3, 5.6)
    ax.set_zlim(-0.9, 0.9)
    ax.set_box_aspect((3.2, 5.9, 1.8))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANEL)
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"].update(color=GRID, linewidth=0.5)
    ax.tick_params(labelsize=7, pad=0)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 2, 4])
    ax.set_zticks([-0.5, 0, 0.5])
    ax.set_xlabel("X / m", labelpad=1, fontsize=8)
    ax.set_ylabel("Z / depth, m", labelpad=1, fontsize=8)
    ax.set_zlabel("Y / m", labelpad=1, fontsize=8)
    ax.scatter(*truth.T, s=42, facecolors="none", edgecolors=MUTED,
               linewidths=0.7, alpha=0.6, depthshade=False)
    ax.scatter(*points.T, s=24, color=TEXT, edgecolors=CORAL,
               linewidths=0.8, depthshade=False)
    for actual, expected in zip(points, truth):
        ax.plot(*np.array([actual, expected]).T, color=CORAL, linewidth=1.0, alpha=0.8)
    for index, (camera, next_camera) in enumerate(zip(current["cameras"], upcoming["cameras"])):
        center = world(interpolate(camera["center"], next_camera["center"], eased))
        corners = world(interpolate(camera["frustum"], next_camera["frustum"], eased))
        color = TEAL if index in data["fixed_cameras"] else CORAL
        for corner in corners:
            ax.plot(*np.array([center, corner]).T, color=color, alpha=0.8, linewidth=1.0)
        outline = np.concatenate((corners, corners[:1]))
        ax.plot(*outline.T, color=color, linewidth=1.3)
        ax.scatter(*center, s=15, color=color, depthshade=False)
    fig.text(0.065, 0.315, "●  fixed anchors", color=TEAL, fontsize=8)
    fig.text(0.235, 0.315, "●  optimized poses", color=CORAL, fontsize=8)
    fig.text(0.427, 0.315, "○  ground truth", color=MUTED, fontsize=8)
    image_ax = fig.add_axes((0.67, 0.395, 0.25, 0.325))
    image_ax.grid(color=GRID, linewidth=0.6)
    for target, estimate in zip(observed, projected):
        image_ax.plot(*np.array([target, estimate]).T, color=CORAL, alpha=0.9, linewidth=1.2)
    image_ax.scatter(*observed.T, s=36, marker="+", color=TEAL, linewidths=1.1, zorder=3)
    image_ax.scatter(*projected.T, s=20, facecolors="none", edgecolors=TEXT,
                     linewidths=0.9, zorder=4)
    all_pixels = np.vstack((observed, np.asarray(states[0]["projected"])))
    image_ax.set_xlim(all_pixels[:, 0].min()-15, all_pixels[:, 0].max()+15)
    image_ax.set_ylim(all_pixels[:, 1].max()+15, all_pixels[:, 1].min()-15)
    image_ax.set_xlabel("u / px", fontsize=8)
    image_ax.set_ylabel("v / px", fontsize=8)
    image_ax.tick_params(labelsize=7)
    fig.text(0.663, 0.318, "+  observed", color=TEAL, fontsize=8)
    fig.text(0.808, 0.318, "○  estimated", color=TEXT, fontsize=8)
    rms = [state["rms"] for state in states]
    cost_ax = fig.add_axes((0.106, 0.112, 0.56, 0.09))
    cost_ax.set_yscale("log")
    cost_ax.set_xlim(0, total)
    cost_ax.set_ylim(1e-12, max(rms)*2)
    cost_ax.set_yticks([1e0, 1e-4, 1e-8, 1e-12])
    cost_ax.set_xticks(range(0, total+1, 2))
    cost_ax.grid(color=GRID, linewidth=0.6)
    cost_ax.tick_params(labelsize=7)
    cost_ax.plot(range(step+1), rms[:step+1], color=TEAL, linewidth=2,
                 marker="o", markersize=3)
    cost_ax.scatter([step], [rms[step]], color=TEAL, s=40, edgecolors=TEXT,
                    linewidths=0.7, zorder=5)
    cost_ax.set_xlabel("LM iteration", fontsize=8, labelpad=1)
    fig.text(0.06, 0.215, "03   CONVERGENCE", fontsize=9, color=MUTED,
             fontfamily="DejaVu Sans Mono")
    label = "INITIAL STATE" if step == 0 else f"ITERATION {step:02d} / {total:02d}"
    if step == total:
        label = "CONVERGED"
    fig.text(0.722, 0.209, label, fontsize=9, color=TEAL,
             fontfamily="DejaVu Sans Mono")
    fig.text(0.722, 0.151, f"{rms[step]:.2e}", fontsize=25, fontweight="bold",
             fontfamily="DejaVu Sans Mono")
    fig.text(0.725, 0.113, "reprojection RMS / pixels", fontsize=8, color=MUTED)
    fig.text(0.04, 0.027, "Synthetic, noiseless BA · 2 known camera poses fixed", fontsize=8, color=MUTED)
    fig.text(0.955, 0.027, "Interpolated playback · metrics from solver checkpoints",
             ha="right", fontsize=8, color=MUTED)
    fig.canvas.draw()
    frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    data = json.loads(args.trace.read_text())
    if data.get("schema") != 1 or not data.get("synthetic"):
        raise ValueError("Expected a synthetic schema-1 Rust solver trace")
    states = data["states"]
    if len(states) < 2 or states[-1]["rms"] >= 1e-7:
        raise ValueError("Trace has no verified convergence")
    phases = [0.0]*10
    for step in range(len(states)-1):
        phases.extend(step+substep/6 for substep in range(1, 7))
    phases.extend([float(len(states)-1)]*15)
    first = render(data, 0.0)
    palette = first.quantize(colors=192, dither=Image.Dither.NONE)
    frames = [first.quantize(palette=palette, dither=Image.Dither.NONE)]
    for phase in phases[1:]:
        frames.append(render(data, phase).quantize(palette=palette, dither=Image.Dither.NONE))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=80, loop=0, optimize=True, disposal=2)
    if args.preview:
        preview = Image.new("RGB", (1100, 2100), BG)
        for index, phase in enumerate([0.0, 2.0, float(len(states)-1)]):
            preview.paste(render(data, phase), (0, 700*index))
        preview.save(args.preview)
    print(f"Wrote {args.output}: {len(frames)} playback frames, {args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
