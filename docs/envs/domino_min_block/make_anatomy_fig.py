"""Schematic: min-block task anatomy — init, calibrated plan, miscalibrated
plan. Pure drawing (numbers match the measured reaches)."""
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

W, D = 0.07, 0.015  # domino width (perp to chain) x depth (along chain)
SPAN = 0.26
TRUE_REACH, BELIEVED_REACH = 0.11, 0.16


def block(ax, x, y, color, label=None):
    ax.add_patch(
        Rectangle((x - D / 2, y - W / 2), D, W, facecolor=color,
                  edgecolor="k", lw=1.2, zorder=3))
    if label:
        ax.text(x, y - W / 2 - 0.03, label, ha="center", va="top", fontsize=9)


def gap_arrow(ax, x0, x1, y, txt, color):
    ax.add_patch(
        FancyArrowPatch((x0 + D / 2, y), (x1 - D / 2, y), arrowstyle="<->",
                        color=color, mutation_scale=10, lw=1.4, zorder=2))
    ax.text((x0 + x1) / 2, y + 0.045, txt, ha="center", fontsize=9,
            color=color)


fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))

# ── Panel 1: the task ────────────────────────────────────────────
ax = axes[0]
block(ax, 0.0, 0.0, "#7fc97f", "start\n(push me)")
block(ax, SPAN, 0.0, "#c599c5", "target\n(topple me)")
gap_arrow(ax, 0.0, SPAN, 0.0, f"span {SPAN} m", "k")
for i, bx in enumerate((0.04, 0.10, 0.16, 0.22)):
    block(ax, bx, -0.16, "#7fb2d9")
ax.text(0.13, -0.235, "4 staged blues (place as few as possible)",
        ha="center", fontsize=9, color="#33658a")
ax.set_title("Task: topple target with ≤ K* blues", fontsize=11)

# ── Panel 2: calibrated plan (true reach 0.11 → K*=2) ────────────
ax = axes[1]
block(ax, 0.0, 0.0, "#7fc97f")
g = SPAN / 3
for i in (1, 2):
    block(ax, i * g, 0.0, "#7fb2d9")
block(ax, SPAN, 0.0, "#c599c5")
gap_arrow(ax, 0.0, g, 0.0, f"{g:.3f} ≤ 0.11 ✓", "#1a7a1a")
ax.text(SPAN / 2, -0.19, "true reach 0.11 → K* = 2 blues\ntopples ✓  uses 2 ≤ K* ✓  REWARD",
        ha="center", fontsize=10, color="#1a7a1a")
ax.set_title("Calibrated model (friction 0.1)", fontsize=11)

# ── Panel 3: miscalibrated plan (believed reach 0.16 → 1 blue) ───
ax = axes[2]
block(ax, 0.0, 0.0, "#7fc97f")
block(ax, SPAN / 2, 0.0, "#7fb2d9")
block(ax, SPAN, 0.0, "#c599c5")
gap_arrow(ax, 0.0, SPAN / 2, 0.0, f"{SPAN/2:.2f} ≤ 0.16 ?", "#b3541e")
ax.text(SPAN * 0.75, 0.09, "✗ chain dies\n(0.13 > 0.11 real reach)",
        ha="center", fontsize=10, color="#a01515")
ax.text(SPAN / 2, -0.19,
        "believed reach 0.16 → plans 1 blue\nvalidates in ITS sim, fails in real",
        ha="center", fontsize=10, color="#a01515")
ax.set_title("Miscalibrated model (believes 0.5)", fontsize=11)

for ax in axes:
    ax.set_xlim(-0.09, 0.36)
    ax.set_ylim(-0.28, 0.17)
    ax.set_aspect("equal")
    ax.axis("off")

fig.tight_layout()
out = str(Path(__file__).parent / "task_anatomy.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
