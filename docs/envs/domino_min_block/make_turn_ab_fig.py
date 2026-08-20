"""Slide figure: the 45-deg turn block's yaw parity is load-bearing.

State yaw is a CCW z-rotation, so a block's fall (thin) axis is
(-sin yaw, cos yaw): d1_yaw = syaw + t*pi/4 leans the corner's fall axis
ALONG the bend's mid-travel (the legacy parity, e.g. pi/2 -> pi/4 -> 0),
while syaw - t*pi/4 lays it ACROSS the bend.

Four schematic panels (footprints + long-axis arrows):
  1. along-travel 45-block (legacy parity)  -> topples (A/B verified)
  2. across-bend 45-block                   -> fails at min-block gaps
  3. search family: stretched single corner (saves a block)
  4. search family: straight-line probe (gated to near-axis lines)
Outcome labels come from the simulated A/B (gaps 0.098-0.13, frictions
0.1/0.5, side offsets {-W/2, 0, +W/2}); this drawing is schematic.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

W, D = 0.07, 0.015
GAP = 0.10


def block(ax, x, y, yaw, color, hl=False):
    tr = (Affine2D().rotate(yaw).translate(x, y) + ax.transData)
    ax.add_patch(
        Rectangle((-W / 2, -D / 2), W, D, facecolor=color, edgecolor="k",
                  lw=1.1, transform=tr, zorder=3))
    fx, fy = 0.045 * np.cos(yaw), 0.045 * np.sin(yaw)
    ax.arrow(x, y, fx, fy, head_width=0.011, color=color, lw=1.1, zorder=4)
    if hl:
        ax.add_patch(
            Circle((x, y), 0.055, fill=False, edgecolor="#d62728", lw=2.2,
                   linestyle="--", zorder=5))


def turn_chain(ax, d1_sign, title, verdict, vcolor, note=None):
    """start -> entry blue -> d1 (45 deg, sign under test) -> d2 -> exit ->
    target."""
    syaw, td = 0.0, 1.0  # travel +y, turning left (exit -x)
    u = np.array([np.sin(syaw), np.cos(syaw)])
    d1_dir = syaw - td * np.pi / 4
    d2_rot = syaw - td * np.pi / 2
    s = np.array([0.0, 0.0])
    block(ax, *s, syaw, "#7fc97f")
    e1 = s + GAP * u
    block(ax, *e1, syaw, "#7fb2d9")
    d1 = e1 + GAP * u + np.array([
        td * -(W / 2) * np.cos(d1_dir), -td * -(W / 2) * np.sin(d1_dir)])
    block(ax, *d1, syaw + d1_sign * td * np.pi / 4, "#7fb2d9", hl=True)
    d2 = d1 + GAP * np.array([np.sin(d1_dir), np.cos(d1_dir)]) + np.array([
        td * -(W / 2) * np.cos(d2_rot), -td * -(W / 2) * np.sin(d2_rot)])
    block(ax, *d2, syaw + td * np.pi / 2, "#7fb2d9")
    e_dir = np.array([np.sin(d2_rot), np.cos(d2_rot)])
    ex = d2 + GAP * e_dir
    block(ax, *ex, d2_rot, "#7fb2d9")
    t = d2 + 2 * GAP * e_dir
    block(ax, *t, d2_rot, "#c599c5")
    ax.set_title(title, fontsize=11)
    ax.text(0.02, -0.13, verdict, fontsize=12, color=vcolor, weight="bold",
            ha="center")
    if note:
        ax.text(0.02, -0.20, note, fontsize=9, color="#555555", ha="center")
    ax.set_xlim(-0.36, 0.18)
    ax.set_ylim(-0.23, 0.36)


fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))

# 1) legacy parity: d1 yaw = syaw + td*45 (fall axis along the bend)
turn_chain(axes[0], +1.0, "Along-travel corner (legacy parity)\n"
           "(fall axis follows the bend)",
           "TOPPLES ✓ (gap ≤ 0.11)", "#1a7a1a")
# 2) opposite parity: d1 yaw = syaw − td*45 (fall axis across the bend)
turn_chain(axes[1], -1.0, "Across-bend corner\n(fall axis lies across "
           "the bend)",
           "FAILS AT MIN-BLOCK GAPS ✗\n(0.098–0.13 · frictions · offsets)",
           "#a01515")

# 3) stretched corner (search family): ONE corner blue leaning into the
# turn, slid toward the start (the agent-buildable corner style; cf. the
# oracle's corner blue), no entry blues needed.
ax = axes[2]
syaw, td = 0.0, 1.0
u = np.array([0.0, 1.0])
d2_rot = -np.pi / 2
s = np.array([0.0, 0.0])
block(ax, *s, syaw, "#7fc97f")
psi = td * 0.5 * np.pi / 2
c_yaw = syaw + psi  # leans halfway into the turn
c_dir = np.array([np.sin(syaw - psi), np.cos(syaw - psi)])  # exit fall dir
c = s + 0.15 * u
block(ax, *c, c_yaw, "#7fb2d9", hl=True)
b1 = c + 0.08 * c_dir
block(ax, *b1, d2_rot, "#7fb2d9")
t = b1 + 0.11 * np.array([np.sin(d2_rot), np.cos(d2_rot)])
block(ax, *t, d2_rot, "#c599c5")
ax.annotate("stretched entry\n(no blue needed)", (0.045, 0.075), fontsize=9,
            color="#b3541e", ha="left")
ax.set_title("Search: stretched single corner\n(slides corner toward start)",
             fontsize=11)
ax.text(-0.09, -0.13, "can SAVE a block vs the even L\n→ K* must search layouts",
        fontsize=10, color="#b3541e", ha="center")
ax.set_xlim(-0.36, 0.18)
ax.set_ylim(-0.23, 0.36)

# 4) straight-line probe (gated to lines within ~30° of the push axis)
ax = axes[3]
s = np.array([0.0, 0.0])
t = np.array([-0.24, 0.24])
d = (t - s) / np.linalg.norm(t - s)
line_yaw = float(np.arctan2(-d[0], d[1]))
block(ax, *s, 0.0, "#7fc97f")
for i in (1, 2):
    p = s + i * np.linalg.norm(t - s) / 3 * d
    block(ax, *p, line_yaw, "#7fb2d9")
block(ax, *t, -np.pi / 2, "#c599c5")
ax.set_title("Search: straight-line probe\n(near-axis targets only)",
             fontsize=11)
ax.text(-0.09, -0.13, "gated to ≤ ~30° off the push axis -\nbeyond that "
        "the oblique first hit\nis contact-history knife-edge",
        fontsize=10, color="#555", ha="center")
ax.set_xlim(-0.36, 0.18)
ax.set_ylim(-0.23, 0.36)

for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")

fig.tight_layout()
out = Path(__file__).parent / "turn_ab.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
