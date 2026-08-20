"""Four-panel top-down comparison answering: 'with this rotation, the sequence
should turn downward instead of upward'.

A. legacy env.py task 10, verbatim (travel +x, bends DOWN).
B. the same task rotated 180 degrees (travel -x, bends UP) - rotating
   adds pi to every yaw: pi/2,pi/4,0 -> -pi/2,-3pi/4,-pi(=pi).
C. the K* family layout used in task_examples_high_friction task 0
   (travel -x, target ABOVE -> bends UP): same motif as B.
D. the K* family layout for the SAME entry but target BELOW: it bends
   DOWN with yaws -pi/2 -> -pi/4 -> 0 - the 'downward' sequence; the
   family picks the turn side from where the target is, like the
   legacy generator's random turn_direction.
"""
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

W, D = 0.07, 0.015

LEGACY = [
    (0.437499, 1.33754, 1.5708, "start"),
    (0.535499, 1.33754, 1.5708, "blue"),
    (0.633499, 1.33754, 1.5708, "blue"),
    (0.706751, 1.31279, 0.785398, "blue"),
    (0.741047, 1.2435, 0.0, "target"),
]
cx = float(np.mean([p[0] for p in LEGACY]))
cy = float(np.mean([p[1] for p in LEGACY]))
ROT180 = [(2 * cx - x, 2 * cy - y, yaw + np.pi, role)
          for x, y, yaw, role in LEGACY]


def family_layout(syaw, t_dir, g_c=0.098, g_e=0.098, k1=2):
    """The (c)-family pair-corner transform (legacy-verbatim)."""
    u = np.array([np.sin(syaw), np.cos(syaw)])
    d1_dir = syaw - t_dir * np.pi / 4
    d1_dir_vec = np.array([np.sin(d1_dir), np.cos(d1_dir)])
    d2_rot = syaw - t_dir * np.pi / 2
    d1_nudge = t_dir * -W / 2 * np.array([np.cos(d1_dir), -np.sin(d1_dir)])
    d2_nudge = t_dir * -W / 2 * np.array([np.cos(d2_rot), -np.sin(d2_rot)])
    s = np.array([0.0, 0.0])
    poses = [(0.0, 0.0, syaw, "start")]
    for i in range(k1):
        p = s + (i + 1) * g_e * u
        poses.append((float(p[0]), float(p[1]), syaw, "blue"))
    last = s + k1 * g_e * u
    d1 = last + g_c * u + d1_nudge
    d2 = d1 + g_c * d1_dir_vec + d2_nudge
    poses.append((float(d1[0]), float(d1[1]), syaw + t_dir * np.pi / 4,
                  "blue"))
    poses.append((float(d2[0]), float(d2[1]), syaw + t_dir * np.pi / 2,
                  "target"))
    return poses


COLORS = {"start": "#7fc97f", "target": "#c599c5", "blue": "#7fb2d9"}


def yaw_label(yaw):
    yaw = float((yaw + np.pi) % (2 * np.pi) - np.pi)
    frac = yaw / np.pi
    names = {0.5: "π/2", 0.25: "π/4", 0.0: "0", -0.25: "-π/4",
             -0.5: "-π/2", -0.75: "-3π/4", 0.75: "3π/4", 1.0: "π",
             -1.0: "π"}
    for num, name in names.items():
        if abs(frac - num) < 5e-3:
            return name
    return f"{frac:.2f}π"


def draw(ax, poses, title, tcolor="k"):
    xs, ys = [], []
    for x, y, yaw, role in poses:
        # State yaw is a CCW z-rotation: long axis (cos, sin); arrow
        # matches the yellow top-triangle in 3D renders.
        tr = (Affine2D().rotate(yaw).translate(x, y) + ax.transData)
        ax.add_patch(
            Rectangle((-W / 2, -D / 2), W, D, facecolor=COLORS[role],
                      edgecolor="k", lw=0.9, transform=tr, zorder=3))
        ax.arrow(x, y, 0.02 * np.cos(yaw), 0.02 * np.sin(yaw),
                 head_width=0.007, color="#b8a000", lw=0.9, zorder=4)
        ax.annotate(yaw_label(yaw), (x, y), textcoords="offset points",
                    xytext=(12, 12), fontsize=8.5, color="#333")
        xs.append(x)
        ys.append(y)
    ax.set_title(title, fontsize=9.5, color=tcolor)
    ax.set_xlim(min(xs) - 0.08, max(xs) + 0.10)
    ax.set_ylim(min(ys) - 0.08, max(ys) + 0.10)
    ax.set_aspect("equal")
    ax.axis("off")


fig, axes = plt.subplots(2, 2, figsize=(11, 8))
draw(axes[0][0], LEGACY,
     "A. legacy env.py task 10 (verbatim)\n"
     "travel +x, bends DOWN · yaws π/2 → π/4 → 0")
draw(axes[0][1], ROT180,
     "B. task 10 rotated 180° (every yaw + π)\n"
     "travel -x, bends UP · yaws -π/2 → -3π/4 → π")
draw(axes[1][0], family_layout(-np.pi / 2, -1.0),
     "C. K* family, travel -x, target ABOVE (up-bend)\n"
     "bends UP · yaws -π/2 → -3π/4 → π   (= B)")
draw(axes[1][1], family_layout(-np.pi / 2, 1.0),
     "D. K* family, travel -x, target BELOW\n"
     "bends DOWN · yaws -π/2 → -π/4 → 0")
fig.suptitle(
    "Turn side follows the target's side; the yaw sequence mirrors with "
    "it (legacy turn_direction semantics)", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = Path(__file__).parent / "legacy_yaw_comparison.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out, flush=True)
