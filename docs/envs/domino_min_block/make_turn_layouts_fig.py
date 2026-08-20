"""Slide figure: the turn-K* candidate family (agent-buildable layouts).

Renders the REAL candidates yielded by ``_candidate_turn_layouts`` for a
canonical turn geometry (k=3): the five single-corner configs
(``_CORNER_CONFIGS``, fall axis leaning f of the way into the turn) and
the legacy 45-degree pair corner (sub-family (c), included in the search
since 2026-07-08). The straight-line probe is gated to lines within ~30
degrees of the start's push axis, so for this 36-degree-off geometry it
is (correctly) absent. Footprints are geometry-exact (poses come from
the search code itself); no simulation is run.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_domino.task_generators import \
    min_block_utils as mbu

utils.reset_config({
    "env": "pybullet_domino",
    "seed": 0,
    "domino_use_domino_blocks_as_target": True,
    "domino_true_friction": 0.1,
})
env = create_new_env("pybullet_domino", do_cache=False, use_gui=False)
comp = env._domino_component  # pylint: disable=protected-access
W, D = comp.domino_width, comp.domino_depth

START = (0.55, 1.20, np.pi / 2)  # travel +x
TARGET = (0.85, 1.42, 0.0)  # one left turn away, faces +y


def block(ax, x, y, yaw, color, hl=False):
    tr = (Affine2D().rotate(yaw).translate(x, y) + ax.transData)
    ax.add_patch(
        Rectangle((-W / 2, -D / 2), W, D, facecolor=color, edgecolor="k",
                  lw=1.0, transform=tr, zorder=3))
    fx, fy = 0.03 * np.cos(yaw), 0.03 * np.sin(yaw)
    ax.arrow(x, y, fx, fy, head_width=0.009, color=color, lw=1.0, zorder=4)
    if hl:
        ax.add_patch(
            Circle((x, y), 0.045, fill=False, edgecolor="#d62728", lw=2.0,
                   linestyle="--", zorder=5))


def draw_candidate(ax, od, title, tcolor="k", corner_yaw=None):
    for obj, pose in od.items():
        x, y, yaw = pose["x"], pose["y"], pose["yaw"]
        if obj.name == "domino_0":
            block(ax, x, y, yaw, "#7fc97f")
        elif obj.name == "domino_1":
            block(ax, x, y, yaw, "#c599c5")
        else:
            hl = corner_yaw is not None and abs(yaw - corner_yaw) < 1e-6
            block(ax, x, y, yaw, "#7fb2d9", hl=hl)
    ax.set_title(title, fontsize=10, color=tcolor)
    ax.set_xlim(0.47, 0.95)
    ax.set_ylim(1.10, 1.52)
    ax.set_aspect("equal")
    ax.axis("off")


cands = list(mbu._candidate_turn_layouts(comp, 3, START, TARGET))


def blue_yaws(od, s_obj, t_obj):
    return [float(p["yaw"]) for o, p in od.items() if o not in (s_obj, t_obj)]


# Classify candidates: single-corner configs (k1=0 come first in yield
# order) and the legacy pair (both d1 = syaw + pi/4 and d2 = syaw + pi/2
# present among the blues).
def is_pair(od, s_obj, t_obj):
    ys = {round(y, 3) for y in blue_yaws(od, s_obj, t_obj)}
    return (round(np.pi / 2 + np.pi / 4, 3) in ys
            and round(np.pi, 3) in {round(abs(y), 3) for y in ys})


corner_cands = cands[:len(mbu._CORNER_CONFIGS)]
pair_cand = next((c for c in cands if is_pair(*c)), None)

fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.2))
axes = axes.ravel()

# Panel 0: the straight-line probe is gated out for this geometry.
ax = axes[0]
ax.axis("off")
ax.text(
    0.05, 0.5, "straight-line probe:\nGATED OUT here\n\n(line is 36° off "
    "the start's\npush axis; the probe is only\noffered within ~30° - "
    "beyond\nit the oblique first hit makes\nthe cascade knife-edge)",
    fontsize=10, va="center", color="#a01515")

# Panels 1-5: the single-corner configs (k1=0 candidates lead the yield
# order; label with their (f, g1, g2)).
for i, ((od, _s, _t), cfg) in enumerate(zip(corner_cands,
                                            mbu._CORNER_CONFIGS)):
    f_yaw, g1, g2 = cfg
    draw_candidate(
        axes[1 + i], od,
        f"single corner\nlean {int(f_yaw * 90)}° · in {g1:.2f} · out "
        f"{g2:.2f}", corner_yaw=np.pi / 2 + f_yaw * np.pi / 2)

# Panel 6: the legacy 45-degree pair corner (sub-family (c)).
ax = axes[6]
if pair_cand is not None:
    od, _s, _t = pair_cand
    d1_yaw = np.pi / 2 + np.pi / 4
    draw_candidate(ax, od,
                   "legacy 45° pair corner\n(the pre-min-block "
                   "generator's turn)", corner_yaw=d1_yaw)
else:
    ax.axis("off")
    ax.set_title("pair corner: no candidate\nfor this geometry",
                 fontsize=10, color="#555")

# Panel 7: legend / notes.
ax = axes[7]
ax.axis("off")
ax.text(0.02, 0.85, "k = 3 candidates for one canonical task", fontsize=11,
        weight="bold")
ax.text(
    0.02, 0.12,
    "green = start (pushed)   purple = target\nblue = movable blues; "
    "dashed circle = corner blue\narrow = long axis (the 3D top "
    "triangle);\nthe fall axis is perpendicular to it\n\nhigher k adds "
    "entry blues\n(per-gap ∈ {0.10, 0.13, 0.15} slides the corner)\nand "
    "evenly-spaced exit blues", fontsize=9.5, va="bottom")

fig.tight_layout()
out = Path(__file__).parent / "turn_layouts.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out, f"({len(cands)} candidates at k=3)")
