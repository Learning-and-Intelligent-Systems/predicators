"""Diagnostic figure: along-travel vs across-bend 45-degree corner at mu=0.5.

Same L geometry (entry 0.30, exit 0.24, k=3: one entry blue + the pair),
identical positions and gaps - ONLY the first turn block's yaw differs.
State yaw maps to a CCW rotation about z (getQuaternionFromEuler), so a
block's fall (thin) axis is (-sin yaw, cos yaw). The legacy parity
d1_yaw = syaw + t*pi/4 puts that fall axis ALONG the bend's mid-travel
(the natural alignment); the opposite parity syaw - t*pi/4 lays it
ACROSS the bend. Each variant is executed with the real Push at true
friction 0.5; the bottom row shows the SETTLED state (toppled blocks
drawn flat/gray).
"""
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_domino.task_generators import \
    min_block_utils as mbu
from predicators.envs.pybullet_domino.task_generators.min_block_utils import \
    _PROBE_ANCHOR

utils.reset_config({
    'env': 'pybullet_domino', 'seed': 0, 'num_train_tasks': 0,
    'num_test_tasks': 0, 'max_initial_demos': 0, 'horizon': 500,
    'domino_initialize_at_finished_state': False,
    'domino_use_domino_blocks_as_target': True,
    'domino_use_continuous_place': True,
    'domino_has_glued_dominos': False,
    'domino_min_block_tasks': True,
    'domino_true_friction': 0.5, 'domino_planning_friction': 0.1,
    'domino_min_block_span_lo': 0.44, 'domino_min_block_span_hi': 0.65,
    'domino_min_block_num_blues': 5,
    'pybullet_birrt_extend_num_interp': 20,
    'pybullet_birrt_path_subsample_ratio': 2,
})
env = create_new_env('pybullet_domino', do_cache=False, use_gui=False)
comp = env._domino_component  # pylint: disable=protected-access
push_opt = mbu._get_push_option(env)  # pylint: disable=protected-access
doms = comp.dominos
W, D = comp.domino_width, comp.domino_depth
AX, AY = _PROBE_ANCHOR
ENTRY, EXIT = 0.30, 0.24
G_C = 0.12
SYAW = np.pi / 2
T = 1.0
U = np.array([1.0, 0.0])


def build(d1_along_travel):
    d1_dir = SYAW - T * np.pi / 4
    d1_dir_vec = np.array([np.sin(d1_dir), np.cos(d1_dir)])
    d2_rot = SYAW - T * np.pi / 2
    d1_nudge = T * -W / 2 * np.array([np.cos(d1_dir), -np.sin(d1_dir)])
    d2_nudge = T * -W / 2 * np.array([np.cos(d2_rot), -np.sin(d2_rot)])
    t_pt = np.array([AX + ENTRY, AY + EXIT])
    pair_adv = G_C * (1.0 + float(np.dot(d1_dir_vec, U))) + \
        float(np.dot(d1_nudge + d2_nudge, U))
    g_e = ENTRY - pair_adv
    last_pt = np.array([AX, AY]) + g_e * U
    d1_pt = last_pt + G_C * U + d1_nudge
    d2_pt = d1_pt + G_C * d1_dir_vec + d2_nudge
    d1_yaw = SYAW + (T if d1_along_travel else -T) * np.pi / 4
    od = {
        doms[0]: comp.place_domino(0, AX, AY, SYAW, is_start_block=True),
        doms[1]: comp.place_domino(1, float(t_pt[0]), float(t_pt[1]), 0.0,
                                   is_target_block=True),
        doms[2]: comp.place_domino(2, float(last_pt[0]), float(last_pt[1]),
                                   SYAW),
        doms[3]: comp.place_domino(3, float(d1_pt[0]), float(d1_pt[1]),
                                   float(d1_yaw)),
        doms[4]: comp.place_domino(4, float(d2_pt[0]), float(d2_pt[1]),
                                   float(d2_rot)),
    }
    return od


def block(ax, x, y, yaw, color, fallen=False):
    face = "#c9c9c9" if fallen else color
    # State yaw is a CCW z-rotation: long axis (cos, sin), matching the
    # yellow top-triangle in 3D renders.
    tr = (Affine2D().rotate(yaw).translate(x, y) + ax.transData)
    ax.add_patch(
        Rectangle((-W / 2, -D / 2), W, D, facecolor=face,
                  edgecolor="#888" if fallen else "k", lw=0.9, transform=tr,
                  zorder=3))
    if not fallen:
        fx, fy = 0.025 * np.cos(yaw), 0.025 * np.sin(yaw)
        ax.arrow(x, y, fx, fy, head_width=0.008, color="#b8a000", lw=0.9,
                 zorder=4)


COLORS = {0: "#7fc97f", 1: "#c599c5"}


def draw(ax, poses, title, tcolor):
    for i, (x, y, yaw, fallen) in enumerate(poses):
        block(ax, x, y, yaw, COLORS.get(i, "#7fb2d9"), fallen)
    ax.set_title(title, fontsize=10, color=tcolor)
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    ax.set_xlim(min(xs) - 0.09, max(xs) + 0.09)
    ax.set_ylim(min(ys) - 0.09, max(ys) + 0.09)
    ax.set_aspect("equal")
    ax.axis("off")


# The corner cascade is knife-edge and sensitive to the simulator's
# contact history, so a single rollout misrepresents either variant:
# run REPEATED interleaved rollouts per variant, count topples, and
# show each variant's most representative settled state (a toppling
# run for a variant that ever topples, else the last dead run).
N_RUNS = 4
results = {}
for rep in range(N_RUNS):
    for along in (True, False):
        od = build(along)
        ok = mbu._layout_topples(env, od, doms[0], doms[1], push_opt)
        state = env._get_state()  # pylint: disable=protected-access
        poses = []
        for i in range(5):
            d = doms[i]
            fallen = abs(float(state.get(d, "roll"))) > 0.6
            poses.append((float(state.get(d, "x")), float(state.get(d, "y")),
                          float(state.get(d, "yaw")), fallen))
        hits, best = results.get(along, (0, None))
        results[along] = (hits + int(ok), poses if
                          (ok or best is None) else best)
        print(f"rep {rep} along_travel={along}: "
              f"{'TOPPLES' if ok else 'dies'}", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
for col, along in enumerate([True, False]):
    od = build(along)
    init_poses = [(od[doms[i]]["x"], od[doms[i]]["y"], od[doms[i]]["yaw"],
                   False) for i in range(5)]
    kind = ("ALONG-TRAVEL corner (legacy yaw = syaw + t·π/4):\n"
            "fall axis follows the bend" if along else
            "ACROSS-BEND corner (yaw = syaw − t·π/4):\n"
            "fall axis lies across the bend")
    draw(axes[0][col], init_poses, f"{kind}\n(identical positions & gaps)",
         "k")
    hits, poses = results[along]
    ok_any = hits > 0
    verdict = (f"target topples in {hits}/{N_RUNS} runs ✓"
               if ok_any else f"chain dies in ALL {N_RUNS} runs ✗")
    draw(axes[1][col], poses,
         f"settled state after Push @ µ=0.5\n→ {verdict}",
         "#1a7a1a" if ok_any else "#a01515")
fig.suptitle(
    "The corner block's fall axis must follow the bend: same layout, "
    "only the corner yaw parity differs (true friction 0.5)",
    fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = Path(__file__).parent / "corner_yaw_variants_mu05.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out, flush=True)
