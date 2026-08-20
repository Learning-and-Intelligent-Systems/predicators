"""Render the PRE-min-block generator's chains (finished state).

Shows exactly how it orients turn blocks - settling whether the legacy
turn used the mirrored yaw or a natural one.
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

utils.reset_config({
    'env': 'pybullet_domino', 'seed': 0, 'num_train_tasks': 0,
    'num_test_tasks': 6,
    'max_initial_demos': 0, 'horizon': 500,
    # The OLD standard-chain config (envs/all.yaml `domino` block), with
    # chains INITIALIZED FINISHED so the full layout incl. turns shows.
    'domino_initialize_at_finished_state': True,
    'domino_use_domino_blocks_as_target': True,
    'domino_use_continuous_place': True,
    'domino_has_glued_dominos': False,
    'domino_min_block_tasks': False,
    'domino_test_num_dominos': [7],
    'domino_test_num_targets': [1],
    'pybullet_birrt_extend_num_interp': 20,
    'pybullet_birrt_path_subsample_ratio': 2,
})
env = create_new_env('pybullet_domino', do_cache=False, use_gui=False)
comp = env._domino_component  # pylint: disable=protected-access
W, D = comp.domino_width, comp.domino_depth

tasks = env.get_test_tasks()
fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 4))
for ti, (task, ax) in enumerate(zip(tasks, np.atleast_1d(axes))):
    state = task.init
    xs, ys = [], []
    for d in state.get_objects(comp.domino_type):
        # pylint: disable=protected-access
        x, y, yaw = (float(state.get(d, f)) for f in ("x", "y", "yaw"))
        role = ("start" if comp._StartBlock_holds(state, [d]) else
                "target" if comp._TargetDomino_holds(state, [d]) else "blue")
        color = {"start": "#7fc97f", "target": "#c599c5",
                 "blue": "#7fb2d9"}[role]
        tr = (Affine2D().rotate(yaw).translate(x, y) + ax.transData)
        ax.add_patch(
            Rectangle((-W / 2, -D / 2), W, D, facecolor=color,
                      edgecolor="k", lw=0.9, transform=tr, zorder=3))
        ax.arrow(x, y, 0.025 * np.cos(yaw), 0.025 * np.sin(yaw),
                 head_width=0.008, color=color, lw=0.9, zorder=4)
        xs.append(x)
        ys.append(y)
    ax.set_title(f"legacy task {ti} (finished chain)", fontsize=10)
    ax.set_xlim(min(xs) - 0.09, max(xs) + 0.09)
    ax.set_ylim(min(ys) - 0.09, max(ys) + 0.09)
    ax.set_aspect("equal")
    ax.axis("off")
out = Path(__file__).parent / "legacy_tasks.png"
fig.tight_layout()
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out, flush=True)
