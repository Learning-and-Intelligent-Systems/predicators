"""ONE derivation, both views: 3D snapshot + top-down of legacy task 10 and the
generated turn task's calibrated layout.

The winning layout is derived ONCE and reused for both renders (the two
earlier figures each re-derived it, which knife-edge rollouts can make
inconsistent). The top-down panels use STANDARD world axes (+x right, +y
up), so they read exactly like the state values. The 3D camera is the
same far-side (+y) view as the user's GUI screenshots, so its screen
orientation is rotated 180 degrees relative to the top-downs - compare
layouts column-by-column, not screen-direction-by-direction. Prints
every block's yaw for ground truth.
"""
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_domino.task_generators import \
    min_block_utils as mbu

utils.reset_config({
    'env': 'pybullet_domino', 'seed': 0, 'num_train_tasks': 1,
    'num_test_tasks': 5, 'test_env_seed_offset': 10000,
    'max_initial_demos': 0, 'horizon': 500,
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
Z = comp.z_lb + comp.domino_height / 2
CID = env._physics_client_id  # pylint: disable=protected-access

LEGACY = [
    (0.437499, 1.33754, 1.5708, "start"),
    (0.535499, 1.33754, 1.5708, "blue"),
    (0.633499, 1.33754, 1.5708, "blue"),
    (0.706751, 1.31279, 0.785398, "target"),
    (0.741047, 1.2435, 0.0, "target"),
]

# Derive the turn task's calibrated layout ONCE.
tasks = env.get_test_tasks()
turn_task = None
for task in tasks:
    st = task.init
    ds = st.get_objects(comp.domino_type)
    s_ = next(d for d in ds if comp._StartBlock_holds(st, [d]))
    t_ = next(d for d in ds if comp._TargetDomino_holds(st, [d]))
    dyaw = float(st.get(s_, "yaw") - st.get(t_, "yaw"))
    if abs((dyaw + np.pi) % (2 * np.pi) - np.pi) > np.pi / 4:
        turn_task = task
        break
assert turn_task is not None
st = turn_task.init
ds = st.get_objects(comp.domino_type)
s_ = next(d for d in ds if comp._StartBlock_holds(st, [d]))
t_ = next(d for d in ds if comp._TargetDomino_holds(st, [d]))
s_pose = tuple(float(st.get(s_, f)) for f in ("x", "y", "yaw"))
t_pose = tuple(float(st.get(t_, f)) for f in ("x", "y", "yaw"))
# The figure compares the LEGACY PAIR construction against legacy task
# 10, so search specifically for the pair signature (d1 = syaw + t*pi/4
# and d2 = syaw + t*pi/2 among the blues; since the 2026-07-09
# yaw-parity fix the K*-optimal build at this cell is the cheaper
# stretched single corner, so the pair is not the first winner).
syaw_ = s_pose[2]
u_ = np.array([np.sin(syaw_), np.cos(syaw_)])
w_ = np.array([t_pose[0] - s_pose[0], t_pose[1] - s_pose[1]])
t_dir_ = 1.0 if float(u_[0] * w_[1] - u_[1] * w_[0]) > 0 else -1.0
pair_sig = {
    round(float(syaw_ + t_dir_ * np.pi / 4), 3),
    round(float(syaw_ + t_dir_ * np.pi / 2), 3),
}
win = None
for k in range(6):
    for od, a_, b_ in mbu._candidate_turn_layouts(comp, k, s_pose, t_pose):
        blue_yaws = {
            round(float(pz["yaw"]), 3)
            for obj, pz in od.items() if obj not in (a_, b_)
        }
        if not pair_sig <= blue_yaws:
            continue
        if mbu._layout_topples(env, od, a_, b_, push_opt):
            win = od
            break
    if win is not None:
        break
assert win is not None
OURS = []
for obj, pz in win.items():
    role = ("start" if obj is s_ else "target" if obj is t_ else "blue")
    OURS.append((float(pz["x"]), float(pz["y"]), float(pz["yaw"]), role))
for x, y, yaw, role in OURS:
    print(f"OURS {role}: x={x:.3f} y={y:.3f} yaw={yaw / np.pi:.3f}pi",
          flush=True)


def assemble(poses):
    od = {}
    for i, (x, y, yaw, role) in enumerate(poses):
        od[doms[i]] = comp.place_domino(i, x, y, yaw,
                                        is_start_block=role == "start",
                                        is_target_block=role == "target")
    target = next(doms[i] for i, pz in enumerate(poses)
                  if pz[3] == "target")
    state = mbu._assembled_state(env, comp, od, target, Z,
                                 comp.target_domino_color)
    env._set_state(state)  # pylint: disable=protected-access


def snapshot():
    # Same far-side camera as the user's GUI screenshots. NOTE: it looks
    # from the +y side, so its screen orientation is 180 degrees off the
    # standard-axes top-downs below (left-right and near-far flipped).
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.66, 1.30, 0.44], distance=1.05,
        yaw=-125, pitch=-38, roll=0, upAxisIndex=2, physicsClientId=CID)
    proj = p.computeProjectionMatrixFOV(fov=55, aspect=4 / 3, nearVal=0.1,
                                        farVal=4.0)
    _, _, rgb, _, _ = p.getCameraImage(1200, 900, view, proj,
                                       renderer=p.ER_TINY_RENDERER,
                                       physicsClientId=CID)
    return np.reshape(rgb, (900, 1200, 4))[:, :, :3].astype(np.uint8)


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


def draw_top(ax, poses, title):
    xs, ys = [], []
    for x, y, yaw, role in poses:
        # State yaw is applied via getQuaternionFromEuler([0,0,yaw]), a CCW
        # rotation about z: the long (width) axis maps to (cos, sin) and
        # the thin (fall) axis to (-sin, cos). The arrow is the long axis,
        # matching the yellow top-triangle in the 3D render.
        tr = (Affine2D().rotate(yaw).translate(x, y) + ax.transData)
        ax.add_patch(
            Rectangle((-W / 2, -D / 2), W, D, facecolor=COLORS[role],
                      edgecolor="k", lw=0.9, transform=tr, zorder=3))
        ax.arrow(x, y, 0.02 * np.cos(yaw), 0.02 * np.sin(yaw),
                 head_width=0.007, color="#b8a000", lw=0.9, zorder=4)
        ax.annotate(yaw_label(yaw), (x, y), textcoords="offset points",
                    xytext=(-24, 12), fontsize=9, color="#333")
        xs.append(x)
        ys.append(y)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(min(xs) - 0.08, max(xs) + 0.10)
    ax.set_ylim(min(ys) - 0.08, max(ys) + 0.10)
    ax.set_aspect("equal")
    ax.axis("off")


assemble(LEGACY)
img_legacy = snapshot()
assemble(OURS)
img_ours = snapshot()

fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.4))
axes[0][0].imshow(img_legacy)
axes[0][0].set_title("legacy env.py task 10 - 3D", fontsize=10)
axes[0][1].imshow(img_ours)
axes[0][1].set_title(
    "generated turn task, legacy pair-corner build\n"
    "(from the K* family) - 3D", fontsize=10)
for ax in (axes[0][0], axes[0][1]):
    ax.axis("off")
draw_top(axes[1][0], LEGACY,
         "same scene, top-down (standard world axes: +x right, +y up)")
draw_top(axes[1][1], OURS,
         "same scene, top-down (standard world axes: +x right, +y up)")
fig.suptitle(
    "One derivation, two views. The 3D camera looks from the far (+y) side, "
    "so its screen orientation is rotated 180 deg from the top-downs",
    fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = Path(__file__).parent / "legacy_3d_comparison.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out, flush=True)
