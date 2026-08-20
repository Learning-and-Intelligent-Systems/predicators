"""Slide figure: heavy-block tasks - believed plan vs calibrated solution.

For each cached heavy-block test task, three rows (same style as
make_task_examples_fig.py):
  1. the staged initial state (blues parked; the GRAY 1000 kg block with
     natural alignment: dead ahead on the line for straight tasks, at
     the L's natural corner for turn tasks);
  2. the calibrated solution at the TRUE physics (gray heavy),
     re-verified by simulation: the half-circle swerve around the gray
     (straight variant) or the skip-around detour with an own corner
     (turn variant);
  3. the miscalibrated plan: the cheapest layout THROUGH the gray that
     the BELIEVED physics accepts (normal gray mass - the same families
     the generation certificates scan), then EXECUTED at the true
     physics: it dies against the gray block.

Everything is produced by the real task-gen machinery: cached tasks are
reloaded through the env, layouts come from the search code, outcomes
from sim rollouts with the real Push. The config matches the experiment
launch flags (common.yaml + envs/all.yaml domino_heavy) and the default
cache dir, so running this script also pre-warms the launch cache.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_domino.task_generators import \
    min_block_utils as mbu
from predicators.envs.pybullet_domino.task_generators.min_block_generation import \
    _believed_physics
from predicators.settings import CFG

utils.reset_config({
    'env': 'pybullet_domino',
    'seed': 0,
    # common.yaml
    'num_train_tasks': 1,
    'num_test_tasks': 5,
    'skill_phase_use_motion_planning': True,
    'pybullet_ik_validate': False,
    'pybullet_camera_height': 900,
    'pybullet_camera_width': 900,
    # envs/all.yaml domino_heavy (mass-only mismatch: no planning friction,
    # true friction stays at the settings default, 0.5 - matching launches)
    'max_initial_demos': 0,
    'excluded_objects_in_state_str': "loc,rot,angle,direction",
    'horizon': 500,
    'domino_initialize_at_finished_state': False,
    'domino_use_domino_blocks_as_target': True,
    'domino_use_continuous_place': True,
    'process_planning_heuristic_weight': 2.0,
    'domino_has_glued_dominos': False,
    'keep_failed_demos': True,
    'predicate_invent_invent_derived_predicates': True,
    'pybullet_birrt_extend_num_interp': 20,
    'pybullet_birrt_path_subsample_ratio': 2,
    'domino_heavy_block_tasks': True,
    'domino_min_block_num_blues': 4,
})
env = create_new_env('pybullet_domino', do_cache=False, use_gui=False)
comp = env._domino_component  # pylint: disable=protected-access
push_opt = mbu._get_push_option(env)
W, D = comp.domino_width, comp.domino_depth
doms = comp.dominos

COLORS = {
    "start": "#7fc97f",
    "target": "#c599c5",
    "blue": "#7fb2d9",
    "heavy": "#5a5a5a",
}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def block(ax, x, y, yaw, color):
    tr = (Affine2D().rotate(yaw).translate(x, y) + ax.transData)
    ax.add_patch(
        Rectangle((-W / 2, -D / 2),
                  W,
                  D,
                  facecolor=color,
                  edgecolor="k",
                  lw=0.9,
                  transform=tr,
                  zorder=3))
    fx, fy = 0.025 * np.cos(yaw), 0.025 * np.sin(yaw)
    ax.arrow(x, y, fx, fy, head_width=0.008, color=color, lw=0.9, zorder=4)


def draw_state(ax, poses, title, tcolor="k"):
    for x, y, yaw, role in poses:
        block(ax, x, y, yaw, COLORS[role])
    ax.set_title(title, fontsize=9.5, color=tcolor)
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    ax.set_xlim(min(xs) - 0.09, max(xs) + 0.09)
    ax.set_ylim(min(ys) - 0.09, max(ys) + 0.09)
    ax.set_aspect("equal")
    ax.axis("off")


def role_of(state, d):
    # pylint: disable=protected-access
    if comp._StartBlock_holds(state, [d]):
        return "start"
    if comp._TargetDomino_holds(state, [d]):
        return "target"
    if comp._HeavyBlock_holds(state, [d]):
        return "heavy"
    return "blue"


def state_poses(state):
    return [(state.get(d, "x"), state.get(d, "y"), state.get(d, "yaw"),
             role_of(state, d)) for d in (state.get_objects(comp.domino_type) +
                                          state.get_objects(comp.block_type))]


def od_poses(od, start, target, gray):
    out = []
    for obj, pose in od.items():
        # Compare by equality, not identity: the env is re-created after
        # loading the tasks, so layouts may be keyed by either the cached
        # tasks' objects or the fresh env's (equal by name).
        role = ("start" if obj == start else "target"
                if obj == target else "heavy" if obj == gray else "blue")
        out.append((pose["x"], pose["y"], pose["yaw"], role))
    return out


def task_geometry(state):
    # pylint: disable=protected-access
    dominoes = (state.get_objects(comp.domino_type) +
                state.get_objects(comp.block_type))
    start = next(d for d in dominoes if comp._StartBlock_holds(state, [d]))
    target = next(d for d in dominoes if comp._TargetDomino_holds(state, [d]))
    gray = next(d for d in dominoes if comp._HeavyBlock_holds(state, [d]))
    pose = lambda d: (state.get(d, "x"), state.get(d, "y"), state.get(
        d, "yaw"))
    return start, target, gray, pose(start), pose(target), pose(gray)


def swerve_solution(s_pose, t_pose, h_pose, k_max):
    """Winning half-circle swerve at the TRUE physics."""
    # pylint: disable=protected-access
    for k in range(2, k_max + 1):
        for od in mbu._candidate_swerve_layouts(comp, k, s_pose, t_pose,
                                                h_pose):
            if mbu._layout_topples(env, od, doms[0], doms[1], push_opt):
                return od, k
    return None


def detour_solution(s_pose, t_pose, gray_od, k_max):
    """Winning NOISE-ROBUST skip-around detour at the TRUE physics - the
    same bar certification holds tasks to (a nominal-only scan would show
    knife-edge cheaper detours the arm cannot actually build, visually
    faking a cost tie with the believed lure)."""
    # pylint: disable=protected-access
    extra_pts = [(d["x"], d["y"]) for d in gray_od.values()]
    h_pose = next((d["x"], d["y"], d["yaw"]) for d in gray_od.values())
    cand_idx = 0
    for k in range(k_max + 1):
        merged = []
        for od, s_, t_ in mbu._candidate_turn_layouts(comp, k, s_pose, t_pose):
            blue_pts = [(d["x"], d["y"]) for o, d in od.items()
                        if o not in (s_, t_)]
            if any(
                    np.hypot(bx - ex, by - ey) < comp.domino_width
                    for bx, by in blue_pts for ex, ey in extra_pts):
                continue
            od.update(gray_od)
            merged.append(od)
        merged.extend(
            mbu._candidate_detour_layouts(comp, k, s_pose, t_pose, h_pose))
        for od in merged:
            cand_idx += 1
            if not mbu._layout_topples(env, od, doms[0], doms[1], push_opt):
                continue
            if not mbu._layout_noise_robust(env, comp, od, push_opt, cand_idx):
                continue
            return od, doms[0], doms[1], k
    return None


def believed_straight(start, target, gray, s_pose, t_pose, h_pose, k_max):
    """Cheapest believed straight-through layout (gray = free link)."""
    # pylint: disable=protected-access
    s_pt, t_pt, h_pt = (np.array(s_pose[:2]), np.array(t_pose[:2]),
                        np.array(h_pose[:2]))
    len1 = float(np.linalg.norm(h_pt - s_pt))
    d1_vec = (h_pt - s_pt) / len1
    yaw1 = float(np.arctan2(d1_vec[0], d1_vec[1]))
    h_dir = np.array([np.sin(h_pose[2]), np.cos(h_pose[2])])

    def _probe():
        for k in range(k_max + 1):
            for k1 in range(k + 1):
                k2 = k - k1
                gap1 = len1 / (k1 + 1)
                if not mbu._MIN_GAP < gap1 < mbu._MAX_GAP:
                    continue
                for g2 in (mbu._DOGLEG_EXIT_GAPS if k2 else (None, )):
                    od = {
                        start: comp.place_domino(0,
                                                 *s_pose,
                                                 is_start_block=True),
                        target: comp.place_domino(1,
                                                  *t_pose,
                                                  is_target_block=True),
                        gray: comp.place_domino(0,
                                                *h_pose,
                                                is_heavy_block=True),
                    }
                    slot = 2
                    for i in range(k1):
                        pt = s_pt + (i + 1) * gap1 * d1_vec
                        od[doms[slot]] = comp.place_domino(
                            slot, float(pt[0]), float(pt[1]), yaw1)
                        slot += 1
                    if k2:
                        b1 = h_pt + g2 * h_dir
                        e_vec = t_pt - b1
                        e_len = float(np.linalg.norm(e_vec))
                        per = e_len / k2
                        if not mbu._MIN_GAP < per < mbu._MAX_GAP:
                            continue
                        e_dir = e_vec / e_len
                        e_yaw = float(np.arctan2(e_dir[0], e_dir[1]))
                        for j in range(k2):
                            pt = b1 + j * per * e_dir
                            od[doms[slot]] = comp.place_domino(
                                slot, float(pt[0]), float(pt[1]), e_yaw)
                            slot += 1
                    elif not mbu._MIN_GAP < float(
                            np.linalg.norm(t_pt - h_pt)) < mbu._MAX_GAP:
                        continue
                    if mbu._layout_topples(env, od, start, target, push_opt):
                        return od, k
        return None

    with _believed_physics(env, believed_heavy_mass=True):
        return _probe()


def believed_gray_corner(start, target, gray, s_pose, t_pose, h_pose, k_max):
    """The believed gray-corner lure: the family layout whose corner pose
    matches the gray, with the gray substituted in (free corner)."""

    # pylint: disable=protected-access
    def _probe():
        cand_idx = 0
        for k in range(2, k_max + 2):
            for od, s_, t_ in mbu._candidate_turn_layouts(
                    comp, k, s_pose, t_pose):
                corner = next((o for o in od if o not in (s_, t_) and np.hypot(
                    od[o]["x"] - h_pose[0], od[o]["y"] - h_pose[1]) < 0.02
                               and abs(wrap(od[o]["yaw"] - h_pose[2])) < 0.1),
                              None)
                if corner is None:
                    continue
                lure = {o: dict(p) for o, p in od.items() if o is not corner}
                lure[gray] = comp.place_domino(0, *h_pose, is_heavy_block=True)
                cand_idx += 1
                # Same noise bar as generation: show the robust lure the
                # believed baseline would actually validate and build.
                if mbu._layout_topples(env, lure, s_, t_, push_opt) \
                        and mbu._layout_noise_robust(env, comp, lure,
                                                     push_opt, cand_idx):
                    return lure, k - 1
        return None

    with _believed_physics(env, believed_heavy_mass=True):
        return _probe()


tasks = env.get_test_tasks()

# Derive the panels in a FRESH simulator: generation may have run in
# this same process, and residual sim context can shift knife-edge
# outcomes even with per-probe state resets - execution also sees a
# fresh simulator, so this is the faithful context for re-derivation.
import pybullet as p

p.disconnect(env._physics_client_id)  # pylint: disable=protected-access
env = create_new_env('pybullet_domino', do_cache=False, use_gui=False)
comp = env._domino_component  # pylint: disable=protected-access
push_opt = mbu._get_push_option(env)
doms = comp.dominos

n = len(tasks)
fig, axes = plt.subplots(3, n, figsize=(3.4 * n, 10.2))
axes = np.atleast_2d(axes).T  # axes[col] = (init, solution, believed)

for col, task in enumerate(tasks):
    budget = int(task.offline_task_metrics["k_star"])
    init = task.init
    start, target, gray, s_pose, t_pose, h_pose = task_geometry(init)
    is_straight = abs(wrap(h_pose[2] - s_pose[2])) < 0.1
    kind = "straight" if is_straight else "turn"
    draw_state(axes[col][0], state_poses(init),
               f"task {col} ({kind}) · budget={budget}\nstaged init")

    k_max = CFG.domino_min_block_num_blues
    gray_od = {gray: comp.place_domino(0, *h_pose, is_heavy_block=True)}
    if is_straight:
        sol = swerve_solution(s_pose, t_pose, h_pose, k_max)
        sol_od = sol[0] if sol else None
        sol_label = f"swerve around: {sol[1]} blues" if sol else None
        bel = believed_straight(start, target, gray, s_pose, t_pose, h_pose,
                                k_max)
        bel_label = "believed through-gray"
    else:
        sol = detour_solution(s_pose, t_pose, gray_od, k_max)
        sol_od = sol[0] if sol else None
        sol_label = f"skip-around detour: {sol[3]} blues" if sol else None
        bel = believed_gray_corner(start, target, gray, s_pose, t_pose, h_pose,
                                   k_max)
        bel_label = "believed gray-corner"

    if sol_od is None:
        axes[col][1].axis("off")
        axes[col][1].set_title("solution not reproducible",
                               fontsize=9.5,
                               color="#555")
    else:
        draw_state(axes[col][1],
                   od_poses(sol_od, doms[0], doms[1], gray),
                   f"{sol_label}\n→ TOPPLES ✓",
                   tcolor="#1a7a1a")

    if bel is None:
        axes[col][2].axis("off")
        axes[col][2].set_title("no believed plan reproducible",
                               fontsize=9.5,
                               color="#555")
        continue
    od_b, k_b = bel
    # Execute the believed plan at the TRUE physics (gray heavy again).
    ok = mbu._layout_topples(env, od_b, start, target, push_opt)
    verdict = "→ TOPPLES (leak!)" if ok else "→ DIES AT GRAY ✗"
    draw_state(axes[col][2],
               od_poses(od_b, start, target, gray),
               f"{bel_label}: {k_b} blues\n{verdict}",
               tcolor="#a01515" if not ok else "#b3541e")

for label, y in (("staged init", 0.86), ("calibrated solution\n@ true physics",
                                         0.53),
                 ("believed (normal-mass) plan\nrun @ true physics", 0.19)):
    fig.text(0.005,
             y,
             label,
             fontsize=11,
             rotation=90,
             va="center",
             weight="bold",
             color="#333")
fig.tight_layout(rect=(0.03, 0, 1, 1))
out = Path(__file__).parent / "heavy_task_examples.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
