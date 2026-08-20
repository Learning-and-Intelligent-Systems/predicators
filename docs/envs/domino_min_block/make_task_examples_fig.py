"""Slide figure: sampled tasks - calibrated vs miscalibrated solutions.

Usage: python make_task_examples_fig.py [low|high]

``low`` (default) renders the domino_low_friction arm (true friction 0.1,
planner believes 0.5 - the over-reach condition), ``high`` the
domino_high_friction arm (true 0.5, believed 0.1 - under-reach). Env
flags mirror the corresponding block in
scripts/configs/predicatorv3/envs/all.yaml.

For each sampled cached test task, three rows:
  1. the staged initial state (blues parked, start/target fixed);
  2. the calibrated solution: the K*-search's winning layout at the TRUE
     friction, re-verified by simulation (should topple);
  3. the miscalibrated build, EXECUTED at the true friction - the
     baseline's predicted behaviour. Over-reach: the cheapest layout the
     planning-friction model accepts (fewer blues; should die short).
     Under-reach: the planning model's over-build (more blues; should
     topple but score below the calibrated reward).

Rows 2-3 also annotate the env reward the DominoEvaluator would grant
the rendered rollout: +1 for toppling the target minus
``domino_block_cost`` per blue the cascade consumed (toppled, or shoved
off its placed spot), read from the settled post-rollout sim state.

Everything is produced by the real task-gen machinery: cached tasks are
reloaded through the env, layouts come from the search code, outcomes from
sim rollouts with the real Push.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_domino.cascade_certificate import \
    RELAY_MIN_SLIDE
from predicators.envs.pybullet_domino.task_generators import \
    min_block_utils as mbu
from predicators.settings import CFG

ARMS = {
    # Over-reach: planner over-estimates reach, under-builds, dies short.
    "low": {
        "true_friction": 0.1,
        "planning_friction": 0.5,
        "span_lo": 0.13,
        "span_hi": 0.30,
        "num_blues": 4,
        "out_name": "task_examples_low_friction.png",
    },
    # Under-reach: planner under-estimates reach, over-builds, topples
    # but exceeds the K* budget. Short-leg geometry (retune 2026-07-12,
    # see envs/all.yaml): straights K*=1 vs believed 2 on spans
    # 0.29-0.31; turns K*=2 via a NATURAL single-corner blue vs believed
    # 3 on legs 0.21-0.24 x 0.17-0.20; 4 staged blues give the believed
    # 3-blue builds a spare.
    "high": {
        "true_friction": 0.5,
        "planning_friction": 0.1,
        "span_lo": 0.29,
        "span_hi": 0.31,
        "num_blues": 4,
        "turn_entry_lo": 0.21,
        "turn_entry_hi": 0.24,
        "turn_exit_lo": 0.17,
        "turn_exit_hi": 0.20,
        "block_cost": 0.1,
        "out_name": "task_examples_high_friction.png",
    },
}
ARM = ARMS[sys.argv[1] if len(sys.argv) > 1 else "low"]
NUM_BLUES = ARM["num_blues"]
OVER_REACH = ARM["planning_friction"] > ARM["true_friction"]

utils.reset_config({
    'env': 'pybullet_domino',
    'seed': 0,
    'num_train_tasks': 1,
    'num_test_tasks': 5,
    'test_env_seed_offset': 10000,
    'max_initial_demos': 0,
    'horizon': 500,
    'domino_initialize_at_finished_state': False,
    'domino_use_domino_blocks_as_target': True,
    'domino_use_continuous_place': True,
    'domino_has_glued_dominos': False,
    'domino_min_block_tasks': True,
    'domino_true_friction': ARM["true_friction"],
    'domino_planning_friction': ARM["planning_friction"],
    'domino_min_block_span_lo': ARM["span_lo"],
    'domino_min_block_span_hi': ARM["span_hi"],
    'domino_min_block_num_blues': NUM_BLUES,
    # Arm-specific turn-leg bands / block cost (retuned high arm); the
    # low arm keeps the generator's legacy direction defaults.
    'domino_min_block_turn_entry_lo': ARM.get("turn_entry_lo"),
    'domino_min_block_turn_entry_hi': ARM.get("turn_entry_hi"),
    'domino_min_block_turn_exit_lo': ARM.get("turn_exit_lo"),
    'domino_min_block_turn_exit_hi': ARM.get("turn_exit_hi"),
    'domino_block_cost': ARM.get("block_cost", 0.05),
    'pybullet_birrt_extend_num_interp': 20,
    'pybullet_birrt_path_subsample_ratio': 2,
})
env = create_new_env('pybullet_domino', do_cache=False, use_gui=False)
comp = env._domino_component  # pylint: disable=protected-access
push_opt = mbu._get_push_option(env)
W, D = comp.domino_width, comp.domino_depth
doms = comp.dominos


def block(ax, x, y, yaw, color, hl=False):
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
    if hl:
        ax.add_patch(
            Circle((x, y),
                   0.042,
                   fill=False,
                   edgecolor="#d62728",
                   lw=1.6,
                   linestyle="--",
                   zorder=5))


def draw_state(ax, poses, title, tcolor="k"):
    """poses: list of (x, y, yaw, role) with role in start/target/blue."""
    colors = {"start": "#7fc97f", "target": "#c599c5", "blue": "#7fb2d9"}
    for x, y, yaw, role in poses:
        block(ax, x, y, yaw, colors[role])
    ax.set_title(title, fontsize=9.5, color=tcolor)
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    ax.set_xlim(min(xs) - 0.09, max(xs) + 0.09)
    ax.set_ylim(min(ys) - 0.09, max(ys) + 0.09)
    ax.set_aspect("equal")
    ax.axis("off")


def od_poses(od, start, target):
    out = []
    for obj, pose in od.items():
        role = ("start"
                if obj is start else "target" if obj is target else "blue")
        out.append((pose["x"], pose["y"], pose["yaw"], role))
    return out


def state_poses(state):
    out = []
    for d in state.get_objects(comp.domino_type):
        # pylint: disable=protected-access
        role = ("start" if comp._StartBlock_holds(state, [d]) else
                "target" if comp._TargetDomino_holds(state, [d]) else "blue")
        out.append(
            (state.get(d, "x"), state.get(d, "y"), state.get(d, "yaw"), role))
    return out


def blues_used(od, start, target):
    """Blues the just-simulated rollout consumed - toppled, or shoved at
    least RELAY_MIN_SLIDE off their placed spot - read from the env's
    settled post-rollout state (the final-state view of the evaluator's
    count_movable_blocks_used)."""
    final = env._get_state()  # pylint: disable=protected-access
    used = 0
    for obj, pose in od.items():
        if obj in (start, target):
            continue
        toppled = abs(final.get(obj, "roll")) >= comp.fallen_threshold
        slid = float(
            np.hypot(
                final.get(obj, "x") - pose["x"],
                final.get(obj, "y") - pose["y"])) >= RELAY_MIN_SLIDE
        if toppled or slid:
            used += 1
    return used


def is_turn(state):
    """A turn task's target faces ~90 degrees off the start (goal_nl no longer
    marks turns, so detect them from the staged geometry)."""
    # pylint: disable=protected-access
    dominoes = state.get_objects(comp.domino_type)
    start = next(d for d in dominoes if comp._StartBlock_holds(state, [d]))
    target = next(d for d in dominoes if comp._TargetDomino_holds(state, [d]))
    dyaw = float(state.get(start, "yaw") - state.get(target, "yaw"))
    return abs((dyaw + np.pi) % (2 * np.pi) - np.pi) > np.pi / 4


def winning_layout(state, k_max, friction):
    """First toppling layout (straight chain or turn candidate) with the fewest
    blues at ``friction``; returns (od, start, target, k) or None."""
    # pylint: disable=protected-access
    dominoes = state.get_objects(comp.domino_type)
    start = next(d for d in dominoes if comp._StartBlock_holds(state, [d]))
    target = next(d for d in dominoes if comp._TargetDomino_holds(state, [d]))
    s_pose = (state.get(start, "x"), state.get(start,
                                               "y"), state.get(start, "yaw"))
    t_pose = (state.get(target, "x"), state.get(target,
                                                "y"), state.get(target, "yaw"))
    env.set_domino_physical_params(lateral_friction=friction)
    try:
        for k in range(k_max + 1):
            for od, s_, t_ in mbu._candidate_turn_layouts(
                    comp, k, s_pose, t_pose):
                if mbu._layout_topples(env, od, s_, t_, push_opt):
                    return od, s_, t_, k
    finally:
        env.set_domino_physical_params(
            lateral_friction=CFG.domino_true_friction)
    return None


def believed_straight_k(span, k_t):
    """Blue count of the planning-friction model's straight chain.

    Over-reach: the cheapest chain STRICTLY below the true count (the
    under-build), or None when no cheaper chain validates. Under-reach:
    the believed minimum over the full budget - the over-build - or None
    when it does not exceed the true count.
    """
    env.set_domino_physical_params(
        lateral_friction=CFG.domino_planning_friction)
    try:
        if OVER_REACH:
            return mbu.straight_span_k_star(env, span, budget=max(k_t - 1, 0))
        k_b = mbu.straight_span_k_star(env, span, budget=NUM_BLUES)
        return k_b if k_b is not None and k_b > k_t else None
    finally:
        env.set_domino_physical_params(
            lateral_friction=CFG.domino_true_friction)


tasks = env.get_test_tasks()
picks = list(range(len(tasks)))  # every task in the live set
# Transposed layout - tasks as columns, stages as rows - so the full set
# fits a widescreen slide.
fig, axes = plt.subplots(3, len(picks), figsize=(3.4 * len(picks), 10.2))
axes = np.atleast_2d(axes).T  # axes[col] = (init, true, believed) per task

for row, ti in enumerate(picks):
    task = tasks[ti]
    k_star = int(task.offline_task_metrics["k_star"])
    turn = is_turn(task.init)
    kind = "turn" if turn else "straight"
    draw_state(axes[row][0], state_poses(task.init),
               f"task {ti} ({kind}) · K*={k_star}\nstaged init")

    # Search up to the full blue budget: staged poses drift a little
    # through the PyBullet round-trip, so the regenerated minimal layout
    # can land one blue off the recorded K*.
    true_win = winning_layout(task.init, NUM_BLUES, CFG.domino_true_friction)
    if true_win is None:
        axes[row][1].axis("off")
        axes[row][1].set_title("layout not reproducible from staged poses",
                               fontsize=9.5,
                               color="#555")
        axes[row][2].axis("off")
        continue
    od, s_, t_, k_t = true_win
    # winning_layout's last rollout is the winner, so the env still holds
    # its settled final state - price it with the evaluator's reward form.
    r_cal = 1.0 - CFG.domino_block_cost * blues_used(od, s_, t_)
    draw_state(axes[row][1],
               od_poses(od, s_, t_),
               f"calibrated: {k_t} blues\n→ TOPPLES ✓ · reward {r_cal:+.2f}",
               tcolor="#1a7a1a")

    # Believed side: what the miscalibrated planning model builds.
    if not turn:
        # Straight: believed evenly-spaced chain over the task's span,
        # count probed at the drift-free canonical anchor.
        s_xy = np.array([od[s_]["x"], od[s_]["y"]])
        t_xy = np.array([od[t_]["x"], od[t_]["y"]])
        span = float(np.linalg.norm(t_xy - s_xy))
        k_b = believed_straight_k(span, k_t)
        if k_b is None:
            axes[row][2].axis("off")
            axes[row][2].set_title(
                f"no {'cheaper' if OVER_REACH else 'dearer'} chain at "
                f"µ={CFG.domino_planning_friction:g}",
                fontsize=9.5,
                color="#555")
            continue
        d_dir = (t_xy - s_xy) / span
        # Fall axis (-sin, cos) along the chain line.
        line_yaw = float(np.arctan2(-d_dir[0], d_dir[1]))
        odb = {s_: od[s_], t_: od[t_]}
        for i in range(k_b):
            pt = s_xy + (i + 1) * span / (k_b + 1) * d_dir
            odb[doms[2 + i]] = comp.place_domino(2 + i, float(pt[0]),
                                                 float(pt[1]), line_yaw)
        n_bel = k_b
        label = f"believed chain: {k_b} blues"
    elif not OVER_REACH:
        # Turn, under-reach: the believed model's ACTUAL build is its own
        # cheapest corner layout in the µ=0.1 sim (certified during task
        # generation) - respacing the calibrated route would break the
        # knife-edge pair corner and misrepresent the baseline.
        bel_win = winning_layout(task.init, NUM_BLUES,
                                 CFG.domino_planning_friction)
        if bel_win is None:
            axes[row][2].axis("off")
            axes[row][2].set_title(
                f"no believed corner plan at "
                f"µ={CFG.domino_planning_friction:g}",
                fontsize=9.5,
                color="#555")
            continue
        odb, _, _, n_bel = bel_win
        label = f"believed corner: {n_bel} blues"
    else:
        # Turn, over-reach: same route with one blue fewer (the believed
        # model's under-build), spread evenly along the winning layout's
        # own path (each blue faces its local travel direction). Uniform
        # across corner and diagonal-probe winning layouts.
        blues = [(o, pp) for o, pp in od.items() if o not in (s_, t_)]
        s_xy = np.array([od[s_]["x"], od[s_]["y"]])
        t_xy = np.array([od[t_]["x"], od[t_]["y"]])
        rest = blues[:]
        path = [s_xy]
        cur = s_xy
        while rest:
            nxt = min(rest,
                      key=lambda e: (e[1]["x"] - cur[0])**2 +
                      (e[1]["y"] - cur[1])**2)
            rest.remove(nxt)
            cur = np.array([nxt[1]["x"], nxt[1]["y"]])
            path.append(cur)
        path.append(t_xy)
        segs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        lens = [float(np.linalg.norm(b - a)) for a, b in segs]
        total = sum(lens)
        n_bel = max(len(blues) - 1, 0)
        odb = {s_: od[s_], t_: od[t_]}
        for i in range(n_bel):
            d_target = (i + 1) / (n_bel + 1) * total
            acc = 0.0
            for (a, b), seg_len in zip(segs, lens):
                if acc + seg_len >= d_target:
                    frac = (d_target - acc) / seg_len
                    pt = a + frac * (b - a)
                    dvec = (b - a) / seg_len
                    yaw = float(np.arctan2(-dvec[0], dvec[1]))
                    odb[doms[2 + i]] = comp.place_domino(
                        2 + i, float(pt[0]), float(pt[1]), yaw)
                    break
                acc += seg_len
        label = f"same route: {n_bel} blues"
    ok = mbu._layout_topples(env, odb, s_, t_, push_opt)
    r_bel = float(ok) - CFG.domino_block_cost * blues_used(odb, s_, t_)
    if OVER_REACH:
        # Expected miscalibrated failure: the under-build dies short.
        verdict = "→ TOPPLES (leak!)" if ok else "→ DIES SHORT ✗"
        tcolor = "#b3541e" if ok else "#a01515"
    else:
        # Expected miscalibrated failure mode: the over-build topples but
        # spends more blues than the calibrated K*, so it scores a lower
        # reward. A believed CORNER build can also legitimately die at
        # the true friction (the µ=0.1 corner geometry is knife-edge at
        # µ=0.5) - still a baseline failure, not a leak; a straight
        # over-build dying is a leak (denser chains only get safer as
        # friction rises).
        if ok:
            over = n_bel > k_star
            verdict = (f"→ TOPPLES, {n_bel} > K*={k_star} ✗"
                       if over else "→ TOPPLES within budget (leak!)")
            tcolor = "#a01515" if over else "#b3541e"
        elif turn:
            verdict, tcolor = "→ DIES SHORT ✗", "#a01515"
        else:
            verdict, tcolor = "→ DIES SHORT (leak!)", "#b3541e"
    draw_state(axes[row][2],
               od_poses(odb, s_, t_),
               f"{label}\n{verdict} · reward {r_bel:+.2f}",
               tcolor=tcolor)

true_mu = CFG.domino_true_friction
plan_mu = CFG.domino_planning_friction
for label, y in (("staged init",
                  0.86), (f"calibrated solution\n@ true µ={true_mu:g}", 0.53),
                 (f"µ={plan_mu:g} model's build\nrun @ true µ={true_mu:g}",
                  0.19)):
    fig.text(0.005,
             y,
             label,
             fontsize=11,
             rotation=90,
             va="center",
             weight="bold",
             color="#333")
fig.tight_layout(rect=(0.03, 0, 1, 1))
out = Path(__file__).parent / ARM["out_name"]
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
