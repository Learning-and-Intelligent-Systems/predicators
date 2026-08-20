"""Test the fan GT hybrid simulator against the real env.

The fan env applies its wind dynamics in ``_domain_specific_step``,
which the approaches' base sims skip (``skip_residual_dynamics=True``),
so the GT residual rules must reproduce the ball's wind-driven motion.
The GT rules act on the physics-command channel: they emit a constant
force on the ball while a fan is on (``cmds.apply_force``) and the base
sim's engine handles contacts. These tests roll the hybrid sim (base
env + GT rules + command queueing, composed exactly like
``AgentSimLearningApproach._build_combined_simulator``) side by side
with the real env under identical no-op action sequences, covering free
runs, obstacle-wall blocking, boundary parking, simultaneous fans, and
the fans-off case.
"""

import numpy as np
import pytest

from predicators import utils
from predicators.code_sim_learning.commands import ApplyForce, CommandBuffer
from predicators.code_sim_learning.utils import apply_rules, \
    has_physics_rules, merge_updates
from predicators.envs import create_new_env
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.ground_truth_models import get_gt_simulator
from predicators.structs import Action

# BallAtTarget tolerance (pos_gap / 2): the hybrid ball must stay within
# this of the real ball or plan validation would disagree with reality.
GOAL_TOL = 0.04


@pytest.fixture(scope="module", name="fan_setup")
def _fan_setup():
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": 0,
        "fan_use_skill_factories": True,
        # The assertions below are written against the curated seed-0
        # task (ball/wall/target aligned in the center column).
        "fan_3x3_strategic_task_gen": True,
    })
    rules, specs, _ = get_gt_simulator("pybullet_fan")
    params = {s.name: s.init_value for s in specs}
    real_env = create_new_env("pybullet_fan", do_cache=False, use_gui=False)
    base_env = create_new_env("pybullet_fan",
                              do_cache=False,
                              use_gui=False,
                              skip_residual_dynamics=True)
    task = real_env.get_train_tasks()[0]
    return real_env, base_env, rules, params, task


def _make_noop(state, env):
    arr = np.array(list(state.joint_positions), dtype=np.float32)
    n = env.action_space.shape[0]
    if arr.shape[0] < n:
        arr = np.concatenate(
            [arr, np.zeros(n - arr.shape[0], dtype=np.float32)])
    return Action(arr)


def _get_obj(state, type_name, pred=None):
    for o in state:
        if o.type.name == type_name and (pred is None or pred(state, o)):
            return o
    raise ValueError(f"no {type_name} in state")


def _rollout_pair(fan_setup, switch_sides, n_steps, ball_xy=None):
    """Roll real and hybrid sims in lockstep; return max ball error and final
    (real, hybrid) states."""
    real_env, base_env, rules, params, task = fan_setup
    state = task.init.copy()
    ball = _get_obj(state, "ball")
    if ball_xy is not None:
        state.set(ball, "x", ball_xy[0])
        state.set(ball, "y", ball_xy[1])
    for side in switch_sides:
        switch = _get_obj(
            state,
            "switch",
            lambda s, o, _side=side: s.get(o, "controls_fan") == _side)
        state.set(switch, "is_on", 1.0)

    def hybrid_simulate(s, a):
        base_state = base_env.simulate(s, a)
        cmds = CommandBuffer()
        updates = apply_rules(base_state, rules, params, cmds=cmds)
        if cmds:
            # Post-step emission, consumed by the next action's substeps
            # - the same cadence _domain_specific_step's force has.
            base_env.queue_residual_commands(cmds.commands)
        return merge_updates(base_state, updates) if updates else base_state

    s_real, s_hyb = state, state
    max_err = 0.0
    for _ in range(n_steps):
        s_real = real_env.simulate(s_real, _make_noop(s_real, real_env))
        s_hyb = hybrid_simulate(s_hyb, _make_noop(s_hyb, base_env))
        err = float(
            np.hypot(
                s_real.get(ball, "x") - s_hyb.get(ball, "x"),
                s_real.get(ball, "y") - s_hyb.get(ball, "y")))
        max_err = max(max_err, err)
    return max_err, s_real, s_hyb


def test_fan_gt_simulator_loads():
    """The factory registry resolves pybullet_fan to a real simulator."""
    utils.reset_config({"env": "pybullet_fan", "seed": 0})
    rules, specs, features = get_gt_simulator("pybullet_fan")
    assert [r.__name__ for r in rules] == ["_wind_blowing"]
    names = {s.name for s in specs}
    assert names == {"wind_force"}
    assert features == {"ball": ["x", "y"]}
    # The rules act through the physics-command channel, so the fitting
    # stack must route them to the rollout objective.
    assert has_physics_rules(rules)


def test_fan_hybrid_free_run_and_boundary(fan_setup):
    """Fan blows the ball across the grid; both sims park at the boundary
    wall."""
    # Seed-0 train task: ball (0.75, 1.694), wall (0.75, 1.614), target
    # (0.75, 1.534); grid x {0.67, 0.75, 0.83}, y {1.534, 1.614, 1.694}.
    max_err, s_real, s_hyb = _rollout_pair(fan_setup, [0.0], 45)
    assert max_err < GOAL_TOL
    ball = _get_obj(s_real, "ball")
    # Both parked at the right boundary (grid edge 0.83).
    assert abs(s_real.get(ball, "x") - 0.83) < 0.005
    assert abs(s_hyb.get(ball, "x") - 0.83) < 0.005


def test_fan_hybrid_wall_blocking(fan_setup):
    """The obstacle wall stops the wind-driven ball in both sims."""
    # Up fan (blows -y) drives the ball from (0.75, 1.694) into the wall
    # at (0.75, 1.614); without wall modeling the hybrid ball would sail
    # through to the target row (1.534) and diverge by ~0.11.
    max_err, s_real, s_hyb = _rollout_pair(fan_setup, [3.0], 20)
    assert max_err < GOAL_TOL
    ball = _get_obj(s_real, "ball")
    for s in (s_real, s_hyb):
        assert s.get(ball, "y") > 1.66  # parked against the wall

    # Same wall approached sideways from a clear cell.
    max_err, _, _ = _rollout_pair(fan_setup, [0.0], 20, ball_xy=(0.67, 1.614))
    assert max_err < GOAL_TOL


def test_fan_hybrid_clear_column_and_multi_fan(fan_setup):
    """Free multi-cell run in a clear column, and two fans at once."""
    max_err, s_real, s_hyb = _rollout_pair(fan_setup, [3.0],
                                           80,
                                           ball_xy=(0.67, 1.694))
    assert max_err < GOAL_TOL
    ball = _get_obj(s_real, "ball")
    # Crossed the full grid to the lower boundary. Derive the grid edge
    # from the (stationary, on-grid) target the same way the GT rules do,
    # rather than hard-coding it: the whole arena translates with
    # PyBulletFanEnv.fan_row_y_shift.
    target = _get_obj(s_real, "target")
    _, y_coords = PyBulletFanEnv._grid_coords_for_point(  # pylint: disable=protected-access
        s_real.get(target, "x"), s_real.get(target, "y"))
    grid_y_lo = min(y_coords)
    assert abs(s_real.get(ball, "y") - grid_y_lo) < 0.005
    assert abs(s_hyb.get(ball, "y") - grid_y_lo) < 0.005

    max_err, _, _ = _rollout_pair(fan_setup, [0.0, 2.0], 35)
    assert max_err < GOAL_TOL


def test_fan_hybrid_fans_off_no_motion(fan_setup):
    """With all fans off the rules leave the ball to the base sim."""
    max_err, s_real, s_hyb = _rollout_pair(fan_setup, [], 10)
    assert max_err < GOAL_TOL
    ball = _get_obj(s_real, "ball")
    task_init = fan_setup[4].init
    assert abs(s_hyb.get(ball, "x") - task_init.get(ball, "x")) < 0.005
    assert abs(s_hyb.get(ball, "y") - task_init.get(ball, "y")) < 0.005


def test_fan_hybrid_slides_along_boundary(fan_setup):
    """A ball pressed into a boundary slab still slides along its face.

    The slabs are long, so a ball touching the left one overlaps it
    slightly in x and across the whole grid in y. A contact rule that
    only asked whether the perpendicular extents overlap would freeze
    the ball's y motion instead of letting it slide.
    """
    task = fan_setup[4]
    target = _get_obj(task.init, "target")
    x_coords, y_coords = PyBulletFanEnv._grid_coords_for_point(  # pylint: disable=protected-access
        task.init.get(target, "x"), task.init.get(target, "y"))
    # Park the ball against the left boundary in the top row, then blow it
    # left (into the slab) and down (along it) at the same time.
    start = (min(x_coords), max(y_coords))
    max_err, s_real, s_hyb = _rollout_pair(fan_setup, [1.0, 3.0],
                                           80,
                                           ball_xy=start)
    assert max_err < GOAL_TOL
    ball = _get_obj(s_real, "ball")
    for s in (s_real, s_hyb):
        assert abs(s.get(ball, "x") - min(x_coords)) < 0.005  # held by slab
        assert abs(s.get(ball, "y") - min(y_coords)) < 0.005  # slid the column


def test_fan_boundary_slabs_are_observed_grid_tight_objects(fan_setup):
    """The arena boundary is in the state, and tracks the task's grid."""
    real_env = fan_setup[0]
    train_init = real_env.get_train_tasks()[0].init
    test_init = real_env.get_test_tasks()[0].init

    for init in (train_init, test_init):
        boundaries = [o for o in init if o.type.name == "boundary"]
        assert {o.name
                for o in boundaries} == {
                    "boundary_left", "boundary_right", "boundary_down",
                    "boundary_up"
                }
        target = _get_obj(init, "target")
        x_coords, y_coords = PyBulletFanEnv._grid_coords_for_point(  # pylint: disable=protected-access
            init.get(target, "x"), init.get(target, "y"))
        gap = PyBulletFanEnv.pos_gap
        by_name = {o.name: o for o in boundaries}
        # Half a grid gap outside the extreme cells, spanning the arena.
        assert np.isclose(init.get(by_name["boundary_left"], "x"),
                          min(x_coords) - gap / 2)
        assert np.isclose(init.get(by_name["boundary_right"], "x"),
                          max(x_coords) + gap / 2)
        assert np.isclose(init.get(by_name["boundary_down"], "y"),
                          min(y_coords) - gap / 2)
        assert np.isclose(init.get(by_name["boundary_up"], "y"),
                          max(y_coords) + gap / 2)
        assert np.isclose(init.get(by_name["boundary_left"], "y_len"),
                          max(y_coords) - min(y_coords) + gap)

    # The two grids differ, so a model that memorized the train boundary
    # would place it wrong at test time - hence it has to be observed.
    def _left_span(init):
        left = next(o for o in init if o.name == "boundary_left")
        return init.get(left, "y_len")

    assert not np.isclose(_left_span(train_init), _left_span(test_init))


def test_fan_rollout_objective_scores_command_rules(fan_setup):
    """The sysID rollout objective runs command rules IN the free-run.

    Records a short real-env trajectory with a fan on, then scores it
    with ``compute_rollout_sse`` against a fresh base env: at the
    calibrated wind force the free-run tracks the recording (small
    SSE); with the wind zeroed the rolled-out ball never moves and the
    SSE grows by orders of magnitude. This is the routing every
    command-emitting artifact takes (``has_physics_rules``) - without
    the in-rollout command feed, both scores would be identical.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        compute_rollout_sse
    _, _, rules, params, task = fan_setup
    state = task.init.copy()
    switch = _get_obj(state, "switch",
                      lambda s, o: s.get(o, "controls_fan") == 0.0)
    state.set(switch, "is_on", 1.0)
    # Record from a FRESH real env, not the module-shared one: earlier
    # tests leave solver warm-start state a state-level reset cannot
    # flush (the same mechanism documented in rollout_states), and the
    # scoring below free-runs fresh envs - the recording must come from
    # the same distribution.
    rec_env = create_new_env("pybullet_fan", do_cache=False, use_gui=False)
    try:
        states, actions = [state], []
        s = state
        for _ in range(15):
            a = _make_noop(s, rec_env)
            s = rec_env.simulate(s, a)
            states.append(s)
            actions.append(a)
    finally:
        rec_env.dispose()
    ball = _get_obj(s, "ball")
    assert s.get(ball, "x") - state.get(ball, "x") > 0.02  # wind acted

    def fit_env():
        return create_new_env("pybullet_fan",
                              do_cache=False,
                              use_gui=False,
                              skip_residual_dynamics=True)

    features = {"ball": ["x", "y"]}
    sse_gt = compute_rollout_sse(fit_env, [(states, actions)], params,
                                 features, [], rules)
    sse_off = compute_rollout_sse(fit_env, [(states, actions)],
                                  {"wind_force": 0.0}, features, [], rules)
    assert sse_gt < 1e-3
    assert sse_off > 10 * max(sse_gt, 1e-6)


def test_residual_commands_expire_after_one_action(fan_setup):
    """Queued commands act on exactly one action, then expire.

    The env-side contract: a queued force is applied across the next
    action's physics substeps and cleared afterwards - a persistent
    process must re-emit per step. A second step WITHOUT re-queueing
    must show (damped) coasting only, not another pushed step.
    """
    _, base_env, _, params, task = fan_setup
    state = task.init.copy()
    ball = _get_obj(state, "ball")
    # Free-field cell (target row, left column) so nothing blocks +x.
    state.set(ball, "x", 0.67)
    state.set(ball, "y", 1.534)

    s0 = base_env.simulate(state, _make_noop(state, base_env))
    buf = CommandBuffer()
    # The force is held across every substep of exactly one action,
    # then expires.
    buf.apply_force(ball, (params["wind_force"], 0.0, 0.0))
    base_env.queue_residual_commands(buf.commands)
    s1 = base_env.simulate(s0, _make_noop(s0, base_env))
    pushed = s1.get(ball, "x") - s0.get(ball, "x")
    assert pushed > 1e-3  # the queued force moved the ball
    s2 = base_env.simulate(s1, _make_noop(s1, base_env))
    coasted = s2.get(ball, "x") - s1.get(ball, "x")
    # Expired: the un-requeued force contributes nothing; the heavily
    # damped ball travels a small fraction of the pushed step.
    assert abs(coasted) < 0.25 * pushed


def test_fan_rule_forces_come_from_observed_fans(fan_setup):
    """The rule's commands are a pure function of observed fan features.

    Contact geometry is the engine's job, so the rule's whole contract
    is the force law: direction from each on-fan's ``rot``, vector
    summation for simultaneous fans, no feature overwrites, and silence
    when every fan is off.
    """
    _, _, rules, params, task = fan_setup
    state = task.init.copy()
    ball = _get_obj(state, "ball")

    # All fans off: no commands, no updates.
    cmds = CommandBuffer()
    updates = apply_rules(state, rules, params, cmds=cmds)
    assert not updates and len(cmds) == 0

    # One fan on: force along (cos rot, sin rot), still no updates -
    # the engine, not merge_updates, moves the ball.
    fan = _get_obj(state, "fan", lambda s, o: s.get(o, "facing_side") == 0.0)
    state.set(fan, "is_on", 1.0)
    cmds = CommandBuffer()
    updates = apply_rules(state, rules, params, cmds=cmds)
    assert not updates
    assert len(cmds) == 1
    cmd = cmds.commands[0]
    assert isinstance(cmd, ApplyForce) and cmd.obj_name == ball.name
    rot = state.get(fan, "rot")
    expected = np.array([np.cos(rot), np.sin(rot), 0.0]) * \
        params["wind_force"]
    assert np.allclose(cmd.force, expected)

    # Two fans on: per-axis vector sum.
    fan2 = _get_obj(state, "fan", lambda s, o: s.get(o, "facing_side") == 3.0)
    state.set(fan2, "is_on", 1.0)
    cmds = CommandBuffer()
    apply_rules(state, rules, params, cmds=cmds)
    assert len(cmds) == 1
    cmd = cmds.commands[0]
    rot2 = state.get(fan2, "rot")
    expected2 = expected + np.array([np.cos(rot2),
                                     np.sin(rot2), 0.0]) * params["wind_force"]
    assert np.allclose(cmd.force, expected2)
