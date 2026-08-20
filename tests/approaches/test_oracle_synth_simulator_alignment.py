"""Refinement vs. real-execution alignment using the SYNTHESIZED simulator
captured by run_20260512_210304.

The original cup-collision happened with the agent's *learned* (not the
oracle GT) simulator wired into option_model. This test loads that
exact ``simulator.py`` snapshot from the failing run's sandbox, builds
the combined simulator (kinematic-only base env + learned step
dynamics), and verifies that:

* The synthesized-simulator option_model and the *real* execution env
  agree on the SwitchBurnerOn outcome for the attempt-2 Place pose
  ``(0.5313, 1.2899, 0.5659, yaw=2.5974)`` — i.e. if refinement says OK,
  execution should also be OK; if refinement says collision, execution
  should also fail.

The point isn't to fix the geometric collision (that's tracked
separately as a Place-sampler clearance fix). The point is to lock in
the invariant the user asked for: refinement / forward-validation success
implies real execution success.
"""
# pylint: disable=protected-access,import-outside-toplevel
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

import predicators.approaches  # noqa: F401  # pylint: disable=unused-import
import predicators.ground_truth_models  # noqa: F401  # pylint: disable=unused-import
from predicators import utils
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import LearnedSimulator, \
    apply_rules, merge_updates, read_simulator_components
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.option_model import _OracleOptionModel

logger = logging.getLogger(__name__)

# The failing run's synthesized simulator snapshot.
_SYNTH_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "logs",
                           "agent_sim_predicate_invention",
                           "boil-agent_predicate_invention", "seed0",
                           "run_20260512_210304", "sandbox", "simulator.py")


def _load_synth_simulator(
        path: str) -> Tuple[List, Dict[str, float], Dict[str, List[str]]]:
    """Execute simulator.py and return (rules, params, features)."""
    if not os.path.exists(path):
        pytest.skip(f"Synthesized simulator snapshot not present at {path}.")
    src = open(path, "r", encoding="utf-8").read()
    exec_ns: Dict[str, Any] = {"np": np, "ParamSpec": ParamSpec}
    exec(src, exec_ns)  # pylint: disable=exec-used
    rules, specs, features = read_simulator_components(exec_ns)
    assert rules and specs and features, (
        f"Snapshot {path} is missing RESIDUAL_RULES/PARAM_SPECS/"
        f"RESIDUAL_FEATURES.")
    params = {s.name: s.init_value for s in specs}
    return rules, params, features


# Attempt-2 plan from info.log:960-970.
_PLAN = [
    ("PickJug", [0.0262]),
    ("Place", [1.0138, 1.4008, 0.5790, -1.9641]),
    ("SwitchFaucetOn", [0.0511, 0.0978]),
    ("Wait", []),
    ("SwitchFaucetOff", [0.0547, 0.1037]),
    ("PickJug", [0.0041]),
    ("Place", [0.5313, 1.2899, 0.5659, 2.5974]),
    ("SwitchBurnerOn", [0.0413, 0.1016]),
]


def _resolve_objs(env, name: str):
    if name == "PickJug":
        return [env._robot, env._jugs[0]]
    if name in ("SwitchFaucetOn", "SwitchFaucetOff"):
        return [env._robot, env._faucet]
    if name == "SwitchBurnerOn":
        return [env._robot, env._burners[0]]
    if name in ("Place", "Wait"):
        return [env._robot]
    raise ValueError(name)


def _run_via_option_model(simulator_fn, options) -> Tuple[int, Optional[str]]:
    """Run the plan via option_model; return (last_step_idx, fail_reason)."""
    om = _OracleOptionModel(set(options.values()), simulator_fn)
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    state = env.get_train_tasks()[0].init
    env.reset("train", 0)
    for i, (name, params) in enumerate(_PLAN):
        opt = options[name].ground(_resolve_objs(env, name),
                                   np.array(params, dtype=np.float32))
        state, na = om.get_next_state_and_num_actions(state, opt)
        if na == 0:
            return i, om.last_execution_failure
    return len(_PLAN), None


def _run_via_env_step() -> Tuple[int, Optional[str]]:
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    options = {o.name: o for o in get_gt_options(env.get_name())}
    env.reset("train", 0)
    for i, (name, params) in enumerate(_PLAN):
        opt = options[name].ground(_resolve_objs(env, name),
                                   np.array(params, dtype=np.float32))
        if not opt.initiable(env._current_state):
            return i, "not initiable"
        try:
            for _ in range(400):
                if opt.terminal(env._current_state):
                    break
                env.step(opt.policy(env._current_state))
        except Exception as e:  # pylint: disable=broad-except
            return i, str(e)
    return len(_PLAN), None


def test_synth_simulator_refinement_agrees_with_real_execution():
    """Lock-in test: refinement using the synthesized simulator must agree with
    real-env execution on the first-failure step.

    Originally diverged because the real env spawned a physical liquid
    body inside the jug during Wait — a mass=0.01 body with collision
    geometry, recreated every fill tick — that pushed the jug a few cm
    over Wait's ~30-50 ticks. The synth simulator (base env with
    skip_residual_dynamics=True + learned step dynamics) never spawned
    that body, so its post-Wait jug pose matched Place exactly, while
    the real env's drifted. The 2nd PickJug's IK target tracks
    ``jug.x + cos(rot)*handle_offset``, so the divergent jug pose
    moved the IK target outside the robot's reachable workspace in
    real execution while option_model still found it reachable —
    exactly the cup-collision bug surfaced by run_20260512_210304.

    Fix: ``_create_liquid_for_jug`` in pybullet_boil.py now sets the
    liquid body's collision-filter mask to 0, so it stays visual-only
    and contributes no contact forces. Both paths now complete the
    attempt-2 plan in lockstep.
    """
    utils.reset_config({
        # Mirror the failing CLI's flags.
        "env": "pybullet_boil",
        "use_gui": False,
        "pybullet_robot": "fetch",
        "boil_use_skill_factories": True,
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
        "skill_phase_use_motion_planning": True,
        "pybullet_ik_validate": False,
        "option_model_terminate_on_repeat": False,
        "boil_goal": "simple",
        "boil_require_jug_full_to_heatup": True,
        "excluded_objects_in_state_str": "switch",
        "max_num_steps_option_rollout": 100,
        "horizon": 500,
        "boil_water_fill_speed": 0.0015,
        "pybullet_birrt_path_subsample_ratio": 2,
        "wait_option_terminate_on_atom_change": True,
        "seed": 0,
    })

    rules, params, _features = _load_synth_simulator(_SYNTH_PATH)
    learned = LearnedSimulator(step_fn=lambda s, c, _r=rules, _p=params:
                               apply_rules(s, _r, _p, cmds=c),
                               name="run_20260512_210304_snapshot")

    # Build the combined simulator the same way
    # AgentSimLearningApproach._build_combined_simulator does: a base
    # env with skip_residual_dynamics=True + the learned step dynamics.
    base_env = create_new_env("pybullet_boil",
                              do_cache=False,
                              use_gui=False,
                              skip_residual_dynamics=True)

    def combined_simulate(state, action):
        base_state = base_env.simulate(state, action)
        updates = learned.predict_step(base_state)
        if not updates:
            return base_state
        return merge_updates(base_state, updates)

    options = {o.name: o for o in get_gt_options("pybullet_boil")}

    om_step, om_reason = _run_via_option_model(combined_simulate, options)
    exec_step, exec_reason = _run_via_env_step()

    logger.info("synth-simulator option_model: stopped at step %d (%r)",
                om_step, om_reason)
    logger.info("real-env execution:          stopped at step %d (%r)",
                exec_step, exec_reason)

    assert om_step == exec_step, (
        f"Refinement (synth simulator) and execution disagree: "
        f"option_model stopped at step {om_step} (reason={om_reason!r}); "
        f"execution stopped at step {exec_step} (reason={exec_reason!r}). "
        f"This is the original cup-collision bug: refinement said the "
        f"plan was feasible but execution failed. Fix the divergence "
        f"or convert this test to xfail with documentation.")
