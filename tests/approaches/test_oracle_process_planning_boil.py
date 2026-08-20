"""End-to-end test: oracle_process_planning solves a boil task.

Mirrors the config from ``predicatorv3/oracle.yaml`` +
``predicatorv3/envs/all.yaml`` + ``predicatorv3/common.yaml`` so that a
regression in either the approach (process planning + bilevel
refinement) or the boil env's skill execution would surface here.

Runs the smallest viable config (1 train task, 1 test task, 1 jug, 1
burner) and asserts:

  - The approach returns a policy (no ApproachTimeout / ApproachFailure).
  - Executing the policy in the env reaches ``task.goal_holds`` within
    the configured horizon.
"""
# pylint: disable=protected-access
from __future__ import annotations

import logging

import predicators.approaches  # noqa: F401  # pylint: disable=unused-import
import predicators.envs  # noqa: F401  # pylint: disable=unused-import
import predicators.ground_truth_models  # noqa: F401  # pylint: disable=unused-import
from predicators import utils
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG

logger = logging.getLogger(__name__)


def _oracle_boil_config() -> dict:
    """Flags from predicatorv3/{common,envs/all,oracle}.yaml flattened.

    Kept minimal: 1 train task and 1 test task, no online learning
    cycles (oracle approach is not learning-based), no LLM (oracle
    doesn't need one).
    """
    return {
        # --- env: boil from envs/all.yaml ---
        "env": "pybullet_boil",
        "excluded_objects_in_state_str": "switch",
        "max_num_steps_option_rollout": 100,
        "horizon": 500,
        "boil_goal": "simple",
        "boil_require_jug_full_to_heatup": True,
        "script_option_file_name": "boil.txt",
        "boil_water_fill_speed": 0.0015,
        "pybullet_birrt_path_subsample_ratio": 2,
        "boil_num_jugs_test": [1],
        "boil_num_jugs_train": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
        # --- common flags relevant to bilevel refinement ---
        "skill_phase_use_motion_planning": True,
        "pybullet_ik_validate": False,
        "planning_filter_unreachable_nsrt": False,
        "no_repeated_arguments_in_grounding": True,
        "terminate_on_goal_reached": False,
        # --- approach: oracle_process_planning from oracle.yaml ---
        "approach": "oracle_process_planning",
        "demonstrator": "oracle_process_planning",
        "terminate_on_goal_reached_and_option_terminated": True,
        "bilevel_plan_without_sim": True,
        # --- test scope: keep it small ---
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "seed": 0,
        "use_gui": False,
        "option_model_use_gui": False,
        # Match the failing run's other knobs that affect Place/push.
        "option_model_terminate_on_repeat": False,
        "wait_option_terminate_on_atom_change": True,
    }


def test_oracle_process_planning_solves_boil_task():
    """Smoke test: oracle_process_planning produces a working policy."""
    utils.reset_config(_oracle_boil_config())
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    options = get_gt_options(env.get_name())
    train_tasks = [t.task for t in env.get_train_tasks()]

    approach = create_approach(
        CFG.approach,
        env.predicates,
        options,
        env.types,
        env.action_space,
        train_tasks,
    )

    # Use the (single) test task — same goal_holds the real pipeline
    # checks at the end of an episode.
    test_task = env.get_test_tasks()[0].task

    # Solve. ApproachFailure / ApproachTimeout propagate.
    policy = approach.solve(test_task, timeout=CFG.timeout)
    assert policy is not None, "oracle_process_planning returned no policy"

    # Execute the policy and confirm the goal is reached within horizon.
    env.reset("test", 0)
    for step in range(CFG.horizon):
        if test_task.goal_holds(env._current_state):
            logger.info("Goal reached after %d env steps.", step)
            return
        action = policy(env._current_state)
        env.step(action)
    assert test_task.goal_holds(env._current_state), (
        f"Policy executed for {CFG.horizon} steps but goal not reached. "
        f"Final state predicates: "
        f"{utils.abstract(env._current_state, env.predicates)}; "
        f"required goal: {test_task.goal}")
