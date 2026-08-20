"""End-to-end test: oracle_process_planning solves a bridge task.

Mirrors the config from ``predicatorv3/oracle.yaml`` +
``predicatorv3/envs/all.yaml`` (bridge entry) + ``predicatorv3/common.yaml``
so that a regression in the approach (process planning + bilevel
refinement), the bridge env's glue/weld machinery, or the skill
factories would surface here.

Runs the smallest viable config (1 train task, 1 test task, "simple"
spec: 4 blocks, 3 glue joints) and asserts:

  - The approach returns a policy (no ApproachTimeout / ApproachFailure).
  - Executing the policy in the env reaches ``task.goal_holds`` within
    the configured horizon (the finished n-bridge: legs at both sites,
    the welded 2-block span seated and cured onto the glued leg tops).
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


def _oracle_bridge_config() -> dict:
    """Flags from predicatorv3/{common,envs/all,oracle}.yaml flattened."""
    return {
        # --- env: bridge from envs/all.yaml ---
        "env": "pybullet_bridge",
        "horizon": 3000,
        "bridge_task_spec_train": ["simple"],
        "bridge_task_spec_test": ["simple"],
        "pybullet_birrt_path_subsample_ratio": 2,
        # The two seat joints cure a few physics steps apart, but each
        # Wait ends on the FIRST atom change, so the tail of a
        # multi-cure plan routinely needs a cheap replan (which reduces
        # to "Wait until the remaining joint cures").
        "process_planning_max_execution_replans": 3,
        "wait_option_max_steps": 120,
        # --- common flags relevant to bilevel refinement ---
        "skill_phase_use_motion_planning": True,
        # Bridge NEEDS validated IK (unlike boil): placement accuracy
        # feeds the cure gates, and the lateral placement error is
        # frozen into the weld. Unvalidated IK leaves the seated span
        # outside the far leg's cure window.
        "pybullet_ik_validate": True,
        "planning_filter_unreachable_nsrt": False,
        "no_repeated_arguments_in_grounding": True,
        "terminate_on_goal_reached": False,
        "sesame_check_expected_atoms": False,
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
        "option_model_terminate_on_repeat": False,
        "wait_option_terminate_on_atom_change": True,
    }


def test_oracle_process_planning_solves_bridge_task():
    """Smoke test: oracle_process_planning builds the simple n-bridge."""
    utils.reset_config(_oracle_bridge_config())
    env = create_new_env("pybullet_bridge", do_cache=False, use_gui=False)
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

    test_task = env.get_test_tasks()[0].task

    policy = approach.solve(test_task, timeout=CFG.timeout)
    assert policy is not None, "oracle_process_planning returned no policy"

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
