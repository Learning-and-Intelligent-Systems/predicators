"""Deterministic, LLM-free reproduction of the domino oracle-samplers failures.

Reproduces the geometric / parsing root causes behind the unsolved tasks in
``domino-agent_oracle_hybrid_sim_oracle_samplers_{demo,no_demo}`` (seeds 0-4),
*without* invoking the LLM sketcher.  The test-task scenes are deterministic
given the seed, so the BiRRT motion-planning infeasibilities and the option-plan
parser bug reproduce exactly.

IMPORTANT: run ONE seed per process.  Generating tasks for several seeds inside
one interpreter advances the shared RNG and changes the scenes (e.g. seed1 would
regenerate as [4,5,5,5,4] dominoes instead of the real [4,4,5,4,4]).  The bash
wrapper at the bottom of the module docstring loops correctly.

Usage:
    # motion-planning reproduction for a single seed (fresh process each):
    #   for s in 0 1 2 3 4; do PYTHONPATH=. python \
    #     scripts/domino_debug/reproduce_domino_failures.py mp $s; done
    # option-plan parser (Push) bug:
    #   PYTHONPATH=. python \
    #     scripts/domino_debug/reproduce_domino_failures.py push 0
"""

import logging
import sys
from typing import List, Optional, Tuple

import numpy as np

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.envs.base_env import BaseEnv
from predicators.ground_truth_models import get_gt_options
from predicators.structs import ParameterizedOption, State, _Option

logging.disable(logging.CRITICAL)

# Geometry-affecting flags copied verbatim from the experiment command line.
_ARGS = {
    "env": "pybullet_domino",
    "approach": "oracle",
    "num_train_tasks": 1,
    "num_test_tasks": 5,
    "pybullet_ik_validate": False,
    "skill_phase_use_motion_planning": True,
    "domino_initialize_at_finished_state": False,
    "domino_use_domino_blocks_as_target": True,
    "domino_use_continuous_place": True,
    "domino_restricted_push": True,
    "domino_has_glued_dominos": False,
    "pybullet_birrt_extend_num_interp": 20,
    "pybullet_birrt_path_subsample_ratio": 2,
}
_GRASP_Z_OFFSET = 0.0825  # value used by the oracle Pick sampler in the runs.
_POS_GAP = 0.098  # domino chain spacing (env.py: domino_width * 1.4).
_MAX_STEPS = 80


def _setup(seed: int) -> Tuple[BaseEnv, List[ParameterizedOption]]:
    """Reset the config and create the env + ground-truth options."""
    args = dict(_ARGS, seed=seed)
    utils.reset_config(args)
    env = get_or_create_env("pybullet_domino")
    options = list(get_gt_options(env.get_name()))
    return env, options


def _run_option(env: BaseEnv, opt: _Option,
                state: State) -> Tuple[Optional[bool], str]:
    """Drive a grounded option to termination; return (ok, failure_msg)."""
    if not opt.initiable(state):
        return None, "not-initiable"
    s = state
    for _ in range(_MAX_STEPS):
        try:
            a = opt.policy(s)
        except utils.OptionExecutionFailure as e:
            return False, str(e)
        s = env.step(a)
        if opt.terminal(s):
            return True, "ok"
    return True, "ran-max-steps"


def reproduce_mp(seed: int) -> None:
    """Report grasp-infeasible dominoes and probe one Place into the gap."""
    env, options = _setup(seed)
    Pick = next(o for o in options if o.name == "Pick")
    tasks = env.get_test_tasks()
    for ti in range(len(tasks)):
        env.reset("test", ti)
        st = env._current_state  # pylint: disable=protected-access
        dominoes = sorted([o for o in st if o.type.name == "domino"],
                          key=lambda o: o.name)
        infeasible = []
        for d in dominoes:
            env.reset("test", ti)
            s = env._current_state  # pylint: disable=protected-access
            rb = next(o for o in s if o.type.name == "robot")
            dd = next(o for o in s if o.name == d.name)
            opt = Pick.ground([rb, dd],
                              np.array([_GRASP_Z_OFFSET], dtype=np.float32))
            ok, _ = _run_option(env, opt, s)
            if ok is False:
                infeasible.append(d.name)
        print(
            f"seed{seed} task{ti} (run Task{ti+1}): {len(dominoes)} dominoes "
            f"| grasp-INFEASIBLE: {infeasible if infeasible else 'none'}")


def reproduce_push_bug(seed: int) -> None:
    """Show the parser drops a Push line that names a target domino."""
    env, options = _setup(seed)
    push = next(o for o in options if o.name == "Push")
    print(f"Push option signature: types={[t.name for t in push.types]}")
    state = env.get_test_tasks()[0].init
    objects = list(state)
    cases = {
        "LLM-style 'Push(robot, domino_0)'":
        "Pick(robot:robot, domino_1:domino)\n"
        "Push(robot:robot, domino_0:domino)\nWait(robot:robot)",
        "legal 'Push(robot)'":
        "Pick(robot:robot, domino_1:domino)\n"
        "Push(robot:robot)\nWait(robot:robot)",
    }
    for label, txt in cases.items():
        plan = utils.parse_model_output_into_option_plan(
            txt, objects, env.types, options, parse_continuous_params=False)
        names = [op.name for op, _, _ in plan]
        flag = "PUSH DROPPED!" if "Push" not in names else "ok"
        print(f"  {label:42s} -> {names}  ({flag})")


def _main() -> None:
    """Dispatch to the requested reproduction mode."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "mp"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if mode == "mp":
        reproduce_mp(seed)
    elif mode == "push":
        reproduce_push_bug(seed)
    else:
        raise SystemExit(f"unknown mode {mode!r} (expected 'mp' or 'push')")


if __name__ == "__main__":
    _main()
