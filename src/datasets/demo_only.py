"""Create offline datasets by collecting demonstrations."""

import os
from typing import List
from predicators.src.approaches import OracleApproach, ApproachTimeout, \
    ApproachFailure
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset, Task, LowLevelTrajectory
from predicators.src.settings import CFG
from predicators.src import utils


def create_demo_data(env: BaseEnv, train_tasks: List[Task]) -> Dataset:
    """Create offline datasets by collecting demos."""
    oracle_approach = OracleApproach(
        env.predicates,
        env.options,
        env.types,
        env.action_space,
        train_tasks,
        task_planning_heuristic=CFG.offline_data_task_planning_heuristic,
        max_skeletons_optimized=CFG.offline_data_max_skeletons_optimized)
    trajectories = []
    for idx, task in enumerate(train_tasks):
        if idx >= CFG.max_initial_demos:
            break
        try:
            oracle_approach.solve(task,
                                  timeout=CFG.offline_data_planning_timeout)
            # Since we're running the oracle approach, we know that the policy
            # is actually a plan under the hood, and we can retrieve it with
            # get_last_plan(). We do this because we want to run the full plan.
            plan = oracle_approach.get_last_plan()
            # Stop run_policy() when OptionPlanExhausted() is hit.
            traj = utils.run_policy(
                utils.option_plan_to_policy(plan),
                env,
                "train",
                idx,
                termination_function=lambda s: False,
                max_num_steps=CFG.horizon,
                exceptions_to_break_on={utils.OptionPlanExhausted})
            # Even though we're running the full plan, we should still check
            # that the goal holds at the end.
            assert task.goal_holds(traj.states[-1]), \
                "Oracle failed on training task"
        except (ApproachTimeout, ApproachFailure, AssertionError) as e:
            # This should be extremely rare, so we only allow the script
            # to continue on supercloud, when running batch experiments
            # with scripts/submit.py.
            print(f"WARNING: Approach failed to solve with error: {e}")
            if not os.getcwd().startswith("/home/gridsan"):
                raise e
            continue
        # Add is_demo flag and task index information into the trajectory.
        traj = LowLevelTrajectory(traj.states,
                                  traj.actions,
                                  _is_demo=True,
                                  _train_task_idx=idx)
        if CFG.option_learner != "no_learning":
            for act in traj.actions:
                act.unset_option()
        trajectories.append(traj)
    return Dataset(trajectories)
