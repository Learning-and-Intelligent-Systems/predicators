"""Create offline datasets by collecting demonstrations."""

import os
from typing import List
from predicators.src.approaches import create_approach, ApproachTimeout, \
    ApproachFailure
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset, Task, LowLevelTrajectory
from predicators.src.settings import CFG
from predicators.src import utils


def create_demo_data(env: BaseEnv, train_tasks: List[Task]) -> Dataset:
    """Create offline datasets by collecting demos."""
    oracle_approach = create_approach("oracle", env.predicates, env.options,
                                      env.types, env.action_space, train_tasks)
    trajectories = []
    for idx, task in enumerate(train_tasks):
        try:
            policy = oracle_approach.solve(
                task, timeout=CFG.offline_data_planning_timeout)
        except (ApproachTimeout, ApproachFailure) as e:  # pragma: no cover
            # This should be extremely rare, so we only allow the script
            # to continue on supercloud, when running batch experiments
            # with analysis/submit.py.
            print(f"WARNING: Approach failed to solve with error: {e}")
            if not os.getcwd().startswith("/home/gridsan"):
                raise e
            continue
        traj, _, solved = utils.run_policy_on_task(
            policy, task, env.simulate, CFG.max_num_steps_check_policy)
        assert solved, "Oracle failed on training task."
        # Add is_demo flag and train task idx into the trajectory.
        traj = LowLevelTrajectory(traj.states,
                                  traj.actions,
                                  _is_demo=True,
                                  _train_task_idx=idx)
        if CFG.option_learner != "no_learning":
            for act in traj.actions:
                act.unset_option()
        trajectories.append(traj)
    return Dataset(trajectories)
