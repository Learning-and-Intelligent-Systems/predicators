"""Create offline datasets by collecting demonstrations.
"""

from typing import List
from predicators.src.approaches import create_approach
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset, Task
from predicators.src.settings import CFG
from predicators.src import utils


def create_demo_data(env: BaseEnv, train_tasks: List[Task]) -> Dataset:
    """Create offline datasets by collecting demos.
    """
    oracle_approach = create_approach("oracle", env.simulate,
        env.predicates, env.options, env.types, env.action_space)
    dataset = []
    for task in train_tasks:
        policy = oracle_approach.solve(
            task, timeout=CFG.offline_data_planning_timeout)
        traj, _, solved = utils.run_policy_on_task(
            policy, task, env.simulate, env.predicates,
            CFG.max_num_steps_check_policy, annotate_traj_with_goal=True)
        if CFG.option_learner != "no_learning":
            for act in traj.actions:
                act.unset_option()
        assert solved, "Oracle failed on training task."
        dataset.append(traj)
    return dataset
