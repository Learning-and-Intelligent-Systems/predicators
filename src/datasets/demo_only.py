"""Create offline datasets by collecting demonstrations.
"""
from typing import List
from predicators.src.approaches import create_approach
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset, Task
from predicators.src import utils


def create_demo_data(env: BaseEnv, train_tasks: List[Task],
                     timeout: int) -> Dataset:
    """Create offline datasets by collecting demos.
    """
    oracle_approach = create_approach("oracle", env.simulate,
        env.predicates, env.options, env.types, env.action_space,
        train_tasks)
    dataset = []
    for task in train_tasks:
        policy = oracle_approach.solve(task, timeout=timeout)
        trajectory, solved = utils.run_policy_on_task(policy, task,
            env.simulate, env.predicates)
        assert solved, "Oracle failed on training task."
        dataset.append(trajectory)
    return dataset
