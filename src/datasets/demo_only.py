"""Create offline datasets by collecting demonstrations.
"""
from typing import List
from predicators.src.approaches import create_approach
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset, Task


def create_demo_data(env: BaseEnv, train_tasks: List[Task],
                     data_config: dict) -> Dataset:
    """Create offline datasets by collecting demos.
    """
    oracle_approach = create_approach("oracle", env.simulate,
        env.predicates, env.options, env.types, env.action_space,
        train_tasks)
    timeout = data_config["planning_timeout"]
    dataset = []
    for i, task in enumerate(train_tasks):
        policy = oracle_approach.solve(task, timeout=timeout)
        # TODO: need to figure out how to get out policy over options,
        # rather than a policy over actions, in the case where
        # data_config["actions_or_options"] == "options"
        import ipdb; ipdb.set_trace()
