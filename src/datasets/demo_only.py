"""Create offline datasets by collecting demonstrations.
"""

from predicators.src.approaches import create_approach
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset
from predicators.src.settings import CFG
from predicators.src import utils


def create_demo_data(env: BaseEnv) -> Dataset:
    """Create offline datasets by collecting demos.
    """
    train_tasks = env.get_train_tasks()
    oracle_approach = create_approach("oracle", env.simulate,
        env.predicates, env.options, env.types, env.action_space,
        train_tasks)
    dataset = []
    for task in train_tasks:
        policy = oracle_approach.solve(
            task, timeout=CFG.offline_data_planning_timeout)
        trajectory, solved = utils.run_policy_on_task(policy, task,
            env.simulate, env.predicates)
        if not CFG.include_options_in_offline_data:
            for act in trajectory[1]:
                act.unset_option()
        assert solved, "Oracle failed on training task."
        dataset.append(trajectory)
    return dataset
