"""Test cases for the operator learning approach.
"""

from predicators.src.envs import CoverEnv
from predicators.src.approaches import OperatorLearningApproach
from predicators.src.datasets import create_dataset
from predicators.src.settings import CFG
from predicators.src import utils


def test_operator_learning_approach():
    """Tests for OperatorLearningApproach class.
    """
    utils.update_config({"env": "cover", "approach": "operator_learning",
                         "timeout": 10, "max_samples_per_step": 1000,
                         "seed": 0})
    env = CoverEnv()
    approach = OperatorLearningApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    dataset = create_dataset(env)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=CFG.timeout)
        assert utils.policy_solves_task(
            policy, task, env.simulate, env.predicates)
