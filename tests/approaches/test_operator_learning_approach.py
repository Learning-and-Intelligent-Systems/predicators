"""Test cases for the operator learning approach.
"""

from predicators.src.envs import CoverEnv
from predicators.src.approaches import OperatorLearningApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.datasets import create_dataset
from predicators.src.settings import CFG
from predicators.src import utils


def test_operator_learning_approach():
    """Tests for OperatorLearningApproach class.
    """
    utils.update_config({"env": "cover", "approach": "operator_learning",
                         "timeout": 10, "max_samples_per_step": 10,
                         "seed": 0, "classifier_max_itr": 500,
                         "regressor_max_itr": 500})
    env = CoverEnv()
    approach = OperatorLearningApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    dataset = create_dataset(env)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    for task in env.get_test_tasks():
        try:
            approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure):
            pass
        # We won't check the policy here because we don't want unit tests to
        # have to train very good models, since that would be slow.
