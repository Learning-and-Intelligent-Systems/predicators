"""Test cases for the operator learning approach.
"""

from predicators.src.envs import CoverEnv
from predicators.src.approaches import create_approach, \
    ApproachTimeout, ApproachFailure
from predicators.src.datasets import create_dataset
from predicators.src.settings import CFG
from predicators.src import utils


def _test_approach(approach_name):
    """Integration test for the given approach.
    """
    utils.update_config({"env": "cover", "approach": approach_name,
                         "timeout": 10, "max_samples_per_step": 10,
                         "seed": 12345, "classifier_max_itr": 500,
                         "regressor_max_itr": 500})
    env = CoverEnv()
    approach = create_approach(approach_name,
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    dataset = create_dataset(env)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    for task in env.get_test_tasks():
        try:
            approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure):  # pragma: no cover
            continue
        # We won't check the policy here because we don't want unit tests to
        # have to train very good models, since that would be slow.
    # Now test loading operators & predicates.
    approach2 = create_approach(approach_name,
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    approach2.load()
    for task in env.get_test_tasks():
        try:
            approach2.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure):  # pragma: no cover
            continue


def test_operator_learning_approach():
    """Tests for OperatorLearningApproach class.
    """
    _test_approach(approach_name="operator_learning")


def test_iterative_invention_approach():
    """Tests for IterativeInventionApproach class.
    """
    _test_approach(approach_name="iterative_invention")
