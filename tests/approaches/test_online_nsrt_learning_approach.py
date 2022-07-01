"""Test cases for the online NSRT learning approach."""

import pytest

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.src.datasets import create_dataset
from predicators.src.envs.cover import CoverEnv
from predicators.src.main import _generate_interaction_results
from predicators.src.settings import CFG
from predicators.src.structs import Dataset
from predicators.src.teacher import Teacher


def test_online_nsrt_learning_approach():
    """Test for OnlineNSRTLearningApproach class, entire pipeline."""
    utils.reset_config({
        "env": "cover",
        "approach": "online_nsrt_learning",
        "timeout": 10,
        "sampler_mlp_classifier_max_itr": 10,
        "predicate_mlp_classifier_max_itr": 10,
        "neural_gaus_regressor_max_itr": 10,
        "num_online_learning_cycles": 1,
        "online_nsrt_learning_requests_per_cycle": 1,
        "num_train_tasks": 3,
        "num_test_tasks": 3,
        "explorer": "random_options",
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    approach = OnlineNSRTLearningApproach(env.predicates, env.options,
                                          env.types, env.action_space,
                                          train_tasks)
    dataset = create_dataset(env, train_tasks, env.options)
    assert approach.is_learning_based
    # Learning with an empty dataset should not crash.
    approach.learn_from_offline_dataset(Dataset([]))
    # Learning with the actual dataset.
    approach.learn_from_offline_dataset(dataset)
    approach.load(online_learning_cycle=None)
    interaction_requests = approach.get_interaction_requests()
    teacher = Teacher(train_tasks)
    interaction_results, _ = _generate_interaction_results(
        env, teacher, interaction_requests)
    approach.learn_from_interaction_results(interaction_results)
    approach.load(online_learning_cycle=0)
    with pytest.raises(FileNotFoundError):
        approach.load(online_learning_cycle=1)
    for task in env.get_test_tasks():
        try:
            approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure):  # pragma: no cover
            pass
        # We won't check the policy here because we don't want unit tests to
        # have to train very good models, since that would be slow.
