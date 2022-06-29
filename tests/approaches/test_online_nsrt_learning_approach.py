"""Test cases for the online NSRT learning approach."""

from contextlib import nullcontext as does_not_raise
from typing import Dict, Sequence

import numpy as np
import pytest

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.src.datasets import create_dataset
from predicators.src.envs.cover import CoverEnv
from predicators.src.main import _generate_interaction_results
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Action, Array, Dataset, Object, State
from predicators.src.teacher import Teacher


@pytest.mark.parametrize("explorer,expectation",
                         [("random_options", does_not_raise()),
                          ("not real", pytest.raises(NotImplementedError))])
def test_online_nsrt_learning_approach(explorer, expectation):
    """Test for OnlineNSRTLearningApproach class."""
    utils.reset_config({
        "env": "cover",
        "approach": "online_nsrt_learning",
        "sampler_mlp_classifier_max_itr": 10,
        "neural_gaus_regressor_max_itr": 10,
        "num_online_learning_cycles": 1,
        "num_train_tasks": 3,
        "num_test_tasks": 1,
        "max_initial_demos": 1,
        "max_num_steps_interaction_request": 3,
        "online_nsrt_learning_tasks_per_cycle": 1,
        "online_nsrt_learning_explorer": explorer,
        "timeout": 1,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    approach = OnlineNSRTLearningApproach(env.predicates, env.options,
                                           env.types, env.action_space,
                                           train_tasks)
    teacher = Teacher(train_tasks)
    dataset = create_dataset(env, train_tasks, env.options)
    approach.learn_from_offline_dataset(dataset)
    approach.load(online_learning_cycle=None)
    with expectation as e:
        interaction_requests = approach.get_interaction_requests()
    if e is not None:
        assert "Unrecognized explorer" in str(e)
        return
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
