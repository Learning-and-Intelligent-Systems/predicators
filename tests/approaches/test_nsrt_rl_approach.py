"""Test cases for the reinforcement learning approach."""

import pytest

from predicators.src import utils
from predicators.src.approaches.nsrt_rl_approach import \
    NSRTReinforcementLearningApproach
from predicators.src.datasets import create_dataset
from predicators.src.envs.cover import CoverMultistepOptions
from predicators.src.main import _generate_interaction_results
from predicators.src.teacher import Teacher


def test_nsrt_reinforcement_learning_approach():
    """Test for NSRTReinforcementLearningApproach class, entire pipeline."""
    config = {
        "env": "cover_multistep_options",
        "cover_multistep_thr_percent": 0.99,
        "cover_multistep_bhr_percent": 0.99,
        "approach": "nsrt_rl",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "option_learner": "direct_bc",
        "sampler_learner": "random",
        "segmenter": "atom_changes",
        "num_online_learning_cycles": 1,
        "mlp_regressor_max_itr": 1,
    }
    utils.reset_config(config)
    env = CoverMultistepOptions()
    train_tasks = env.get_train_tasks()
    options = utils.parse_config_included_options(env)
    approach = NSRTReinforcementLearningApproach(env.predicates, options,
                                                 env.types, env.action_space,
                                                 train_tasks)
    teacher = Teacher(train_tasks)
    dataset = create_dataset(env, train_tasks, options)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    approach.load(online_learning_cycle=None)
    # Reduce the timeout for planning in the interaction loop, to make the
    # test faster.
    config["timeout"] = 0.01
    utils.reset_config(config)
    interaction_requests = approach.get_interaction_requests()
    interaction_results, _ = _generate_interaction_results(
        env, teacher, interaction_requests)
    approach.learn_from_interaction_results(interaction_results)
    approach.load(online_learning_cycle=1)
    with pytest.raises(FileNotFoundError):
        approach.load(online_learning_cycle=2)
