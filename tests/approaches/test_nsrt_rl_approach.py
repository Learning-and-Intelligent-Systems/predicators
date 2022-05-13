"""Test cases for the reinforcement learning approach."""

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.approaches.nsrt_rl_approach import \
    NSRTReinforcementLearningApproach
from predicators.src.datasets import create_dataset
from predicators.src.envs.cover import CoverMultistepOptions
from predicators.src.main import _generate_interaction_results
from predicators.src.settings import CFG
from predicators.src.teacher import Teacher


def test_nsrt_reinforcement_learning_approach():
    """Test for NSRTReinforcementLearningApproach class, entire pipeline."""

    utils.reset_config({
        "env": "cover_multistep_options",
        "cover_multistep_thr_percent": 0.99,
        "cover_multistep_bhr_percent": 0.99,
        "approach": "nsrt_rl",
        "num_train_tasks": 10,
        "num_test_tasks": 10,
        "option_learner": "direct_bc",
        "sampler_learner": "neural",
        "num_online_learning_cycles": 1
    })
    env = CoverMultistepOptions()
    train_tasks = env.get_train_tasks()
    approach = NSRTReinforcementLearningApproach(env.predicates, env.options,
                                                 env.types, env.action_space,
                                                 train_tasks)
    teacher = Teacher(train_tasks)
    dataset = create_dataset(env, train_tasks)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    interaction_requests = approach.get_interaction_requests()
    interaction_results, _ = _generate_interaction_results(
        env, teacher, interaction_requests)
    approach.learn_from_interaction_results(interaction_results)
    # We won't check the policy here because we don't want unit tests to
    # have to train very good models, since that would be slow.
    for task in env.get_test_tasks():
        try:
            approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure):  # pragma: no cover
            pass
