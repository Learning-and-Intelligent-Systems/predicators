"""Test cases for the reinforcement learning approach."""

import pytest

from predicators.src import utils
from predicators.src.approaches.nsrt_rl_approach import \
    NSRTReinforcementLearningApproach
from predicators.src.datasets import create_dataset
from predicators.src.envs.cover import CoverMultistepOptions
from predicators.src.main import _generate_interaction_results
from predicators.src.structs import InteractionResult, Task
from predicators.src.teacher import Teacher


class _MockNSRTReinforcementLearningApproach(NSRTReinforcementLearningApproach
                                             ):
    """Mock class that exposes self._requests_info for testing."""

    def get_requests_info(self):
        """Get the current self._requests_info."""
        return self._requests_info

    def set_requests_info_idx(self, idx, val):
        """Set the value of self._requests_info at a current index."""
        self._requests_info[idx] = val


@pytest.mark.parametrize("nsrt_rl_reward_epsilon", [1e-2, 1e6])
def test_nsrt_reinforcement_learning_approach(nsrt_rl_reward_epsilon):
    """Test for NSRTReinforcementLearningApproach class, entire pipeline."""

    utils.reset_config({
        "env": "cover_multistep_options",
        "cover_multistep_thr_percent": 0.99,
        "cover_multistep_bhr_percent": 0.99,
        "approach": "nsrt_rl",
        "num_train_tasks": 3,
        "num_test_tasks": 3,
        "option_learner": "direct_bc",
        "sampler_learner": "neural",
        "num_online_learning_cycles": 1,
        "segmenter": "contacts",
        "mlp_regressor_max_itr": 10,
        "sampler_mlp_classifier_max_itr": 10,
        "neural_gaus_regressor_max_itr": 10,
        "timeout": 0.1,
        "disable_harmlessness_check": True,
        "nsrt_rl_reward_epsilon": nsrt_rl_reward_epsilon,
    })
    env = CoverMultistepOptions()
    train_tasks = env.get_train_tasks()
    # Make the last train task have a trivial goal so that it can be solved by
    # get_interaction_requests() even though we're not learning good models.
    train_tasks[-1] = Task(train_tasks[-1].init, set())
    approach = _MockNSRTReinforcementLearningApproach(env.predicates, {},
                                                      env.types,
                                                      env.action_space,
                                                      train_tasks)
    teacher = Teacher(train_tasks)
    dataset = create_dataset(env, train_tasks, {})
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    interaction_requests = approach.get_interaction_requests()
    interaction_results, _ = _generate_interaction_results(
        env, teacher, interaction_requests)
    # Hack the last interaction result to be non-trivial. Note that this
    # requires hacking approach._requests_info as well, since that is used
    # in learn_from_interaction_results().
    assert len(interaction_results[-1].states) == 1
    state = interaction_results[-1].states[0]
    arbitrary_action = interaction_results[0].actions[0]
    arbitrary_plan = approach.get_requests_info()[0][1]
    interaction_results[-1] = InteractionResult([state, state],
                                                [arbitrary_action],
                                                [None, None])
    approach.set_requests_info_idx(-1, (2, arbitrary_plan))
    approach.learn_from_interaction_results(interaction_results)
    arbitrary_plan[0].parent.effect_based_terminal = lambda s, o: True
    # Now learn from the interaction results (including the hacked last one).
    approach.learn_from_interaction_results([interaction_results[-1]])
