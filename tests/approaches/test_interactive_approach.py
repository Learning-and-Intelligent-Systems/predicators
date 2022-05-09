"""Test cases for the interactive learning approach."""

from typing import Dict, Sequence

import numpy as np
import pytest

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.approaches.interactive_learning_approach import \
    InteractiveLearningApproach
from predicators.src.datasets import create_dataset
from predicators.src.envs.cover import CoverEnv
from predicators.src.main import _generate_interaction_results
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Action, Array, Dataset, Object, State
from predicators.src.teacher import Teacher


def test_interactive_learning_approach():
    """Test for InteractiveLearningApproach class, entire pipeline."""
    utils.reset_config({
        "env": "cover",
        "approach": "interactive_learning",
        "offline_data_method": "demo+ground_atoms",
        "excluded_predicates": "Covers,Holding",
        "timeout": 10,
        "sampler_mlp_classifier_max_itr": 100,
        "predicate_mlp_classifier_max_itr": 100,
        "neural_gaus_regressor_max_itr": 100,
        "num_online_learning_cycles": 1,
        "teacher_dataset_num_examples": 5,
        "num_train_tasks": 5,
        "num_test_tasks": 5,
        "interactive_num_ensemble_members": 1,
        "interactive_num_requests_per_cycle": 1,
        # old default settings, for test coverage
        "interactive_action_strategy": "glib",
        "interactive_query_policy": "strict_best_seen",
        "interactive_score_function": "frequency",
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    initial_predicates = {
        p
        for p in env.predicates if p.name not in ["Covers", "Holding"]
    }
    approach = InteractiveLearningApproach(initial_predicates, env.options,
                                           env.types, env.action_space,
                                           train_tasks)
    teacher = Teacher(train_tasks)
    dataset = create_dataset(env, train_tasks, env.options)
    assert approach.is_learning_based
    # Learning with an empty dataset should not crash.
    approach.learn_from_offline_dataset(Dataset([]))
    # Learning with the actual dataset.
    approach.learn_from_offline_dataset(dataset)
    approach.load(online_learning_cycle=None)
    interaction_requests = approach.get_interaction_requests()
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
    # Test interactive_action_strategy random.
    utils.update_config({
        "interactive_action_strategy": "random",
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(env, teacher, interaction_requests)
    # Test interactive_action_strategy do nothing.
    utils.update_config({
        "interactive_action_strategy": "do_nothing",
    })
    interaction_requests = approach.get_interaction_requests()
    assert interaction_requests
    _generate_interaction_results(env, teacher, interaction_requests)
    # Test that glib falls back to random if no solvable task can be found.
    utils.update_config({
        "interactive_action_strategy": "glib",
        "timeout": 0.0,
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(env, teacher, interaction_requests)
    # Test that glib also falls back when there are no non-static predicates.
    approach2 = InteractiveLearningApproach(initial_predicates, env.options,
                                            env.types, env.action_space,
                                            train_tasks)
    approach2.learn_from_offline_dataset(Dataset([]))
    approach2.get_interaction_requests()
    # Test with a query policy that always queries about every atom.
    utils.update_config({
        "interactive_query_policy": "nonstrict_best_seen",
        "interactive_score_function": "trivial",
    })
    approach._best_score = -np.inf  # pylint: disable=protected-access
    interaction_requests = approach.get_interaction_requests()
    interaction_results, query_cost = _generate_interaction_results(
        env, teacher, interaction_requests)
    assert len(interaction_results) == 1
    interaction_result = interaction_results[0]
    predicates_to_learn = {
        p
        for p in env.predicates if p.name in ["Covers", "Holding"]
    }
    expected_query_cost = 0
    for s in interaction_result.states:
        ground_atoms = utils.all_possible_ground_atoms(s, predicates_to_learn)
        expected_query_cost += len(ground_atoms)
    assert query_cost == expected_query_cost
    # Cover random query policy
    utils.update_config({
        "interactive_query_policy": "random",
        "interactive_random_query_prob": 0.1,
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(env, teacher, interaction_requests)
    # Test with entropy score function and score threshold.
    utils.update_config({
        "interactive_query_policy": "threshold",
        "interactive_score_function": "entropy",
        "interactive_score_threshold": 0.5,
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(env, teacher, interaction_requests)
    # Test with BALD score function and score threshold.
    utils.update_config({
        "interactive_score_function": "BALD",
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(env, teacher, interaction_requests)
    # Test with variance score function and score threshold.
    utils.update_config({
        "interactive_score_function": "variance",
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(env, teacher, interaction_requests)
    # Test with greedy lookahead action strategy.
    utils.update_config({
        "interactive_action_strategy": "greedy_lookahead",
        "interactive_max_num_trajectories": 1,
        "interactive_max_trajectory_length": 1,
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(env, teacher, interaction_requests)

    # Cover greedy lookahead edge cases.
    def _policy(s: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del s, memory, objects, params  # unused
        raise utils.OptionExecutionFailure("Mock error")

    # Force 0 actions in trajectory
    new_nsrts = set()
    for nsrt in approach._nsrts:  # pylint: disable=protected-access
        new_option = utils.SingletonParameterizedOption(
            "LearnedMockOption",
            _policy,
            types=nsrt.option.types,
            params_space=nsrt.option.params_space,
            initiable=nsrt.option.initiable)
        new_nsrt = NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions,
                        nsrt.add_effects, nsrt.delete_effects, set(),
                        new_option, nsrt.option_vars, nsrt._sampler)  # pylint: disable=protected-access
        new_nsrts.add(new_nsrt)
    approach._nsrts = new_nsrts  # pylint: disable=protected-access
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(env, teacher, interaction_requests)
    # Force no applicable NSRTs
    approach._nsrts = set()  # pylint: disable=protected-access
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(env, teacher, interaction_requests)
    # Cover unrecognized interactive_action_strategy.
    utils.update_config({
        "interactive_action_strategy": "not a real action strategy",
        "interactive_query_policy": "strict_best_seen",
        "interactive_score_function": "frequency",
        "timeout": 10.0,
    })
    with pytest.raises(NotImplementedError) as e:
        approach.get_interaction_requests()
    assert "Unrecognized interactive_action_strategy" in str(e)
    # Cover unrecognized interactive_query_policy.
    utils.update_config({
        "interactive_action_strategy": "glib",
        "interactive_query_policy": "not a real query policy",
        "interactive_score_function": "frequency",
    })
    with pytest.raises(NotImplementedError) as e:
        approach.get_interaction_requests()
    assert "Unrecognized interactive_query_policy" in str(e)
    # Cover unrecognized interactive_score_function.
    utils.update_config({
        "interactive_action_strategy":
        "glib",
        "interactive_query_policy":
        "strict_best_seen",
        "interactive_score_function":
        "not a real score function",
    })
    with pytest.raises(NotImplementedError) as e:
        approach._score_atom_set(set(), train_tasks[0].init)  # pylint: disable=protected-access
    assert "Unrecognized interactive_score_function" in str(e)
    # Test assertion that all predicates are seen in the data
    utils.update_config({
        "approach": "interactive_learning",
        "teacher_dataset_num_examples": 0,
    })
    with pytest.raises(AssertionError):
        create_dataset(env, train_tasks, env.options)
