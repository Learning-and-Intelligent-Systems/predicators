"""Test cases for the interactive learning approach."""
from contextlib import nullcontext as does_not_raise
from typing import Dict, Sequence

import numpy as np
import pytest

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.interactive_learning_approach import \
    InteractiveLearningApproach
from predicators.cogman import CogMan
from predicators.datasets import create_dataset
from predicators.envs.cover import CoverEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.main import _generate_interaction_results
from predicators.perception import create_perceiver
from predicators.settings import CFG
from predicators.structs import NSRT, Action, Array, Dataset, Object, State
from predicators.teacher import Teacher


@pytest.mark.parametrize("predicate_classifier_model,expectation",
                         [("mlp", does_not_raise()), ("knn", does_not_raise()),
                          ("not a real model", pytest.raises(ValueError))])
def test_interactive_learning_approach(predicate_classifier_model,
                                       expectation):
    """Test for InteractiveLearningApproach class, entire pipeline."""
    utils.reset_config({
        "env": "cover",
        "approach": "interactive_learning",
        "offline_data_method": "demo+ground_atoms",
        "excluded_predicates": "Covers,Holding",
        "timeout": 10,
        "sampler_mlp_classifier_max_itr": 100,
        "predicate_classifier_model": predicate_classifier_model,
        "predicate_mlp_classifier_max_itr": 100,
        "neural_gaus_regressor_max_itr": 100,
        "num_online_learning_cycles": 1,
        "teacher_dataset_num_examples": 3,
        "num_train_tasks": 3,
        "num_test_tasks": 3,
        "interactive_num_ensemble_members": 1,
        "interactive_num_requests_per_cycle": 1,
        # old default settings, for test coverage
        "explorer": "glib",
        "interactive_query_policy": "strict_best_seen",
        "interactive_score_function": "frequency",
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    initial_predicates = {
        p
        for p in env.predicates if p.name not in ["Covers", "Holding"]
    }
    stripped_train_tasks = [
        utils.strip_task(task, initial_predicates) for task in train_tasks
    ]
    approach = InteractiveLearningApproach(initial_predicates,
                                           get_gt_options(env.get_name()),
                                           env.types, env.action_space,
                                           stripped_train_tasks)
    teacher = Teacher(train_tasks)
    predicates, _ = utils.parse_config_excluded_predicates(env)
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()),
                             predicates)
    assert approach.is_learning_based
    # Learning with an empty dataset should not crash.
    approach.learn_from_offline_dataset(Dataset([]))
    # Learning with the actual dataset.
    with expectation as e:
        approach.learn_from_offline_dataset(dataset)
    if e is not None:
        assert "Unrecognized predicate_classifier_model" in str(e)
        return
    approach.load(online_learning_cycle=None)
    interaction_requests = approach.get_interaction_requests()
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, exec_monitor)
    interaction_results, _ = _generate_interaction_results(
        cogman, env, teacher, interaction_requests)
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
    # Test explorer random.
    utils.update_config({
        "explorer": "random_options",
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(cogman, env, teacher, interaction_requests)
    # Test explorer do nothing.
    utils.update_config({
        "explorer": "no_explore",
    })
    interaction_requests = approach.get_interaction_requests()
    assert interaction_requests
    _generate_interaction_results(cogman, env, teacher, interaction_requests)
    # Test that glib falls back to random if no solvable task can be found.
    utils.update_config({
        "explorer": "glib",
        "timeout": 0.0,
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(cogman, env, teacher, interaction_requests)
    # Test that glib also falls back when there are no non-static predicates.
    approach2 = InteractiveLearningApproach(initial_predicates,
                                            get_gt_options(env.get_name()),
                                            env.types, env.action_space,
                                            stripped_train_tasks)
    approach2.learn_from_offline_dataset(Dataset([]))
    approach2.get_interaction_requests()
    # Test with a query policy that always queries about every atom.
    utils.update_config({
        "interactive_query_policy": "nonstrict_best_seen",
        "interactive_score_function": "trivial",
    })
    approach._best_score = -np.inf  # pylint: disable=protected-access
    interaction_requests = approach.get_interaction_requests()
    cogman = CogMan(approach, perceiver, exec_monitor)
    interaction_results, query_cost = _generate_interaction_results(
        cogman, env, teacher, interaction_requests)
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
    cogman = CogMan(approach, perceiver, exec_monitor)
    _generate_interaction_results(cogman, env, teacher, interaction_requests)
    # Test with entropy score function and score threshold.
    utils.update_config({
        "interactive_query_policy": "threshold",
        "interactive_score_function": "entropy",
        "interactive_score_threshold": 0.5,
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(cogman, env, teacher, interaction_requests)
    # Test with BALD score function and score threshold.
    utils.update_config({
        "interactive_score_function": "BALD",
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(cogman, env, teacher, interaction_requests)
    # Test with variance score function and score threshold.
    utils.update_config({
        "interactive_score_function": "variance",
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(cogman, env, teacher, interaction_requests)
    # Test with greedy lookahead action strategy.
    utils.update_config({
        "explorer": "greedy_lookahead",
        "greedy_lookahead_max_num_trajectories": 1,
        "greedy_lookahead_max_traj_length": 1,
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(cogman, env, teacher, interaction_requests)

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
    cogman = CogMan(approach, perceiver, exec_monitor)
    _generate_interaction_results(cogman, env, teacher, interaction_requests)
    # Force no applicable NSRTs
    approach._nsrts = set()  # pylint: disable=protected-access
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(cogman, env, teacher, interaction_requests)
    # Cover unrecognized explorer.
    utils.update_config({
        "explorer": "not a real action strategy",
        "interactive_query_policy": "strict_best_seen",
        "interactive_score_function": "frequency",
        "timeout": 10.0,
    })
    with pytest.raises(NotImplementedError) as e:
        approach.get_interaction_requests()
    assert "Unrecognized explorer" in str(e)
    # Cover unrecognized interactive_query_policy.
    utils.update_config({
        "explorer": "glib",
        "interactive_query_policy": "not a real query policy",
        "interactive_score_function": "frequency",
    })
    with pytest.raises(NotImplementedError) as e:
        approach.get_interaction_requests()
    assert "Unrecognized interactive_query_policy" in str(e)
    # Cover unrecognized interactive_score_function.
    utils.update_config({
        "explorer":
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
        create_dataset(env, train_tasks, get_gt_options(env.get_name()),
                       predicates)
