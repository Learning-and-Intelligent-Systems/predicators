"""Test cases for the interactive learning approach."""

import pytest
from predicators.src.approaches import InteractiveLearningApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.datasets import create_dataset
from predicators.src.envs import CoverEnv
from predicators.src.settings import CFG
from predicators.src.main import _generate_interaction_results
from predicators.src.teacher import Teacher
from predicators.src import utils


def test_interactive_learning_approach():
    """Test for InteractiveLearningApproach class, entire pipeline."""
    utils.update_config({"env": "cover"})
    utils.update_config({
        "approach": "interactive_learning",
        "interactive_action_strategy": "glib",
        "interactive_query_policy": "strict_best_seen",
        "interactive_score_function": "frequency",
        "interactive_num_babbles": 10,
        "offline_data_method": "demo+ground_atoms",
        "excluded_predicates": "IsBlock,Covers",
        "timeout": 10,
        "max_samples_per_step": 10,
        "seed": 123,
        # These need to be large enough that the classifier for Covers is good;
        # otherwise, incorrect operators can lead to glib failing.
        "teacher_dataset_label_ratio": 1.0,
        "predicate_mlp_classifier_max_itr": 1000,
        "sampler_mlp_classifier_max_itr": 200,
        "neural_gaus_regressor_max_itr": 200,
        "num_online_learning_cycles": 1,
        "num_train_tasks": 5,
        "num_test_tasks": 5,
        "cover_initial_holding_prob": 0.0,
        "min_data_for_nsrt": 0,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    initial_predicates = {
        p
        for p in env.predicates if p.name not in ["IsBlock", "Covers"]
    }
    approach = InteractiveLearningApproach(initial_predicates, env.options,
                                           env.types, env.action_space,
                                           train_tasks)
    teacher = Teacher()
    dataset = create_dataset(env, train_tasks)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    # Cover unrecognized interactive_action_strategy.
    utils.update_config({
        "interactive_action_strategy": "not a real action strategy",
        "interactive_query_policy": "strict_best_seen",
        "interactive_score_function": "frequency",
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
    # Successfully generate interaction requests.
    utils.update_config({
        "interactive_action_strategy": "glib",
        "interactive_query_policy": "strict_best_seen",
        "interactive_score_function": "not a real score function",
    })
    interaction_requests = approach.get_interaction_requests()
    # Cover unrecognized interactive_score_function.
    with pytest.raises(NotImplementedError) as e:
        _generate_interaction_results(env.simulate, teacher, train_tasks,
                                      interaction_requests)
    assert "Unrecognized interactive_score_function" in str(e)
    # Successfully generate interaction results.
    utils.update_config({
        "interactive_action_strategy": "glib",
        "interactive_query_policy": "strict_best_seen",
        "interactive_score_function": "frequency",
    })
    interaction_results = _generate_interaction_results(
        env.simulate, teacher, train_tasks, interaction_requests)
    approach.learn_from_interaction_results(interaction_results)
    for task in env.get_test_tasks():
        try:
            approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure):  # pragma: no cover
            pass
        # We won't check the policy here because we don't want unit tests to
        # have to train very good models, since that would be slow.    
