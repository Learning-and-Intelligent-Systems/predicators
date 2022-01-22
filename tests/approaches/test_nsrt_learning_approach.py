"""Test cases for the NSRT learning approach."""

import pytest
from predicators.src.envs import create_env
from predicators.src.approaches import create_approach
from predicators.src.datasets import create_dataset
from predicators.src.settings import CFG
from predicators.src import utils


def _test_approach(env_name,
                   approach_name,
                   excluded_predicates="",
                   try_solving=True,
                   check_solution=False,
                   sampler_learner="neural",
                   option_learner="no_learning",
                   learn_side_predicates=False):
    """Integration test for the given approach."""
    utils.flush_cache()  # Some extremely nasty bugs arise without this.
    utils.update_config({
        "env": env_name,
        "approach": approach_name,
        "seed": 12345,
        "experiment_id": "",
    })
    utils.update_config({
        "env": env_name,
        "approach": approach_name,
        "timeout": 10,
        "max_samples_per_step": 10,
        "seed": 12345,
        "neural_gaus_regressor_max_itr": 200,
        "sampler_mlp_classifier_max_itr": 200,
        "predicate_mlp_classifier_max_itr": 200,
        "mlp_regressor_max_itr": 200,
        "excluded_predicates": excluded_predicates,
        "learn_side_predicates": learn_side_predicates,
        "option_learner": option_learner,
        "sampler_learner": sampler_learner
    })
    env = create_env(env_name)
    assert env.goal_predicates.issubset(env.predicates)
    if CFG.excluded_predicates:
        excludeds = set(CFG.excluded_predicates.split(","))
        assert excludeds.issubset({pred.name for pred in env.predicates}), \
            "Unrecognized excluded_predicates!"
        preds = {pred for pred in env.predicates if pred.name not in excludeds}
        assert env.goal_predicates.issubset(preds), \
            "Can't exclude a goal predicate!"
    else:
        preds = env.predicates
    approach = create_approach(approach_name, env.simulate, preds, env.options,
                               env.types, env.action_space)
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    task = env.get_test_tasks()[0]
    if try_solving:
        policy = approach.solve(task, timeout=CFG.timeout)
        if check_solution:
            assert utils.policy_solves_task(policy, task, env.simulate)
    # We won't check the policy here because we don't want unit tests to
    # have to train very good models, since that would be slow.
    # Now test loading NSRTs & predicates.
    approach2 = create_approach(approach_name, env.simulate, preds,
                                env.options, env.types, env.action_space)
    approach2.load()
    if try_solving:
        policy = approach2.solve(task, timeout=CFG.timeout)
        if check_solution:
            assert utils.policy_solves_task(policy, task, env.simulate)


def test_nsrt_learning_approach():
    """Tests for NSRTLearningApproach class."""
    _test_approach(env_name="blocks", approach_name="nsrt_learning")
    with pytest.raises(NotImplementedError):  # bad sampler_learner
        _test_approach(env_name="cover_multistep_options",
                       approach_name="nsrt_learning",
                       try_solving=False,
                       sampler_learner="not a real sampler learner")
    _test_approach(env_name="cover_multistep_options",
                   approach_name="nsrt_learning",
                   try_solving=False,
                   sampler_learner="random")
    with pytest.raises(NotImplementedError):
        _test_approach(env_name="repeated_nextto",
                       approach_name="nsrt_learning",
                       try_solving=False,
                       sampler_learner="random",
                       learn_side_predicates=True)
    # Test neural option learning.
    _test_approach(env_name="cover_multistep_options",
                   approach_name="nsrt_learning",
                   try_solving=False,
                   sampler_learner="random",
                   option_learner="neural",
                   check_solution=False)


def test_oracle_samplers():
    """Test NSRTLearningApproach with oracle samplers."""
    # Oracle sampler learning should work (and be fast) in cover and blocks.
    # We can even check that the policy succeeds!
    _test_approach(env_name="cover",
                   approach_name="nsrt_learning",
                   sampler_learner="oracle",
                   check_solution=True)
    _test_approach(env_name="blocks",
                   approach_name="nsrt_learning",
                   sampler_learner="oracle",
                   check_solution=True)
    # Test oracle samplers + option learning.
    _test_approach(env_name="cover",
                   approach_name="nsrt_learning",
                   sampler_learner="oracle",
                   option_learner="oracle",
                   check_solution=True)
    with pytest.raises(Exception) as e:
        # In painting, we learn operators that are different from the
        # oracle ones, so oracle sampler learning is not possible.
        _test_approach(env_name="painting",
                       approach_name="nsrt_learning",
                       sampler_learner="oracle",
                       check_solution=True)
    assert "no match for ground truth NSRT" in str(e)


def test_iterative_invention_approach():
    """Tests for IterativeInventionApproach class."""
    _test_approach(env_name="cover",
                   approach_name="iterative_invention",
                   excluded_predicates="Holding",
                   try_solving=False,
                   sampler_learner="random")
    _test_approach(env_name="blocks",
                   approach_name="iterative_invention",
                   excluded_predicates="Holding",
                   try_solving=False,
                   sampler_learner="random")


def test_grammar_search_invention_approach():
    """Tests for GrammarSearchInventionApproach class.

    Keeping this here because we can't import test files in github
    checks.
    """
    utils.update_config({
        "grammar_search_true_pos_weight": 10,
        "grammar_search_false_pos_weight": 1,
        "grammar_search_operator_size_weight": 1e-2,
        "grammar_search_max_predicates": 10,
        "grammar_search_predicate_cost_upper_bound": 6,
        "grammar_search_score_function": "prediction_error",
    })
    _test_approach(env_name="cover",
                   approach_name="grammar_search_invention",
                   excluded_predicates="Holding",
                   try_solving=False,
                   sampler_learner="random")
    # Test that the pipeline doesn't crash when no predicates are learned
    # involving a certain option argument (robot in this case).
    utils.update_config({"grammar_search_max_predicates": 0})
    _test_approach(env_name="blocks",
                   approach_name="grammar_search_invention",
                   excluded_predicates="GripperOpen",
                   try_solving=False,
                   sampler_learner="random")
