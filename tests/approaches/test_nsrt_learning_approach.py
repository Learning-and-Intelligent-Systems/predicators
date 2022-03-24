"""Test cases for the NSRT learning approach."""

import pytest

from predicators.src import utils
from predicators.src.approaches import create_approach
from predicators.src.datasets import create_dataset
from predicators.src.envs import create_new_env
from predicators.src.settings import CFG


def _test_approach(env_name,
                   approach_name,
                   excluded_predicates="",
                   try_solving=True,
                   check_solution=False,
                   sampler_learner="neural",
                   option_learner="no_learning",
                   side_predicate_learner="no_learning",
                   num_train_tasks=1,
                   offline_data_method="demo+replay",
                   additional_settings=None):
    """Integration test for the given approach."""
    if additional_settings is None:
        additional_settings = {}
    utils.flush_cache()  # Some extremely nasty bugs arise without this.
    utils.reset_config({
        "env": env_name,
        "approach": approach_name,
        "neural_gaus_regressor_max_itr": 50,
        "sampler_mlp_classifier_max_itr": 50,
        "predicate_mlp_classifier_max_itr": 50,
        "mlp_regressor_max_itr": 50,
        "num_train_tasks": num_train_tasks,
        "num_test_tasks": 1,
        "offline_data_method": offline_data_method,
        "sesame_allow_noops": False,
        "offline_data_num_replays": 50,
        "excluded_predicates": excluded_predicates,
        "side_predicate_learner": side_predicate_learner,
        "option_learner": option_learner,
        "sampler_learner": sampler_learner,
        "cover_initial_holding_prob": 0.0,
        **additional_settings,
    })
    env = create_new_env(env_name)
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
    train_tasks = env.get_train_tasks()
    approach = create_approach(approach_name, preds, env.options, env.types,
                               env.action_space, train_tasks)
    dataset = create_dataset(env, train_tasks)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    task = env.get_test_tasks()[0]
    if try_solving:
        policy = approach.solve(task, timeout=CFG.timeout)
        if check_solution:
            traj = utils.run_policy_with_simulator(policy,
                                                   env.simulate,
                                                   task.init,
                                                   task.goal_holds,
                                                   max_num_steps=CFG.horizon)
            assert task.goal_holds(traj.states[-1])
    # We won't check the policy here because we don't want unit tests to
    # have to train very good models, since that would be slow.
    # Now test loading NSRTs & predicates.
    approach2 = create_approach(approach_name, preds, env.options, env.types,
                                env.action_space, train_tasks)
    approach2.load(online_learning_cycle=None)
    if try_solving:
        policy = approach2.solve(task, timeout=CFG.timeout)
        if check_solution:
            traj = utils.run_policy_with_simulator(policy,
                                                   env.simulate,
                                                   task.init,
                                                   task.goal_holds,
                                                   max_num_steps=CFG.horizon)
            assert task.goal_holds(traj.states[-1])


def test_nsrt_learning_approach():
    """Tests for NSRTLearningApproach class."""
    _test_approach(env_name="blocks",
                   approach_name="nsrt_learning",
                   try_solving=False)
    for side_predicate_learner in [
            "prediction_error_hill_climbing",
            "preserve_skeletons_hill_climbing",
            "backchaining",
    ]:
        _test_approach(env_name="repeated_nextto",
                       approach_name="nsrt_learning",
                       try_solving=False,
                       sampler_learner="random",
                       side_predicate_learner=side_predicate_learner)


def test_unknown_side_predicate_learner():
    """Test that arbitrary sidelining approach throws an error."""
    with pytest.raises(ValueError) as e:
        _test_approach(env_name="repeated_nextto",
                       approach_name="nsrt_learning",
                       try_solving=False,
                       sampler_learner="random",
                       side_predicate_learner="not_a_real_sidelining_strat")
    assert "not implemented" in str(e)


def test_neural_option_learning():
    """Tests for NeuralOptionLearner class."""
    _test_approach(env_name="cover_multistep_options",
                   approach_name="nsrt_learning",
                   try_solving=False,
                   sampler_learner="random",
                   option_learner="neural",
                   check_solution=False,
                   additional_settings={
                       "cover_multistep_thr_percent": 0.99,
                       "cover_multistep_bhr_percent": 0.99,
                   })


def test_oracle_samplers():
    """Test NSRTLearningApproach with oracle samplers."""
    # Oracle sampler learning should work (and be fast) in cover and blocks.
    # We can even check that the policy succeeds!
    _test_approach(env_name="cover",
                   approach_name="nsrt_learning",
                   sampler_learner="oracle",
                   check_solution=True,
                   num_train_tasks=3)
    _test_approach(env_name="blocks",
                   approach_name="nsrt_learning",
                   sampler_learner="oracle",
                   check_solution=True,
                   num_train_tasks=3)
    # Test oracle samplers + option learning.
    _test_approach(env_name="cover",
                   approach_name="nsrt_learning",
                   sampler_learner="oracle",
                   option_learner="oracle",
                   check_solution=True,
                   num_train_tasks=3)
    with pytest.raises(Exception) as e:
        # In painting, we learn operators that are different from the
        # oracle ones, so oracle sampler learning is not possible.
        _test_approach(env_name="painting",
                       approach_name="nsrt_learning",
                       sampler_learner="oracle",
                       check_solution=True,
                       num_train_tasks=3)
    assert "no match for ground truth NSRT" in str(e)


def test_grammar_search_invention_approach():
    """Tests for GrammarSearchInventionApproach class.

    Keeping this here because we can't import test files in github
    checks.
    """
    additional_settings = {
        "grammar_search_true_pos_weight": 10,
        "grammar_search_false_pos_weight": 1,
        "grammar_search_operator_size_weight": 1e-2,
        "grammar_search_max_predicates": 10,
        "grammar_search_predicate_cost_upper_bound": 6,
        "grammar_search_score_function": "prediction_error",
        "grammar_search_search_algorithm": "hill_climbing",
        "pretty_print_when_loading": True,
    }
    _test_approach(env_name="cover",
                   approach_name="grammar_search_invention",
                   excluded_predicates="Holding",
                   try_solving=False,
                   sampler_learner="random",
                   num_train_tasks=3,
                   additional_settings=additional_settings)
    # Test approach with unrecognized search algorithm.
    additional_settings = {
        "grammar_search_search_algorithm": "not a real search algorithm",
        "grammar_search_gbfs_num_evals": 10,
    }
    with pytest.raises(Exception) as e:
        _test_approach(env_name="cover",
                       approach_name="grammar_search_invention",
                       excluded_predicates="Holding",
                       try_solving=False,
                       sampler_learner="random",
                       additional_settings=additional_settings)
    assert "Unrecognized grammar_search_search_algorithm" in str(e.value)
    # Test approach with gbfs.
    additional_settings = {
        "grammar_search_search_algorithm": "gbfs",
        "grammar_search_gbfs_num_evals": 10,
    }
    _test_approach(env_name="cover",
                   approach_name="grammar_search_invention",
                   excluded_predicates="Holding",
                   try_solving=False,
                   sampler_learner="random",
                   additional_settings=additional_settings)


def test_sampler_learning_with_goals():
    """Tests for NSRT learning when samplers learn with goals."""
    _test_approach(env_name="cluttered_table_place",
                   approach_name="nsrt_learning",
                   try_solving=False,
                   sampler_learner="neural",
                   offline_data_method="demo",
                   additional_settings={"sampler_learning_use_goals": True})
