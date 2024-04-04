"""Test cases for the NSRT learning approach."""

import pytest

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, create_approach
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
                   strips_learner="cluster_and_intersect",
                   segmenter="option_changes",
                   num_train_tasks=1,
                   offline_data_method="demo+replay",
                   solve_exceptions=None,
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
        "strips_learner": strips_learner,
        "option_learner": option_learner,
        "sampler_learner": sampler_learner,
        "segmenter": segmenter,
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
    if option_learner == "no_learning":
        options = env.options
    else:
        options = utils.parse_config_included_options(env)
    approach = create_approach(approach_name, preds, options, env.types,
                               env.action_space, train_tasks)
    dataset = create_dataset(env, train_tasks, options)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    task = env.get_test_tasks()[0]
    if try_solving:
        if solve_exceptions is not None:
            assert not check_solution
            try:
                policy = approach.solve(task, timeout=CFG.timeout)
            except solve_exceptions:
                pass
        else:
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
        if solve_exceptions is not None:
            assert not check_solution
            try:
                policy = approach2.solve(task, timeout=CFG.timeout)
            except solve_exceptions:
                pass
        else:
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
    _test_approach(env_name="blocks",
                   approach_name="nsrt_learning",
                   strips_learner="cluster_and_search",
                   try_solving=False)
    for strips_learner in [
            "cluster_and_intersect_sideline_prederror",
            "cluster_and_intersect_sideline_harmlessness",
            "backchaining",
    ]:
        _test_approach(env_name="repeated_nextto",
                       approach_name="nsrt_learning",
                       try_solving=False,
                       sampler_learner="random",
                       strips_learner=strips_learner)
        if strips_learner == "backchaining":
            _test_approach(env_name="repeated_nextto_single_option",
                           approach_name="nsrt_learning",
                           try_solving=False,
                           sampler_learner="random",
                           strips_learner=strips_learner)


def test_unknown_strips_learner():
    """Test that arbitrary STRIPS learning approach throws an error."""
    with pytest.raises(ValueError) as e:
        _test_approach(env_name="repeated_nextto",
                       approach_name="nsrt_learning",
                       try_solving=False,
                       sampler_learner="random",
                       strips_learner="not a real strips learner")
    assert "Unrecognized STRIPS learner" in str(e)


def test_neural_option_learning():
    """Tests for NeuralOptionLearner class."""
    # Test with some, but not all, options given.
    _test_approach(env_name="cover_multistep_options",
                   approach_name="nsrt_learning",
                   try_solving=False,
                   sampler_learner="random",
                   option_learner="direct_bc",
                   segmenter="atom_changes",
                   check_solution=False,
                   additional_settings={
                       "cover_multistep_thr_percent": 0.99,
                       "cover_multistep_bhr_percent": 0.99,
                       "included_options": "Pick"
                   })
    # Test with oracle samplers.
    _test_approach(env_name="cover_multistep_options",
                   approach_name="nsrt_learning",
                   try_solving=True,
                   sampler_learner="oracle",
                   option_learner="direct_bc",
                   segmenter="atom_changes",
                   check_solution=False,
                   offline_data_method="demo",
                   solve_exceptions=(ApproachFailure, ),
                   additional_settings={
                       "cover_multistep_thr_percent": 0.99,
                       "cover_multistep_bhr_percent": 0.99,
                   })
    # Test with implicit bc option learning.
    _test_approach(env_name="touch_point",
                   approach_name="nsrt_learning",
                   try_solving=False,
                   sampler_learner="random",
                   option_learner="implicit_bc",
                   segmenter="atom_changes",
                   check_solution=False,
                   additional_settings={
                       "implicit_mlp_regressor_max_itr": 10,
                       "implicit_mlp_regressor_num_negative_data_per_input": 1,
                       "implicit_mlp_regressor_num_samples_per_inference": 1,
                   })
    # Test with direct bc nonparameterized option learning.
    _test_approach(env_name="touch_point",
                   approach_name="nsrt_learning",
                   try_solving=True,
                   sampler_learner="random",
                   option_learner="direct_bc_nonparameterized",
                   segmenter="atom_changes",
                   check_solution=False)


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
                   segmenter="atom_changes",
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


def test_degenerate_mlp_sampler_learning():
    """Tests for NSRTLearningApproach() with a degenerate MLP sampler."""
    _test_approach(env_name="cover",
                   approach_name="nsrt_learning",
                   try_solving=False,
                   sampler_learner="neural",
                   additional_settings={
                       "sampler_learning_regressor_model": "degenerate_mlp",
                   })


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
        "grammar_search_gbfs_num_evals": 1,
    }
    _test_approach(env_name="cover",
                   approach_name="grammar_search_invention",
                   excluded_predicates="Holding",
                   try_solving=False,
                   sampler_learner="random",
                   num_train_tasks=3,
                   additional_settings=additional_settings)
    # Test approach with unrecognized search algorithm.
    additional_settings["grammar_search_search_algorithm"] = \
        "not a real search algorithm"
    with pytest.raises(Exception) as e:
        _test_approach(env_name="cover",
                       approach_name="grammar_search_invention",
                       excluded_predicates="Holding",
                       try_solving=False,
                       sampler_learner="random",
                       additional_settings=additional_settings)
    assert "Unrecognized grammar_search_search_algorithm" in str(e.value)
    # Test approach with gbfs.
    additional_settings["grammar_search_search_algorithm"] = "gbfs"
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


def test_oracle_strips_and_segmenter_learning():
    """Test for oracle strips and segmenter learning in the stick button env
    with direct BC option learning."""
    additional_settings = {
        "stick_button_num_buttons_train": [1],
        "stick_button_num_buttons_test": [1],
        "segmenter": "oracle",
    }
    # The expected behavior is that segmentation and STRIPS learning will go
    # through, but then because there is such a limited number of demos, we
    # will not see data for all operators, which will lead to a crash
    # during option learning. This test still covers an important case, which
    # is recomputing datastores in STRIPS learning when options are unknown.
    with pytest.raises(Exception) as e:
        _test_approach(env_name="stick_button",
                       approach_name="nsrt_learning",
                       strips_learner="oracle",
                       option_learner="direct_bc",
                       offline_data_method="demo",
                       num_train_tasks=1,
                       try_solving=False,
                       additional_settings=additional_settings)
    assert "No data found for learning an option." in str(e)
