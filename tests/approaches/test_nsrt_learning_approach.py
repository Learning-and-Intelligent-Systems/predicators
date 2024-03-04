"""Test cases for the NSRT learning approach."""
import os

import dill as pkl
import pytest

from predicators import utils
from predicators.approaches import ApproachFailure, create_approach
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options, \
    parse_config_included_options
from predicators.settings import CFG

longrun = pytest.mark.skipif("not config.getoption('longrun')")


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
                   save_atoms=False,
                   load_atoms=False,
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
        "save_atoms": save_atoms,
        "load_atoms": load_atoms,
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
    train_tasks = [t.task for t in env.get_train_tasks()]
    if option_learner == "no_learning":
        options = get_gt_options(env.get_name())
    else:
        options = parse_config_included_options(env)
    approach = create_approach(approach_name, preds, options, env.types,
                               env.action_space, train_tasks)
    dataset = create_dataset(env, train_tasks, options)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    task = env.get_test_tasks()[0].task
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
    approach2 = create_approach(approach_name, preds,
                                get_gt_options(env.get_name()), env.types,
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
    return approach


def test_nsrt_learning_approach():
    """Tests for NSRTLearningApproach class."""
    approach = _test_approach(env_name="blocks",
                              approach_name="nsrt_learning",
                              try_solving=False)
    approach = _test_approach(env_name="blocks",
                              approach_name="nsrt_learning",
                              try_solving=False,
                              additional_settings={
                                  "compute_sidelining_objective_value": True,
                              })
    assert "sidelining_obj_num_plans_up_to_n" in approach.metrics
    assert "sidelining_obj_complexity" in approach.metrics
    assert approach.metrics["sidelining_obj_num_plans_up_to_n"] == 1.0
    assert approach.metrics["sidelining_obj_complexity"] == 34.0
    _test_approach(env_name="blocks",
                   approach_name="nsrt_learning",
                   strips_learner="cluster_and_search",
                   try_solving=False)
    for strips_learner in [
            "cluster_and_intersect_sideline_prederror",
            "cluster_and_intersect_sideline_harmlessness",
            "backchaining",
    ]:
        _test_approach(env_name="blocks",
                       approach_name="nsrt_learning",
                       try_solving=False,
                       sampler_learner="random",
                       strips_learner=strips_learner)


@longrun
def test_nsrt_learning_approach_longrun():
    """Tests for NSRTLearningApproach class."""
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


def test_saving_and_loading_atoms():
    """Test learning with saving and loading groudn atoms functionality."""
    # First, call the approach with load_atoms=False so that the
    # atoms get saved.
    approach = _test_approach(env_name="blocks",
                              approach_name="nsrt_learning",
                              try_solving=False,
                              save_atoms=True,
                              load_atoms=False)
    # Next, try to manually load these saved atoms.
    dataset_fname, _ = utils.create_dataset_filename_str(
        saving_ground_atoms=True, online_learning_cycle=None)
    assert os.path.exists(dataset_fname)
    with open(dataset_fname, "rb") as f:
        ground_atom_dataset_atoms = pkl.load(f)
    # Check that each of the loaded atoms is linked to one of the
    # environment predicates.
    env = create_new_env("blocks")
    for atoms_seq in ground_atom_dataset_atoms:
        assert all(atom.predicate in env.predicates for atom_set in atoms_seq
                   for atom in atom_set)
    # Next, call NSRT learning with load_atoms=True to test whether loading
    # works.
    approach = _test_approach(env_name="blocks",
                              approach_name="nsrt_learning",
                              try_solving=False,
                              load_atoms=True,
                              additional_settings={
                                  "compute_sidelining_objective_value": True,
                              })
    # Test assertions to check that learning occurs smoothly after loading.
    assert "sidelining_obj_num_plans_up_to_n" in approach.metrics
    assert "sidelining_obj_complexity" in approach.metrics
    assert approach.metrics["sidelining_obj_num_plans_up_to_n"] == 1.0
    assert approach.metrics["sidelining_obj_complexity"] == 34.0
    # Remove the file and test that error is thrown when loading
    # non-existent file.
    os.remove(dataset_fname)
    with pytest.raises(ValueError) as e:
        approach = _test_approach(env_name="blocks",
                                  approach_name="nsrt_learning",
                                  try_solving=False,
                                  load_atoms=True)
    assert "Cannot load ground atoms" in str(e)


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
    _test_approach(env_name="cover_handempty",
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
    # In painting, we learn operators that are different from the oracle ones.
    # The expected behavior is that the learned operators will have random
    # samplers, so we don't expected planning to necessarily work.
    _test_approach(env_name="painting",
                   approach_name="nsrt_learning",
                   sampler_learner="oracle",
                   try_solving=False,
                   num_train_tasks=3)


def test_full_oracle_no_data():
    """Test NSRTLearningApproach with oracle everything and no data."""
    # Oracle sampler learning should work (and be fast) in cover and blocks.
    # We can even check that the policy succeeds!
    _test_approach(env_name="cover",
                   approach_name="nsrt_learning",
                   strips_learner="oracle",
                   sampler_learner="oracle",
                   offline_data_method="demo",
                   check_solution=True,
                   num_train_tasks=0)


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
        "grammar_search_operator_complexity_weight": 1e-2,
        "grammar_search_max_predicates": 10,
        "grammar_search_predicate_cost_upper_bound": 6,
        "grammar_search_score_function": "prediction_error",
        "grammar_search_search_algorithm": "hill_climbing",
        "pretty_print_when_loading": True,
        "grammar_search_gbfs_num_evals": 1,
        "save_atoms": True
    }
    _test_approach(env_name="cover",
                   approach_name="grammar_search_invention",
                   excluded_predicates="Holding",
                   try_solving=False,
                   sampler_learner="random",
                   num_train_tasks=3,
                   additional_settings=additional_settings)
    # Now test loading.
    additional_settings.update({"load_atoms": True})
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
    additional_settings["load_atoms"] = False
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
    # This test still covers recomputing datastores in STRIPS learning when
    # options are unknown.
    _test_approach(env_name="stick_button",
                   approach_name="nsrt_learning",
                   strips_learner="oracle",
                   option_learner="direct_bc",
                   offline_data_method="demo",
                   num_train_tasks=1,
                   try_solving=False,
                   additional_settings=additional_settings)


def test_predicate_invention_with_oracle_clustering():
    """Test for predicate invention via clustering assuming access to clusters
    from ground truth operators."""
    additional_settings = {
        "grammar_search_pred_selection_approach": "clustering",
        "grammar_search_pred_clusterer": "oracle",
        "segmenter": "option_changes",
    }
    _test_approach(env_name="blocks",
                   num_train_tasks=1,
                   approach_name="grammar_search_invention",
                   strips_learner="oracle_clustering",
                   offline_data_method="demo+gt_operators",
                   solve_exceptions=ApproachFailure,
                   additional_settings=additional_settings)


def test_predicate_invention_with_custom_clustering():
    """Test for predicate invention with a custom clustering algorithm."""
    additional_settings = {
        "grammar_search_pred_selection_approach": "clustering",
        "grammar_search_pred_clusterer": "option-type-number-sample",
        "segmenter": "option_changes",
    }
    _test_approach(env_name="blocks",
                   num_train_tasks=10,
                   approach_name="grammar_search_invention",
                   strips_learner="cluster_and_intersect",
                   offline_data_method="demo",
                   solve_exceptions=ApproachFailure,
                   additional_settings=additional_settings)
    _test_approach(env_name="painting",
                   num_train_tasks=10,
                   approach_name="grammar_search_invention",
                   strips_learner="cluster_and_intersect",
                   offline_data_method="demo",
                   solve_exceptions=ApproachFailure,
                   additional_settings=additional_settings)
    _test_approach(env_name="repeated_nextto",
                   num_train_tasks=10,
                   approach_name="grammar_search_invention",
                   strips_learner="pnad_search",
                   offline_data_method="demo",
                   solve_exceptions=ApproachFailure,
                   additional_settings=additional_settings)
