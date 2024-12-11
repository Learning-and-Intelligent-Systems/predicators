"""Test cases for the grammar search invention approach."""

from operator import gt

import numpy as np
import pytest

from predicators import utils
from predicators.approaches.grammar_search_invention_approach import \
    GrammarSearchInventionApproach, _AttributeDiffCompareClassifier, \
    _create_grammar, _DataBasedPredicateGrammar, \
    _EuclideanAttributeDiffCompareClassifier, \
    _EuclideanDistancePredicateGrammar, \
    _FeatureDiffInequalitiesPredicateGrammar, _ForallClassifier, \
    _halving_constant_generator, _NegationClassifier, _PredicateGrammar, \
    _SingleAttributeCompareClassifier, \
    _SingleFeatureInequalitiesPredicateGrammar, _UnaryFreeForallClassifier
from predicators.datasets import create_dataset
from predicators.envs.cover import CoverEnv
from predicators.envs.stick_button import StickButtonMovementEnv
from predicators.envs.vlm_envs import IceTeaMakingEnv
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG
from predicators.structs import Action, Dataset, LowLevelTrajectory, Object, \
    Predicate, State, Type


@pytest.mark.parametrize("segmenter", ["atom_changes", "contacts"])
def test_predicate_grammar(segmenter):
    """Tests for _PredicateGrammar class."""
    utils.reset_config({"env": "cover", "segmenter": segmenter})
    env = CoverEnv()
    train_task = env.get_train_tasks()[0].task
    state = train_task.init
    other_state = state.copy()
    robby = [o for o in state if o.type.name == "robot"][0]
    block = [o for o in state if o.name == "block0"][0]
    state.set(robby, "hand", 0.5)
    other_state.set(robby, "hand", 0.8)
    state.set(block, "grasp", -1)
    other_state.set(block, "grasp", 1)
    dataset = Dataset([
        LowLevelTrajectory([state, other_state],
                           [Action(np.zeros(1, dtype=np.float32))])
    ])
    base_grammar = _PredicateGrammar()
    assert not base_grammar.generate(max_num=0)
    with pytest.raises(NotImplementedError):
        base_grammar.generate(max_num=1)
    data_based_grammar = _DataBasedPredicateGrammar(dataset)
    assert data_based_grammar.types == env.types
    with pytest.raises(NotImplementedError):
        data_based_grammar.generate(max_num=1)
    env = CoverEnv()
    single_ineq_grammar = _SingleFeatureInequalitiesPredicateGrammar(dataset)
    diff_ineq_grammar = _FeatureDiffInequalitiesPredicateGrammar(dataset)
    euclidean_grammar = _EuclideanDistancePredicateGrammar(
        dataset, "x", "y", "x", "y")
    assert len(single_ineq_grammar.generate(max_num=1)) == 1
    sing_feature_ranges = single_ineq_grammar._get_feature_ranges()  # pylint: disable=protected-access
    assert sing_feature_ranges[robby.type]["hand"] == (0.5, 0.8)
    assert sing_feature_ranges[block.type]["grasp"] == (-1, 1)
    doub_feature_ranges = diff_ineq_grammar._get_feature_ranges()  # pylint: disable=protected-access
    assert doub_feature_ranges[robby.type]["hand"] == (0.5, 0.8)
    assert doub_feature_ranges[block.type]["grasp"] == (-1, 1)
    euclidean_feature_ranges = euclidean_grammar._get_feature_ranges()  # pylint: disable=protected-access
    assert euclidean_feature_ranges[block.type]["is_block"] == (1.0, 1.0)
    assert euclidean_feature_ranges[robby.type]["hand"] == (0.5, 0.8)

    # Generate from the diff ineq grammar and verify that the number of
    # candidates generated is under the limit.
    preds = diff_ineq_grammar.generate(max_num=100)
    assert len(preds) <= 100

    forall_grammar = _create_grammar(dataset, env.predicates)
    # Test edge case where there are no low-level features in the dataset.
    dummy_type = Type("dummy", [])
    dummy_obj = Object("dummy", dummy_type)
    dummy_state = State({dummy_obj: []})
    dummy_dataset = Dataset([
        LowLevelTrajectory([dummy_state, dummy_state],
                           [np.zeros(1, dtype=np.float32)])
    ])
    dummy_sing_grammar = _SingleFeatureInequalitiesPredicateGrammar(
        dummy_dataset)
    dummy_doub_grammar = _FeatureDiffInequalitiesPredicateGrammar(
        dummy_dataset)
    dummy_euc_grammar = _EuclideanDistancePredicateGrammar(
        dummy_dataset, "x", "y", "x", "y")
    assert len(dummy_sing_grammar.generate(max_num=1)) == 0
    assert len(dummy_doub_grammar.generate(max_num=1)) == 0
    assert len(dummy_euc_grammar.generate(max_num=1)) == 0
    # There are only so many unique predicates possible under the grammar.
    # Non-unique predicates are pruned. Note that with a larger dataset,
    # more predicates would appear unique.
    assert len(forall_grammar.generate(max_num=100)) == 12
    # Test the same thing, but using a forall grammar with
    # on 2-arity predicates.
    utils.reset_config({
        "grammar_search_grammar_use_diff_features": True,
        "segmenter": segmenter,
        "env": "cover",
    })
    forall_grammar = _create_grammar(dataset, env.predicates)
    assert len(forall_grammar.generate(max_num=100)) == 9
    # Test CFG.grammar_search_predicate_cost_upper_bound.
    default = CFG.grammar_search_predicate_cost_upper_bound
    utils.reset_config({"grammar_search_predicate_cost_upper_bound": 0})
    assert len(single_ineq_grammar.generate(max_num=10)) == 0
    assert len(diff_ineq_grammar.generate(max_num=10)) == 0
    # With an empty dataset, all predicates should look the same, so zero
    # predicates should be enumerated. The reason that it's zero and not one
    # is because the given predicates are considered too when determining
    # if a candidate predicate is unique.
    # Set a small upper bound so that this terminates quickly.
    utils.update_config({"grammar_search_predicate_cost_upper_bound": 3})
    empty_data_grammar = _create_grammar(Dataset([]), env.predicates)
    assert len(empty_data_grammar.generate(max_num=10)) == 0
    # Reset to default just in case.
    utils.update_config({"grammar_search_predicate_cost_upper_bound": default})
    # Test debug grammar.
    utils.reset_config({"env": "unittest"})
    utils.update_config({"grammar_search_use_handcoded_debug_grammar": True})
    debug_grammar = _create_grammar(dataset, set())
    assert len(debug_grammar.generate(max_num=10)) == 3
    utils.update_config({"grammar_search_use_handcoded_debug_grammar": False})


def test_labelled_atoms_invention():
    """Tests for _PredicateGrammar class."""
    utils.reset_config({
        "env": "cover",
        "offline_data_method": "demo+labelled_atoms"
    })
    env = CoverEnv()
    train_task = env.get_train_tasks()[0].task
    state = train_task.init
    other_state = state.copy()
    robby = [o for o in state if o.type.name == "robot"][0]
    block = [o for o in state if o.name == "block0"][0]
    state.set(robby, "hand", 0.5)
    other_state.set(robby, "hand", 0.8)
    state.set(block, "grasp", -1)
    other_state.set(block, "grasp", 1)
    preds = env.predicates
    assert len(preds) == 5
    ground_atoms = []
    for s in [state, other_state]:
        curr_state_atoms = utils.abstract(s, preds)
        ground_atoms.append(curr_state_atoms)

    ll_trajs = [
        LowLevelTrajectory([state, other_state],
                           [Action(np.zeros(1, dtype=np.float32))])
    ]
    dataset = Dataset(ll_trajs, [ground_atoms])

    approach = GrammarSearchInventionApproach(env.predicates,
                                              get_gt_options(env.get_name()),
                                              env.types, env.action_space,
                                              [train_task])

    with pytest.raises(AssertionError):
        # The below command should fail because even though it should be able
        # to extract predicates from the dataset, the trajectories' actions
        # don't have options that can be used.
        approach.learn_from_offline_dataset(dataset)


def test_invention_from_txt_file():
    """Test loading a dataset from a txt file."""
    utils.reset_config({
        "env":
        "ice_tea_making",
        "num_train_tasks":
        1,
        "num_test_tasks":
        0,
        "offline_data_method":
        "demo+labelled_atoms",
        "data_dir":
        "tests/datasets/mock_vlm_datasets",
        "handmade_demo_filename":
        "ice_tea_making__demo+labelled_atoms__manual__1.txt"
    })
    env = IceTeaMakingEnv()
    train_tasks = env.get_train_tasks()
    predicates, _ = utils.parse_config_excluded_predicates(env)
    loaded_dataset = create_dataset(env, train_tasks,
                                    get_gt_options(env.get_name()), predicates)
    approach = GrammarSearchInventionApproach(env.goal_predicates,
                                              get_gt_options(env.get_name()),
                                              env.types, env.action_space,
                                              train_tasks)
    approach.learn_from_offline_dataset(loaded_dataset)
    # The ice_tea_making__demo+labelled_atoms__manual__1.txt happens to
    # set all atoms to True at all timesteps, and so we expect predicate
    # invention to not select any of the predicates (only select the goal)
    # predicates.
    assert len(approach._get_current_predicates()) == 1  # pylint:disable=protected-access
    assert approach._get_current_predicates() == env.goal_predicates  # pylint:disable=protected-access


def test_no_select_invention():
    """Test loading a dataset from a txt file using the no_select method."""
    utils.reset_config({
        "env": "ice_tea_making",
        "num_train_tasks": 1,
        "num_test_tasks": 0,
        "offline_data_method": "demo+labelled_atoms",
        "data_dir": "tests/datasets/mock_vlm_datasets",
        "handmade_demo_filename":
        "ice_tea_making__demo+labelled_atoms__manual__1.txt",
        "grammar_search_pred_selection_approach": "no_select",
        "disable_harmlessness_check": True
    })
    env = IceTeaMakingEnv()
    train_tasks = env.get_train_tasks()
    predicates, _ = utils.parse_config_excluded_predicates(env)
    loaded_dataset = create_dataset(env, train_tasks,
                                    get_gt_options(env.get_name()), predicates)
    approach = GrammarSearchInventionApproach(env.goal_predicates,
                                              get_gt_options(env.get_name()),
                                              env.types, env.action_space,
                                              train_tasks)
    approach.learn_from_offline_dataset(loaded_dataset)
    # The ice_tea_making__demo+labelled_atoms__manual__1.txt happens to
    # set all atoms to True at all timesteps. Normal predicate invention
    # would only select the goal predicate, but in our case, we get more!
    assert len(approach._get_current_predicates()) == 7  # pylint:disable=protected-access


def test_geo_and_vlm_invention():
    """Test constructing an atom dataset with both geo and vlm predicates."""
    utils.reset_config({
        "env": "ice_tea_making",
        "num_train_tasks": 1,
        "num_test_tasks": 0,
        "offline_data_method": "geo_and_demo+labelled_atoms",
        "data_dir": "tests/datasets/mock_vlm_datasets",
        "handmade_demo_filename":
        "ice_tea_making__demo+labelled_atoms__manual__1.txt",
        "grammar_search_select_all_debug": True,
        "grammar_search_use_handcoded_debug_grammar": True
    })
    env = IceTeaMakingEnv()
    train_tasks = env.get_train_tasks()
    predicates, _ = utils.parse_config_excluded_predicates(env)
    loaded_dataset = create_dataset(env, train_tasks,
                                    get_gt_options(env.get_name()), predicates)
    approach = GrammarSearchInventionApproach(env.goal_predicates,
                                              get_gt_options(env.get_name()),
                                              env.types, env.action_space,
                                              train_tasks)
    approach.learn_from_offline_dataset(loaded_dataset)
    # The ice_tea_making__demo+labelled_atoms__manual__1.txt happens to
    # set all atoms to True at all timesteps, and so we expect predicate
    # invention to not select any of the predicates (only select the goal)
    # predicates.
    # If you investigate the atom_dataset created inside
    # learn_from_offline_dataset() you'll see some grammar-based predicates
    # invented that are based on the DummyGoal, but they don't get selected.
    # A better test would alter the dataset such that some grammar-based
    # predicates actually get selected, so we can verify that geo + vlm
    # predicate invention works more explicitly.
    assert len(approach._get_current_predicates()) == 1  # pylint:disable=protected-access
    assert approach._get_current_predicates() == env.goal_predicates  # pylint:disable=protected-access


def test_geo_only_invention_with_all_vlm():
    """Test invention where we only want to invent geo predicates, and use all
    vlm predicates directly."""
    utils.reset_config({
        "env": "ice_tea_making",
        "num_train_tasks": 1,
        "num_test_tasks": 0,
        "offline_data_method": "geo_and_demo+labelled_atoms",
        "data_dir": "tests/datasets/mock_vlm_datasets",
        "handmade_demo_filename":
        "ice_tea_making__demo+labelled_atoms__manual__1.txt",
        "grammar_search_select_all_debug": True,
        "grammar_search_use_handcoded_debug_grammar": True,
        "grammar_search_invent_geo_predicates_only": True,
        "disable_harmlessness_check": True
    })
    env = IceTeaMakingEnv()
    train_tasks = env.get_train_tasks()
    predicates, _ = utils.parse_config_excluded_predicates(env)
    loaded_dataset = create_dataset(env, train_tasks,
                                    get_gt_options(env.get_name()), predicates)
    approach = GrammarSearchInventionApproach(env.goal_predicates,
                                              get_gt_options(env.get_name()),
                                              env.types, env.action_space,
                                              train_tasks)
    approach.learn_from_offline_dataset(loaded_dataset)
    # There are 6 predicates aside from the goal predicate in this txt file,
    # so we expect the total number of predicates to be 7.
    assert len(approach._get_current_predicates()) == 7  # pylint:disable=protected-access
    assert env.goal_predicates.issubset(approach._get_current_predicates())  # pylint:disable=protected-access


def test_select_all_debug_predicates():
    """Test selecting all debug predicates from the debug grammar."""
    utils.reset_config({
        "env": "ice_tea_making",
        "num_train_tasks": 1,
        "num_test_tasks": 0,
        "offline_data_method": "geo_and_demo+labelled_atoms",
        "data_dir": "tests/datasets/mock_vlm_datasets",
        "handmade_demo_filename":
        "ice_tea_making__demo+labelled_atoms__manual__1.txt",
        "grammar_search_select_all_debug": True,
        "grammar_search_use_handcoded_debug_grammar": True
    })
    env = IceTeaMakingEnv()
    train_tasks = env.get_train_tasks()
    predicates, _ = utils.parse_config_excluded_predicates(env)
    loaded_dataset = create_dataset(env, train_tasks,
                                    get_gt_options(env.get_name()), predicates)
    approach = GrammarSearchInventionApproach(env.goal_predicates,
                                              get_gt_options(env.get_name()),
                                              env.types, env.action_space,
                                              train_tasks)
    approach.learn_from_offline_dataset(loaded_dataset)
    # Because we have not listed any debug predicates for this environment,
    # no predicates will be selected.
    # A better test would test this on some environment that actually has both
    # geometric and VLM debug predicates.
    assert len(approach._learned_predicates) == 0  # pylint:disable=protected-access


def test_euclidean_grammar():
    """Tests for the EuclideanGrammar."""
    utils.reset_config({"env": "stick_button_move"})
    env = StickButtonMovementEnv()
    train_task = env.get_train_tasks()[0].task
    state = train_task.init
    other_state = state.copy()
    robby = [o for o in state if o.type.name == "robot"][0]
    curr_x = state.get(robby, "x")
    curr_y = state.get(robby, "y")
    other_state.set(robby, "x", curr_x + 0.05)
    other_state.set(robby, "y", curr_y + 0.05)
    dataset = Dataset([
        LowLevelTrajectory([state, other_state],
                           [Action(np.zeros(4, dtype=np.float32))])
    ])
    utils.reset_config({
        "grammar_search_grammar_use_euclidean_dist": True,
        "segmenter": "atom_changes",
    })
    grammar = _create_grammar(dataset, env.predicates)
    assert len(grammar.generate(max_num=100)) == 28
    utils.reset_config({
        "grammar_search_grammar_use_euclidean_dist": False,
        "segmenter": "contacts"
    })


def test_halving_constant_generator():
    """Tests for _halving_constant_generator()."""
    expected_constants = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875]
    expected_costs = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    generator = _halving_constant_generator(0., 1.)
    for (expected_constant, expected_cost, (constant, cost)) in \
        zip(expected_constants, expected_costs, generator):
        assert abs(expected_constant - constant) < 1e-6
        assert abs(expected_cost - cost) < 1e-6


def test_single_attribute_compare_classifier():
    """Tests for _SingleAttributeCompareClassifier."""
    cup_type = Type("cup_type", ["feat1"])
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    cup3 = cup_type("cup3")
    classifier = _SingleAttributeCompareClassifier(2, cup_type, "feat1", 1.0,
                                                   5, gt, ">")
    state0 = State({cup1: [0.0], cup2: [1.0], cup3: [2.0]})
    assert not classifier(state0, [cup1])
    assert not classifier(state0, [cup2])
    assert classifier(state0, [cup3])
    assert str(classifier) == "((2:cup_type).feat1>[idx 5]1.0)"
    assert classifier.pretty_str() == ("?z:cup_type", "(?z.feat1 > 1.0)")


def test_diff_attribute_compare_classifier():
    """Tests for _AttributeDiffCompareClassifier."""
    cup_type = Type("cup_type", ["feat1"])
    saucer_type = Type("saucer_type", ["feat1"])
    cup1 = cup_type("cup1")
    saucer1 = saucer_type("saucer1")

    classifier = _AttributeDiffCompareClassifier(2, cup_type, "feat1", 0,
                                                 saucer_type, "feat1", 1.0, 5,
                                                 gt, ">")
    state0 = State({cup1: [0.0], saucer1: [2.0]})
    assert classifier(state0, [cup1, saucer1])
    assert str(classifier
               ) == "(|(2:cup_type).feat1 - (0:saucer_type).feat1|>[idx 5]1.0)"
    assert classifier.pretty_str() == ('?z:cup_type, ?x:saucer_type',
                                       '(|?z.feat1 - ?x.feat1| > 1.0)')


def test_forall_classifier():
    """Tests for _ForallClassifier()."""
    cup_type = Type("cup_type", ["feat1"])
    pred = Predicate("Pred", [cup_type],
                     lambda s, o: s.get(o[0], "feat1") > 0.5)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    state0 = State({cup1: [0.], cup2: [0.]})
    state1 = State({cup1: [0.], cup2: [1.]})
    state2 = State({cup1: [1.], cup2: [1.]})
    classifier = _ForallClassifier(pred)
    assert not classifier(state0, [])
    assert not classifier(state1, [])
    assert classifier(state2, [])
    assert str(classifier) == "Forall[0:cup_type].[Pred(0)]"
    assert classifier.pretty_str() == ("", "(∀ ?x:cup_type . Pred(?x))")


def test_unary_free_forall_classifier():
    """Tests for _UnaryFreeForallClassifier()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    cup0 = cup_type("cup0")
    plate0 = plate_type("plate0")
    state0 = State({cup0: [0.], plate0: [0.]})
    classifier0 = _UnaryFreeForallClassifier(on, 0)
    assert classifier0(state0, [cup0])
    assert str(classifier0) == "Forall[1:plate_type].[On(0,1)]"
    assert classifier0.pretty_str() == ("?x:cup_type",
                                        "(∀ ?y:plate_type . On(?x, ?y))")
    classifier1 = _UnaryFreeForallClassifier(on, 1)
    assert classifier1(state0, [plate0])
    assert str(classifier1) == "Forall[0:cup_type].[On(0,1)]"
    assert classifier1.pretty_str() == ("?y:plate_type",
                                        "(∀ ?x:cup_type . On(?x, ?y))")
    noton_classifier = _NegationClassifier(on)
    noton = Predicate(str(noton_classifier), [cup_type, plate_type],
                      noton_classifier)
    classifier2 = _UnaryFreeForallClassifier(noton, 0)
    assert not classifier2(state0, [cup0])
    assert str(classifier2) == "Forall[1:plate_type].[NOT-On(0,1)]"
    assert classifier2.pretty_str() == ("?x:cup_type",
                                        "(∀ ?y:plate_type . ¬On(?x, ?y))")
    forallnoton = Predicate(str(classifier2), [cup_type], classifier2)
    classifier3 = _NegationClassifier(forallnoton)
    assert classifier3(state0, [cup0])
    assert str(classifier3) == "NOT-Forall[1:plate_type].[NOT-On(0,1)]"
    assert classifier3.pretty_str() == ("?x:cup_type",
                                        "¬(∀ ?y:plate_type . ¬On(?x, ?y))")


def test_euclidean_classifier_and_grammar():
    """Tests for the _EuclideanAttributeDiffCompareClassifier and certain
    aspects of the euclidean grammar."""
    a_type = Type("a_type", ["x", "y"])
    b_type = Type("b_type", ["x", "y"])
    classifier0 = _EuclideanAttributeDiffCompareClassifier(
        0, a_type, "x", "y", 1, b_type, "x", "y", 1.0, 0, gt, ">")
    assert classifier0.pretty_str() == (
        '?x:a_type, ?y:b_type', '((?x.x - ?y.x)^2  + ((?x.y - ?y.y)^2 > 1.0)')


def test_unrecognized_clusterer():
    """Tests that a dummy name for the 'clusterer' argument will trigger a
    failure.

    Note that most of the coverage for the clusterer comes from
    test_nsrt_learning_approach.py.
    """
    utils.update_config({
        "env": "cover",
        "segmenter": "atom_changes",
        "grammar_search_pred_selection_approach": "clustering",
        "grammar_search_pred_clusterer": "NotARealClusterer"
    })
    env = CoverEnv()
    train_task = env.get_train_tasks()[0].task
    state = train_task.init
    other_state = state.copy()
    robby = [o for o in state if o.type.name == "robot"][0]
    block = [o for o in state if o.name == "block0"][0]
    state.set(robby, "hand", 0.5)
    other_state.set(robby, "hand", 0.8)
    state.set(block, "grasp", -1)
    other_state.set(block, "grasp", 1)
    dataset = Dataset([
        LowLevelTrajectory([state, other_state],
                           [Action(np.zeros(1, dtype=np.float32))])
    ])
    approach = GrammarSearchInventionApproach(env.predicates,
                                              get_gt_options(env.get_name()),
                                              env.types, env.action_space,
                                              [train_task])
    with pytest.raises(NotImplementedError) as e:
        approach.learn_from_offline_dataset(dataset)
    assert "Unrecognized clusterer" in str(e)
