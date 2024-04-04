"""Test cases for the grammar search invention approach."""

from operator import gt

import numpy as np
import pytest

from predicators.src import utils
from predicators.src.approaches.grammar_search_invention_approach import \
    _create_grammar, _DataBasedPredicateGrammar, _ForallClassifier, \
    _halving_constant_generator, _NegationClassifier, _PredicateGrammar, \
    _SingleAttributeCompareClassifier, \
    _SingleFeatureInequalitiesPredicateGrammar, _UnaryFreeForallClassifier
from predicators.src.envs.cover import CoverEnv
from predicators.src.settings import CFG
from predicators.src.structs import Dataset, LowLevelTrajectory, Object, \
    Predicate, State, Type


def test_predicate_grammar():
    """Tests for _PredicateGrammar class."""
    utils.reset_config({"env": "cover", "segmenter": "atom_changes"})
    env = CoverEnv()
    train_task = env.get_train_tasks()[0]
    state = train_task.init
    other_state = state.copy()
    robby = [o for o in state if o.type.name == "robot"][0]
    state.set(robby, "hand", 0.5)
    other_state.set(robby, "hand", 0.8)
    dataset = Dataset([
        LowLevelTrajectory([state, other_state],
                           [np.zeros(1, dtype=np.float32)])
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
    assert len(single_ineq_grammar.generate(max_num=1)) == 1
    feature_ranges = single_ineq_grammar._get_feature_ranges()  # pylint: disable=protected-access
    assert feature_ranges[robby.type]["hand"] == (0.5, 0.8)
    forall_grammar = _create_grammar(dataset, env.predicates)
    # Test edge case where there are no low-level features in the dataset.
    dummy_type = Type("dummy", [])
    dummy_obj = Object("dummy", dummy_type)
    dummy_state = State({dummy_obj: []})
    dummy_dataset = Dataset([
        LowLevelTrajectory([dummy_state, dummy_state],
                           [np.zeros(1, dtype=np.float32)])
    ])
    dummy_grammar = _SingleFeatureInequalitiesPredicateGrammar(dummy_dataset)
    assert len(dummy_grammar.generate(max_num=1)) == 0
    # There are only so many unique predicates possible under the grammar.
    # Non-unique predicates are pruned. Note that with a larger dataset,
    # more predicates would appear unique.
    assert len(forall_grammar.generate(max_num=100)) == 12
    # Test CFG.grammar_search_predicate_cost_upper_bound.
    default = CFG.grammar_search_predicate_cost_upper_bound
    utils.reset_config({"grammar_search_predicate_cost_upper_bound": 0})
    assert len(single_ineq_grammar.generate(max_num=10)) == 0
    # With an empty dataset, all predicates should look the same, so zero
    # predicates should be enumerated. The reason that it's zero and not one
    # is because the given predicates are considered too when determining
    # if a candidate predicate is unique.
    # Set a small upper bound so that this terminates quickly.
    utils.update_config({"grammar_search_predicate_cost_upper_bound": 2})
    empty_data_grammar = _create_grammar(Dataset([]), env.predicates)
    assert len(empty_data_grammar.generate(max_num=10)) == 0
    # Reset to default just in case.
    utils.update_config({"grammar_search_predicate_cost_upper_bound": default})
    # Test debug grammar.
    utils.reset_config({"env": "unittest"})
    utils.update_config({"grammar_search_use_handcoded_debug_grammar": True})
    debug_grammar = _create_grammar(dataset, set())
    assert len(debug_grammar.generate(max_num=10)) == 2
    utils.update_config({"grammar_search_use_handcoded_debug_grammar": False})


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
