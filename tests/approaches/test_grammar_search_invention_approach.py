"""Test cases for the grammar search invention approach.
"""

import pytest
import numpy as np
from predicators.src.approaches.grammar_search_invention_approach import \
    _PredicateGrammar, _count_positives_for_ops, _create_grammar, \
    _halving_constant_generator, _ForallClassifier
from predicators.src.envs import CoverEnv
from predicators.src.structs import Type, Predicate, STRIPSOperator, State, \
    Action
from predicators.src import utils


def test_predicate_grammar():
    """Tests for _PredicateGrammar class.
    """
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    train_task = env.get_train_tasks()[0]
    state = train_task.init
    other_state = state.copy()
    robby = [o for o in state if o.type.name == "robot"][0]
    state.set(robby, "hand", 0.5)
    other_state.set(robby, "hand", 0.8)
    dataset = [([state, other_state], [np.zeros(1, dtype=np.float32)])]
    base_grammar = _PredicateGrammar(dataset)
    assert base_grammar.types == env.types
    with pytest.raises(NotImplementedError):
        base_grammar.generate(max_num=1)
    with pytest.raises(NotImplementedError):
        _create_grammar("not a real grammar name", dataset)
    env = CoverEnv()
    holding_dummy_grammar = _create_grammar("holding_dummy", dataset)
    assert len(holding_dummy_grammar.generate(max_num=1)) == 1
    assert len(holding_dummy_grammar.generate(max_num=3)) == 2
    single_ineq_grammar = _create_grammar("single_feat_ineqs", dataset)
    assert len(single_ineq_grammar.generate(max_num=1)) == 1
    feature_ranges = single_ineq_grammar._get_feature_ranges()  # pylint: disable=protected-access
    assert feature_ranges[robby.type]["hand"] == (0.5, 0.8)
    candidates = single_ineq_grammar.generate(max_num=4)
    assert str(sorted(candidates)) == \
        "[(0.pose<=2.33), (0.pose>=2.33), (0.width<=19.0), (0.width>=19.0)]"


def test_count_positives_for_ops():
    """Tests for _count_positives_for_ops().
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    preconditions = {not_on([cup_var, plate_var])}
    add_effects = {on([cup_var, plate_var])}
    delete_effects = {not_on([cup_var, plate_var])}
    strips_operator = STRIPSOperator("Pick", parameters, preconditions,
                                     add_effects, delete_effects)
    cup = cup_type("cup")
    plate = plate_type("plate")
    state = State({cup: [0.5], plate: [1.0]})
    action = Action(np.zeros(1, dtype=np.float32))
    states = [state, state]
    actions = [action]
    strips_ops = {strips_operator}
    pruned_atom_data = [
        # Test empty sequence.
        ([state], [], [{on([cup, plate])}]),
        # Test not positive.
        (states, actions, [{on([cup, plate])}, set()]),
        # Test true positive.
        (states, actions, [{not_on([cup, plate])}, {on([cup, plate])}]),
        # Test false positive.
        (states, actions, [{not_on([cup, plate])}, set()]),
    ]

    num_true, num_false = _count_positives_for_ops(strips_ops, pruned_atom_data)
    assert num_true == 1
    assert num_false == 1


def test_halving_constant_generator():
    """Tests for _halving_constant_generator().
    """
    expected_sequence = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875]
    generator = _halving_constant_generator(0., 1.)
    for i, x in zip(range(len(expected_sequence)), generator):
        assert abs(expected_sequence[i] - x) < 1e-6
