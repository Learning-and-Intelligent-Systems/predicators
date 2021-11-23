"""Test cases for the grammar search invention approach.
"""

import pytest
import numpy as np
from predicators.src.approaches.grammar_search_invention_approach import \
    _PredicateGrammar, _count_positives_for_ops, _create_grammar
from predicators.src.envs import CoverEnv
from predicators.src.structs import Type, Predicate, STRIPSOperator, State, \
    Action


def test_predicate_grammar():
    """Tests for _PredicateGrammar class.
    """
    env = CoverEnv()
    types = env.types
    dataset = []
    base_grammar = _PredicateGrammar(types, dataset)
    with pytest.raises(NotImplementedError):
        base_grammar.generate(max_num=1)
    with pytest.raises(NotImplementedError):
        _create_grammar("not a real grammar name", types, dataset)
    holding_dummy_grammar = _create_grammar("holding_dummy", types, dataset)
    assert len(holding_dummy_grammar.generate(max_num=1)) == 1
    assert len(holding_dummy_grammar.generate(max_num=3)) == 2


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
