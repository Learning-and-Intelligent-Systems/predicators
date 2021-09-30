"""Test cases for utils.
"""

import pytest
from gym.spaces import Box  # type: ignore
from predicators.src.structs import State, Type, ParameterizedOption, \
    Predicate, Operator
from predicators.src import utils


def test_option_to_trajectory():
    """Tests for option_to_trajectory().
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state = State({cup: [0.5], plate: [1.0, 1.2]})
    def _simulator(s, a):
        ns = s.copy()
        assert a.shape == (1,)
        ns[cup][0] += a.item()
        return ns
    params_space = Box(0, 1, (1,))
    def _policy(_, p):
        return p
    def _initiable(_1, p):
        return p > 0.25
    def _terminal(s, _):
        return s[cup][0] > 9.9
    parameterized_option = ParameterizedOption(
        "Move", params_space, _policy, _initiable, _terminal)
    params = [0.1]
    option = parameterized_option.ground(params)
    with pytest.raises(AssertionError):
        # option is not initiable from start state
        utils.option_to_trajectory(state, _simulator, option,
                                   max_num_steps=5)
    params = [0.5]
    option = parameterized_option.ground(params)
    states, actions = utils.option_to_trajectory(
        state, _simulator, option, max_num_steps=100)
    assert len(actions) == len(states)-1 == 19
    states, actions = utils.option_to_trajectory(
        state, _simulator, option, max_num_steps=10)
    assert len(actions) == len(states)-1 == 10


def test_abstract():
    """Tests for abstract().
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    def _classifier1(state, objects):
        cup, plate = objects
        return state[cup][0] + state[plate][0] < 2
    pred1 = Predicate("On", [cup_type, plate_type], _classifier1)
    def _classifier2(state, objects):
        cup, _, plate = objects
        return state[cup][0] + state[plate][0] < -1
    pred2 = Predicate("Is", [cup_type, plate_type, plate_type], _classifier2)
    cup = cup_type("cup")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    state = State({cup: [0.5], plate1: [1.0, 1.2], plate2: [-9.0, 1.0]})
    atoms = utils.abstract(state, {pred1, pred2})
    assert len(atoms) == 3
    assert atoms == {pred1([cup, plate1]),
                     pred1([cup, plate2]),
                     # predicates with duplicate arguments are filtered out
                     pred2([cup, plate1, plate2])}


def test_operator_methods():
    """Tests for all_ground_operators(), extract_preds_and_types().
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate1_var = plate_type("?plate1")
    plate2_var = plate_type("?plate1")
    parameters = [cup_var, plate1_var, plate2_var]
    preconditions = {not_on([cup_var, plate1_var])}
    add_effects = {on([cup_var, plate1_var])}
    delete_effects = {not_on([cup_var, plate1_var])}
    params_space = Box(-10, 10, (2,))
    parameterized_option = ParameterizedOption("Pick",
        params_space, lambda s, p: 2*p, lambda s, p: True, lambda s, p: True)
    operator = Operator("PickOperator", parameters, preconditions, add_effects,
                        delete_effects, parameterized_option, _sampler=None)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_ops = utils.all_ground_operators(operator, objects)
    assert len(ground_ops) == 8
    all_obj = [op.objects for op in ground_ops]
    assert [cup1, plate1, plate1] in all_obj
    assert [cup1, plate2, plate1] in all_obj
    assert [cup2, plate1, plate1] in all_obj
    assert [cup2, plate2, plate1] in all_obj
    assert [cup1, plate1, plate2] in all_obj
    assert [cup1, plate2, plate2] in all_obj
    assert [cup2, plate1, plate2] in all_obj
    assert [cup2, plate2, plate2] in all_obj
    preds, types = utils.extract_preds_and_types({operator})
    assert preds == {"NotOn": not_on, "On": on}
    assert types == {"plate_type": plate_type, "cup_type": cup_type}
