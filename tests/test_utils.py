"""Test cases for utils.
"""

import os
import pytest
import numpy as np
from gym.spaces import Box
from predicators.src.structs import State, Type, ParameterizedOption, \
    Predicate, Operator, Action
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
        assert a.arr.shape == (1,)
        ns[cup][0] += a.arr.item()
        return ns
    params_space = Box(0, 1, (1,))
    def _policy(_1, _2, p):
        return Action(p)
    def _initiable(_1, _2, p):
        return p > 0.25
    def _terminal(s, _1, _2):
        return s[cup][0] > 9.9
    parameterized_option = ParameterizedOption(
        "Move", [], params_space, _policy, _initiable, _terminal)
    params = [0.1]
    option = parameterized_option.ground([], params)
    with pytest.raises(AssertionError):
        # option is not initiable from start state
        utils.option_to_trajectory(state, _simulator, option,
                                   max_num_steps=5)
    params = [0.5]
    option = parameterized_option.ground([], params)
    states, actions = utils.option_to_trajectory(
        state, _simulator, option, max_num_steps=100)
    assert len(actions) == len(states)-1 == 19
    states, actions = utils.option_to_trajectory(
        state, _simulator, option, max_num_steps=10)
    assert len(actions) == len(states)-1 == 10


def test_action_to_option_trajectory():
    """Tests for action_to_option_trajectory.
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state = State({cup: [0.5], plate: [1.0, 1.2]})
    def _simulator(s, a):
        ns = s.copy()
        assert a.arr.shape == (1,)
        ns[cup][0] += a.arr.item()
        return ns
    params_space = Box(0, 1, (1,))
    def _policy(_1, _2, p):
        return Action(p)
    def _initiable(_1, _2, p):
        return p > 0.25
    def _terminal(s, _1, _2):
        return s[cup][0] > 9.9
    parameterized_option = ParameterizedOption(
        "Move", [], params_space, _policy, _initiable, _terminal)
    params = [0.5]
    option = parameterized_option.ground([], params)
    act_traj = utils.option_to_trajectory(
        state, _simulator, option, max_num_steps=100)
    opt_traj = utils.action_to_option_trajectory(act_traj)
    assert len(opt_traj) == 2
    assert repr(opt_traj[1][0]) == (
        "_Option(name='Move', objects=[], "
        "params=array([0.5], dtype=float32))")
    state_only_traj = (act_traj[0][:1], [])
    opt_traj = utils.action_to_option_trajectory(state_only_traj)
    assert len(opt_traj[0]) == 1
    assert len(opt_traj[1]) == 0
    params = [0.6]
    other_option = parameterized_option.ground([], params)
    other_act_traj = utils.option_to_trajectory(
        act_traj[0][-1], _simulator, other_option, max_num_steps=100)
    states = act_traj[0] + other_act_traj[0]
    actions = act_traj[1] + other_act_traj[1]
    opt_traj = utils.action_to_option_trajectory((states, actions))
    assert len(opt_traj) == 2
    assert len(opt_traj[1]) == 2
    assert repr(opt_traj[1][0]) == (
        "_Option(name='Move', objects=[], "
        "params=array([0.5], dtype=float32))")
    assert repr(opt_traj[1][1]) == (
        "_Option(name='Move', objects=[], "
        "params=array([0.6], dtype=float32))")
    assert len(opt_traj[0]) == 3


def test_strip_predicate():
    """Test for strip_predicate().
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    def _classifier1(state, objects):
        cup, plate = objects
        return state[cup][0] + state[plate][0] < 2
    pred = Predicate("On", [cup_type, plate_type], _classifier1)
    cup = cup_type("cup")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    state = State({cup: [0.5], plate1: [1.0, 1.2], plate2: [-9.0, 1.0]})
    pred_stripped = utils.strip_predicate(pred)
    assert pred.name == pred_stripped.name
    assert pred.types == pred_stripped.types
    assert pred.holds(state, (cup, plate1))
    assert pred.holds(state, (cup, plate2))
    assert not pred_stripped.holds(state, (cup, plate1))
    assert not pred_stripped.holds(state, (cup, plate2))


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


def test_unify():
    """Tests for unify().
    """
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    var0 = cup_type("?var0")
    var1 = cup_type("?var1")
    var2 = cup_type("?var2")
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: True)
    pred1 = Predicate("Pred1", [cup_type, cup_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type], lambda s, o: True)

    kb0 = frozenset({pred0([cup0])})
    q0 = frozenset({pred0([var0])})
    found, assignment = utils.unify(kb0, q0)
    assert found
    assert assignment == {cup0: var0}

    q1 = frozenset({pred0([var0]), pred0([var1])})
    found, assignment = utils.unify(kb0, q1)
    assert not found
    assert assignment == {}

    kb1 = frozenset({pred0([cup0]), pred0([cup1])})
    found, assignment = utils.unify(kb1, q0)
    assert not found  # different number of predicates/objects
    assert assignment == {}

    kb2 = frozenset({pred0([cup0]), pred2([cup2])})
    q2 = frozenset({pred0([var0]), pred2([var2])})
    found, assignment = utils.unify(kb2, q2)
    assert found
    assert assignment == {cup0: var0, cup2: var2}

    kb3 = frozenset({pred0([cup0])})
    q3 = frozenset({pred0([var0]), pred2([var2])})
    found, assignment = utils.unify(kb3, q3)
    assert not found
    assert assignment == {}

    kb4 = frozenset({pred1([cup0, cup1]), pred1([cup1, cup2])})
    q4 = frozenset({pred1([var0, var1])})
    found, assignment = utils.unify(kb4, q4)
    assert not found  # different number of predicates
    assert assignment == {}

    kb5 = frozenset({pred0([cup2]), pred1([cup0, cup1]), pred1([cup1, cup2])})
    q5 = frozenset({pred1([var0, var1]), pred0([var1]), pred0([var0])})
    found, assignment = utils.unify(kb5, q5)
    assert not found
    assert assignment == {}

    kb6 = frozenset({pred0([cup0]), pred2([cup1]), pred1([cup0, cup2]),
                     pred1([cup2, cup1])})
    q6 = frozenset({pred0([var0]), pred2([var1]), pred1([var0, var1])})
    found, assignment = utils.unify(kb6, q6)
    assert not found
    assert assignment == {}

    kb7 = frozenset({pred0([cup0]), pred2([cup1])})
    q7 = frozenset({pred0([var0]), pred2([var0])})
    found, assignment = utils.unify(kb7, q7)
    assert not found  # different number of objects
    assert assignment == {}

    kb8 = frozenset({pred0([cup0]), pred2([cup0])})
    q8 = frozenset({pred0([var0]), pred2([var0])})
    found, assignment = utils.unify(kb8, q8)
    assert found
    assert assignment == {cup0: var0}

    kb9 = frozenset({pred1([cup0, cup1]), pred1([cup1, cup2]), pred2([cup0])})
    q9 = frozenset({pred1([var0, var1]), pred1([var2, var0]), pred2([var0])})
    found, assignment = utils.unify(kb9, q9)
    assert not found
    assert assignment == {}


def test_find_substitution():
    """Tests for find_substitution().
    """
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    var0 = cup_type("?var0")
    var1 = cup_type("?var1")
    var2 = cup_type("?var2")
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: True)
    pred1 = Predicate("Pred1", [cup_type, cup_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type], lambda s, o: True)

    kb0 = [pred0([cup0])]
    q0 = [pred0([var0])]
    found, assignment = utils.find_substitution(kb0, q0)
    assert found
    assert assignment == {var0: cup0}

    q1 = [pred0([var0]), pred0([var1])]
    found, assignment = utils.find_substitution(kb0, q1)
    assert not found
    assert assignment == {}

    q1 = [pred0([var0]), pred0([var1])]
    found, assignment = utils.find_substitution(kb0, q1,
                                                allow_redundant=True)
    assert found
    assert assignment == {var0: cup0, var1: cup0}

    kb1 = [pred0([cup0]), pred0([cup1])]
    found, assignment = utils.find_substitution(kb1, q0)
    assert found
    assert assignment == {var0: cup0}

    kb2 = [pred0([cup0]), pred2([cup2])]
    q2 = [pred0([var0]), pred2([var2])]
    found, assignment = utils.find_substitution(kb2, q2)
    assert found
    assert assignment == {var0: cup0, var2: cup2}

    kb3 = [pred0([cup0])]
    q3 = [pred0([var0]), pred2([var2])]
    found, assignment = utils.find_substitution(kb3, q3)
    assert not found
    assert assignment == {}

    kb4 = [pred1([cup0, cup1]), pred1([cup1, cup2])]
    q4 = [pred1([var0, var1])]
    found, assignment = utils.find_substitution(kb4, q4)
    assert found
    assert assignment == {var0: cup0, var1: cup1}

    kb5 = [pred0([cup2]), pred1([cup0, cup1]), pred1([cup1, cup2])]
    q5 = [pred1([var0, var1]), pred0([var1]), pred0([var0])]
    found, assignment = utils.find_substitution(kb5, q5)
    assert not found
    assert assignment == {}

    kb6 = [pred0([cup0]), pred2([cup1]), pred1([cup0, cup2]),
           pred1([cup2, cup1])]
    q6 = [pred0([var0]), pred2([var1]), pred1([var0, var1])]
    found, assignment = utils.find_substitution(kb6, q6)
    assert not found
    assert assignment == {}

    kb7 = [pred1([cup0, cup0])]
    q7 = [pred1([var0, var0])]
    found, assignment = utils.find_substitution(kb7, q7)
    assert found
    assert assignment == {var0: cup0}

    kb8 = [pred1([cup0, cup0])]
    q8 = [pred1([var0, var1])]
    found, assignment = utils.find_substitution(kb8, q8)
    assert not found
    assert assignment == {}

    found, assignment = utils.find_substitution(kb8, q8,
                                                allow_redundant=True)
    assert found
    assert assignment == {var0: cup0, var1: cup0}

    kb9 = [pred1([cup0, cup1])]
    q9 = [pred1([var0, var0])]
    found, assignment = utils.find_substitution(kb9, q9)
    assert not found
    assert assignment == {}

    found, assignment = utils.find_substitution(kb9, q9,
                                                allow_redundant=True)
    assert not found
    assert assignment == {}

    kb10 = [pred1([cup0, cup1]), pred1([cup1, cup0])]
    q10 = [pred1([var0, var1]), pred1([var0, var2])]
    found, assignment = utils.find_substitution(kb10, q10)
    assert not found
    assert assignment == {}

    kb11 = [pred1([cup0, cup1]), pred1([cup1, cup0])]
    q11 = [pred1([var0, var1]), pred1([var1, var0])]
    found, assignment = utils.find_substitution(kb11, q11)
    assert found
    assert assignment == {var0: cup0, var1: cup1}

    plate_type = Type("plate_type", ["feat1"])
    plate0 = plate_type("plate0")
    var3 = plate_type("?var3")
    pred4 = Predicate("Pred4", [plate_type], lambda s, o: True)
    pred5 = Predicate("Pred5", [cup_type, plate_type], lambda s, o: True)

    kb12 = [pred4([plate0])]
    q12 = [pred0([var0])]
    found, assignment = utils.find_substitution(kb12, q12)
    assert not found
    assert assignment == {}

    kb13 = [pred4([plate0]), pred5([plate0, cup0])]
    q13 = [pred4([var3]), pred5([var3, var0])]
    found, assignment = utils.find_substitution(kb13, q13)
    assert found
    assert assignment == {var3: plate0, var0: cup0}


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
    parameterized_option = ParameterizedOption(
        "Pick", [cup_type], params_space, lambda s, o, p: 2*p,
        lambda s, o, p: True, lambda s, o, p: True)
    operator = Operator("PickOperator", parameters, preconditions, add_effects,
                        delete_effects, parameterized_option, [parameters[0]],
                        _sampler=None)
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


def test_static_operator_filtering():
    """Tests for filter_static_operators().
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    pred1 = Predicate("Pred1", [cup_type, plate_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type, plate_type], lambda s, o: True)
    pred3 = Predicate("Pred3", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    # pred1 is static, pred2/pred3 are not
    preconditions1 = {pred1([cup_var, plate_var])}
    add_effects1 = {pred2([cup_var, plate_var])}
    delete_effects1 = {}
    preconditions2 = {pred1([cup_var, plate_var])}
    add_effects2 = {}
    delete_effects2 = {pred3([cup_var, plate_var])}
    operator1 = Operator("Pick", parameters, preconditions1, add_effects1,
                         delete_effects1, option=None, option_vars=[],
                         _sampler=None)
    operator2 = Operator("Place", parameters, preconditions2, add_effects2,
                         delete_effects2, option=None, option_vars=[],
                         _sampler=None)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_ops = (utils.all_ground_operators(operator1, objects) |
                  utils.all_ground_operators(operator2, objects))
    assert len(ground_ops) == 8
    atoms = {pred1([cup1, plate1]), pred1([cup1, plate2]),
             pred2([cup1, plate1]), pred2([cup1, plate2]),
             pred2([cup2, plate1]), pred2([cup2, plate2])}
    assert utils.atom_to_tuple(pred1([cup1, plate1])) == (
        "Pred1", "cup1:cup_type", "plate1:plate_type")
    with pytest.raises(AttributeError):
        # Can't call atom_to_tuple on a lifted atom.
        utils.atom_to_tuple(pred1([cup_var, plate_var]))
    assert utils.atoms_to_tuples(
        {pred1([cup1, plate1]), pred2([cup2, plate2])}) == {
            ("Pred1", "cup1:cup_type", "plate1:plate_type"),
            ("Pred2", "cup2:cup_type", "plate2:plate_type")}
    # All operators with cup2 in the args should get filtered out,
    # since pred1 doesn't hold on cup2.
    ground_ops = utils.filter_static_operators(ground_ops, atoms)
    all_obj = [(op.name, op.objects) for op in ground_ops]
    assert ("Pick", [cup1, plate1]) in all_obj
    assert ("Pick", [cup1, plate2]) in all_obj
    assert ("Place", [cup1, plate1]) in all_obj
    assert ("Place", [cup1, plate2]) in all_obj


def test_is_dr_reachable():
    """Tests for is_dr_reachable().
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    pred1 = Predicate("Pred1", [cup_type, plate_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type, plate_type], lambda s, o: True)
    pred3 = Predicate("Pred3", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    # pred3 is unreachable
    preconditions1 = {pred1([cup_var, plate_var])}
    add_effects1 = {pred2([cup_var, plate_var])}
    delete_effects1 = {}
    preconditions2 = {pred1([cup_var, plate_var])}
    add_effects2 = {}
    delete_effects2 = {pred3([cup_var, plate_var])}
    operator1 = Operator("Pick", parameters, preconditions1, add_effects1,
                         delete_effects1, option=None, option_vars=[],
                         _sampler=None)
    operator2 = Operator("Place", parameters, preconditions2, add_effects2,
                         delete_effects2, option=None, option_vars=[],
                         _sampler=None)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_ops = (utils.all_ground_operators(operator1, objects) |
                  utils.all_ground_operators(operator2, objects))
    assert len(ground_ops) == 8
    atoms = {pred1([cup1, plate1]), pred1([cup1, plate2])}
    ground_ops = utils.filter_static_operators(ground_ops, atoms)
    assert utils.is_dr_reachable(ground_ops, atoms, {pred1([cup1, plate1])})
    assert utils.is_dr_reachable(ground_ops, atoms, {pred1([cup1, plate2])})
    assert utils.is_dr_reachable(ground_ops, atoms, {pred2([cup1, plate1])})
    assert utils.is_dr_reachable(ground_ops, atoms, {pred2([cup1, plate2])})
    assert not utils.is_dr_reachable(ground_ops, atoms, {pred3([cup1, plate1])})
    assert not utils.is_dr_reachable(ground_ops, atoms, {pred3([cup1, plate2])})
    assert not utils.is_dr_reachable(ground_ops, atoms, {pred1([cup2, plate1])})
    assert not utils.is_dr_reachable(ground_ops, atoms, {pred1([cup2, plate2])})
    assert not utils.is_dr_reachable(ground_ops, atoms, {pred2([cup2, plate1])})
    assert not utils.is_dr_reachable(ground_ops, atoms, {pred2([cup2, plate2])})
    assert not utils.is_dr_reachable(ground_ops, atoms, {pred3([cup2, plate1])})
    assert not utils.is_dr_reachable(ground_ops, atoms, {pred3([cup2, plate2])})


def test_operator_application():
    """Tests for get_applicable_operators(), apply_operator().
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    pred1 = Predicate("Pred1", [cup_type, plate_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type, plate_type], lambda s, o: True)
    pred3 = Predicate("Pred3", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    preconditions1 = {pred1([cup_var, plate_var])}
    add_effects1 = {pred2([cup_var, plate_var])}
    delete_effects1 = {}
    preconditions2 = {pred1([cup_var, plate_var])}
    add_effects2 = {}
    delete_effects2 = {pred3([cup_var, plate_var])}
    operator1 = Operator("Pick", parameters, preconditions1, add_effects1,
                         delete_effects1, option=None, option_vars=[],
                         _sampler=None)
    operator2 = Operator("Place", parameters, preconditions2, add_effects2,
                         delete_effects2, option=None, option_vars=[],
                         _sampler=None)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_ops = (utils.all_ground_operators(operator1, objects) |
                  utils.all_ground_operators(operator2, objects))
    assert len(ground_ops) == 8
    applicable = list(utils.get_applicable_operators(
        ground_ops, {pred1([cup1, plate1])}))
    assert len(applicable) == 2
    all_obj = [(op.name, op.objects) for op in applicable]
    assert ("Pick", [cup1, plate1]) in all_obj
    assert ("Place", [cup1, plate1]) in all_obj
    next_atoms = [utils.apply_operator(op, {pred1([cup1, plate1])})
                  for op in applicable]
    assert {pred1([cup1, plate1])} in next_atoms
    assert {pred1([cup1, plate1]), pred2([cup1, plate1])} in next_atoms
    assert list(utils.get_applicable_operators(
        ground_ops, {pred1([cup1, plate2])}))
    assert list(utils.get_applicable_operators(
        ground_ops, {pred1([cup2, plate1])}))
    assert list(utils.get_applicable_operators(
        ground_ops, {pred1([cup2, plate2])}))
    assert not list(utils.get_applicable_operators(
        ground_ops, {pred2([cup1, plate1])}))
    assert not list(utils.get_applicable_operators(
        ground_ops, {pred2([cup1, plate2])}))
    assert not list(utils.get_applicable_operators(
        ground_ops, {pred2([cup2, plate1])}))
    assert not list(utils.get_applicable_operators(
        ground_ops, {pred2([cup2, plate2])}))
    assert not list(utils.get_applicable_operators(
        ground_ops, {pred3([cup1, plate1])}))
    assert not list(utils.get_applicable_operators(
        ground_ops, {pred3([cup1, plate2])}))
    assert not list(utils.get_applicable_operators(
        ground_ops, {pred3([cup2, plate1])}))
    assert not list(utils.get_applicable_operators(
        ground_ops, {pred3([cup2, plate2])}))


def test_hadd_heuristic():
    """Tests for hAddHeuristic.
    """
    initial_state = frozenset({("IsBlock", "block0:block"),
                               ("IsTarget", "target0:target"),
                               ("IsTarget", "target1:target"),
                               ("HandEmpty",),
                               ("IsBlock", "block1:block")})
    operators = [
        utils.RelaxedOperator(
            "Pick", frozenset({("HandEmpty",), ("IsBlock", "block1:block")}),
            frozenset({("Holding", "block1:block")})),
        utils.RelaxedOperator(
            "Pick", frozenset({("IsBlock", "block0:block"), ("HandEmpty",)}),
            frozenset({("Holding", "block0:block")})),
        utils.RelaxedOperator(
            "Place", frozenset({("Holding", "block0:block"),
                                ("IsBlock", "block0:block"),
                                ("IsTarget", "target0:target")}),
            frozenset({("HandEmpty",),
                       ("Covers", "block0:block", "target0:target")})),
        utils.RelaxedOperator(
            "Place", frozenset({("IsTarget", "target0:target"),
                                ("Holding", "block1:block"),
                                ("IsBlock", "block1:block")}),
            frozenset({("HandEmpty",),
                       ("Covers", "block1:block", "target0:target")})),
        utils.RelaxedOperator(
            "Place", frozenset({("IsTarget", "target1:target"),
                                ("Holding", "block1:block"),
                                ("IsBlock", "block1:block")}),
            frozenset({("Covers", "block1:block", "target1:target"),
                       ("HandEmpty",)})),
        utils.RelaxedOperator(
            "Place", frozenset({("IsTarget", "target1:target"),
                                ("Holding", "block0:block"),
                                ("IsBlock", "block0:block")}),
            frozenset({("Covers", "block0:block", "target1:target"),
                       ("HandEmpty",)})),
        utils.RelaxedOperator(
            "Dummy", frozenset({}), frozenset({}))]
    goals = frozenset({("Covers", "block0:block", "target0:target"),
                       ("Covers", "block1:block", "target1:target")})
    heuristic = utils.HAddHeuristic(initial_state, goals, operators)
    assert heuristic(initial_state) == 4
    assert heuristic(goals) == 0
    goals = frozenset({("Covers", "block0:block", "target0:target")})
    heuristic = utils.HAddHeuristic(initial_state, goals, operators)
    assert heuristic(initial_state) == 2
    assert heuristic(goals) == 0


def test_save_video():
    """Tests for save_video().
    """
    dirname = "_fake_tmp_video_dir"
    filename = "video.mp4"
    utils.update_config({"video_dir": dirname})
    rng = np.random.default_rng(123)
    video = [rng.integers(255, size=(3, 3), dtype=np.uint8)
             for _ in range(3)]
    utils.save_video(filename, video)
    os.remove(os.path.join(dirname, filename))
    os.rmdir(dirname)


def test_get_config_path_str():
    """Tests for get_config_path_str().
    """
    utils.update_config({
        "env": "dummyenv",
        "approach": "dummyapproach",
        "seed": 321,
    })
    s = utils.get_config_path_str()
    assert s == "dummyenv__dummyapproach__321"
