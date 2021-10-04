"""Test cases for utils.
"""

import pytest
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
    def _policy(_, p):
        return Action(p)
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
                         delete_effects1, option=None, _sampler=None)
    operator2 = Operator("Place", parameters, preconditions2, add_effects2,
                         delete_effects2, option=None, _sampler=None)
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
                         delete_effects1, option=None, _sampler=None)
    operator2 = Operator("Place", parameters, preconditions2, add_effects2,
                         delete_effects2, option=None, _sampler=None)
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
                         delete_effects1, option=None, _sampler=None)
    operator2 = Operator("Place", parameters, preconditions2, add_effects2,
                         delete_effects2, option=None, _sampler=None)
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


def test_action():
    """Tests for Action class.
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
    def _policy(_, p):
        return Action(p)
    def _initiable(_1, p):
        return p > 0.25
    def _terminal(s, _):
        return s[cup][0] > 9.9
    parameterized_option = ParameterizedOption(
        "Move", params_space, _policy, _initiable, _terminal)
    params = [0.5]
    option = parameterized_option.ground(params)
    states, actions = utils.option_to_trajectory(state, _simulator, option,
                                                 max_num_steps=5)
    assert len(actions) == len(states)-1 == 5
    next_ind = 0
    for act in actions:
        assert act.has_option()
        opt, ind = act.get_option()
        assert opt is option
        assert ind == next_ind
        next_ind += 1
