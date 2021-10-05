"""Test cases for structs.
"""

import pytest
import numpy as np
from gym.spaces import Box
from predicators.src.structs import Type, Object, Variable, State, Predicate, \
    _Atom, LiftedAtom, GroundAtom, Task, ParameterizedOption, _Option, \
    Operator, _GroundOperator, Action
from predicators.src import utils


def test_object_type():
    """Tests for Type class.
    """
    name = "test"
    feats = ["feat1", "feat2"]
    my_type = Type(name, feats)
    assert my_type.name == name
    assert my_type.dim == len(my_type.feature_names) == len(feats)
    assert my_type.feature_names == feats
    assert isinstance(hash(my_type), int)


def test_object():
    """Tests for Object class.
    """
    my_name = "obj"
    my_type = Type("type", ["feat1", "feat2"])
    obj = my_type(my_name)
    assert isinstance(obj, Object)
    assert obj.name == my_name
    assert obj.type == my_type
    assert str(obj) == repr(obj) == "obj:type"
    assert isinstance(hash(obj), int)
    with pytest.raises(AssertionError):
        Object("?obj", my_type)  # name cannot start with ?


def test_variable():
    """Tests for Variable class.
    """
    my_name = "?var"
    my_type = Type("type", ["feat1", "feat2"])
    var = my_type(my_name)
    assert isinstance(var, Variable)
    assert var.name == my_name
    assert var.type == my_type
    assert str(var) == repr(var) == "?var:type"
    assert isinstance(hash(var), int)
    with pytest.raises(AssertionError):
        Variable("var", my_type)  # name must start with ?


def test_state():
    """Tests for State class.
    """
    type1 = Type("type1", ["feat1", "feat2"])
    type2 = Type("type2", ["feat3", "feat4", "feat5"])
    obj3 = type1("obj3")
    obj7 = type1("obj7")
    obj1 = type2("obj1")
    obj1_dup = type2("obj1")
    obj4 = type2("obj4")
    obj9 = type2("obj9")
    assert obj7 > obj1
    assert obj1 < obj4
    assert obj1 < obj3
    assert obj1 != obj9
    assert obj1 == obj1_dup
    with pytest.raises(AssertionError):
        State({obj3: [1, 2, 3]})  # bad feature vector dimension
    state = State({obj3: [1, 2],
                   obj7: [3, 4],
                   obj1: [5, 6, 7],
                   obj4: [8, 9, 10],
                   obj9: [11, 12, 13]})
    sorted_objs = list(state)
    assert sorted_objs == [obj1, obj3, obj4, obj7, obj9]
    assert state[obj9] == state.data[obj9] == [11, 12, 13]
    vec = state.vec([obj3, obj1])
    assert vec.shape == (5,)
    assert list(vec) == [1, 2, 5, 6, 7]
    state2 = state.copy()
    assert state == state2
    state2[obj1][0] = 999
    assert state != state2  # changing copy doesn't change original
    state3 = State({obj3: np.array([1, 2])})
    state3.copy()  # try copying with numpy array
    return state


def test_predicate_and_atom():
    """Tests for Predicate, LiftedAtom, GroundAtom classes.
    """
    # Predicates
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    def _classifier(state, objects):
        cup, plate = objects
        return state[cup][0] + state[plate][0] < 2
    pred = Predicate("On", [cup_type, plate_type], _classifier)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    cup_var = cup_type("?cup")
    plate = plate_type("plate")
    state = State({cup1: [0.5], cup2: [1.5], plate: [1.0]})
    with pytest.raises(AssertionError):
        pred.holds(state, [cup1])  # too few arguments
    with pytest.raises(AssertionError):
        pred.holds(state, [cup1, cup2, plate])  # too many arguments
    with pytest.raises(AssertionError):
        pred.holds(state, [cup1, cup2])  # wrong object types
    with pytest.raises(AssertionError):
        pred.holds(state, [cup_var, plate])  # variable as argument
    assert pred.holds(state, [cup1, plate])
    assert not pred.holds(state, [cup2, plate])
    assert str(pred) == repr(pred) == "On"
    assert {pred, pred} == {pred}
    pred2 = Predicate("On2", [cup_type, plate_type], _classifier)
    assert pred != pred2
    assert pred < pred2
    plate_var = plate_type("?plate")
    # Lifted atoms
    lifted_atom = pred([cup_var, plate_var])
    lifted_atom2 = pred([cup_var, plate_var])
    assert lifted_atom.predicate == pred
    assert lifted_atom.variables == [cup_var, plate_var]
    assert {lifted_atom, lifted_atom2} == {lifted_atom}
    assert lifted_atom == lifted_atom2
    assert isinstance(lifted_atom, LiftedAtom)
    assert (str(lifted_atom) == repr(lifted_atom) ==
            "On(?cup:cup_type, ?plate:plate_type)")
    # Ground atoms
    ground_atom = pred([cup1, plate])
    assert ground_atom.predicate == pred
    assert ground_atom.objects == [cup1, plate]
    assert {ground_atom} == {ground_atom}
    assert (str(ground_atom) == repr(ground_atom) ==
            "On(cup1:cup_type, plate:plate_type)")
    assert isinstance(ground_atom, GroundAtom)
    lifted_atom3 = ground_atom.lift({cup1: cup_var, plate: plate_var})
    assert lifted_atom3 == lifted_atom
    with pytest.raises(ValueError):
        pred([cup_var, plate])  # mix of variables and objects
    atom = _Atom(pred, [cup1, plate])
    with pytest.raises(NotImplementedError):
        str(atom)  # abstract class


def test_task():
    """Tests for Task class.
    """
    state = test_state()
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    pred = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    cup = cup_type("cup")
    cup_var = cup_type("?cup")
    plate = plate_type("plate")
    plate_var = plate_type("?plate")
    lifted_goal = {pred([cup_var, plate_var])}
    with pytest.raises(AssertionError):
        Task(state, lifted_goal)  # tasks require ground goals
    goal = {pred([cup, plate])}
    task = Task(state, goal)
    assert task.init == state
    assert task.goal == goal


def test_option():
    """Tests for ParameterizedOption, Option classes.
    """
    state = test_state()
    params_space = Box(-10, 10, (2,))
    def _policy(s, p):
        del s  # unused
        return p*2
    def _initiable(s, p):
        obj = list(s)[0]
        return p[0] < s[obj][0]
    def _terminal(s, p):
        obj = list(s)[0]
        return p[1] > s[obj][2]
    parameterized_option = ParameterizedOption(
        "Pick", params_space, _policy, _initiable, _terminal)
    assert (repr(parameterized_option) == str(parameterized_option) ==
            "ParameterizedOption(name='Pick')")
    params = [-15, 5]
    with pytest.raises(AssertionError):
        parameterized_option.ground(params)  # params not in params_space
    assert not hasattr(parameterized_option, "policy")
    assert not hasattr(parameterized_option, "initiable")
    assert not hasattr(parameterized_option, "terminal")
    params = [-5, 5]
    option = parameterized_option.ground(params)
    assert isinstance(option, _Option)
    assert repr(option) == str(option) == "_Option(name='Pick(-5.0, 5.0)')"
    assert option.name == "Pick(-5.0, 5.0)"
    assert option.parent.name == "Pick"
    assert option.parent is parameterized_option
    assert np.all(option.policy(state) == np.array(params)*2)
    assert option.initiable(state)
    assert not option.terminal(state)
    params = [5, -5]
    option = parameterized_option.ground(params)
    assert isinstance(option, _Option)
    assert repr(option) == str(option) == "_Option(name='Pick(5.0, -5.0)')"
    assert option.name == "Pick(5.0, -5.0)"
    assert option.parent.name == "Pick"
    assert option.parent is parameterized_option
    assert np.all(option.policy(state) == np.array(params)*2)
    assert not option.initiable(state)
    assert not option.terminal(state)


def test_operators():
    """Tests for Operator and _GroundOperator classes.
    """
    # Operator
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
    params_space = Box(-10, 10, (2,))
    parameterized_option = ParameterizedOption("Pick",
        params_space, lambda s, p: 2*p, lambda s, p: True, lambda s, p: True)
    def sampler(s, rng, objs):
        del s  # unused
        del rng  # unused
        del objs  # unused
        return params_space.sample()
    operator = Operator("PickOperator", parameters, preconditions, add_effects,
                        delete_effects, parameterized_option, sampler)
    assert str(operator) == repr(operator) == """PickOperator:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Add Effects: [On(?cup:cup_type, ?plate:plate_type)]
    Delete Effects: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Option: ParameterizedOption(name='Pick')"""
    assert isinstance(hash(operator), int)
    operator2 = Operator("PickOperator", parameters, preconditions, add_effects,
                         delete_effects, parameterized_option, sampler)
    assert operator == operator2
    # _GroundOperator
    cup = cup_type("cup")
    plate = plate_type("plate")
    ground_op = operator.ground([cup, plate])
    assert isinstance(ground_op, _GroundOperator)
    assert str(ground_op) == repr(ground_op) == """PickOperator:
    Parameters: [cup:cup_type, plate:plate_type]
    Preconditions: [NotOn(cup:cup_type, plate:plate_type)]
    Add Effects: [On(cup:cup_type, plate:plate_type)]
    Delete Effects: [NotOn(cup:cup_type, plate:plate_type)]
    Option: ParameterizedOption(name='Pick')"""
    assert isinstance(hash(ground_op), int)
    ground_op2 = operator2.ground([cup, plate])
    assert ground_op == ground_op2
    state = test_state()
    _ = ground_op.sampler(state, np.random.default_rng(123))
    filtered_op = operator.filter_predicates({on})
    assert len(filtered_op.parameters) == 2
    assert len(filtered_op.preconditions) == 0
    assert len(filtered_op.add_effects) == 1
    assert len(filtered_op.delete_effects) == 0
    filtered_op = operator.filter_predicates({not_on})
    assert len(filtered_op.parameters) == 2
    assert len(filtered_op.preconditions) == 1
    assert len(filtered_op.add_effects) == 0
    assert len(filtered_op.delete_effects) == 1


def test_datasets():
    """Tests for ActionDatasets and OptionDatasets.
    """
    state = test_state()
    action = np.zeros(3, dtype=np.float32)
    transition = [state, action, state]
    dataset = [transition]
    assert len(dataset) == 1
    assert dataset[0] == transition


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
        act.unset_option()
        assert not act.has_option()
        next_ind += 1
    act = Action([0.5])
    assert not act.has_option()
