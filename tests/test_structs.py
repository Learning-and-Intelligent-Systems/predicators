"""Test cases for structs.
"""

import pytest
import numpy as np
from gym.spaces import Box
from predicators.src.structs import Type, Object, Variable, State, Predicate, \
    _Atom, LiftedAtom, GroundAtom, Task, ParameterizedOption, _Option, \
    STRIPSOperator, NSRT, _GroundNSRT, Action
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
    name = "test2"
    feats = ["feat3"]
    my_type2 = Type(name, feats, parent=my_type)
    assert my_type2.name == name
    assert my_type2.dim == len(my_type2.feature_names) == len(feats)
    assert my_type2.feature_names == feats
    assert isinstance(hash(my_type2), int)
    assert my_type2.parent == my_type
    name = "test2"
    feats = ["feat3"]
    my_type3 = Type(name, feats, parent=my_type)  # same as my_type2
    obj = my_type("obj1")
    assert obj.is_instance(my_type)
    assert not obj.is_instance(my_type2)
    assert not obj.is_instance(my_type3)
    obj = my_type2("obj2")
    assert obj.is_instance(my_type)
    assert obj.is_instance(my_type2)
    assert obj.is_instance(my_type3)


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
    obj2 = type2("obj2")
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
    assert state.get(obj3, "feat2") == 2
    assert state.get(obj1, "feat4") == 6
    with pytest.raises(ValueError):
        state.get(obj3, "feat3")  # feature not in list
    with pytest.raises(ValueError):
        state.get(obj1, "feat1")  # feature not in list
    vec = state.vec([obj3, obj1])
    assert vec.shape == (5,)
    assert list(vec) == [1, 2, 5, 6, 7]
    state.set(obj3, "feat2", 122)
    assert state.get(obj3, "feat2") == 122
    state2 = state.copy()
    assert state.allclose(state2)
    state2[obj1][0] = 999
    state2.set(obj1, "feat5", 991)
    assert state != state2  # changing copy doesn't change original
    assert state2.get(obj1, "feat3") == 999
    assert state2[obj1][2] == 991
    state3 = State({obj3: np.array([1, 2])})
    state3.copy()  # try copying with numpy array
    # Test state vec with no objects
    vec = state.vec([])
    assert vec.shape == (0,)
    # Test allclose
    state2 = State({obj3: [1, 122],
                    obj7: [3, 4],
                    obj1: [5, 6, 7],
                    obj4: [8, 9, 10],
                    obj9: [11, 12, 13]})
    assert state.allclose(state2)
    state2 = State({obj3: [1, 122],
                    obj7: [3, 4],
                    obj1: [5, 6, 7],
                    obj4: [8.3, 9, 10],
                    obj9: [11, 12, 13]})
    assert not state.allclose(state2)  # obj4 state is different
    state2 = State({obj3: [1, 122],
                    obj7: [3, 4],
                    obj4: [8, 9, 10],
                    obj9: [11, 12, 13]})
    assert not state.allclose(state2)  # obj1 is missing
    state2 = State({obj3: [1, 122],
                    obj7: [3, 4],
                    obj1: [5, 6, 7],
                    obj2: [5, 6, 7],
                    obj4: [8, 9, 10],
                    obj9: [11, 12, 13]})
    assert not state.allclose(state2)  # obj2 is extra
    # Test including simulator_state
    state_with_sim = State({}, "simulator_state")
    assert state_with_sim.simulator_state == "simulator_state"
    assert state.simulator_state is None
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
    def _classifier2(state, objects):
        cup, plate = objects
        return state[cup][0] + state[plate][0] < 1
    pred = Predicate("On", [cup_type, plate_type], _classifier)
    other_pred = Predicate("On", [cup_type, plate_type], _classifier2)
    assert pred == other_pred
    assert len({pred, other_pred}) == 1
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
    assert not other_pred.holds(state, [cup1, plate])
    assert not pred.holds(state, [cup2, plate])
    neg_pred = pred.get_negation()
    assert not neg_pred.holds(state, [cup1, plate])
    assert neg_pred.holds(state, [cup2, plate])
    assert str(pred) == repr(pred) == "On"
    assert {pred, pred} == {pred}
    pred2 = Predicate("On2", [cup_type, plate_type], _classifier)
    assert pred != pred2
    assert pred < pred2
    plate_var = plate_type("?plate")
    # Lifted atoms
    lifted_atom = pred([cup_var, plate_var])
    lifted_atom2 = pred([cup_var, plate_var])
    lifted_atom3 = pred2([cup_var, plate_var])
    with pytest.raises(AssertionError):
        pred2([cup_var])  # bad arity
    with pytest.raises(AssertionError):
        pred2([plate_var, cup_var])  # bad types
    assert lifted_atom.predicate == pred
    assert lifted_atom.variables == [cup_var, plate_var]
    assert {lifted_atom, lifted_atom2} == {lifted_atom}
    assert lifted_atom == lifted_atom2
    assert lifted_atom < lifted_atom3
    assert sorted([lifted_atom3, lifted_atom]) == [lifted_atom, lifted_atom3]
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
    assert task.init.allclose(state)
    assert task.goal == goal


def test_option():
    """Tests for ParameterizedOption, Option classes.
    """
    type1 = Type("type1", ["feat1", "feat2"])
    type2 = Type("type2", ["feat3", "feat4", "feat5"])
    obj7 = type1("obj7")
    obj1 = type2("obj1")
    state = test_state()
    params_space = Box(-10, 10, (2,))
    def _policy(s, o, p):
        del s, o  # unused
        return Action(p*2)
    def _initiable(s, o, p):
        del o  # unused
        obj = list(s)[0]
        return p[0] < s[obj][0]
    def _terminal(s, o, p):
        del o  # unused
        obj = list(s)[0]
        return p[1] > s[obj][2]
    parameterized_option = ParameterizedOption(
        "Pick", [], params_space, _policy, _initiable, _terminal)
    assert (repr(parameterized_option) == str(parameterized_option) ==
            "ParameterizedOption(name='Pick', types=[])")
    params = [-15, 5]
    with pytest.raises(AssertionError):
        parameterized_option.ground([], params)  # params not in params_space
    assert not hasattr(parameterized_option, "policy")
    assert not hasattr(parameterized_option, "initiable")
    assert not hasattr(parameterized_option, "terminal")
    params = [-5, 5]
    option = parameterized_option.ground([], params)
    assert isinstance(option, _Option)
    assert repr(option) == str(option) == (
        "_Option(name='Pick', objects=[], "
        "params=array([-5.,  5.], dtype=float32))")
    assert option.name == "Pick"
    assert option.parent.name == "Pick"
    assert option.parent is parameterized_option
    assert np.all(option.policy(state).arr == np.array(params)*2)
    assert option.initiable(state)
    assert not option.terminal(state)
    assert option.params[0] == -5 and option.params[1] == 5
    params = [5, -5]
    option = parameterized_option.ground([], params)
    assert isinstance(option, _Option)
    assert repr(option) == str(option) == (
        "_Option(name='Pick', objects=[], params=array([ 5., -5.], "
        "dtype=float32))")
    assert option.name == "Pick"
    assert option.parent.name == "Pick"
    assert option.parent is parameterized_option
    assert np.all(option.policy(state).arr == np.array(params)*2)
    assert not option.initiable(state)
    assert not option.terminal(state)
    assert option.params[0] == 5 and option.params[1] == -5
    parameterized_option = ParameterizedOption(
        "Pick", [type1], params_space, _policy, _initiable, _terminal)
    assert (repr(parameterized_option) == str(parameterized_option) ==
            "ParameterizedOption(name='Pick', types=[Type(name='type1')])")
    with pytest.raises(AssertionError):
        parameterized_option.ground([], params)  # grounding type mismatch
    with pytest.raises(AssertionError):
        parameterized_option.ground([obj1], params)  # grounding type mismatch
    option = parameterized_option.ground([obj7], params)
    assert isinstance(option, _Option)
    assert repr(option) == str(option) == (
        "_Option(name='Pick', objects=[obj7:type1], "
        "params=array([ 5., -5.], dtype=float32))")


def test_nsrts():
    """Tests for STRIPSOperator and NSRT and _GroundNSRT classes.
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
    params_space = Box(-10, 10, (2,))
    parameterized_option = ParameterizedOption(
        "Pick", [], params_space, lambda s, o, p: 2*p, lambda s, o, p: True,
        lambda s, o, p: True)
    def sampler(s, rng, objs):
        del s  # unused
        del rng  # unused
        del objs  # unused
        return params_space.sample()
    # STRIPSOperator
    strips_operator = STRIPSOperator("PickOperator", parameters, preconditions,
                                     add_effects, delete_effects)
    assert str(strips_operator) == repr(strips_operator) == \
        """STRIPS-PickOperator:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Add Effects: [On(?cup:cup_type, ?plate:plate_type)]
    Delete Effects: [NotOn(?cup:cup_type, ?plate:plate_type)]"""
    assert isinstance(hash(strips_operator), int)
    strips_operator2 = STRIPSOperator("PickOperator", parameters, preconditions,
                                      add_effects, delete_effects)
    assert strips_operator == strips_operator2
    # NSRT
    nsrt = NSRT("PickNSRT", parameters, preconditions, add_effects,
                delete_effects, parameterized_option, [], sampler)
    assert str(nsrt) == repr(nsrt) == """PickNSRT:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Add Effects: [On(?cup:cup_type, ?plate:plate_type)]
    Delete Effects: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Option: ParameterizedOption(name='Pick', types=[])
    Option Variables: []"""
    assert isinstance(hash(nsrt), int)
    nsrt2 = NSRT("PickNSRT", parameters, preconditions, add_effects,
                 delete_effects, parameterized_option, [], sampler)
    assert nsrt == nsrt2
    # _GroundNSRT
    cup = cup_type("cup")
    plate = plate_type("plate")
    ground_nsrt = nsrt.ground([cup, plate])
    assert isinstance(ground_nsrt, _GroundNSRT)
    assert str(ground_nsrt) == repr(ground_nsrt) == """PickNSRT:
    Parameters: [cup:cup_type, plate:plate_type]
    Preconditions: [NotOn(cup:cup_type, plate:plate_type)]
    Add Effects: [On(cup:cup_type, plate:plate_type)]
    Delete Effects: [NotOn(cup:cup_type, plate:plate_type)]
    Option: ParameterizedOption(name='Pick', types=[])
    Option Objects: []"""
    assert isinstance(hash(ground_nsrt), int)
    ground_nsrt2 = nsrt2.ground([cup, plate])
    assert ground_nsrt == ground_nsrt2
    state = test_state()
    ground_nsrt.sample_option(state, np.random.default_rng(123))
    filtered_nsrt = nsrt.filter_predicates({on})
    assert len(filtered_nsrt.parameters) == 2
    assert len(filtered_nsrt.preconditions) == 0
    assert len(filtered_nsrt.add_effects) == 1
    assert len(filtered_nsrt.delete_effects) == 0
    filtered_nsrt = nsrt.filter_predicates({not_on})
    assert len(filtered_nsrt.parameters) == 2
    assert len(filtered_nsrt.preconditions) == 1
    assert len(filtered_nsrt.add_effects) == 0
    assert len(filtered_nsrt.delete_effects) == 1


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
    states, actions = utils.option_to_trajectory(state, _simulator, option,
                                                 max_num_steps=5)
    assert len(actions) == len(states)-1 == 5
    for act in actions:
        assert act.has_option()
        opt = act.get_option()
        assert opt is option
        act.unset_option()
        assert not act.has_option()
    act = Action([0.5])
    assert not act.has_option()
