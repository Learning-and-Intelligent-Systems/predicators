"""Test cases for structs."""

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.structs import NSRT, Action, DefaultState, \
    DemonstrationQuery, GroundAtom, InteractionRequest, InteractionResult, \
    LDLRule, LiftedAtom, LiftedDecisionList, LowLevelTrajectory, Object, \
    ParameterizedOption, PartialNSRTAndDatastore, Predicate, Query, Segment, \
    State, STRIPSOperator, Task, Type, Variable, _Atom, _GroundNSRT, \
    _GroundSTRIPSOperator, _Option


def test_object_type():
    """Tests for Type class."""
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
    """Tests for Object class."""
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
    """Tests for Variable class."""
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


@pytest.fixture(scope="module", name="state")
def test_state():
    """Tests for State class."""
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
    state = State({
        obj3: [1, 2],
        obj7: [3, 4],
        obj1: [5, 6, 7],
        obj4: [8, 9, 10],
        obj9: [11, 12, 13]
    })
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
    assert vec.shape == (5, )
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
    # Test state copy with a simulator state.
    state4 = State({obj3: np.array([1, 2])}, simulator_state="dummy")
    assert state4.simulator_state == "dummy"
    assert state4.copy().simulator_state == "dummy"
    # Cannot use allclose with non-None simulator states.
    with pytest.raises(NotImplementedError):
        state4.allclose(state3)
    with pytest.raises(NotImplementedError):
        state3.allclose(state4)
    # Test state vec with no objects
    vec = state.vec([])
    assert vec.shape == (0, )
    # Test allclose
    state2 = State({
        obj3: [1, 122],
        obj7: [3, 4],
        obj1: [5, 6, 7],
        obj4: [8, 9, 10],
        obj9: [11, 12, 13]
    })
    assert state.allclose(state2)
    state2 = State({
        obj3: [1, 122],
        obj7: [3, 4],
        obj1: [5, 6, 7],
        obj4: [8.3, 9, 10],
        obj9: [11, 12, 13]
    })
    assert not state.allclose(state2)  # obj4 state is different
    state2 = State({
        obj3: [1, 122],
        obj7: [3, 4],
        obj4: [8, 9, 10],
        obj9: [11, 12, 13]
    })
    assert not state.allclose(state2)  # obj1 is missing
    state2 = State({
        obj3: [1, 122],
        obj7: [3, 4],
        obj1: [5, 6, 7],
        obj2: [5, 6, 7],
        obj4: [8, 9, 10],
        obj9: [11, 12, 13]
    })
    assert not state.allclose(state2)  # obj2 is extra
    # Test pretty_str
    assert state2.pretty_str() == """################# STATE ################
type: type1      feat1    feat2
-------------  -------  -------
obj3                 1      122
obj7                 3        4

type: type2      feat3    feat4    feat5
-------------  -------  -------  -------
obj1                 5        6        7
obj2                 5        6        7
obj4                 8        9       10
obj9                11       12       13
########################################
"""
    # Test including simulator_state
    state_with_sim = State({}, "simulator_state")
    assert state_with_sim.simulator_state == "simulator_state"
    assert state.simulator_state is None
    return state


def test_predicate_and_atom():
    """Tests for Predicate, LiftedAtom, GroundAtom classes."""
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
    assert ground_atom.holds(state)
    ground_atom2 = pred([cup2, plate])
    assert not ground_atom2.holds(state)
    lifted_atom3 = ground_atom.lift({cup1: cup_var, plate: plate_var})
    assert lifted_atom3 == lifted_atom
    with pytest.raises(ValueError):
        pred([cup_var, plate])  # mix of variables and objects
    atom = _Atom(pred, [cup1, plate])
    with pytest.raises(NotImplementedError):
        str(atom)  # abstract class
    zero_arity_pred = Predicate("NoArity", [], _classifier)
    with pytest.raises(ValueError):
        zero_arity_pred([])  # ambiguous whether lifted or ground
    unary_predicate = Predicate("Unary", [cup_type], _classifier)
    with pytest.raises(ValueError) as e:
        GroundAtom(unary_predicate, cup1)  # expecting a sequence of atoms
    assert "Atoms expect a sequence of entities" in str(e)
    with pytest.raises(ValueError) as e:
        LiftedAtom(unary_predicate, cup_var)  # expecting a sequence of atoms
    assert "Atoms expect a sequence of entities" in str(e)


def test_task(state):
    """Tests for Task class."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    pred = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    pred2 = Predicate("On", [cup_type, plate_type], lambda s, o: False)
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
    assert task.goal_holds(task.init)
    goal2 = {pred2([cup, plate])}
    task2 = Task(state, goal2)
    assert task2.init.allclose(state)
    assert task2.goal == goal2
    assert not task2.goal_holds(task.init)


def test_option(state):
    """Tests for ParameterizedOption, Option classes."""
    type1 = Type("type1", ["feat1", "feat2"])
    type2 = Type("type2", ["feat3", "feat4", "feat5"])
    obj7 = type1("obj7")
    obj1 = type2("obj1")
    params_space = Box(-10, 10, (2, ))

    def policy(s, m, o, p):
        del s, m, o  # unused
        return Action(p * 2)

    def initiable(s, m, o, p):
        del o  # unused
        m["test_key"] = "test_string"
        obj = list(s)[0]
        return p[0] < s[obj][0]

    def terminal(s, m, o, p):
        del m, o  # unused
        obj = list(s)[0]
        return p[1] > s[obj][2]

    parameterized_option = ParameterizedOption("Pick", [], params_space,
                                               policy, initiable, terminal)
    assert (repr(parameterized_option) == str(parameterized_option) ==
            "ParameterizedOption(name='Pick', types=[])")
    params = [-15, 5]
    with pytest.raises(AssertionError):
        parameterized_option.ground([], params)  # params not in params_space
    params = [-5, 5]
    option = parameterized_option.ground([], params)
    assert isinstance(option, _Option)
    assert repr(option) == str(option) == (
        "_Option(name='Pick', objects=[], "
        "params=array([-5.,  5.], dtype=float32))")
    assert option.name == "Pick"
    assert option.memory == {}
    assert option.parent.name == "Pick"
    assert option.parent is parameterized_option
    assert np.all(option.policy(state).arr == np.array(params) * 2)
    assert option.initiable(state)
    assert option.memory == {"test_key": "test_string"}  # set by initiable()
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
    assert np.all(option.policy(state).arr == np.array(params) * 2)
    assert not option.initiable(state)
    assert not option.terminal(state)
    assert option.params[0] == 5 and option.params[1] == -5
    parameterized_option = ParameterizedOption("Pick", [type1], params_space,
                                               policy, initiable, terminal)
    assert (repr(parameterized_option) == str(parameterized_option) ==
            "ParameterizedOption(name='Pick', types=[Type(name='type1')])")
    parameterized_option2 = ParameterizedOption("Pick2", [type1], params_space,
                                                policy, initiable, terminal)
    assert parameterized_option2 > parameterized_option
    assert parameterized_option < parameterized_option2
    with pytest.raises(AssertionError):
        parameterized_option.ground([], params)  # grounding type mismatch
    with pytest.raises(AssertionError):
        parameterized_option.ground([obj1], params)  # grounding type mismatch
    option = parameterized_option.ground([obj7], params)
    assert isinstance(option, _Option)
    assert repr(option) == str(option) == (
        "_Option(name='Pick', objects=[obj7:type1], "
        "params=array([ 5., -5.], dtype=float32))")
    parameterized_option = utils.SingletonParameterizedOption(
        "Pick", policy, types=[type1], params_space=params_space)
    option = parameterized_option.ground([obj7], params)
    with pytest.raises(AssertionError):
        assert not option.terminal(state)  # must call initiable() first
    assert option.initiable(state.copy())
    assert option.initiable(state)
    assert not option.terminal(state)
    assert not option.terminal(state)  # try it again
    assert option.terminal(state.copy())  # should be True on a copy


def test_option_memory_incorrect():
    """Tests for doing option memory the WRONG way.

    Ensures that it fails in the way we'd expect.
    """

    def _make_option():
        value = 0.0

        def policy(s, m, o, p):
            del s, o  # unused
            del m  # the correct way of doing memory is unused here
            nonlocal value
            value += p[0]  # add the param to value
            return Action(p)

        return ParameterizedOption(
            "Dummy", [], Box(0, 1, (1, )), policy, lambda s, m, o, p: True,
            lambda s, m, o, p: value > 1.0)  # terminate when value > 1.0

    param_opt = _make_option()
    opt1 = param_opt.ground([], [0.7])
    opt2 = param_opt.ground([], [0.4])
    state = DefaultState
    assert abs(opt1.policy(state).arr[0] - 0.7) < 1e-6
    assert abs(opt2.policy(state).arr[0] - 0.4) < 1e-6
    # Since memory is shared between the two ground options, both will be
    # terminal now, since they'll share a value of 1.1 -- this is BAD, but
    # we include this test as an example of what NOT to do.
    assert opt1.terminal(state)
    assert opt2.terminal(state)


def test_option_memory_correct():
    """Tests for doing option memory the RIGHT way.

    Uses the memory dict.
    """

    def _make_option():

        def initiable(s, m, o, p):
            del s, o, p  # unused
            m["value"] = 0.0  # initialize value
            return True

        def policy(s, m, o, p):
            del s, o  # unused
            assert "value" in m, "Call initiable() first!"
            m["value"] += p[0]  # add the param to value
            return Action(p)

        return ParameterizedOption(
            "Dummy", [], Box(0, 1, (1, )), policy, initiable,
            lambda s, m, o, p: m["value"] > 1.0)  # terminate when value > 1.0

    param_opt = _make_option()
    opt1 = param_opt.ground([], [0.7])
    opt2 = param_opt.ground([], [0.4])
    state = DefaultState
    assert opt1.initiable(state)
    assert opt2.initiable(state)
    assert abs(opt1.policy(state).arr[0] - 0.7) < 1e-6
    assert abs(opt2.policy(state).arr[0] - 0.4) < 1e-6
    # Since memory is NOT shared between the two ground options, neither
    # will be terminal now.
    assert not opt1.terminal(state)
    assert not opt2.terminal(state)
    # Now make opt1 terminal.
    assert abs(opt1.policy(state).arr[0] - 0.7) < 1e-6
    assert opt1.terminal(state)
    assert not opt2.terminal(state)
    # opt2 is not quite terminal yet...value is 0.8
    opt2.policy(state)
    assert not opt2.terminal(state)
    # Make opt2 terminal.
    opt2.policy(state)
    assert opt2.terminal(state)


def test_operators_and_nsrts(state):
    """Tests for STRIPSOperator, _GroundSTRIPSOperator, NSRT and
    _GroundNSRT."""
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
    ignore_effects = {on}
    params_space = Box(-10, 10, (2, ))
    parameterized_option = ParameterizedOption("Pick", [], params_space,
                                               lambda s, m, o, p: 2 * p,
                                               lambda s, m, o, p: True,
                                               lambda s, m, o, p: True)

    def sampler(s, g, rng, objs):
        del s, g, rng, objs  # unused
        return params_space.sample()

    # STRIPSOperator
    strips_operator = STRIPSOperator("Pick", parameters, preconditions,
                                     add_effects, delete_effects,
                                     ignore_effects)
    assert str(strips_operator) == repr(strips_operator) == \
        """STRIPS-Pick:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Add Effects: [On(?cup:cup_type, ?plate:plate_type)]
    Delete Effects: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Ignore Effects: [On]"""
    assert strips_operator.pddl_str() == \
        """(:action Pick
    :parameters (?cup - cup_type ?plate - plate_type)
    :precondition (and (NotOn ?cup ?plate))
    :effect (and (On ?cup ?plate)
        (not (NotOn ?cup ?plate))
        (forall (?x0 - cup_type ?x1 - plate_type) (not (On ?x0 ?x1)))
        )
  )"""
    assert strips_operator.get_complexity() == 4.0  # 2^2
    assert isinstance(hash(strips_operator), int)
    strips_operator2 = STRIPSOperator("Pick", parameters, preconditions,
                                      add_effects, delete_effects,
                                      ignore_effects)
    assert strips_operator == strips_operator2
    strips_operator3 = STRIPSOperator("PickDuplicate", parameters,
                                      preconditions, add_effects,
                                      delete_effects, ignore_effects)
    assert strips_operator < strips_operator3
    assert strips_operator3 > strips_operator
    with pytest.raises(AssertionError):
        strips_operator.effect_to_ignore_effect(next(
            iter(add_effects)), [], "dummy")  # invalid last argument
    with pytest.raises(AssertionError):
        strips_operator.effect_to_ignore_effect(next(
            iter(add_effects)), [], "delete")  # not a delete effect!
    with pytest.raises(AssertionError):
        strips_operator.effect_to_ignore_effect(next(iter(delete_effects)), [],
                                                "add")  # not an add effect!
    strips_operator_zero_params = strips_operator.copy_with(parameters=[])
    assert strips_operator_zero_params.get_complexity() == 1.0  # 2^0
    strips_operator_three_params = strips_operator.copy_with(
        parameters=[1, 2, 3])
    assert strips_operator_three_params.get_complexity() == 8.0  # 2^3
    sidelined_add = strips_operator.effect_to_ignore_effect(
        next(iter(add_effects)), [], "add")
    assert str(sidelined_add) == repr(sidelined_add) == \
        """STRIPS-Pick:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Add Effects: []
    Delete Effects: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Ignore Effects: [On]"""
    sidelined_delete = strips_operator.effect_to_ignore_effect(
        next(iter(delete_effects)), [], "delete")
    assert str(sidelined_delete) == repr(sidelined_delete) == \
        """STRIPS-Pick:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Add Effects: [On(?cup:cup_type, ?plate:plate_type)]
    Delete Effects: []
    Ignore Effects: [NotOn, On]"""
    # Test copy_with().
    strips_operator4 = strips_operator.copy_with(preconditions=set())
    assert str(strips_operator4) == repr(strips_operator4) == \
        """STRIPS-Pick:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: []
    Add Effects: [On(?cup:cup_type, ?plate:plate_type)]
    Delete Effects: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Ignore Effects: [On]"""
    assert str(strips_operator) == repr(strips_operator) == \
        """STRIPS-Pick:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Add Effects: [On(?cup:cup_type, ?plate:plate_type)]
    Delete Effects: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Ignore Effects: [On]"""
    # _GroundSTRIPSOperator
    cup = cup_type("cup")
    plate = plate_type("plate")
    ground_op = strips_operator.ground((cup, plate))
    assert isinstance(ground_op, _GroundSTRIPSOperator)
    assert ground_op.parent is strips_operator
    assert str(ground_op) == repr(ground_op) == """GroundSTRIPS-Pick:
    Parameters: [cup:cup_type, plate:plate_type]
    Preconditions: [NotOn(cup:cup_type, plate:plate_type)]
    Add Effects: [On(cup:cup_type, plate:plate_type)]
    Delete Effects: [NotOn(cup:cup_type, plate:plate_type)]
    Ignore Effects: [On]"""
    ground_op2 = strips_operator2.ground((cup, plate))
    ground_op3 = strips_operator3.ground((cup, plate))
    assert ground_op == ground_op2
    assert ground_op < ground_op3
    assert ground_op3 > ground_op
    assert hash(ground_op) == hash(ground_op2)
    # NSRT
    nsrt = NSRT("Pick", parameters, preconditions, add_effects, delete_effects,
                ignore_effects, parameterized_option, [], sampler)
    assert str(nsrt) == repr(nsrt) == """NSRT-Pick:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Add Effects: [On(?cup:cup_type, ?plate:plate_type)]
    Delete Effects: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Ignore Effects: [On]
    Option Spec: Pick()"""
    assert str(nsrt.op) == repr(nsrt.op) == """STRIPS-Pick:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Add Effects: [On(?cup:cup_type, ?plate:plate_type)]
    Delete Effects: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Ignore Effects: [On]"""
    assert isinstance(hash(nsrt), int)
    nsrt2 = NSRT("Pick", parameters, preconditions, add_effects,
                 delete_effects, ignore_effects, parameterized_option, [],
                 sampler)
    assert nsrt == nsrt2
    nsrt3 = strips_operator.make_nsrt(parameterized_option, [], sampler)
    assert nsrt == nsrt3
    assert nsrt.sampler is sampler
    # _GroundNSRT
    ground_nsrt = nsrt.ground([cup, plate])
    assert isinstance(ground_nsrt, _GroundNSRT)
    assert ground_nsrt.parent is nsrt
    assert str(ground_nsrt) == repr(ground_nsrt) == """GroundNSRT-Pick:
    Parameters: [cup:cup_type, plate:plate_type]
    Preconditions: [NotOn(cup:cup_type, plate:plate_type)]
    Add Effects: [On(cup:cup_type, plate:plate_type)]
    Delete Effects: [NotOn(cup:cup_type, plate:plate_type)]
    Ignore Effects: [On]
    Option: ParameterizedOption(name='Pick', types=[])
    Option Objects: []"""
    assert isinstance(hash(ground_nsrt), int)
    ground_nsrt2 = nsrt2.ground([cup, plate])
    assert ground_nsrt == ground_nsrt2
    # Test less than comparison for grounded options
    nsrt4 = NSRT("Pick-Cup", parameters, preconditions, add_effects,
                 delete_effects, ignore_effects, parameterized_option, [],
                 sampler)
    assert nsrt2 > nsrt4
    assert nsrt4 < nsrt2
    ground_nsrt4 = nsrt4.ground([cup, plate])
    assert ground_nsrt4 < ground_nsrt2
    assert ground_nsrt2 > ground_nsrt4
    ground_nsrt.sample_option(state, set(), np.random.default_rng(123))
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
    # Test copy_with().
    ground_nsrt_copy1 = ground_nsrt.copy_with()
    assert ground_nsrt == ground_nsrt_copy1
    ground_nsrt_copy2 = ground_nsrt.copy_with(preconditions=set())
    assert str(ground_nsrt_copy2) == """GroundNSRT-Pick:
    Parameters: [cup:cup_type, plate:plate_type]
    Preconditions: []
    Add Effects: [On(cup:cup_type, plate:plate_type)]
    Delete Effects: [NotOn(cup:cup_type, plate:plate_type)]
    Ignore Effects: [On]
    Option: ParameterizedOption(name='Pick', types=[])
    Option Objects: []"""
    ground_nsrt_copy3 = ground_nsrt.copy_with(add_effects=set())
    assert str(ground_nsrt_copy3) == """GroundNSRT-Pick:
    Parameters: [cup:cup_type, plate:plate_type]
    Preconditions: [NotOn(cup:cup_type, plate:plate_type)]
    Add Effects: []
    Delete Effects: [NotOn(cup:cup_type, plate:plate_type)]
    Ignore Effects: [On]
    Option: ParameterizedOption(name='Pick', types=[])
    Option Objects: []"""


def test_action():
    """Tests for Action class."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state = State({cup: [0.5], plate: [1.0, 1.2]})

    def _simulator(s, a):
        ns = s.copy()
        assert a.arr.shape == (1, )
        ns[cup][0] += a.arr.item()
        return ns

    params_space = Box(0, 1, (1, ))

    def policy(_1, _2, _3, p):
        return Action(p)

    def initiable(_1, _2, _3, p):
        return p > 0.25

    def terminal(s, _1, _2, _3):
        return s[cup][0] > 9.9

    parameterized_option = ParameterizedOption("Move", [], params_space,
                                               policy, initiable, terminal)
    params = [0.5]
    option = parameterized_option.ground([], params)
    assert option.initiable(state)
    traj = utils.run_policy_with_simulator(option.policy,
                                           _simulator,
                                           state,
                                           option.terminal,
                                           max_num_steps=5)
    assert len(traj.actions) == len(traj.states) - 1 == 5
    for act in traj.actions:
        assert act.has_option()
        opt = act.get_option()
        assert opt is option
        act.unset_option()
        assert not act.has_option()
    act = Action([0.5])
    assert not act.has_option()


def test_low_level_trajectory():
    """Tests for LowLevelTrajectory class."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state0 = State({cup: [0.5], plate: [1.0, 1.2]})
    state1 = State({cup: [0.5], plate: [1.1, 1.2]})
    state2 = State({cup: [0.8], plate: [1.5, 1.2]})
    states = [state0, state1, state2]
    action0 = Action([0.4])
    action1 = Action([0.6])
    actions = [action0, action1]
    traj = LowLevelTrajectory(states, actions)
    assert traj.states == states
    assert traj.actions == actions
    assert not traj.is_demo
    with pytest.raises(AssertionError):
        print(traj.train_task_idx)  # no train task idx in this traj
    with pytest.raises(AssertionError):
        # Demo must have train task idx.
        traj = LowLevelTrajectory(states, actions, True)
    traj = LowLevelTrajectory(states,
                              actions,
                              _is_demo=True,
                              _train_task_idx=0)
    assert traj.is_demo
    assert traj.train_task_idx == 0
    with pytest.raises(AssertionError):
        # Incompatible lengths of states and actions.
        traj = LowLevelTrajectory(states[:-1], actions)


def test_segment():
    """Tests for Segment class."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    state0 = State({cup: [0.5], plate: [1.0, 1.2]})
    state1 = State({cup: [0.5], plate: [1.1, 1.2]})
    state2 = State({cup: [0.8], plate: [1.5, 1.2]})
    states = [state0, state1, state2]
    action0 = Action([0.4])
    action1 = Action([0.6])
    actions = [action0, action1]
    traj = LowLevelTrajectory(states, actions)
    init_atoms = {on([cup, plate])}
    final_atoms = {not_on([cup, plate])}
    parameterized_option = ParameterizedOption("Move", [], Box(0, 1, (1, )),
                                               lambda s, m, o, p: Action(p),
                                               lambda s, m, o, p: True,
                                               lambda s, m, o, p: True)
    params = [0.5]
    option = parameterized_option.ground([], params)
    action0.set_option(option)
    action1.set_option(option)
    # First create segment without the option.
    segment = Segment(traj, init_atoms, final_atoms)
    assert len(segment.states) == len(states)
    assert all(ss.allclose(s) for ss, s in zip(segment.states, states))
    assert len(segment.actions) == len(actions)
    assert all(np.allclose(sa.arr, a.arr) \
               for sa, a in zip(segment.actions, actions))
    assert segment.init_atoms == init_atoms
    assert segment.final_atoms == final_atoms
    assert segment.add_effects == {not_on([cup, plate])}
    assert segment.delete_effects == {on([cup, plate])}
    assert not segment.has_option()
    segment.set_option(option)
    assert segment.has_option()
    assert segment.get_option() == option
    # Test adding goals to segments and accessing them.
    assert not segment.has_goal()
    with pytest.raises(AssertionError):
        segment.get_goal()
    clear = Predicate("Clear", [plate_type], lambda s, o: True)
    goal = {clear([plate])}
    segment.set_goal(goal)
    assert segment.has_goal()
    assert segment.get_goal() == goal


def test_pnad():
    """Tests for PartialNSRTAndDatastore class."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    state0 = State({cup: [0.5], plate: [1.0, 1.2]})
    state1 = State({cup: [0.5], plate: [1.1, 1.2]})
    state2 = State({cup: [0.8], plate: [1.5, 1.2]})
    states = [state0, state1, state2]
    action0 = Action([0.4])
    action1 = Action([0.6])
    actions = [action0, action1]
    traj = LowLevelTrajectory(states, actions)
    init_atoms = {on([cup, plate])}
    final_atoms = {not_on([cup, plate])}
    parameterized_option = ParameterizedOption("Move", [], Box(0, 1, (1, )),
                                               lambda s, m, o, p: Action(p),
                                               lambda s, m, o, p: True,
                                               lambda s, m, o, p: True)
    params = [0.5]
    option = parameterized_option.ground([], params)
    segment1 = Segment(traj, init_atoms, final_atoms, option)
    var_to_obj = {cup_var: cup, plate_var: plate}
    segment2 = Segment(traj, init_atoms, final_atoms, option)
    segment3 = Segment(traj, init_atoms, set(), option)
    datastore = [(segment1, var_to_obj)]
    parameters = [cup_var, plate_var]
    preconditions = {on([cup_var, plate_var])}
    add_effects = {not_on([cup_var, plate_var])}
    delete_effects = {on([cup_var, plate_var])}
    ignore_effects = {on}
    strips_operator = STRIPSOperator("Pick", parameters, preconditions,
                                     add_effects, delete_effects,
                                     ignore_effects)
    pnad = PartialNSRTAndDatastore(strips_operator, datastore,
                                   (parameterized_option, []))
    assert len(pnad.datastore) == 1
    pnad.add_to_datastore((segment2, var_to_obj))
    assert len(pnad.datastore) == 2
    var_to_obj2 = {cup_var: plate, plate_var: cup}
    with pytest.raises(AssertionError):  # doesn't fit add effects
        pnad.add_to_datastore((segment3, var_to_obj2))
    pnad.add_to_datastore((segment3, var_to_obj2), check_effect_equality=False)
    assert len(pnad.datastore) == 3
    assert repr(pnad) == str(pnad) == """STRIPS-Pick:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: [On(?cup:cup_type, ?plate:plate_type)]
    Add Effects: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Delete Effects: [On(?cup:cup_type, ?plate:plate_type)]
    Ignore Effects: [On]
    Option Spec: Move()"""
    with pytest.raises(AssertionError):  # no sampler
        pnad.make_nsrt()
    pnad.sampler = lambda _1, _2, _3: Box(0, 1, (1, )).sample()
    nsrt = pnad.make_nsrt()
    assert repr(nsrt) == str(nsrt) == """NSRT-Pick:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Preconditions: [On(?cup:cup_type, ?plate:plate_type)]
    Add Effects: [NotOn(?cup:cup_type, ?plate:plate_type)]
    Delete Effects: [On(?cup:cup_type, ?plate:plate_type)]
    Ignore Effects: [On]
    Option Spec: Move()"""


def test_interaction_request_and_result():
    """Tests for InteractionRequest, InteractionResult classes."""
    InteractionRequest(None, None, None, None)
    with pytest.raises(AssertionError):  # wrong lengths
        InteractionResult([None], [None], [None])
    with pytest.raises(AssertionError):  # wrong lengths
        InteractionResult([None, None], [None], [None])
    InteractionResult([None, None], [None], [None, None])
    InteractionResult([None], [], [None])


def test_query():
    """Test for Query classes."""
    query = Query()
    with pytest.raises(NotImplementedError):
        _ = query.cost
    demo_query = DemonstrationQuery(0)
    assert demo_query.cost == 1


def test_lifted_decision_lists():
    """Tests for LDLRule, _GroundLDLRule, LiftedDecisionList."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    robot_type = Type("robot_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    on_table = Predicate("OnTable", [cup_type], lambda s, o: True)
    holding = Predicate("Holding", [cup_type], lambda s, o: True)
    hand_empty = Predicate("HandEmpty", [robot_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    robot_var = robot_type("?robot")
    pick_option = utils.SingletonParameterizedOption(
        "Pick", lambda _1, _2, _3, _4: Action(np.zeros(1)))
    place_option = utils.SingletonParameterizedOption(
        "Place",
        lambda _1, _2, _3, _4: Action(np.zeros(1)),
        types=[plate_type])

    pick_nsrt = NSRT("Pick",
                     parameters=[cup_var],
                     preconditions={on_table([cup_var])},
                     add_effects={holding([cup_var])},
                     delete_effects={on_table([cup_var])},
                     ignore_effects=set(),
                     option=pick_option,
                     option_vars=[],
                     _sampler=utils.null_sampler)

    place_nsrt = NSRT("Place",
                      parameters=[cup_var, plate_var],
                      preconditions={holding([cup_var])},
                      add_effects={on([cup_var, plate_var])},
                      delete_effects={not_on([cup_var, plate_var])},
                      ignore_effects=set(),
                      option=place_option,
                      option_vars=[plate_var],
                      _sampler=utils.null_sampler)

    # LDLRule
    pick_rule = LDLRule(
        "MyPickRule",
        parameters=[cup_var, plate_var, robot_var],
        pos_state_preconditions={on_table([cup_var]),
                                 hand_empty([robot_var])},
        neg_state_preconditions={holding([cup_var])},
        goal_preconditions={on([cup_var, plate_var])},
        nsrt=pick_nsrt)

    assert str(pick_rule) == repr(pick_rule) == """LDLRule-MyPickRule:
    Parameters: [?cup:cup_type, ?plate:plate_type, ?robot:robot_type]
    Pos State Pre: [HandEmpty(?robot:robot_type), OnTable(?cup:cup_type)]
    Neg State Pre: [Holding(?cup:cup_type)]
    Goal Pre: [On(?cup:cup_type, ?plate:plate_type)]
    NSRT: Pick(?cup:cup_type)"""

    place_rule = LDLRule("MyPlaceRule",
                         parameters=[cup_var, plate_var],
                         pos_state_preconditions={holding([cup_var])},
                         neg_state_preconditions=set(),
                         goal_preconditions={on([cup_var, plate_var])},
                         nsrt=place_nsrt)

    assert str(place_rule) == repr(place_rule) == """LDLRule-MyPlaceRule:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Pos State Pre: [Holding(?cup:cup_type)]
    Neg State Pre: []
    Goal Pre: [On(?cup:cup_type, ?plate:plate_type)]
    NSRT: Place(?cup:cup_type, ?plate:plate_type)"""

    assert pick_rule != place_rule

    pick_rule2 = LDLRule(
        "MyPickRule",
        parameters=[cup_var, plate_var, robot_var],
        pos_state_preconditions={on_table([cup_var]),
                                 hand_empty([robot_var])},
        neg_state_preconditions={holding([cup_var])},
        goal_preconditions={on([cup_var, plate_var])},
        nsrt=pick_nsrt)

    assert pick_rule == pick_rule2
    assert pick_rule < place_rule
    assert place_rule > pick_rule

    # Make sure rules are hashable.
    rules = {pick_rule, place_rule}
    assert rules == {pick_rule, place_rule}

    # Test that errors are raised if rules are malformed.
    with pytest.raises(AssertionError):
        _ = LDLRule("MissingStatePreconditionsRule",
                    parameters=[cup_var, plate_var, robot_var],
                    pos_state_preconditions=set(),
                    neg_state_preconditions=set(),
                    goal_preconditions={on([cup_var, plate_var])},
                    nsrt=pick_nsrt)
    with pytest.raises(AssertionError):
        _ = LDLRule("MissingParametersRule",
                    parameters=[plate_var, robot_var],
                    pos_state_preconditions={
                        on_table([cup_var]),
                        hand_empty([robot_var])
                    },
                    neg_state_preconditions=set(),
                    goal_preconditions={on([cup_var, plate_var])},
                    nsrt=pick_nsrt)

    # _GroundLDLRule
    cup1 = cup_type("cup1")
    plate1 = plate_type("plate1")
    robot = robot_type("robot")
    ground_pick_rule = pick_rule.ground((cup1, plate1, robot))

    assert str(ground_pick_rule) == repr(
        ground_pick_rule) == """GroundLDLRule-MyPickRule:
    Parameters: [cup1:cup_type, plate1:plate_type, robot:robot_type]
    Pos State Pre: [HandEmpty(robot:robot_type), OnTable(cup1:cup_type)]
    Neg State Pre: [Holding(cup1:cup_type)]
    Goal Pre: [On(cup1:cup_type, plate1:plate_type)]
    NSRT: Pick(cup1:cup_type)"""

    ground_place_rule = place_rule.ground((cup1, plate1))

    assert ground_pick_rule != ground_place_rule
    ground_pick_rule2 = pick_rule.ground((cup1, plate1, robot))
    assert ground_pick_rule == ground_pick_rule2
    assert ground_pick_rule < ground_place_rule
    assert ground_place_rule > ground_pick_rule

    # Make sure ground rules are hashable.
    rule_set = {ground_pick_rule, ground_place_rule}
    assert rule_set == {ground_pick_rule, ground_place_rule}

    # LiftedDecisionList
    rules = [place_rule, pick_rule]
    ldl = LiftedDecisionList(rules)
    assert ldl.rules == rules

    assert str(ldl) == """LiftedDecisionList[
LDLRule-MyPlaceRule:
    Parameters: [?cup:cup_type, ?plate:plate_type]
    Pos State Pre: [Holding(?cup:cup_type)]
    Neg State Pre: []
    Goal Pre: [On(?cup:cup_type, ?plate:plate_type)]
    NSRT: Place(?cup:cup_type, ?plate:plate_type)
LDLRule-MyPickRule:
    Parameters: [?cup:cup_type, ?plate:plate_type, ?robot:robot_type]
    Pos State Pre: [HandEmpty(?robot:robot_type), OnTable(?cup:cup_type)]
    Neg State Pre: [Holding(?cup:cup_type)]
    Goal Pre: [On(?cup:cup_type, ?plate:plate_type)]
    NSRT: Pick(?cup:cup_type)
]"""

    atoms = {on_table([cup1]), hand_empty([robot])}
    goal = {on([cup1, plate1])}
    objects = {cup1, plate1, robot}

    expected_nsrt = pick_nsrt.ground([cup1])
    assert utils.query_ldl(ldl, atoms, objects, goal) == expected_nsrt

    atoms = {holding([cup1])}

    expected_nsrt = place_nsrt.ground([cup1, plate1])
    assert utils.query_ldl(ldl, atoms, objects, goal) == expected_nsrt

    atoms = set()
    assert utils.query_ldl(ldl, atoms, objects, goal) is None

    ldl2 = LiftedDecisionList(rules)
    assert ldl == ldl2

    ldl3 = LiftedDecisionList(rules[::-1])
    assert ldl != ldl3

    ldl4 = LiftedDecisionList([place_rule])
    assert ldl != ldl4

    ldl5 = LiftedDecisionList(rules[:])
    assert ldl == ldl5

    # Make sure lifted decision lists are hashable.
    assert len({ldl, ldl2}) == 1
