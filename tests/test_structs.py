"""Test cases for structs.
"""

import pytest
from predicators.src.structs import Type, Object, Variable, State, Predicate, \
    Atom, LiftedAtom, GroundAtom


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
    with pytest.raises(ValueError):
        pred([cup_var, plate])  # mix of variables and objects
    with pytest.raises(NotImplementedError):
        atom = Atom(pred, [cup1, plate])  # abstract class
        str(atom)
