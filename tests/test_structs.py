"""Test cases for structs.
"""

from copy import copy, deepcopy
import pytest
from predicators.src.structs import Type, Object, Variable, State, Predicate


def test_object_type():
    """Tests for Type class.
    """
    name = "test"
    feats = ["feat1", "feat2"]
    my_type = Type(name, feats)
    assert my_type.name == name
    assert my_type.dim == len(my_type.feature_names) == len(feats)
    assert my_type.feature_names == feats


def test_object():
    """Tests for Object class.
    """
    my_name = "obj"
    my_type = Type("type", ["feat1", "feat2"])
    obj = Object(my_name, my_type)
    assert obj.name == my_name
    assert obj.type == my_type
    assert str(obj) == repr(obj) == "obj:type"
    with pytest.raises(AssertionError):
        Object("?obj", my_type)  # name cannot start with ?


def test_variable():
    """Tests for Variable class.
    """
    my_name = "?var"
    my_type = Type("type", ["feat1", "feat2"])
    var = Variable(my_name, my_type)
    assert var.name == my_name
    assert var.type == my_type
    assert str(var) == repr(var) == "?var:type"
    with pytest.raises(AssertionError):
        Variable("var", my_type)  # name must start with ?


def test_state():
    """Tests for State class.
    """
    type1 = Type("type1", ["feat1", "feat2"])
    type2 = Type("type2", ["feat3", "feat4", "feat5"])
    obj3 = Object("obj3", type1)
    obj7 = Object("obj7", type1)
    obj1 = Object("obj1", type2)
    obj1_dup = Object("obj1", type2)
    obj4 = Object("obj4", type2)
    obj9 = Object("obj9", type2)
    assert obj7 > obj1
    assert obj1 < obj4
    assert obj1 < obj3
    assert obj1 != obj9
    assert obj1 == obj1_dup
    assert copy(obj1) is obj1
    assert copy(obj1) is not obj1_dup
    assert copy(obj1) == obj1_dup
    assert deepcopy(obj1) is obj1
    assert deepcopy(obj1) is not obj1_dup
    assert deepcopy(obj1) == obj1_dup
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


def test_predicate():
    """Tests for Predicate class.
    """
    cup_type = Type("cup", ["feat1"])
    plate_type = Type("plate", ["feat1"])
    def _classifier(state, objects):
        cup, plate = objects
        return state[cup][0] + state[plate][0] < 2
    pred = Predicate("On", [cup_type, plate_type], _classifier)
    cup1 = Object("cup1", cup_type)
    cup2 = Object("cup2", cup_type)
    cup_var = Variable("?cup", cup_type)
    plate = Object("plate", plate_type)
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
    assert copy(pred) is pred
    assert deepcopy(pred) is pred
    pred2 = Predicate("On2", [cup_type, plate_type], _classifier)
    assert pred != pred2
    assert pred < pred2
