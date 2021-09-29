"""Test cases for structs.
"""

import pytest
from predicators.src.structs import ObjectType, Object, State
from copy import copy, deepcopy


def test_object_type():
    """Tests for ObjectType class.
    """
    name = "test"
    feats = ["feat1", "feat2"]
    my_type = ObjectType(name, feats)
    assert my_type.name == name
    assert my_type.dim == len(my_type.feature_names) == len(feats)
    assert my_type.feature_names == feats


def test_object():
    """Tests for Object class.
    """
    my_name = "obj"
    my_type = ObjectType("type", ["feat1", "feat2"])
    obj = Object(my_name, my_type)
    assert obj.name == my_name
    assert obj.type == my_type
    assert str(obj) == repr(obj) == "obj:type"


def test_state():
    """Tests for State class.
    """
    type1 = ObjectType("type1", ["feat1", "feat2"])
    type2 = ObjectType("type2", ["feat3", "feat4", "feat5"])
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
    assert state[obj9] == [11, 12, 13]
    vec = state.vec([obj3, obj1])
    assert vec.shape == (5,)
    assert list(vec) == [1, 2, 5, 6, 7]
