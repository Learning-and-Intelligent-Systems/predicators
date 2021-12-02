"""Test cases for the painting environment.
"""

import pytest
from predicators.src.envs import PaintingEnv
from predicators.src import utils


def test_painting():
    """Tests for PaintingEnv class.
    """
    utils.update_config({"env": "painting"})
    env = PaintingEnv()
    env.seed(123)
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 13
    assert {pred.name for pred in env.goal_predicates} == \
        {"InBox", "IsBoxColor", "InShelf", "IsShelfColor"}
    assert len(env.options) == 6
    assert len(env.types) == 5
    obj_type = [t for t in env.types if t.name == "obj"][0]
    robot_type = [t for t in env.types if t.name == "robot"][0]
    box_type = [t for t in env.types if t.name == "box"][0]
    shelf_type = [t for t in env.types if t.name == "shelf"][0]
    lid_type = [t for t in env.types if t.name == "lid"][0]
    assert env.action_space.shape == (8,)
    for i, task in enumerate(env.get_test_tasks()):
        state = task.init
        robot = None
        box = None
        shelf = None
        lid = None
        for item in state:
            if item.type == robot_type:
                robot = item
                continue
            if item.type == box_type:
                box = item
                continue
            if item.type == shelf_type:
                shelf = item
                continue
            if item.type == lid_type:
                lid = item
                continue
            assert item.type == obj_type
            assert not state.get(item, "held")
        assert robot is not None
        assert box is not None
        assert shelf is not None
        assert lid is not None
        if i == 0:
            # Test rendering
            with pytest.raises(NotImplementedError):
                env.render(state, task)

def test_painting_failure_cases():
    """Tests for the cases where simulate() is a no-op.
    """
    utils.update_config({"env": "painting"})
    env = PaintingEnv()
    env.seed(123)
