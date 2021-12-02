"""Test cases for the painting environment.
"""

import pytest
import numpy as np
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
    utils.update_config({"env": "painting",
                         "approach": "nsrt_learning",
                         "seed": 123})
    env = PaintingEnv()
    env.seed(123)
    Pick = [o for o in env.options if o.name == "Pick"][0]
    Wash = [o for o in env.options if o.name == "Wash"][0]
    Dry = [o for o in env.options if o.name == "Dry"][0]
    Paint = [o for o in env.options if o.name == "Paint"][0]
    Place = [o for o in env.options if o.name == "Place"][0]
    OnTable = [o for o in env.predicates if o.name == "OnTable"][0]
    obj_type = [t for t in env.types if t.name == "obj"][0]
    robot_type = [t for t in env.types if t.name == "robot"][0]
    obj0 = obj_type("obj0")
    obj1 = obj_type("obj1")
    obj2 = obj_type("obj2")
    robot = robot_type("robot")
    task = env.get_train_tasks()[0]
    state = task.init
    atoms = utils.abstract(state, env.predicates)
    assert OnTable([obj0]) in atoms
    assert OnTable([obj1]) in atoms
    assert OnTable([obj2]) in atoms
    # No object at this pose, pick fails
    act = Pick.ground([robot, obj0], np.array(
        [0, -1, 0, 0], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot wash without holding
    act = Wash.ground([robot], np.array(
        [1], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot dry without holding
    act = Dry.ground([robot], np.array(
        [1], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot paint without holding
    act = Paint.ground([robot], np.array(
        [0], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot place without holding
    act = Place.ground([robot], np.array(
        [PaintingEnv.obj_x, PaintingEnv.shelf_lb, PaintingEnv.obj_z],
        dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Perform valid pick with gripper_rot = 0.5
    act = Pick.ground([robot, obj0], np.array(
        [0, 0, 0, 0.5], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Cannot pick twice in a row
    act = Pick.ground([robot, obj1], np.array(
        [0, 0, 0, 0], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot paint because not dry/clean
    act = Paint.ground([robot], np.array(
        [0], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot place outside of shelf/box
    act = Place.ground([robot], np.array(
        [PaintingEnv.obj_x, PaintingEnv.shelf_lb-0.1, PaintingEnv.obj_z],
        dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    # Cannot place because gripper_rot is 0.5
    act = Place.ground([robot], np.array(
        [PaintingEnv.obj_x, PaintingEnv.shelf_lb+1e-3, PaintingEnv.obj_z],
        dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Reset state
    state = task.init
    # Perform valid pick with gripper_rot = 1 (top grasp)
    act = Pick.ground([robot, obj0], np.array(
        [0, 0, 0, 1], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Cannot place in shelf because gripper_rot is 1
    act = Place.ground([robot], np.array(
        [PaintingEnv.obj_x, PaintingEnv.shelf_lb+1e-3, PaintingEnv.obj_z],
        dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
