"""Test cases for the repeated_nextto_painting environment."""

import numpy as np
import pytest

from predicators.src import utils
from predicators.src.envs.repeated_nextto_painting import \
    RepeatedNextToPaintingEnv


def test_repeated_nextto_painting():
    """Tests for RepeatedNextToPaintingEnv class."""
    utils.reset_config({
        "env": "repeated_nextto_painting",
    })
    env = RepeatedNextToPaintingEnv()
    env.seed(123)
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 17
    assert {pred.name for pred in env.goal_predicates} == \
        {"InBox", "IsBoxColor", "InShelf", "IsShelfColor"}
    assert len(env.options) == 9
    assert len(env.types) == 5
    obj_type = [t for t in env.types if t.name == "obj"][0]
    robot_type = [t for t in env.types if t.name == "robot"][0]
    box_type = [t for t in env.types if t.name == "box"][0]
    shelf_type = [t for t in env.types if t.name == "shelf"][0]
    lid_type = [t for t in env.types if t.name == "lid"][0]
    assert env.action_space.shape == (8, )
    for i, task in enumerate(env.get_test_tasks()):
        state = task.init
        assert {box_type, shelf_type, obj_type, robot_type,
                lid_type} == {item.type
                              for item in state}
        if i < 3:
            # Test rendering
            env.render(state, task)


def test_repeated_nextto_painting_failure_cases():
    """Tests for the cases where simulate() is a no-op or
    EnvironmentFailure."""
    utils.reset_config({
        "env": "painting",
        "approach": "nsrt_learning",
        "painting_initial_holding_prob": 1.0,
        "painting_lid_open_prob": 0.0,
    })
    env = RepeatedNextToPaintingEnv()
    env.seed(123)
    Pick = [o for o in env.options if o.name == "Pick"][0]
    Wash = [o for o in env.options if o.name == "Wash"][0]
    Dry = [o for o in env.options if o.name == "Dry"][0]
    Paint = [o for o in env.options if o.name == "Paint"][0]
    Place = [o for o in env.options if o.name == "Place"][0]
    MoveToObj = [o for o in env.options if o.name == "MoveToObj"][0]
    MoveToBox = [o for o in env.options if o.name == "MoveToBox"][0]
    MoveToShelf = [o for o in env.options if o.name == "MoveToShelf"][0]
    OpenLid = [o for o in env.options if o.name == "OpenLid"][0]
    OnTable = [o for o in env.predicates if o.name == "OnTable"][0]
    Holding = [o for o in env.predicates if o.name == "Holding"][0]
    obj_type = [t for t in env.types if t.name == "obj"][0]
    box_type = [t for t in env.types if t.name == "box"][0]
    shelf_type = [t for t in env.types if t.name == "shelf"][0]
    robot_type = [t for t in env.types if t.name == "robot"][0]
    lid_type = [t for t in env.types if t.name == "lid"][0]
    obj0 = obj_type("obj0")
    obj1 = obj_type("obj1")
    box = box_type("receptacle_box")
    shelf = shelf_type("receptacle_shelf")
    robot = robot_type("robby")
    lid = lid_type("box_lid")
    task = env.get_train_tasks()[0]
    state = task.init
    atoms = utils.abstract(state, env.predicates)
    assert OnTable([obj1]) in atoms
    # In the first initial state, we are holding an object, because
    # painting_initial_holding_prob = 1.0
    assert Holding([obj0]) in atoms
    before_move_y = state.get(obj0, "pose_y")
    # Perform valid move to obj and change the state
    act = MoveToObj.ground([robot, obj1],
                           np.array([state.get(obj1, "pose_y")],
                                    dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    # Change the state
    state = next_state
    # Placing it on another object causes a collision
    with pytest.raises(utils.EnvironmentFailure):
        x = state.get(obj1, "pose_x")
        y = state.get(obj1, "pose_y")
        z = state.get(obj1, "pose_z")
        act = Place.ground([robot], np.array([x, y, z],
                                             dtype=np.float32)).policy(state)
        env.simulate(state, act)
    # Change the y back to initial y
    state.set(robot, "pose_y", before_move_y)
    state.set(obj0, "pose_y", before_move_y)
    # Advance to a state where we are not holding anything
    x = state.get(obj0, "pose_x")
    y = state.get(obj0, "pose_y")
    z = state.get(obj0, "pose_z") - 1.0
    act = Place.ground([robot], np.array([x, y, z],
                                         dtype=np.float32)).policy(state)
    handempty_state = env.simulate(state, act)
    state = handempty_state
    assert Holding([obj0]) not in utils.abstract(state, env.predicates)
    # Perform invalid pick with grasp = 1 (top grasp) too far away
    # (state should remain the same)
    act = Pick.ground([robot, obj1], np.array([0, 0, 0, 1],
                                              dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # No object at this pose, pick fails
    act = Pick.ground([robot, obj0], np.array([0, -1, 0, 0],
                                              dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot wash without holding
    act = Wash.ground([robot], np.array([1], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot dry without holding
    act = Dry.ground([robot], np.array([1], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot paint without holding
    act = Paint.ground([robot], np.array([0], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot place without holding
    act = Place.ground(
        [robot],
        np.array([
            RepeatedNextToPaintingEnv.obj_x,
            RepeatedNextToPaintingEnv.shelf_lb, RepeatedNextToPaintingEnv.obj_z
        ],
                 dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot pick with grasp = 0.5
    act = Pick.ground([robot, obj0], np.array([0, 0, 0, 0.5],
                                              dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Perform valid move to the same object (so state doesn't change)
    act = MoveToObj.ground([robot, obj0],
                           np.array([state.get(obj0, "pose_y")],
                                    dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Change the state
    state = next_state
    # Perform valid pick with grasp = 1 (top grasp)
    act = Pick.ground([robot, obj0], np.array([0, 0, 0, 1],
                                              dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Render with holding
    env.render(state, task)
    # Cannot pick twice in a row
    act = Pick.ground([robot, obj1], np.array([0, 0, 0, 0],
                                              dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot paint because not dry/clean
    act = Paint.ground([robot], np.array([0], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot place outside of shelf/box
    act = Place.ground([robot],
                       np.array([
                           RepeatedNextToPaintingEnv.obj_x,
                           RepeatedNextToPaintingEnv.shelf_lb - 0.1,
                           RepeatedNextToPaintingEnv.obj_z
                       ],
                                dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot place in shelf because grasp is 1
    act = Place.ground([robot],
                       np.array([
                           RepeatedNextToPaintingEnv.obj_x,
                           RepeatedNextToPaintingEnv.shelf_lb + 1e-3,
                           RepeatedNextToPaintingEnv.obj_z
                       ],
                                dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Render with a forced color of an object
    state.set(obj0, "color", 0.6)
    env.render(state, task)
    # Cannot place in box because lid is closed
    assert state[lid].item() == 0.0
    act = Place.ground([robot],
                       np.array([
                           RepeatedNextToPaintingEnv.obj_x,
                           RepeatedNextToPaintingEnv.box_lb + 1e-3,
                           RepeatedNextToPaintingEnv.obj_z
                       ],
                                dtype=np.float32)).policy(state)
    with pytest.raises(utils.EnvironmentFailure):
        env.simulate(state, act)
    # Open the box lid
    act = OpenLid.ground([robot, lid],
                         np.array([], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Perform invalid place into box (box too far)
    # state should not be changed
    act = Place.ground([robot],
                       np.array([
                           RepeatedNextToPaintingEnv.obj_x,
                           RepeatedNextToPaintingEnv.box_lb + 1e-3,
                           RepeatedNextToPaintingEnv.obj_z
                       ],
                                dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Perform valid move to box and change the state
    act = MoveToBox.ground([robot, box],
                           np.array([state.get(box, "pose_y")],
                                    dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Perform valid place into box
    act = Place.ground([robot],
                       np.array([
                           RepeatedNextToPaintingEnv.obj_x,
                           RepeatedNextToPaintingEnv.box_lb + 1e-3,
                           RepeatedNextToPaintingEnv.obj_z
                       ],
                                dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    # Reset state
    state = handempty_state
    # Perform valid move to obj, but since we're already near
    # it doesn't change state
    act = MoveToObj.ground([robot, obj0],
                           np.array([state.get(obj0, "pose_y")],
                                    dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Change the state
    state = next_state
    # Perform valid pick with grasp = 0 (side grasp)
    act = Pick.ground([robot, obj0], np.array([0, 0, 0, 0],
                                              dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Open the box lid
    act = OpenLid.ground([robot, lid],
                         np.array([], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Cannot place in box because grasp is 0
    act = Place.ground([robot],
                       np.array([
                           RepeatedNextToPaintingEnv.obj_x,
                           RepeatedNextToPaintingEnv.box_lb + 1e-3,
                           RepeatedNextToPaintingEnv.obj_z
                       ],
                                dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Perform valid move to shelf and change the state
    act = MoveToShelf.ground([robot, shelf],
                             np.array([state.get(shelf, "pose_y")],
                                      dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Perform valid place into shelf
    act = Place.ground([robot],
                       np.array([
                           RepeatedNextToPaintingEnv.obj_x,
                           RepeatedNextToPaintingEnv.shelf_lb + 1e-3,
                           RepeatedNextToPaintingEnv.obj_z
                       ],
                                dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Picking from shelf should fail
    act = Pick.ground([robot, obj0], np.array([0, 0, 0, 0],
                                              dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Make sure painting_initial_holding_prob = 0.0 works too.
    utils.update_config({"painting_initial_holding_prob": 0.0})
    env = RepeatedNextToPaintingEnv()
    env.seed(123)
    task = env.get_train_tasks()[0]
    state = task.init
    assert not utils.abstract(state, {Holding})
    # Make sure painting_lid_open_prob = 1.0 works too.
    utils.update_config({"painting_lid_open_prob": 1.0})
    env = RepeatedNextToPaintingEnv()
    env.seed(123)
    task = env.get_train_tasks()[0]
    state = task.init
    assert state[lid].item() == 1.0
