"""Test cases for the repeated_nextto_painting environment."""

import numpy as np

from predicators.src import utils
from predicators.src.envs.repeated_nextto_painting import \
    RepeatedNextToPaintingEnv


def test_repeated_nextto_painting():
    """Tests for RepeatedNextToPaintingEnv class."""
    utils.reset_config({
        "env": "repeated_nextto_painting",
    })
    env = RepeatedNextToPaintingEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 18
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
            env.render_state(state, task)


def test_repeated_nextto_painting_failure_cases():
    """Tests for the cases where simulate() is a no-op or
    EnvironmentFailure."""
    utils.reset_config({
        "env": "repeated_nextto_painting",
        "approach": "nsrt_learning",
        "painting_initial_holding_prob": 1.0,
        "painting_lid_open_prob": 0.0,
    })
    env = RepeatedNextToPaintingEnv()
    Pick = [o for o in env.options if o.name == "Pick"][0]
    Place = [o for o in env.options if o.name == "Place"][0]
    Holding = [o for o in env.predicates if o.name == "Holding"][0]
    OnTable = [o for o in env.predicates if o.name == "OnTable"][0]
    obj_type = [t for t in env.types if t.name == "obj"][0]
    robot_type = [t for t in env.types if t.name == "robot"][0]
    obj0 = obj_type("obj0")
    obj1 = obj_type("obj1")
    robot = robot_type("robby")
    task = env.get_train_tasks()[0]
    state = task.init
    x = state.get(obj0, "pose_x")
    y = state.get(obj0, "pose_y")
    # Set z to be slightly above the table to test
    # whether env snaps to the table on placement.
    z = env.table_height + env.obj_height / 2 + 0.004
    # Perform invalid place because we are not NextTo the target
    # (state should remain the same)
    act = Place.ground([robot], np.array([x, y - 3.0, z],
                                         dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Advance to a state where we are not holding anything
    act = Place.ground([robot], np.array([x, y - 1.0, z],
                                         dtype=np.float32)).policy(state)
    handempty_state = env.simulate(state, act)
    assert not state.allclose(handempty_state)
    state = handempty_state
    assert Holding([obj0]) not in utils.abstract(state, env.predicates)
    assert OnTable([obj0]) in utils.abstract(state, env.predicates)
    # Perform invalid pick because we are not NextTo the object
    # (state should remain the same)
    act = Pick.ground([robot, obj1], np.array([1],
                                              dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
