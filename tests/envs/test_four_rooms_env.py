"""Test cases for the four rooms environment."""

import numpy as np

from predicators.src import utils
from predicators.src.envs.four_rooms import FourRoomsEnv
from predicators.src.structs import Action, GroundAtom


def test_four_rooms():
    """Tests for TouchFourRooms class."""
    utils.reset_config({"env": "four_rooms"})
    env = FourRoomsEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 2
    Connected, InRoom = sorted(env.predicates)
    assert Connected.name == "Connected"
    assert InRoom.name == "InRoom"
    assert env.goal_predicates == {InRoom}
    assert len(env.options) == 1
    Move, = env.options
    assert Move.name == "Move"
    assert len(env.types) == 2
    robot_type, room_type = sorted(env.types)
    assert robot_type.name == "robot"
    assert room_type.name == "room"
    assert env.action_space.shape == (2, )
    task = env.get_train_tasks()[0]
    state = task.init.copy()
    robots = state.get_objects(robot_type)
    assert len(robots) == 1
    robot = robots[0]
    rooms = state.get_objects(room_type)
    assert len(rooms) == 4
    assert len(utils.abstract(state, {InRoom})) == 1
    assert len(utils.abstract(state, {Connected})) == 8
    # Select the bottom left room.
    room = min(rooms, key=lambda r: (state.get(r, "x"), state.get(r, "y")))
    assert room.name == "room1-0"
    env.render_state(state, task)
    # Make the width longer than the hallway size.
    state.set(robot, "width", 1.1 * env.hallway_width)
    w = state.get(robot, "width")
    # Position the robot in the middle of the room.
    rx, ry = state.get(room, "x"), state.get(room, "y")
    cx = rx + env.room_size / 2 - w / 2
    cy = ry + env.room_size / 2 - env.robot_height / 2
    state.set(robot, "x", cx)
    state.set(robot, "y", cy)
    state.set(robot, "rot", 0.0)
    in_room = GroundAtom(InRoom, [robot, room])
    assert in_room.holds(state)
    # Test moving up.
    up_action = Action(np.array([0.0, np.pi / 2.], dtype=np.float32))
    state2 = env.simulate(state, up_action)
    assert abs(state2.get(robot, "x") - cx) < 1e-6
    assert abs(state2.get(robot, "y") - (cy + env.action_magnitude)) < 1e-6
    assert abs(state2.get(robot, "rot") - 0.0) < 1e-6
    # Test rotating.
    rot_action1 = Action(np.array([np.pi / 10.0, 0.0], dtype=np.float32))
    state3 = env.simulate(state, rot_action1)
    assert abs(state3.get(robot, "rot") - np.pi / 10.0) < 1e-6
    rot_action2 = Action(np.array([-np.pi / 10.0, np.pi], dtype=np.float32))
    state4 = env.simulate(state3, rot_action2)
    assert state4.allclose(state)
    # Test that we can't leave the room through the top hallway because of
    # collisions.
    for _ in range(20):
        state = env.simulate(state, up_action)
    assert in_room.holds(state)
    # Now if we make the width smaller, it should fit through.
    state.set(robot, "width", 0.5 * env.hallway_width)
    for _ in range(5):
        state = env.simulate(state, up_action)
    assert not in_room.holds(state)
