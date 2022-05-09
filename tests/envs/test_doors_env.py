"""Test cases for the doors environment."""

import numpy as np

from predicators.src import utils
from predicators.src.envs.doors import DoorsEnv
from predicators.src.structs import Action, GroundAtom, State, Task


def test_doors():
    """Tests for DoorsEnv()."""
    utils.reset_config({
        "env": "doors",
        "doors_room_map_size": 2,
        "doors_min_obstacles_per_room": 1,
        "doors_max_obstacles_per_room": 1,
        "doors_min_room_exists_frac": 1.0,
        "doors_max_room_exists_frac": 1.0,
    })
    env = DoorsEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 6
    DoorInRoom, DoorIsOpen, InDoorway, InMainRoom, InRoom, TouchingDoor = \
        sorted(env.predicates)
    assert DoorInRoom.name == "DoorInRoom"
    assert DoorIsOpen.name == "DoorIsOpen"
    assert InDoorway.name == "InDoorway"
    assert InMainRoom.name == "InMainRoom"
    assert InRoom.name == "InRoom"
    assert TouchingDoor.name == "TouchingDoor"
    assert env.goal_predicates == {InRoom}
    assert len(env.options) == 3
    assert len(env.types) == 4
    door_type, obstacle_type, robot_type, room_type = sorted(env.types)
    assert door_type.name == "door"
    assert obstacle_type.name == "obstacle"
    assert robot_type.name == "robot"
    assert room_type.name == "room"
    assert env.action_space.shape == (3, )
    # Create a custom initial state, with all rooms in the 2x2 grid, with the
    # robot starting out in the top left, and obstacles in the top right and
    # bottom left rooms.
    state = env.get_train_tasks()[0].init.copy()
    robot, = state.get_objects(robot_type)
    rooms = state.get_objects(room_type)
    assert len(rooms) == 4
    # Recall that the obstacles include the walls.
    expected_num_walls = 24
    obstacles = state.get_objects(obstacle_type)
    # Walls + 1 obstacle per room.
    assert len(obstacles) == expected_num_walls + len(rooms)
    doors = state.get_objects(door_type)
    assert len(doors) == 4
    # Remove the obstacle from the top left room.
    top_left_obstacles = [o for o in obstacles if "-0-0-obstacle" in o.name]
    assert len(top_left_obstacles) == 1
    top_left_obstacle = top_left_obstacles[0]
    state = State({o: state[o] for o in state if o != top_left_obstacle})
    # Put the robot in the middle of the top left room.
    top_left_room, top_right_room, _, bottom_right_room = sorted(rooms)
    room_cx = state.get(top_left_room, "x") + env.room_size / 2
    room_cy = state.get(top_left_room, "y") + env.room_size / 2
    state.set(robot, "x", room_cx)
    state.set(robot, "y", room_cy)
    # For later tests, make sure that the obstacle in the top right room is
    # exactly in the center.
    top_right_obstacles = [o for o in obstacles if "-0-1-obstacle" in o.name]
    assert len(top_right_obstacles) == 1
    top_right_obstacle = top_right_obstacles[0]
    w = state.get(top_right_obstacle, "width")
    h = state.get(top_right_obstacle, "height")
    x = state.get(top_right_room, "x") + env.room_size / 2 - w
    y = state.get(top_right_room, "y") + env.room_size / 2 - h
    state.set(top_right_obstacle, "x", x)
    state.set(top_right_obstacle, "y", y)
    state.set(top_right_obstacle, "theta", 0.0)
    # Since we moved obstacles around, the caches in the original env will be
    # wrong. Make a new env to be safe.
    env = DoorsEnv()
    # Since we removed the obstacle, there should be no collisions.
    assert not env._state_has_collision(state)  # pylint: disable=protected-access
    assert GroundAtom(InRoom, [robot, top_left_room]).holds(state)
    # Create a task with a goal to move to the bottom right room.
    goal = {GroundAtom(InRoom, [robot, bottom_right_room])}
    task = Task(state, goal)
    env.render_state(state, task)

    ## Test simulate ##

    # Test that the robot is contained within the walls when moving in any
    # direction, because the doors are initially closed.
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        action = Action(env.action_magnitude *
                        np.array([dx, dy, 0.0], dtype=np.float32))
        assert env.action_space.contains(action.arr)
        s = state.copy()
        for _ in range(50):
            s = env.simulate(s, action)
        # Should still be in the room.
        assert GroundAtom(InRoom, [robot, top_left_room]).holds(state)

    # Test opening the door on the right. First, find the door.
    top_doors = [d for d in doors \
        if GroundAtom(DoorInRoom, [d, top_left_room]).holds(state) and \
           GroundAtom(DoorInRoom, [d, top_right_room]).holds(state)
    ]
    assert len(top_doors) == 1
    top_door = top_doors[0]
    # Move more than enough steps to touch the door.
    action = Action(env.action_magnitude *
                    np.array([1.0, 0.0, 0.0], dtype=np.float32))
    s = state.copy()
    for _ in range(50):
        s = env.simulate(s, action)
    # The door should start off closed.
    assert not GroundAtom(DoorIsOpen, [top_door]).holds(s)
    # The robot should now be in the doorway and touching the door.
    assert GroundAtom(InDoorway, [robot, top_door]).holds(s)
    assert GroundAtom(TouchingDoor, [robot, top_door]).holds(s)
    # Now, open the door.
    mass = state.get(top_door, "mass")
    friction = state.get(top_door, "friction")
    target_rot = state.get(top_door, "target_rot")
    target_val = env._get_open_door_target_value(mass, friction, target_rot)  # pylint: disable=protected-access
    action = Action(np.array([0.0, 0.0, target_val], dtype=np.float32))
    s = env.simulate(s, action)
    # The door should now be open.
    assert GroundAtom(DoorIsOpen, [top_door]).holds(s)
    # The robot should still be in the doorway, but not touching the door.
    assert GroundAtom(InDoorway, [robot, top_door]).holds(s)
    assert not GroundAtom(TouchingDoor, [robot, top_door]).holds(s)

    # Test obstacle collisions. Continuing from the previous state, if we
    # move to the right, we should run into the obstacle at the center of
    # the room, and not pass it.
    action = Action(env.action_magnitude *
                    np.array([1.0, 0.0, 0.0], dtype=np.float32))
    for _ in range(50):
        s = env.simulate(s, action)
    # The robot should still be on the left of the obstacle.
    obstacle_x = s.get(top_right_obstacle, "x")
    robot_x = s.get(robot, "x")
    assert robot_x < obstacle_x
