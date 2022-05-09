"""Test cases for the doors environment."""

import numpy as np
import pytest

from predicators.src import utils
from predicators.src.envs.doors import DoorsEnv
from predicators.src.structs import Action, GroundAtom, Task


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
    assert len(env.goal_predicates) == 1
    assert {pred.name for pred in env.goal_predicates} == {"InRoom"}
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
    # bottom left rooms. The goal will be to get to the bottom right.
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
