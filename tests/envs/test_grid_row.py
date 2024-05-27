"""Test cases for the Grid Row environment."""

import numpy as np
import pytest

from predicators import utils
from predicators.envs.grid_row import GridRowDoorEnv, GridRowEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options


def test_grid_row():
    """Tests for the Grid Row environment."""
    utils.reset_config({
        "env": "grid_row",
        "num_train_tasks": 1,
        "num_test_tasks": 2,
        "grid_row_num_cells": 10,
    })
    env = GridRowEnv()
    assert env.get_name() == "grid_row"
    for env_task in env.get_train_tasks():
        task = env_task.task
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for env_task in env.get_test_tasks():
        task = env_task.task
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 5
    Adjacent, LightInCell, LightOff, LightOn, RobotInCell = \
        sorted(env.predicates)
    assert Adjacent.name == "Adjacent"
    assert LightInCell.name == "LightInCell"
    assert LightOff.name == "LightOff"
    assert LightOn.name == "LightOn"
    assert RobotInCell.name == "RobotInCell"
    assert env.goal_predicates == {LightOn}
    options = get_gt_options(env.get_name())
    assert len(env.types) == 3
    cell_type, light_type, robot_type = sorted(env.types)
    assert cell_type.name == "cell"
    assert light_type.name == "light"
    assert robot_type.name == "robot"
    assert env.action_space.shape == (2, )
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    assert len(nsrts) == 4
    assert len(options) == len(nsrts)
    env_train_tasks = env.get_train_tasks()
    assert len(env_train_tasks) == 1
    env_test_tasks = env.get_test_tasks()
    assert len(env_test_tasks) == 2
    env_task = env_test_tasks[1]
    env.reset("test", 1)
    with pytest.raises(NotImplementedError):
        env.render(caption="Test")

    # Test NSRTs.
    JumpToLight, MoveRobot, TurnOffLight, TurnOnLight = sorted(nsrts)
    assert JumpToLight.name == "JumpToLight"
    assert MoveRobot.name == "MoveRobot"
    assert TurnOffLight.name == "TurnOffLight"
    assert TurnOnLight.name == "TurnOnLight"

    init_state = env_test_tasks[0].task.init
    rng = np.random.default_rng(123)

    # Test successful turning on the light.
    robot, = init_state.get_objects(robot_type)
    light, = init_state.get_objects(light_type)
    cell_order = sorted(init_state.get_objects(cell_type),
                        key=lambda o: int(o.name[len("cell"):]))
    ground_nsrt_plan = []
    for cell, next_cell in zip(cell_order[:-1], cell_order[1:]):
        ground_nsrt_plan.append(MoveRobot.ground([robot, cell, next_cell]))
    # First move to the light.
    state = init_state.copy()
    rng = np.random.default_rng(123)
    for ground_nsrt in ground_nsrt_plan:
        assert all(a.holds(state) for a in ground_nsrt.preconditions)
        option = ground_nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        action = option.policy(state)
        state = env.simulate(state, action)
        assert option.terminal(state)
        assert all(a.holds(state) for a in ground_nsrt.add_effects)
        assert not any(a.holds(state) for a in ground_nsrt.delete_effects)
    # Now repeatedly turn on the light until it succeeds.
    ground_nsrt = TurnOnLight.ground([robot, cell_order[-1], light])
    for _ in range(100):
        assert all(a.holds(state) for a in ground_nsrt.preconditions)
        option = ground_nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        action = option.policy(state)
        state = env.simulate(state, action)
        assert option.terminal(state)
        if all(a.holds(state) for a in ground_nsrt.add_effects):
            break
    assert all(a.holds(state) for a in ground_nsrt.add_effects)
    assert not any(a.holds(state) for a in ground_nsrt.delete_effects)
    # Now repeatedly turn off the light until it succeeds.
    ground_nsrt = TurnOffLight.ground([robot, cell_order[-1], light])
    for _ in range(100):
        assert all(a.holds(state) for a in ground_nsrt.preconditions)
        option = ground_nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        action = option.policy(state)
        state = env.simulate(state, action)
        assert option.terminal(state)
        if all(a.holds(state) for a in ground_nsrt.add_effects):
            break
    assert all(a.holds(state) for a in ground_nsrt.add_effects)
    assert not any(a.holds(state) for a in ground_nsrt.delete_effects)
    # Go back to the shortcut square and show that jumping always fails.
    cell_n2, cell_n1, cell_n0 = cell_order[-3:]
    ground_nsrt_plan = [
        MoveRobot.ground([robot, cell_n0, cell_n1]),
        MoveRobot.ground([robot, cell_n1, cell_n2]),
    ]
    for ground_nsrt in ground_nsrt_plan:
        assert all(a.holds(state) for a in ground_nsrt.preconditions)
        option = ground_nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        action = option.policy(state)
        state = env.simulate(state, action)
        assert option.terminal(state)
        assert all(a.holds(state) for a in ground_nsrt.add_effects)
        assert not any(a.holds(state) for a in ground_nsrt.delete_effects)
    ground_nsrt = JumpToLight.ground([robot, cell_n2, cell_n1, cell_n0, light])
    for _ in range(100):
        assert all(a.holds(state) for a in ground_nsrt.preconditions)
        option = ground_nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        action = option.policy(state)
        state = env.simulate(state, action)
        assert option.terminal(state)
    assert not all(a.holds(state) for a in ground_nsrt.add_effects)


def test_grid_row_door():
    """Tests for the Grid Row Door environment."""
    utils.reset_config({
        "env": "grid_row_door",
        "num_train_tasks": 1,
        "num_test_tasks": 2,
        "grid_row_num_cells": 10,
    })
    env = GridRowDoorEnv()
    assert env.get_name() == "grid_row_door"
    for env_task in env.get_train_tasks():
        task = env_task.task
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for env_task in env.get_test_tasks():
        task = env_task.task
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 6
    Adjacent, DoorInCell, LightInCell, LightOff, LightOn, RobotInCell = \
        sorted(env.predicates)
    assert Adjacent.name == "Adjacent"
    assert LightInCell.name == "LightInCell"
    assert LightOff.name == "LightOff"
    assert LightOn.name == "LightOn"
    assert RobotInCell.name == "RobotInCell"
    assert DoorInCell.name == "DoorInCell"
    assert env.goal_predicates == {LightOn}
    options = get_gt_options(env.get_name())
    assert len(env.types) == 4
    cell_type, door_type, light_type, robot_type = sorted(env.types)
    assert cell_type.name == "cell"
    assert light_type.name == "light"
    assert robot_type.name == "robot"
    assert door_type.name == "door"
    assert env.action_space.shape == (3, )
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    assert len(nsrts) == 4
    assert len(options) == len(nsrts)
    env_train_tasks = env.get_train_tasks()
    assert len(env_train_tasks) == 1
    env_test_tasks = env.get_test_tasks()
    assert len(env_test_tasks) == 2
    env_task = env_test_tasks[1]
    env.reset("test", 1)
    with pytest.raises(NotImplementedError):
        env.render(caption="Test")

    # Test NSRTs.
    MoveRobot, OpenDoor, TurnOffLight, TurnOnLight = sorted(nsrts)
    assert OpenDoor.name == "OpenDoor"
    assert MoveRobot.name == "MoveRobot"
    assert TurnOffLight.name == "TurnOffLight"
    assert TurnOnLight.name == "TurnOnLight"

    init_state = env_test_tasks[0].task.init
    rng = np.random.default_rng(123)

    # Test successful turning on the light.
    robot, = init_state.get_objects(robot_type)
    light, = init_state.get_objects(light_type)
    door, = init_state.get_objects(door_type)
    cell_order = sorted(init_state.get_objects(cell_type),
                        key=lambda o: int(o.name[len("cell"):]))
    ground_nsrt_plan = []
    # First move to the light.
    state = init_state.copy()
    for cell, next_cell in zip(cell_order[:-1], cell_order[1:]):
        if len(ground_nsrt_plan) == state.get(door, "x") - 0.5:
            ground_nsrt_plan.append(OpenDoor.ground([robot, cell, door]))
            ground_nsrt_plan.append(MoveRobot.ground([robot, cell, next_cell]))
        else:
            ground_nsrt_plan.append(MoveRobot.ground([robot, cell, next_cell]))
    print(ground_nsrt_plan)
    print(state.get(door, "x"))
    rng = np.random.default_rng(123)
    for ground_nsrt in ground_nsrt_plan:
        print(state)
        print(ground_nsrt.preconditions)
        assert all(a.holds(state) for a in ground_nsrt.preconditions)
        option = ground_nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        action = option.policy(state)
        state = env.simulate(state, action)
        assert option.terminal(state)
        assert all(a.holds(state) for a in ground_nsrt.add_effects)
        assert not any(a.holds(state) for a in ground_nsrt.delete_effects)
    # Now repeatedly turn on the light until it succeeds.
    ground_nsrt = TurnOnLight.ground([robot, cell_order[-1], light])
    for _ in range(100):
        assert all(a.holds(state) for a in ground_nsrt.preconditions)
        option = ground_nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        action = option.policy(state)
        state = env.simulate(state, action)
        assert option.terminal(state)
        if all(a.holds(state) for a in ground_nsrt.add_effects):
            break
    assert all(a.holds(state) for a in ground_nsrt.add_effects)
    assert not any(a.holds(state) for a in ground_nsrt.delete_effects)
    # Now repeatedly turn off the light until it succeeds.
    ground_nsrt = TurnOffLight.ground([robot, cell_order[-1], light])
    for _ in range(100):
        assert all(a.holds(state) for a in ground_nsrt.preconditions)
        option = ground_nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        action = option.policy(state)
        state = env.simulate(state, action)
        assert option.terminal(state)
        if all(a.holds(state) for a in ground_nsrt.add_effects):
            break
    assert all(a.holds(state) for a in ground_nsrt.add_effects)
    assert not any(a.holds(state) for a in ground_nsrt.delete_effects)
