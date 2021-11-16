"""Test cases for the boring room vs. playroom environment.
"""

import numpy as np
from predicators.src.envs import PlayroomEnv
from predicators.src import utils
from predicators.src.structs import Action


def test_playroom():
    """Tests for PlayroomEnv class: properties and rendering.
    """
    utils.update_config({"env": "playroom"})
    env = PlayroomEnv()
    env.seed(123)
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 5
    assert {pred.name for pred in env.goal_predicates} == {"On", "OnTable"}
    assert len(env.options) == 3
    assert len(env.types) == 4
    block_type = [t for t in env.types if t.name == "block"][0]
    robot_type = [t for t in env.types if t.name == "robot"][0]
    assert env.action_space.shape == (5,)
    assert abs(env.action_space.low[0]-PlayroomEnv.x_lb) < 1e-3
    assert abs(env.action_space.high[0]-PlayroomEnv.x_ub) < 1e-3
    assert abs(env.action_space.low[1]-PlayroomEnv.y_lb) < 1e-3
    assert abs(env.action_space.high[1]-PlayroomEnv.y_ub) < 1e-3
    assert abs(env.action_space.low[2]) < 1e-3
    assert abs(env.action_space.low[3]+2) < 1e-3
    assert abs(env.action_space.high[3]-2) < 1e-3
    for i, task in enumerate(env.get_test_tasks()):
        state = task.init
        robot = None
        for item in state:
            if item.type == robot_type:
                robot = item
                continue
            if item.type == block_type:
                assert not (state.get(item, "held") and
                            state.get(item, "clear"))
        assert robot is not None
        if i == 0:
            # Open a door to test rendering
            act = Action(np.array([29.8, 15, 1, 0, 1]).astype(np.float32))
            state = env.simulate(state, act)
            # Force initial pick to test rendering with holding
            Pick = [o for o in env.options if o.name == "Pick"][0]
            block = sorted([o for o in state if o.type.name == "block" and \
                            state.get(o, 'clear') > env.clear_tol])[0]
            act = Pick.ground([robot, block], np.zeros(3)).policy(state)
            state = env.simulate(state, act)
            env.render(state, task)

def test_playroom_failure_cases():
    """Tests for the cases where simulate() is a no-op.
    """
    utils.update_config({"env": "playroom"})
    env = PlayroomEnv()
    env.seed(123)
    On = [o for o in env.predicates if o.name == "On"][0]
    OnTable = [o for o in env.predicates if o.name == "OnTable"][0]
    block_type = [t for t in env.types if t.name == "block"][0]
    robot_type = [t for t in env.types if t.name == "robot"][0]
    block0 = block_type("block0")
    block1 = block_type("block1")
    block2 = block_type("block2")
    task = env.get_train_tasks()[0]
    state = task.init
    atoms = utils.abstract(state, env.predicates)
    robot = None
    for item in state:
        if item.type == robot_type:
            robot = item
            break
    assert robot is not None
    # block1 is on block0 is on the table, block2 is on the table
    assert OnTable([block0]) in atoms
    assert OnTable([block1]) not in atoms
    assert OnTable([block2]) in atoms
    assert On([block1, block0]) in atoms
    # Cannot pick while not facing table
    act = Action(np.array([11.8, 18, 0.45, 1.7, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # No block at this pose, pick fails
    act = Action(np.array([19, 19, 0.45, -1.5, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # Object not clear, pick fails
    act = Action(np.array([12.2, 11.8, 0.45, 0.7, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # Cannot putontable or stack without picking first
    act = Action(np.array([12.2, 11.8, 5, 0.7, 0.7]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    act = Action(np.array([19, 14, 0.45, 1.9, 0.8]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # Perform valid pick
    act = Action(np.array([11.8, 18, 0.45, -0.3, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[robot] != next_state[robot])
    assert np.any(state[block2] != next_state[block2])
    # Change the state
    state = next_state
    # Cannot pick twice in a row
    act = Action(np.array([11.8, 18, 0.45, -0.3, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot stack onto non-clear block
    act = Action(np.array([12.2, 11.8, 0.8, 0.7, 0.7]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # Cannot stack onto no block
    act = Action(np.array([15, 16, 0.8, -1.0, 0.7]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # Cannot stack onto yourself
    act = Action(np.array([11.8, 18, 1.5, -0.3, 0.7]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot put on table when not clear
    act = Action(np.array([12.2, 11.8, 0.5, 0.7, 0.7]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # Cannot move to invalid location
    act = Action(np.array([40, 5, 0, 0, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)

def test_playroom_simulate_blocks():
    """Tests for the cases where simulate() allows the robot to interact
    with blocks.
    """
    utils.update_config({"env": "playroom"})
    env = PlayroomEnv()
    env.seed(123)
    block_type = [t for t in env.types if t.name == "block"][0]
    robot_type = [t for t in env.types if t.name == "robot"][0]
    block2 = block_type("block2")
    task = env.get_train_tasks()[0]
    state = task.init
    robot = None
    for item in state:
        if item.type == robot_type:
            robot = item
            break
    assert robot is not None
    # Perform valid pick
    act = Action(np.array([11.8, 18, 0.45, -0.3, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[robot] != next_state[robot])
    assert np.any(state[block2] != next_state[block2])
    state = next_state
    # Perform valid put on table
    act = Action(np.array([19, 14, 0.45, 1.9, 0.8]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[block2] != next_state[block2])
    state = next_state
    # Pick up block again
    act = Action(np.array([19, 14, 0.45, 1.9, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[block2] != next_state[block2])
    state = next_state
    # Perform valid stack
    act = Action(np.array([12.2, 11.8, 5, 0.7, 0.7]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[block2] != next_state[block2])
    state = next_state

def test_playroom_simulate_doors_and_dial():
    """Tests for the cases where simulate() allows the robot to interact
    with doors and the dial.
    """
    utils.update_config({"env": "playroom"})
    env = PlayroomEnv()
    env.seed(123)
    door_type = [t for t in env.types if t.name == "door"][0]
    robot_type = [t for t in env.types if t.name == "robot"][0]
    dial_type = [t for t in env.types if t.name == "dial"][0]
    door1 = door_type("door1")
    door2 = door_type("door2")
    door3 = door_type("door3")
    door4 = door_type("door4")
    door5 = door_type("door5")
    door6 = door_type("door6")
    task = env.get_train_tasks()[0]
    state = task.init
    robot = None
    dial = None
    for item in state:
        if item.type == robot_type:
            robot = item
        if item.type == dial_type:
            dial = item
    assert robot is not None
    assert dial is not None
    # Move somewhere without interacting with anything
    act = Action(np.array([25, 25, 1, 0, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    state = next_state
    # Open boring room door
    act = Action(np.array([29.8, 15, 1, 0, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[robot] != next_state[robot])
    assert np.any(state[door1] != next_state[door1])
    state = next_state
    # Open all the other doors from left to right
    door_locs = [49.8, 59.8, 79.8, 99.8, 109.8]
    doors = [door2, door3, door4, door5, door6]
    for x, door in zip(door_locs, doors):
        act = Action(np.array([x, 15, 1, 0, 1]).astype(np.float32))
        next_state = env.simulate(state, act)
        assert np.any(state[door] != next_state[door])
        state = next_state
    # Shut door to playroom
    act = Action(np.array([110.2, 15, 1, 2, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[door6] != next_state[door6])
    state = next_state
    # Cannot go through closed door
    act = Action(np.array([105, 15, 1, 2, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # Turn dial on, facing S
    act = Action(np.array([125, 15.1, 1, -1, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[dial] != next_state[dial])
    state = next_state
    # Turn dial off, facing E
    act = Action(np.array([125, 15, 1, 0, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[dial] != next_state[dial])
    state = next_state
    # Turn dial on, facing N
    act = Action(np.array([125, 14.9, 1, 1, 1]).astype(np.float32))
    state = env.simulate(state, act)
    # Turn dial off, facing W
    act = Action(np.array([125.1, 15, 1, 2, 1]).astype(np.float32))
    state = env.simulate(state, act)
    # Can't toggle when not facing dial
    act = Action(np.array([125.1, 15, 1, 0, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
