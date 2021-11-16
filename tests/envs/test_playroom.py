"""Test cases for the boring room vs. playroom environment.
"""

import numpy as np
from predicators.src.envs import PlayroomEnv
from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.structs import Action


def test_playroom():
    """Tests for PlayroomEnv class.
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
    Pick = [o for o in env.options if o.name == "Pick"][0]
    Stack = [o for o in env.options if o.name == "Stack"][0]
    PutOnTable = [o for o in env.options if o.name == "PutOnTable"][0]
    On = [o for o in env.predicates if o.name == "On"][0]
    OnTable = [o for o in env.predicates if o.name == "OnTable"][0]
    block_type = [t for t in env.types if t.name == "block"][0]
    robot_type = [t for t in env.types if t.name == "robot"][0]
    block0 = block_type("block0")
    block1 = block_type("block1")
    block2 = block_type("block2")
    task = env.get_train_tasks()[0]
    state = task.init
    print("\nINITIAL STATE:", state)
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
    # No block at this pose, pick fails
    act = Action(np.array([20, 20, 0.45, -1.5, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # TODO
    # Object not clear, pick fails
    act = Action(np.array([12.2, 11.8, 0.45, 0.7, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.all(state[o] == next_state[o] for o in state)
    
    # Cannot putontable or stack without picking first
    act = Stack.ground([robot, block1], np.array(
        [0, 0, 0], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert np.all(state[o] == next_state[o] for o in state)
    act = PutOnTable.ground([robot], np.array(
        [0.5, 0.5], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert np.all(state[o] == next_state[o] for o in state)
    # Perform valid pick
    # TODO: this doesn't work bc rotation isn't set to face the table
    # print("STATE BEFORE:", state)
    act = Pick.ground([robot, block2], np.array(
        [0, 0, 0], dtype=np.float32)).policy(state)
    # print("\nACTION FROM GROUNDING:", act)
    next_state = env.simulate(state, act)
    # print("\nNEXT STATE:", next_state)
    assert np.any(state[block2] != next_state[block2])
    # Change the state
    state = next_state
    # Cannot pick twice in a row
    act = Pick.ground([robot, block2], np.array(
        [0, 0, 0], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert np.all(state[o] == next_state[o] for o in state)
    # Cannot stack onto non-clear block
    act = Stack.ground([robot, block0], np.array(
        [0, 0, CFG.playroom_block_size], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert np.all(state[o] == next_state[o] for o in state)
    # Cannot stack onto no block
    act = Stack.ground([robot, block1], np.array(
        [0, -1, CFG.playroom_block_size], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert np.all(state[o] == next_state[o] for o in state)
    # Cannot stack onto yourself
    act = Stack.ground([robot, block2], np.array(
        [0, 0, CFG.playroom_block_size], dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert np.all(state[o] == next_state[o] for o in state)
    # Cannot move to invalid location
    act = Action(np.array([40, 5, 0, 0, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.all(state[o] == next_state[o] for o in state)

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
    # Open boring room door
    act = Action(np.array([29.8, 15, 1, 0, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[robot] != next_state[robot])
    assert np.any(state[door1] != next_state[door1])
    state = next_state
    # Open all the other doors
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
    # Turn dial on
    act = Action(np.array([125, 15, 1, 0, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[dial] != next_state[dial])
    state = next_state
    # Turn dial off
    act = Action(np.array([125, 15, 1, 0, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[dial] != next_state[dial])
    state = next_state
