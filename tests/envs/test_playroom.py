"""Test cases for the boring room vs. playroom environment.
"""

import numpy as np
import pytest
from predicators.src.envs import PlayroomEnv
from predicators.src import utils
from predicators.src.structs import Action, State

def test_playroom():
    """Tests for PlayroomEnv class properties.
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
    assert len(env.predicates) == 12
    assert {pred.name for pred in env.goal_predicates} == \
        {"On", "OnTable", "LightOn", "LightOff"}
    assert len(env.options) == 8
    assert len(env.types) == 4
    assert env.action_space.shape == (5,)
    assert abs(env.action_space.low[0]-PlayroomEnv.x_lb) < 1e-3
    assert abs(env.action_space.high[0]-PlayroomEnv.x_ub) < 1e-3
    assert abs(env.action_space.low[1]-PlayroomEnv.y_lb) < 1e-3
    assert abs(env.action_space.high[1]-PlayroomEnv.y_ub) < 1e-3
    assert abs(env.action_space.low[2]) < 1e-3
    assert abs(env.action_space.high[2]-10) < 1e-3
    assert abs(env.action_space.low[3]+1) < 1e-3
    assert abs(env.action_space.high[3]-1) < 1e-3

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
    # Check robot is not next to any door
    with pytest.raises(RuntimeError):
        env._get_door_next_to(state)  # pylint: disable=protected-access
    # block1 is on block0 is on the table, block2 is on the table
    assert OnTable([block0]) in atoms
    assert OnTable([block1]) not in atoms
    assert OnTable([block2]) in atoms
    assert On([block1, block0]) in atoms
    # Cannot pick while not facing table
    act = Action(np.array([11.8, 18, 0.45, 0.85, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # No block at this pose, pick fails
    act = Action(np.array([19, 19, 0.45, -0.75, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # Object not clear, pick fails
    act = Action(np.array([12.2, 11.8, 0.45, 0.35, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # Cannot putontable or stack without picking first
    act = Action(np.array([12.2, 11.8, 5, 0.35, 0.7]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    act = Action(np.array([19, 14, 0.45, 0.95, 0.8]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # Perform valid pick
    act = Action(np.array([11.8, 18, 0.45, -0.15, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[robot] != next_state[robot])
    assert np.any(state[block2] != next_state[block2])
    state = next_state
    atoms = utils.abstract(state, env.predicates)
    assert OnTable([block2]) not in atoms
    # Cannot pick twice in a row
    act = Action(np.array([11.8, 18, 0.45, -0.15, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot stack onto non-clear block
    act = Action(np.array([12.2, 11.8, 0.8, 0.35, 0.7]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # Cannot stack onto no block
    act = Action(np.array([15, 16, 0.8, -0.5, 0.7]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # Cannot stack onto yourself
    act = Action(np.array([11.8, 18, 1.5, -0.15, 0.7]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot put on table when not clear
    act = Action(np.array([12.2, 11.8, 0.5, 0.35, 0.7]).astype(np.float32))
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
    block1 = block_type("block1")
    block2 = block_type("block2")
    task = env.get_train_tasks()[0]
    state = task.init
    robot = None
    for item in state:
        if item.type == robot_type:
            robot = item
            break
    assert robot is not None
    # Perform valid pick of block 1
    act = Action(np.array([12, 11.8, 0.95, 0.35, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[robot] != next_state[robot])
    assert np.any(state[block1] != next_state[block1])
    state = next_state
    # Perform valid put on table
    act = Action(np.array([19, 14, 0.45, 0.95, 0.8]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[block1] != next_state[block1])
    state = next_state
    # Perform valid pick of block 2
    act = Action(np.array([11.8, 18, 0.45, -0.15, 0]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[robot] != next_state[robot])
    assert np.any(state[block2] != next_state[block2])
    state = next_state
    # Perform valid stack
    act = Action(np.array([12.2, 11.8, 5, 0.35, 0.7]).astype(np.float32))
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
    act = Action(np.array([29.8, 15, 3, 0, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[robot] != next_state[robot])
    assert np.any(state[door1] != next_state[door1])
    state = next_state
    # Open all the other doors from left to right
    door_locs = [49.8, 59.8, 79.8, 99.8, 109.8]
    doors = [door2, door3, door4, door5, door6]
    for x, door in zip(door_locs, doors):
        act = Action(np.array([x, 15, 3, 0, 1]).astype(np.float32))
        next_state = env.simulate(state, act)
        assert np.any(state[door] != next_state[door])
        state = next_state
    # Shut door to playroom
    act = Action(np.array([110.2, 15, 3, 1, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[door6] != next_state[door6])
    state = next_state
    # Cannot go through closed door
    act = Action(np.array([105, 15, 3, 1, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])
    # Turn dial on, facing S
    act = Action(np.array([125, 15.1, 1, -0.5, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[dial] != next_state[dial])
    state = next_state
    # Turn dial off, facing E
    act = Action(np.array([125, 15, 1, 0, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    assert np.any(state[dial] != next_state[dial])
    state = next_state
    # Turn dial on, facing N
    act = Action(np.array([125, 14.9, 1, 0.5, 1]).astype(np.float32))
    state = env.simulate(state, act)
    # Turn dial off, facing W
    act = Action(np.array([125.1, 15, 1, 1, 1]).astype(np.float32))
    state = env.simulate(state, act)
    # Can't toggle when not facing dial
    act = Action(np.array([125.1, 15, 1, 0, 1]).astype(np.float32))
    next_state = env.simulate(state, act)
    for o in state:
        if o.type != robot_type:
            assert np.all(state[o] == next_state[o])

def test_playroom_options():
    """Tests for predicate option policies.
    """
    utils.update_config({"env": "playroom"})
    env = PlayroomEnv()
    env.seed(123)
    robot_type = [t for t in env.types if t.name == "robot"][0]
    block_type = [t for t in env.types if t.name == "block"][0]
    door_type = [t for t in env.types if t.name == "door"][0]
    dial_type = [t for t in env.types if t.name == "dial"][0]
    On = [p for p in env.predicates if p.name == "On"][0]
    OnTable = [p for p in env.predicates if p.name == "OnTable"][0]
    Clear = [p for p in env.predicates if p.name == "Clear"][0]
    LightOn = [p for p in env.predicates if p.name == "LightOn"][0]
    robot = robot_type("robby")
    block0 = block_type("block0")
    block1 = block_type("block1")
    block2 = block_type("block2")
    door1 = door_type("door1")
    door2 = door_type("door2")
    door3 = door_type("door3")
    door4 = door_type("door4")
    door5 = door_type("door5")
    door6 = door_type("door6")
    dial = dial_type("dial")
    task = env.get_train_tasks()[0]
    state = task.init
    # Run through a specific plan of options.
    Pick = [o for o in env.options if o.name == "Pick"][0]
    Stack = [o for o in env.options if o.name == "Stack"][0]
    PutOnTable = [o for o in env.options if o.name == "PutOnTable"][0]
    Move = [o for o in env.options if o.name == "Move"][0]
    OpenDoor = [o for o in env.options if o.name == "OpenDoor"][0]
    CloseDoor = [o for o in env.options if o.name == "CloseDoor"][0]
    TurnOnDial = [o for o in env.options if o.name == "TurnOnDial"][0]
    TurnOffDial = [o for o in env.options if o.name == "TurnOffDial"][0]
    plan = [
        Move.ground([robot], [2.0, 30.0, 0.0]),
        Pick.ground([block1], [0.0, 0.0, 0.0, 0.35]),
        PutOnTable.ground([], [0.1, 0.5, 0.0]),  # put block1 on table
        Pick.ground([block2], [0.0, 0.0, 0.0, -0.15]),
        Stack.ground([block1], [0.0, 0.0, 1.0, 0.0]),  # stack block2 on block1
        OpenDoor.ground([door1], [-0.2, 0.0, 0.0, 0.0]),
        OpenDoor.ground([door2], [-0.2, 0.0, 0.0, 0.0]),
        OpenDoor.ground([door3], [-0.2, 0.0, 0.0, 0.0]),
        OpenDoor.ground([door4], [-0.2, 0.0, 0.0, 0.0]),
        OpenDoor.ground([door5], [-0.2, 0.0, 0.0, 0.0]),
        OpenDoor.ground([door6], [-0.2, 0.0, 0.0, 0.0]),
        CloseDoor.ground([door6], [0.2, 0.0, 0.0, 1.0]),
        TurnOffDial.ground([dial], [0.0, 0.0, 0.0, 0.0]),
        TurnOnDial.ground([dial], [-0.2, 0.0, 0.0, 0.0])
    ]
    assert plan[0].initiable(state)
    make_video = False  # Can toggle to true for debugging
    (states, _), video, _ = utils.run_policy_on_task(
        utils.option_plan_to_policy(plan), task, env.simulate,
        env.predicates, 14, make_video, env.render)
    if make_video:
        outfile = "hardcoded_options_playroom.mp4"  # pragma: no cover
        utils.save_video(outfile, video)  # pragma: no cover
    final_atoms = utils.abstract(states[-1], env.predicates)
    assert LightOn([dial]) in final_atoms
    assert OnTable([block1]) in final_atoms
    assert On([block2, block1]) in final_atoms
    assert Clear([block0]) in final_atoms
    assert Clear([block1]) not in final_atoms
    assert Clear([block2]) in final_atoms

def test_playroom_action_sequence_video():
    """Test to sanity check rendering.
    """
    utils.update_config({"env": "playroom"})
    env = PlayroomEnv()
    env.seed(123)
    # Run through a specific plan of low-level actions.
    task = env.get_train_tasks()[0]
    action_arrs = [
        # Pick up a block
        np.array([11.8, 18, 0.45, -0.15, 0]).astype(np.float32),
        # Move down hallway from left to right and open all doors
        np.array([29.8, 15, 3, 0, 1]).astype(np.float32),
        np.array([49.8, 15, 3, 0, 1]).astype(np.float32),
        np.array([59.8, 15, 3, 0, 1]).astype(np.float32),
        np.array([79.8, 15, 3, 0, 1]).astype(np.float32),
        np.array([99.8, 15, 3, 0, 1]).astype(np.float32),
        np.array([109.8, 15, 3, 0, 1]).astype(np.float32),
        # Shut playroom door
        np.array([110.2, 15, 1, 1, 1]).astype(np.float32),
        # Turn dial on
        np.array([125, 15.1, 1, -1, 1]).astype(np.float32),
    ]
    make_video = False  # Can toggle to true for debugging
    def policy(s: State) -> Action:
        del s  # unused
        return Action(action_arrs.pop(0))
    (states, _), video, _ = utils.run_policy_on_task(
        policy, task, env.simulate, env.predicates,
        len(action_arrs), make_video, env.render)
    if make_video:
        outfile = "hardcoded_actions_playroom.mp4"  # pragma: no cover
        utils.save_video(outfile, video)  # pragma: no cover
    # Render a state where we're grasping
    env.render(states[1], task)
    # Render end state with open and closed doors
    env.render(states[-1], task)
