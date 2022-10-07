"""Test cases for the blocks environment."""

import json
import tempfile
from pathlib import Path

import numpy as np

from predicators import utils
from predicators.envs.blocks import BlocksEnv


def test_blocks():
    """Tests for BlocksEnv class."""
    utils.reset_config({"env": "blocks"})
    env = BlocksEnv()
    clear = env._block_is_clear  # pylint: disable=protected-access
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 5
    assert {pred.name for pred in env.goal_predicates} == {"On", "OnTable"}
    assert len(env.options) == 3
    assert len(env.types) == 2
    block_type = [t for t in env.types if t.name == "block"][0]
    assert env.action_space.shape == (4, )
    assert abs(env.action_space.low[0] - BlocksEnv.x_lb) < 1e-3
    assert abs(env.action_space.high[0] - BlocksEnv.x_ub) < 1e-3
    assert abs(env.action_space.low[1] - BlocksEnv.y_lb) < 1e-3
    assert abs(env.action_space.high[1] - BlocksEnv.y_ub) < 1e-3
    assert abs(env.action_space.low[2]) < 1e-3
    assert abs(env.action_space.low[3]) < 1e-3
    assert abs(env.action_space.high[3] - 1) < 1e-3
    for i, task in enumerate(env.get_test_tasks()):
        state = task.init
        robot = None
        for item in state:
            if item.type != block_type:
                robot = item
                continue
            assert not (state.get(item, "held") and clear(item, state))
        assert robot is not None
        if i == 0:
            # Force initial pick to test rendering with holding
            Pick = [o for o in env.options if o.name == "Pick"][0]
            block = sorted([o for o in state if o.type.name == "block" and \
                            clear(o, state)])[0]
            act = Pick.ground([robot, block], np.zeros(0)).policy(state)
            state = env.simulate(state, act)
            env.render_state(state, task, caption="caption")


def test_blocks_failure_cases():
    """Tests for the cases where simulate() is a noop."""
    utils.reset_config({"env": "blocks"})
    env = BlocksEnv()
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
    robot = robot_type("robot")
    task = env.get_train_tasks()[0]
    state = task.init
    atoms = utils.abstract(state, env.predicates)
    assert OnTable([block0]) in atoms
    assert OnTable([block1]) in atoms
    assert OnTable([block2]) not in atoms
    assert On([block2, block1]) in atoms
    # No block at this pose, pick fails
    act = Pick.ground([robot, block0], np.zeros(0)).policy(state)
    fake_state = state.copy()
    fake_state.set(block0, "pose_y", state.get(block0, "pose_y") - 1)
    next_state = env.simulate(fake_state, act)
    assert fake_state.allclose(next_state)
    # Object not clear, pick fails
    act = Pick.ground([robot, block1], np.zeros(0)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot putontable or stack without picking first
    act = Stack.ground([robot, block1], np.zeros(0)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    act = PutOnTable.ground([robot], np.array([0.5, 0.5],
                                              dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Perform valid pick
    act = Pick.ground([robot, block0], np.zeros(0)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Cannot pick twice in a row
    act = Pick.ground([robot, block2], np.zeros(0)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot stack onto non-clear block
    act = Stack.ground([robot, block1], np.zeros(0)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot stack onto no block
    act = Stack.ground([robot, block1], np.zeros(0)).policy(state)
    next_state = env.simulate(fake_state, act)
    assert fake_state.allclose(next_state)
    # Cannot stack onto yourself
    act = Stack.ground([robot, block0], np.zeros(0)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)


def test_blocks_load_task_from_json():
    """Tests for loading blocks test tasks from a JSON file."""
    # Set up the JSON file.
    task_spec = {
        "problem_name": "blocks_test_problem1",
        "blocks": {
            "red_block": {
                "position": [1.36409716, 1.0389289, 0.2225],
                "color": [1, 0, 0]
            },
            "green_block": {
                "position": [1.36409716, 1.0389289, 0.2675],
                "color": [0, 1, 0]
            },
            "blue_block": {
                "position": [1.35479861, 0.91064759, 0.2225],
                "color": [0, 0, 1]
            }
        },
        "block_size": 0.045,
        "goal": {
            "On": [["red_block", "green_block"], ["green_block",
                                                  "blue_block"]],
            "OnTable": [["blue_block"]]
        }
    }

    with tempfile.TemporaryDirectory() as json_dir:
        json_file = Path(json_dir) / "example_task1.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(task_spec, f)

        utils.reset_config({
            "env": "blocks",
            "num_test_tasks": 1,
            "blocks_test_task_json_dir": json_dir
        })

        env = BlocksEnv()
        test_tasks = env.get_test_tasks()

    assert len(test_tasks) == 1
    task = test_tasks[0]

    # pylint: disable=line-too-long
    assert task.init.pretty_str(
    ) == """####################################### STATE ######################################
type: block      pose_x    pose_y    pose_z    held    color_r    color_g    color_b
-------------  --------  --------  --------  ------  ---------  ---------  ---------
blue_block       1.3548  0.910648    0.2225       0          0          0          1
green_block      1.3641  1.03893     0.2675       0          0          1          0
red_block        1.3641  1.03893     0.2225       0          1          0          0

type: robot      pose_x    pose_y    pose_z    fingers
-------------  --------  --------  --------  ---------
robby              1.35      0.75      0.75          1
####################################################################################
"""
    assert str(
        sorted(task.goal)
    ) == "[On(green_block:block, blue_block:block), On(red_block:block, green_block:block), OnTable(blue_block:block)]"
