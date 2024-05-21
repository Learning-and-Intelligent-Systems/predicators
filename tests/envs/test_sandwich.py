"""Test cases for the sandwich env."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import predicators.pretrained_model_interface
from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.sandwich import SandwichEnv
from predicators.ground_truth_models import get_gt_options
from predicators.structs import Action, GroundAtom

_LLM_MODULE_PATH = predicators.pretrained_model_interface.__name__


def test_sandwich_properties():
    """Test env object initialization and properties."""
    utils.reset_config({"env": "sandwich"})
    env = SandwichEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 15
    BoardClear, Clear, GripperOpen, Holding, InHolder, IsBread, IsCheese, \
        IsEgg, IsGreenPepper, IsHam, IsLettuce, IsPatty, IsTomato, On, \
        OnBoard = sorted(env.predicates)
    assert BoardClear.name == "BoardClear"
    assert Clear.name == "Clear"
    assert GripperOpen.name == "GripperOpen"
    assert Holding.name == "Holding"
    assert InHolder.name == "InHolder"
    assert IsBread.name == "IsBread"
    assert IsPatty.name == "IsPatty"
    assert IsCheese.name == "IsCheese"
    assert IsEgg.name == "IsEgg"
    assert IsGreenPepper.name == "IsGreenPepper"
    assert IsHam.name == "IsHam"
    assert IsLettuce.name == "IsLettuce"
    assert IsTomato.name == "IsTomato"
    assert On.name == "On"
    assert OnBoard.name == "OnBoard"
    assert env.goal_predicates == {
        IsBread, IsPatty, IsCheese, IsEgg, IsGreenPepper, IsHam, IsLettuce,
        IsTomato, On, OnBoard
    }
    assert len(get_gt_options(env.get_name())) == 3
    Pick, PutOnBoard, Stack = sorted(get_gt_options(env.get_name()))
    assert Pick.name == "Pick"
    assert PutOnBoard.name == "PutOnBoard"
    assert Stack.name == "Stack"
    assert len(env.types) == 4
    board_type, holder_type, ingredient_type, robot_type = sorted(env.types)
    assert board_type.name == "board"
    assert holder_type.name == "holder"
    assert ingredient_type.name == "ingredient"
    assert robot_type.name == "robot"
    assert env.action_space.shape == (4, )


@pytest.mark.parametrize("env_name", ["sandwich", "sandwich_clear"])
def test_sandwich_options(env_name):
    """Tests for sandwich parameterized options, predicates, and rendering."""
    # Set up environment
    utils.reset_config({
        "env": env_name,
        # "render_state_dpi": 150,  # uncomment for higher-res test videos
    })
    env = create_new_env(env_name)
    BoardClear, Clear, GripperOpen, _, InHolder, _, _, _, _, _, _, _, _, On, \
        OnBoard = sorted(env.predicates)
    Pick, PutOnBoard, Stack = sorted(get_gt_options(env.get_name()))
    board_type, holder_type, _, robot_type = sorted(env.types)

    task = env.get_train_tasks()[0]
    state = task.init
    obj_name_to_obj = {o.name: o for o in state}
    # Select one cuboid and one cylinder to cover the different rendering cases
    ing0 = obj_name_to_obj["bread0"]
    ing1 = obj_name_to_obj["tomato0"]
    ing2 = obj_name_to_obj["bread1"]
    robot, = state.get_objects(robot_type)
    board, = state.get_objects(board_type)
    holder, = state.get_objects(holder_type)

    # Test a successful trajectory involving all the options
    option_plan = [
        Pick.ground([robot, ing0], []),
        PutOnBoard.ground([robot, board], []),
        Pick.ground([robot, ing1], []),
        Stack.ground([robot, ing0], [])
    ]
    policy = utils.option_plan_to_policy(option_plan)
    monitor = utils.SimulateVideoMonitor(task, env.render_state)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure},
        monitor=monitor,
    )
    # Save video of run
    video = monitor.get_video()
    assert len(video) == 5  # each option just takes 1 step
    # outfile = "hardcoded_options_sandwich.mp4"
    # utils.save_video(outfile, video)
    state0, state1, state2, state3, state4 = traj.states

    assert GroundAtom(BoardClear, [board]).holds(state0)
    assert not GroundAtom(BoardClear, [board]).holds(state4)
    assert GroundAtom(GripperOpen, [robot]).holds(state0)
    assert GroundAtom(GripperOpen, [robot]).holds(state4)
    assert GroundAtom(InHolder, [ing0, holder]).holds(state0)
    assert not GroundAtom(InHolder, [ing0, holder]).holds(state4)
    assert GroundAtom(InHolder, [ing1, holder]).holds(state0)
    assert not GroundAtom(InHolder, [ing1, holder]).holds(state4)
    assert not GroundAtom(On, [ing1, ing0]).holds(state0)
    assert GroundAtom(On, [ing1, ing0]).holds(state4)
    assert not GroundAtom(OnBoard, [ing0, board]).holds(state0)
    assert GroundAtom(OnBoard, [ing0, board]).holds(state4)
    assert not GroundAtom(Clear, [ing1]).holds(state0)
    assert not GroundAtom(Clear, [ing0]).holds(state1)
    assert GroundAtom(Clear, [ing1]).holds(state4)
    assert not GroundAtom(Clear, [ing0]).holds(state4)

    # Cover simulate "failure" cases.

    # Can only pick if fingers are open.
    state = state1
    option = Pick.ground([robot, ing1], [])
    action = option.policy(state)
    next_state = env.simulate(state, action)
    assert state.allclose(next_state)

    # No ingredient at this pose.
    state = state2
    x = state0.get(ing0, "pose_x")
    y = state0.get(ing0, "pose_y")
    z = state0.get(ing0, "pose_z")
    action = Action(np.array([x, y, z, 0.0], dtype=np.float32))
    next_state = env.simulate(state, action)
    assert state.allclose(next_state)

    # Can only pick if ingredient is in the holder.
    state = state2
    option = Pick.ground([robot, ing0], [])
    action = option.policy(state)
    next_state = env.simulate(state, action)
    assert state.allclose(next_state)

    # Can only putonboard if fingers are closed.
    state = state0
    option = PutOnBoard.ground([robot, board], [])
    action = option.policy(state)
    next_state = env.simulate(state, action)
    assert state.allclose(next_state)

    # Can only putonboard if nothing is on the board.
    state = state3
    option = PutOnBoard.ground([robot, board], [])
    action = option.policy(state)
    next_state = env.simulate(state, action)
    assert state.allclose(next_state)

    # Can only stack if fingers are closed.
    state = state0
    x, y, z = env.x_ub, env.y_ub, env.z_ub
    action = Action(np.array([x, y, z, 1.0], dtype=np.float32))
    next_state = env.simulate(state, action)
    assert state.allclose(next_state)

    # No object to stack onto.
    state = state1
    x, y, z = env.x_ub, env.y_ub, env.z_ub
    action = Action(np.array([x, y, z, 1.0], dtype=np.float32))
    next_state = env.simulate(state, action)
    assert state.allclose(next_state)

    # Can't stack onto yourself!
    state = state3
    option = Stack.ground([robot, ing1], [])
    action = option.policy(state)
    next_state = env.simulate(state, action)
    assert state.allclose(next_state)

    # Need object we're stacking onto to be clear.
    state = state4
    option = Pick.ground([robot, ing2], [])
    action = option.policy(state)
    state = env.simulate(state, action)
    option = Stack.ground([robot, ing0], [])
    action = option.policy(state)
    next_state = env.simulate(state, action)
    assert state.allclose(next_state)

    # Rendering with caption.
    env.render_state(state, task, caption="Test caption")


def test_sandwich_load_task_from_json():
    """Tests for loading sandwich test tasks from a JSON file."""
    # Set up the JSON file.
    task_spec = {
        "goal": {
            "On": [["bread1", "cheese0"], ["cheese0", "bread0"]],
            "OnBoard": [["bread0", "board"]]
        },
        "init": {
            "board": {
                "pose_x": 1.3224584114361717,
                "pose_y": 0.9605784672434986,
                "length": 0.14,
                "width": 0.24000000000000005,
                "thickness": 0.01
            },
            "bread0": {
                "pose_x": 1.3582177012392873,
                "pose_y": 0.478885069963471,
                "pose_z": 0.266,
                "rot": 1.5707963267948966,
                "held": 0,
                "color_r": 0.58,
                "color_g": 0.29,
                "color_b": 0,
                "thickness": 0.02,
                "radius": 0.05600000000000001,
                "shape": 0
            },
            "bread1": {
                "pose_x": 1.3582177012392873,
                "pose_y": 0.7788850699634711,
                "pose_z": 0.266,
                "rot": 1.5707963267948966,
                "held": 0,
                "color_r": 0.58,
                "color_g": 0.29,
                "color_b": 0,
                "thickness": 0.02,
                "radius": 0.05600000000000001,
                "shape": 0
            },
            "cheese0": {
                "pose_x": 1.3582177012392873,
                "pose_y": 0.628885069963471,
                "pose_z": 0.2609090909090909,
                "rot": 1.5707963267948966,
                "held": 0,
                "color_r": 0.937,
                "color_g": 0.737,
                "color_b": 0.203,
                "thickness": 0.02,
                "radius": 0.05090909090909091,
                "shape": 0
            },
            "egg0": {
                "pose_x": 1.3582177012392873,
                "pose_y": 0.5788850699634711,
                "pose_z": 0.2530769230769231,
                "rot": 1.5707963267948966,
                "held": 0,
                "color_r": 0.937,
                "color_g": 0.898,
                "color_b": 0.384,
                "thickness": 0.02,
                "radius": 0.04307692307692308,
                "shape": 1
            },
            "green_pepper0": {
                "pose_x": 1.3582177012392873,
                "pose_y": 0.42888506996347103,
                "pose_z": 0.25,
                "rot": 1.5707963267948966,
                "held": 0,
                "color_r": 0.156,
                "color_g": 0.541,
                "color_b": 0.16,
                "thickness": 0.02,
                "radius": 0.04,
                "shape": 1
            },
            "ham0": {
                "pose_x": 1.3582177012392873,
                "pose_y": 0.8288850699634711,
                "pose_z": 0.2609090909090909,
                "rot": 1.5707963267948966,
                "held": 0,
                "color_r": 0.937,
                "color_g": 0.384,
                "color_b": 0.576,
                "thickness": 0.02,
                "radius": 0.05090909090909091,
                "shape": 0
            },
            "holder": {
                "pose_x": 1.3582177012392873,
                "pose_y": 0.628885069963471,
                "length": 0.42000000000000004,
                "width": 0.24000000000000005,
                "thickness": 0.01
            },
            "lettuce0": {
                "pose_x": 1.3582177012392873,
                "pose_y": 0.728885069963471,
                "pose_z": 0.2566666666666667,
                "rot": 1.5707963267948966,
                "held": 0,
                "color_r": 0.203,
                "color_g": 0.937,
                "color_b": 0.431,
                "thickness": 0.02,
                "radius": 0.04666666666666667,
                "shape": 1
            },
            "patty0": {
                "pose_x": 1.3582177012392873,
                "pose_y": 0.678885069963471,
                "pose_z": 0.2566666666666667,
                "rot": 1.5707963267948966,
                "held": 0,
                "color_r": 0.32,
                "color_g": 0.15,
                "color_b": 0,
                "thickness": 0.02,
                "radius": 0.04666666666666667,
                "shape": 1
            },
            "robby": {
                "pose_x": 1.35,
                "pose_y": 0.75,
                "pose_z": 0.7,
                "fingers": 1
            },
            "tomato0": {
                "pose_x": 1.3582177012392873,
                "pose_y": 0.5288850699634711,
                "pose_z": 0.2566666666666667,
                "rot": 1.5707963267948966,
                "held": 0,
                "color_r": 0.917,
                "color_g": 0.18,
                "color_b": 0.043,
                "thickness": 0.02,
                "radius": 0.04666666666666667,
                "shape": 1
            }
        },
        "objects": {
            "board": "board",
            "bread0": "ingredient",
            "bread1": "ingredient",
            "cheese0": "ingredient",
            "egg0": "ingredient",
            "green_pepper0": "ingredient",
            "ham0": "ingredient",
            "holder": "holder",
            "lettuce0": "ingredient",
            "patty0": "ingredient",
            "robby": "robot",
            "tomato0": "ingredient"
        }
    }

    with tempfile.TemporaryDirectory() as json_dir:
        json_file = Path(json_dir) / "example_task1.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(task_spec, f)

        utils.reset_config({
            "env": "sandwich",
            "num_test_tasks": 1,
            "test_task_json_dir": json_dir
        })

        env = SandwichEnv()
        test_tasks = [t.task for t in env.get_test_tasks()]

    assert len(test_tasks) == 1
    task = test_tasks[0]

    assert str(
        sorted(task.goal)
    ) == "[On(bread1:ingredient, cheese0:ingredient), On(cheese0:ingredient, bread0:ingredient), OnBoard(bread0:ingredient, board:board)]"  # pylint:disable=line-too-long

    # Test language-based goal specification.
    task_spec = task_spec.copy()
    del task_spec["goal"]
    task_spec["language_goal"] = "Make me a cheese sandwich."

    with tempfile.TemporaryDirectory() as json_dir:
        json_file = Path(json_dir) / "example_task2.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(task_spec, f)

        utils.reset_config({
            "env": "sandwich",
            "num_test_tasks": 1,
            "test_task_json_dir": json_dir
        })

        env = SandwichEnv()

        with patch(f"{_LLM_MODULE_PATH}.OpenAILLM.sample_completions") as \
            mock_sample_completions:
            mock_sample_completions.return_value = [
                """
{"On": [["bread1", "cheese0"], ["cheese0", "bread0"]],
 "OnBoard": [["bread0", "board"]]}"""
            ]
            test_tasks = [t.task for t in env.get_test_tasks()]

    assert len(test_tasks) == 1
    task = test_tasks[0]
    assert str(
        sorted(task.goal)
    ) == "[On(bread1:ingredient, cheese0:ingredient), On(cheese0:ingredient, bread0:ingredient), OnBoard(bread0:ingredient, board:board)]"  # pylint:disable=line-too-long
