"""Test cases for the Spot Env environments."""

import json
import tempfile
from pathlib import Path

from predicators import utils
from predicators.envs.spot_env import SpotBikeEnv


def test_spot_bike_env():
    """Tests for SpotBikeEnv class."""
    utils.reset_config({
        "env": "spot_bike_env",
        "num_train_tasks": 0,
        "num_test_tasks": 1
    })
    env = SpotBikeEnv()
    assert {pred.name
            for pred in env.goal_predicates
            } == {pred.name
                  for pred in env.predicates}


def test_spot_bike_env_load_task_from_json():
    """Tests for loading SpotBikeEnv tasks from a JSON file."""
    # Set up the JSON file.
    task_spec = {
        "objects": {
            "hammer": "tool",
            "brush": "tool",
            "hex_key": "tool",
            "hex_screwdriver": "tool",
            "low_wall_rack": "flat_surface",
            "tool_room_table": "flat_surface",
            "toolbag": "bag",
            "spot": "robot",
        },
        "init": {
            "hammer": {
                "x": 9.88252,
                "y": -7.10786,
                "z": 0.622855,
                "lost": 0.0,
                "in_view": 0.0
            },
            "brush": {
                "x": 6.43948,
                "y": -6.02389,
                "z": 0.174947,
                "lost": 0.0,
                "in_view": 0.0
            },
            "hex_key": {
                "x": 9.90738,
                "y": -6.84972,
                "z": 0.643172,
                "lost": 0.0,
                "in_view": 0.0
            },
            "hex_screwdriver": {
                "x": 6.57559,
                "y": -5.87017,
                "z": 0.286362,
                "lost": 0.0,
                "in_view": 0.0
            },
            "low_wall_rack": {
                "x": 10.0275,
                "y": -6.96979,
                "z": 0.275323,
            },
            "tool_room_table": {
                "x": 6.49849,
                "y": -6.25279,
                "z": -0.0138028,
            },
            "toolbag": {
                "x": 6.85457,
                "y": -8.19294,
                "z": -0.189187,
            },
            "spot": {
                "gripper_open_percentage": 0.42733,
                "curr_held_item_id": 0,
                "x": 8.46583,
                "y": -6.94704,
                "z": 0.131564,
            }
        },
        "goal": {
            "InBag": [["hammer", "toolbag"], ["brush", "toolbag"],
                      ["hex_key", "toolbag"], ["hex_screwdriver", "toolbag"]]
        }
    }

    with tempfile.TemporaryDirectory() as json_dir:
        json_file = Path(json_dir) / "example_task1.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(task_spec, f)

        utils.reset_config({
            "env": "spot_bike_env",
            "num_train_tasks": 0,
            "num_test_tasks": 1,
            "test_task_json_dir": json_dir
        })

        env = SpotBikeEnv()
        test_tasks = env.get_test_tasks()

    assert len(test_tasks) == 1
    task = test_tasks[0]

    # pylint: disable=line-too-long
    assert str(
        task.init_obs
    ) == "_SpotObservation(images={}, objects_in_view={brush:tool: (6.43948, -6.02389, 0.174947), hammer:tool: (9.88252, -7.10786, 0.622855), hex_key:tool: (9.90738, -6.84972, 0.643172), hex_screwdriver:tool: (6.57559, -5.87017, 0.286362), low_wall_rack:flat_surface: (10.0275, -6.96979, 0.275323), tool_room_table:flat_surface: (6.49849, -6.25279, -0.0138028), toolbag:bag: (6.85457, -8.19294, -0.189187)}, objects_in_hand_view=[], robot=spot:robot, gripper_open_percentage=0.42733, robot_pos=(8.46583, -6.94704, 0.131564), nonpercept_atoms=set(), nonpercept_predicates={InBag, ReachableSurface, PlatformNear, OnFloor, HoldingPlatformLeash, HoldingBag})"

    assert str(
        sorted(task.goal)
    ) == "[InBag(brush:tool, toolbag:bag), InBag(hammer:tool, toolbag:bag), InBag(hex_key:tool, toolbag:bag), InBag(hex_screwdriver:tool, toolbag:bag)]"
