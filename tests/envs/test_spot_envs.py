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
        "approach": "spot_wrapper[oracle]",
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
            "measuring_tape": "tool",
            "low_wall_rack": "flat_surface",
            "tool_room_table": "flat_surface",
            "bucket": "bag",
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
            "measuring_tape": {
                "x": 9.90738,
                "y": -6.84972,
                "z": 0.643172,
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
            "bucket": {
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
                "yaw": 0.0,
            }
        },
        "goal": {
            "InBag": [["hammer", "bucket"], ["brush", "bucket"],
                      ["measuring_tape", "bucket"]]
        }
    }

    with tempfile.TemporaryDirectory() as json_dir:
        json_file = Path(json_dir) / "example_task1.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(task_spec, f)

        utils.reset_config({
            "env": "spot_bike_env",
            "approach": "spot_wrapper[oracle]",
            "num_train_tasks": 0,
            "num_test_tasks": 1,
            "test_task_json_dir": json_dir
        })

        env = SpotBikeEnv()
        test_tasks = env.get_test_tasks()

    assert len(test_tasks) == 1
    task = test_tasks[0]

    # pylint:disable=line-too-long
    assert str(
        sorted(task.goal)
    ) == "[InBag(brush:tool, bucket:bag), InBag(hammer:tool, bucket:bag), InBag(measuring_tape:tool, bucket:bag)]"
