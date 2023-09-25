"""Test cases for the Spot Env environments."""

import json
import tempfile
from pathlib import Path

import numpy as np

from predicators import utils
from predicators.envs.spot_env import SpotBikeEnv, SpotCubeEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.perception.spot_bike_perceiver import SpotBikePerceiver
from predicators.settings import CFG


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


def real_robot_cube_env_test():
    """A real robot test, not to be run by unit tests!

    Run this test by running the file directly, i.e.,

    python tests/envs/test_spot_envs.py --spot_robot_ip <ip address>

    TODO: support the below, which currently will crash.

    Optionally load the last initial state:

    python tests/envs/test_spot_envs.py --spot_robot_ip <ip address> \
        --test_task_json_dir predicators/envs/assets/task_jsons/spot_bike_env/
    """
    args = utils.parse_args(env_required=False,
                            seed_required=False,
                            approach_required=False)
    utils.reset_config({
        "env":
        "spot_cube_env",
        "approach":
        "spot_wrapper[oracle]",
        "num_train_tasks":
        0,
        "num_test_tasks":
        1,
        "seed":
        123,
        "spot_robot_ip":
        args["spot_robot_ip"],
        "test_task_json_dir":
        args.get("test_task_json_dir", None),
    })
    rng = np.random.default_rng(123)
    env = SpotCubeEnv()
    perceiver = SpotBikePerceiver()
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))

    # This should have the robot spin around and locate all objects.
    task = env.get_test_tasks()[0]
    obs = env.reset("test", 0)
    perceiver.reset(task)
    assert len(obs.objects_in_view) == 3
    cube, table1, table2 = sorted(obs.objects_in_view)
    assert cube.name == "cube"
    assert "table" in table1.name
    assert "table" in table2.name
    state = perceiver.step(obs)
    spot = next(o for o in state if o.type.name == "robot")

    # The robot gripper should be empty, and the cube should be on a table.
    pred_name_to_pred = {p.name: p for p in env.predicates}
    HandEmpty = pred_name_to_pred["HandEmpty"]
    On = pred_name_to_pred["On"]
    InViewTool = pred_name_to_pred["InViewTool"]
    HoldingTool = pred_name_to_pred["HoldingTool"]
    assert HandEmpty([spot]).holds(state)
    on_atoms = [On([cube, t]) for t in [table1, table2]]
    true_on_atoms = [a for a in on_atoms if a.holds(state)]
    assert len(true_on_atoms) == 1
    _, init_table = true_on_atoms[0].objects
    target_table = table1 if init_table is table2 else table2

    # Find the applicable NSRTs.
    ground_nsrts = []
    for nsrt in sorted(nsrts):
        ground_nsrts.extend(utils.all_ground_nsrts(nsrt, set(state)))
    atoms = utils.abstract(state, env.predicates)
    applicable_nsrts = list(utils.get_applicable_operators(
        ground_nsrts, atoms))
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    MoveToToolOnSurface = nsrt_name_to_nsrt["MoveToToolOnSurface"]
    MoveToSurface = nsrt_name_to_nsrt["MoveToSurface"]
    move_to_cube_nsrt = MoveToToolOnSurface.ground((spot, cube, init_table))
    assert set(applicable_nsrts) == {
        move_to_cube_nsrt,
        MoveToSurface.ground((spot, init_table)),
        MoveToSurface.ground((spot, target_table)),
    }

    # Sample and run an option to move to the surface.
    option = move_to_cube_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):  # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        state = perceiver.step(obs)
        perceiver.update_perceiver_with_action(action)
        if option.terminal(state):
            break

    # Check that moving succeeded.
    assert InViewTool([spot, cube]).holds(state)

    # Now sample and run an option to pick from the surface.
    GraspToolFromSurface = nsrt_name_to_nsrt["GraspToolFromSurface"]
    grasp_cube_nrst = GraspToolFromSurface.ground([spot, cube, init_table])
    assert all(a.holds(state) for a in grasp_cube_nrst.preconditions)
    option = grasp_cube_nrst.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):  # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that picking succeeded.
    assert not HandEmpty([spot]).holds(state)
    assert HoldingTool([spot, cube]).holds(state)

    # Sample and run an option to move to the surface.
    move_to_target_table_nsrt = MoveToSurface.ground([spot, target_table])
    assert all(a.holds(state) for a in move_to_target_table_nsrt.preconditions)
    option = move_to_target_table_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):  # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Sample and run an option to place on the surface.
    PlaceToolOnSurface = nsrt_name_to_nsrt["PlaceToolOnSurface"]
    place_on_table_nsrt = PlaceToolOnSurface.ground([spot, cube, target_table])
    assert all(a.holds(state) for a in place_on_table_nsrt.preconditions)
    option = place_on_table_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):  # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that placing on the table succeeded.
    assert HandEmpty([spot]).holds(state)
    assert not HoldingTool([spot, cube]).holds(state)
    assert On([cube, target_table]).holds(state)


if __name__ == "__main__":
    real_robot_cube_env_test()
