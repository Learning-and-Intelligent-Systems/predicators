"""Test cases for the Spot Env environments."""

import tempfile
from typing import List

import dill as pkl
import numpy as np
import pytest

from predicators import utils
from predicators.approaches import create_approach
from predicators.cogman import CogMan
from predicators.envs import create_new_env
from predicators.envs.spot_env import SpotCubeEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.perception.spot_perceiver import SpotPerceiver
from predicators.settings import CFG
from predicators.spot_utils.skills.spot_navigation import go_home
from predicators.structs import Action, GroundAtom, _GroundNSRT


@pytest.mark.parametrize("env", ["spot_cube_env", "spot_soda_sweep_env"])
def test_spot_env_dry_run(env) -> None:
    """Dry run tests (do not require access to robot)."""
    utils.reset_config({
        "env": env,
        "approach": "spot_wrapper[oracle]",
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "seed": 123,
        "spot_run_dry": True,
        "bilevel_plan_without_sim": True,
    })
    env = create_new_env(env)
    perceiver = SpotPerceiver()
    execution_monitor = create_execution_monitor("expected_atoms")
    env_train_tasks = env.get_train_tasks()
    env_test_tasks = env.get_test_tasks()
    train_tasks = [perceiver.reset(t) for t in env_train_tasks]
    options = get_gt_options(env.get_name())
    approach = create_approach(CFG.approach, env.predicates, options,
                               env.types, env.action_space, train_tasks)
    cogman = CogMan(approach, perceiver, execution_monitor)
    env_task = env_test_tasks[0]
    cogman.reset(env_task)
    obs = env.reset("test", 0)
    for _ in range(100):
        if env.goal_reached():
            break
        act = cogman.step(obs)
        assert act is not None
        obs = env.step(act)
    assert env.goal_reached()


def real_robot_cube_env_test() -> None:
    """A real robot test, not to be run by unit tests!

    Run this test by running the file directly, i.e.,

    python tests/envs/test_spot_envs.py --spot_robot_ip <ip address>

    Optionally load the last initial state:

    python tests/envs/test_spot_envs.py --spot_robot_ip <ip address> \
        --test_task_json_dir predicators/envs/assets/task_jsons/spot/
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
    perceiver = SpotPerceiver()
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
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    MoveToToolOnSurface = nsrt_name_to_nsrt["MoveToToolOnSurface"]
    MoveToSurface = nsrt_name_to_nsrt["MoveToSurface"]
    ground_nsrts: List[_GroundNSRT] = []
    for nsrt in sorted(nsrts):
        ground_nsrts.extend(utils.all_ground_nsrts(nsrt, set(state)))

    # The robot gripper should be empty, and the cube should be on a table.
    pred_name_to_pred = {p.name: p for p in env.predicates}
    HandEmpty = pred_name_to_pred["HandEmpty"]
    On = pred_name_to_pred["On"]
    InViewTool = pred_name_to_pred["InViewTool"]
    HoldingTool = pred_name_to_pred["HoldingTool"]
    OnFloor = pred_name_to_pred["OnFloor"]
    assert GroundAtom(HandEmpty, [spot]).holds(state)
    on_atoms = [GroundAtom(On, [cube, t]) for t in [table1, table2]]
    true_on_atoms = [a for a in on_atoms if a.holds(state)]
    assert len(true_on_atoms) == 1
    _, init_table = true_on_atoms[0].objects
    target_table = table1 if init_table is table2 else table2

    # Find the applicable NSRTs.
    atoms = utils.abstract(state, env.predicates)
    applicable_nsrts = list(utils.get_applicable_operators(
        ground_nsrts, atoms))
    move_to_cube_nsrt = MoveToToolOnSurface.ground((spot, cube, init_table))
    assert set(applicable_nsrts) == {
        move_to_cube_nsrt,
        MoveToSurface.ground((spot, init_table)),
        MoveToSurface.ground((spot, target_table)),
    }

    # Sample and run an option to move to the surface.
    option = move_to_cube_nsrt.sample_option(state, set(), rng)
    actions: List[Action] = []  # to test pickling
    assert option.initiable(state)
    for _ in range(100):  # should terminate much earlier
        action = option.policy(state)
        actions.append(action)
        obs = env.step(action)
        state = perceiver.step(obs)
        perceiver.update_perceiver_with_action(action)
        if option.terminal(state):
            break

    with tempfile.NamedTemporaryFile(mode="wb") as f:
        pkl.dump((nsrts, task, state, actions), f)

    # Check that moving succeeded.
    assert GroundAtom(InViewTool, [spot, cube]).holds(state)

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
    assert not GroundAtom(HandEmpty, [spot]).holds(state)
    assert GroundAtom(HoldingTool, [spot, cube]).holds(state)

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
    assert GroundAtom(HandEmpty, [spot]).holds(state)
    assert not GroundAtom(HoldingTool, [spot, cube]).holds(state)
    assert GroundAtom(On, [cube, target_table]).holds(state)

    # Sample and run an option to move to the surface.
    option = move_to_cube_nsrt.sample_option(state, set(), rng)
    actions = []  # to test pickling
    assert option.initiable(state)
    for _ in range(100):  # should terminate much earlier
        action = option.policy(state)
        actions.append(action)
        obs = env.step(action)
        state = perceiver.step(obs)
        perceiver.update_perceiver_with_action(action)
        if option.terminal(state):
            break

    # Check that moving succeeded.
    assert GroundAtom(InViewTool, [spot, cube]).holds(state)
    assert GroundAtom(On, [cube, target_table]).holds(state)
    assert GroundAtom(HandEmpty, [spot]).holds(state)

    # Sample an option to pick from the surface again.
    GraspToolFromSurface = nsrt_name_to_nsrt["GraspToolFromSurface"]
    grasp_cube_nsrt = GraspToolFromSurface.ground([spot, cube, target_table])
    assert all(a.holds(state) for a in grasp_cube_nsrt.preconditions)
    option = grasp_cube_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):  # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that picking succeeded.
    assert not GroundAtom(HandEmpty, [spot]).holds(state)
    assert GroundAtom(HoldingTool, [spot, cube]).holds(state)

    # Navigate home.
    localizer = env._localizer  # pylint: disable=protected-access
    assert localizer is not None
    localizer.localize()
    robot = env._robot  # pylint: disable=protected-access
    assert robot is not None
    go_home(robot, localizer)

    # Drop the object onto the floor.
    PlaceToolOnFloor = nsrt_name_to_nsrt["PlaceToolOnFloor"]
    place_cube_nrst = PlaceToolOnFloor.ground([spot, cube])
    assert all(a.holds(state) for a in place_cube_nrst.preconditions)
    option = place_cube_nrst.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):  # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that placing on the floor succeeded.
    assert GroundAtom(HandEmpty, [spot]).holds(state)
    assert not GroundAtom(HoldingTool, [spot, cube]).holds(state)
    assert GroundAtom(OnFloor, [cube]).holds(state)

    # Move to the object on the floor.
    MoveToToolOnFloor = nsrt_name_to_nsrt["MoveToToolOnFloor"]
    move_to_cube_on_floor = MoveToToolOnFloor.ground([spot, cube])
    assert all(a.holds(state) for a in move_to_cube_on_floor.preconditions)
    option = move_to_cube_on_floor.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(100):  # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Check that moving succeeded.
    assert GroundAtom(InViewTool, [spot, cube]).holds(state)

    # Now pick from floor.
    GraspToolFromFloorOp = nsrt_name_to_nsrt["GraspToolFromFloor"]
    grasp_cube_nrst = GraspToolFromFloorOp.ground([spot, cube])
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
    assert not GroundAtom(HandEmpty, [spot]).holds(state)
    assert GroundAtom(HoldingTool, [spot, cube]).holds(state)


if __name__ == "__main__":
    real_robot_cube_env_test()
