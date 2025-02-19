"""Test cases for the Spot Env environments."""

import tempfile
from typing import List

import dill as pkl
import numpy as np
import pytest
from bosdyn.client import math_helpers

from predicators import utils
from predicators.approaches import create_approach
from predicators.cogman import CogMan
from predicators.envs import create_new_env
from predicators.envs.spot_env import SpotBallAndCupStickyTableEnv, \
    SpotCubeEnv, SpotMainSweepEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.perception.spot_perceiver import SpotPerceiver
from predicators.settings import CFG
from predicators.spot_utils.skills.spot_hand_move import close_gripper, \
    move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_navigation import go_home
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.structs import Action, GroundAtom, _GroundNSRT


def test_spot_env_dry_run():
    """Dry run tests (do not require access to robot)."""
    utils.reset_config({
        "env": "spot_cube_env",
        "approach": "spot_wrapper[oracle]",
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "seed": 123,
        "spot_run_dry": True,
        "bilevel_plan_without_sim": True,
        "spot_use_perfect_samplers": True,
        "spot_graph_nav_map": "floor8-v2",
    })
    env = create_new_env(CFG.env)
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


@pytest.mark.parametrize("graph_nav_map", ["floor8-v2", "floor8-sweeping"])
def test_spot_main_sweep_env_dry_run(graph_nav_map):
    """Tests specific to the main sweeping environment."""
    utils.reset_config({
        "env": "spot_main_sweep_env",
        "approach": "spot_wrapper[oracle]",
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "seed": 123,
        "spot_run_dry": True,
        "bilevel_plan_without_sim": True,
        "spot_use_perfect_samplers": True,
        "spot_graph_nav_map": graph_nav_map,
    })
    # Need to flush cache due to cached graph nav map.
    utils.flush_cache()
    env = create_new_env(CFG.env)
    perceiver = SpotPerceiver()
    execution_monitor = create_execution_monitor("expected_atoms")
    env_train_tasks = env.get_train_tasks()
    env_test_tasks = env.get_test_tasks()
    train_tasks = [perceiver.reset(t) for t in env_train_tasks]
    options = get_gt_options(env.get_name())
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    approach = create_approach(CFG.approach, env.predicates, options,
                               env.types, env.action_space, train_tasks)
    cogman = CogMan(approach, perceiver, execution_monitor)
    env_task = env_test_tasks[0]
    cogman.reset(env_task)
    init_obs = env.reset("test", 0)
    state = perceiver.step(init_obs)

    # Test that we can sweep the train_toy into the bucket, dump it out, then
    # do the whole thing again.
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    MoveToReachObject = nsrt_name_to_nsrt["MoveToReachObject"]
    MoveToHandViewObject = nsrt_name_to_nsrt["MoveToHandViewObject"]
    PickObjectFromTop = nsrt_name_to_nsrt["PickObjectFromTop"]
    PickObjectToDrag = nsrt_name_to_nsrt["PickObjectToDrag"]
    PlaceObjectOnTop = nsrt_name_to_nsrt["PlaceObjectOnTop"]
    DragToUnblockObject = nsrt_name_to_nsrt["DragToUnblockObject"]
    SweepIntoContainer = nsrt_name_to_nsrt["SweepIntoContainer"]
    PrepareContainerForSweeping = nsrt_name_to_nsrt[
        "PrepareContainerForSweeping"]
    PickAndDumpContainer = nsrt_name_to_nsrt["PickAndDumpContainer"]

    rng = np.random.default_rng(123)

    obj_name_to_obj = {o.name: o for o in state}
    robot = obj_name_to_obj["robot"]
    bucket = obj_name_to_obj["bucket"]
    brush = obj_name_to_obj["brush"]
    train_toy = obj_name_to_obj["train_toy"]
    chair = obj_name_to_obj["chair"]
    table = obj_name_to_obj["black_table"]
    floor = obj_name_to_obj["floor"]

    def _run_ground_nsrt(ground_nsrt,
                         state,
                         override_params=None,
                         assert_add_effects=True,
                         assert_delete_effects=True):

        def obs_to_state(obs):
            option = ground_nsrt.sample_option(state, set(), rng)
            assert option.initiable(state)
            action = option.policy(state)
            perceiver.update_perceiver_with_action(action)
            perceiver.step(obs)

            # Uncomment for debugging. Also add "render_state_dpi": 150.
            # imgs = perceiver.render_mental_images(obs, env_task)
            # import cv2
            # cv2.imshow("Mental image", imgs[0])
            # cv2.waitKey(0)

            return perceiver._create_state()  # pylint: disable=protected-access

        return utils.run_ground_nsrt_with_assertions(
            ground_nsrt,
            state,
            env,
            rng,
            override_params,
            obs_to_state,
            assert_add_effects=assert_add_effects,
            assert_delete_effects=assert_delete_effects)

    # Set up all the NSRTs for the following tests.
    move_to_hand_view_bucket = MoveToHandViewObject.ground([robot, bucket])
    pick_bucket = PickObjectFromTop.ground([robot, bucket, floor])
    prepare_bucket = PrepareContainerForSweeping.ground(
        [robot, bucket, train_toy, table])
    move_to_hand_view_chair = MoveToHandViewObject.ground([robot, chair])
    pick_chair = PickObjectToDrag.ground([robot, chair])
    drag_chair = DragToUnblockObject.ground([robot, chair, train_toy])
    move_to_hand_view_brush = MoveToHandViewObject.ground([robot, brush])
    pick_brush = PickObjectFromTop.ground([robot, brush, floor])
    move_to_reach_train_toy = MoveToReachObject.ground([robot, train_toy])
    sweep = SweepIntoContainer.ground([robot, brush, train_toy, table, bucket])
    move_to_reach_floor = MoveToReachObject.ground([robot, floor])
    place_brush = PlaceObjectOnTop.ground([robot, brush, floor])
    dump_bucket = PickAndDumpContainer.ground(
        [robot, bucket, floor, train_toy])
    move_to_hand_view_train_toy = MoveToHandViewObject.ground(
        [robot, train_toy])
    pick_train_toy = PickObjectFromTop.ground([robot, train_toy, floor])
    move_to_reach_table = MoveToReachObject.ground([robot, table])
    place_train_toy = PlaceObjectOnTop.ground([robot, train_toy, table])

    # Assertions will be raised in _run_ground_nsrt if there are any issues.
    state = _run_ground_nsrt(move_to_hand_view_bucket, state)
    state = _run_ground_nsrt(pick_bucket, state)
    state = _run_ground_nsrt(prepare_bucket, state)
    state = _run_ground_nsrt(move_to_hand_view_chair, state)
    state = _run_ground_nsrt(pick_chair, state, assert_delete_effects=False)
    state = _run_ground_nsrt(drag_chair, state)
    state = _run_ground_nsrt(move_to_hand_view_brush, state)
    state = _run_ground_nsrt(pick_brush, state)
    state = _run_ground_nsrt(move_to_reach_train_toy, state)
    state = _run_ground_nsrt(sweep, state, assert_delete_effects=False)
    state = _run_ground_nsrt(move_to_reach_floor, state)
    state = _run_ground_nsrt(place_brush, state)
    state = _run_ground_nsrt(move_to_hand_view_bucket, state)
    state = _run_ground_nsrt(dump_bucket, state)
    state = _run_ground_nsrt(move_to_reach_floor, state)
    state = _run_ground_nsrt(move_to_hand_view_train_toy, state)
    state = _run_ground_nsrt(pick_train_toy, state)
    state = _run_ground_nsrt(move_to_reach_table, state)
    state = _run_ground_nsrt(place_train_toy, state)
    state = _run_ground_nsrt(move_to_hand_view_bucket, state)
    state = _run_ground_nsrt(pick_bucket, state)
    state = _run_ground_nsrt(prepare_bucket, state)
    state = _run_ground_nsrt(move_to_hand_view_brush, state)
    state = _run_ground_nsrt(pick_brush, state)
    state = _run_ground_nsrt(move_to_reach_train_toy, state)
    state = _run_ground_nsrt(sweep, state, assert_delete_effects=False)
    state = _run_ground_nsrt(move_to_reach_floor, state)
    state = _run_ground_nsrt(place_brush, state)
    state = _run_ground_nsrt(move_to_hand_view_bucket, state)
    _run_ground_nsrt(dump_bucket, state)


def test_json_loading():
    """Tests specific to the main sweeping environment."""
    utils.reset_config({
        "env":
        "spot_soda_floor_env",
        "approach":
        "spot_wrapper[oracle]",
        "num_train_tasks":
        0,
        "num_test_tasks":
        1,
        "seed":
        123,
        "spot_run_dry":
        True,
        "bilevel_plan_without_sim":
        True,
        "spot_use_perfect_samplers":
        True,
        "spot_graph_nav_map":
        "floor8-sweeping",
        "test_task_json_dir":
        "predicators/envs/assets/task_jsons/spot/test/"
    })
    # Need to flush cache due to cached graph nav map.
    utils.flush_cache()
    env = create_new_env(CFG.env)
    env_test_tasks = env.get_test_tasks()
    assert len(env_test_tasks) == 1


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
    assert len(obs.objects_in_view) == 4
    cube, floor, table1, table2 = sorted(obs.objects_in_view)
    assert cube.name == "cube"
    assert "table" in table1.name
    assert "table" in table2.name
    state = perceiver.step(obs)
    spot = next(o for o in state if o.type.name == "robot")
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    MoveToHandViewObject = nsrt_name_to_nsrt["MoveToHandViewObject"]
    MoveToReachObject = nsrt_name_to_nsrt["MoveToReachObject"]
    ground_nsrts: List[_GroundNSRT] = []
    for nsrt in sorted(nsrts):
        ground_nsrts.extend(utils.all_ground_nsrts(nsrt, set(state)))

    # The robot gripper should be empty, and the cube should be on a table.
    pred_name_to_pred = {p.name: p for p in env.predicates}
    HandEmpty = pred_name_to_pred["HandEmpty"]
    On = pred_name_to_pred["On"]
    InHandView = pred_name_to_pred["InHandView"]
    Holding = pred_name_to_pred["Holding"]
    OnFloor = pred_name_to_pred["On"]
    assert GroundAtom(HandEmpty, [spot]).holds(state)
    on_atoms = [GroundAtom(On, [cube, t]) for t in [table1, table2]]
    true_on_atoms = [a for a in on_atoms if a.holds(state)]
    assert len(true_on_atoms) == 1
    _, init_table = true_on_atoms[0].objects
    target_table = table1 if init_table is table2 else table2

    # Find the applicable NSRTs.
    move_to_cube_nsrt = MoveToHandViewObject.ground((spot, cube))
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
    assert GroundAtom(InHandView, [spot, cube]).holds(state)

    # Now sample and run an option to pick from the surface.
    PickObjectFromTop = nsrt_name_to_nsrt["PickObjectFromTop"]
    grasp_cube_nrst = PickObjectFromTop.ground([spot, cube, init_table])
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
    assert GroundAtom(Holding, [spot, cube]).holds(state)

    # Sample and run an option to move to the surface.
    move_to_target_table_nsrt = MoveToReachObject.ground([spot, target_table])
    for a in move_to_target_table_nsrt.preconditions:
        print(a, a.holds(state))
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
    PlaceObjectOnTop = nsrt_name_to_nsrt["PlaceObjectOnTop"]
    place_on_table_nsrt = PlaceObjectOnTop.ground([spot, cube, target_table])
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
    assert not GroundAtom(Holding, [spot, cube]).holds(state)
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
    assert GroundAtom(InHandView, [spot, cube]).holds(state)
    assert GroundAtom(On, [cube, target_table]).holds(state)
    assert GroundAtom(HandEmpty, [spot]).holds(state)

    # Sample an option to pick from the surface again.
    PickObjectFromTop = nsrt_name_to_nsrt["PickObjectFromTop"]
    grasp_cube_nsrt = PickObjectFromTop.ground([spot, cube, target_table])
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
    assert GroundAtom(Holding, [spot, cube]).holds(state)

    # Navigate home.
    localizer = env._localizer  # pylint: disable=protected-access
    assert localizer is not None
    localizer.localize()
    robot = env._robot  # pylint: disable=protected-access
    assert robot is not None
    go_home(robot, localizer)

    # Drop the object onto the floor.
    PlaceToolOnFloor = nsrt_name_to_nsrt["PlaceObjectOnTop"]
    place_cube_nrst = PlaceToolOnFloor.ground([spot, cube, floor])
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
    assert not GroundAtom(Holding, [spot, cube]).holds(state)
    assert GroundAtom(OnFloor, [cube, floor]).holds(state)

    # Move to the object on the floor.
    MoveToHandViewObject = nsrt_name_to_nsrt["MoveToHandViewObject"]
    move_to_cube_on_floor = MoveToHandViewObject.ground([spot, cube])
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
    assert GroundAtom(InHandView, [spot, cube]).holds(state)

    # Now pick from floor.
    PickObjectFromTop = nsrt_name_to_nsrt["PickObjectFromTop"]
    grasp_cube_nrst = PickObjectFromTop.ground([spot, cube, floor])
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
    assert GroundAtom(Holding, [spot, cube]).holds(state)


def real_robot_drafting_table_placement_test() -> None:
    """Another real robot test, not to be run by unit tests! Mostly for
    debugging the place sampler on the drafting table, which seems to be
    surprisingly biased to the left side of the table. Note that this test
    doesn't assert anything: a user must manually check that the agent is
    sampling a different point on the surface every time.

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
        "spot_ball_and_cup_sticky_table_env",
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
        "spot_graph_nav_map":
        args["spot_graph_nav_map"],
        "test_task_json_dir":
        args.get("test_task_json_dir", None),
    })
    rng = np.random.default_rng(123)
    env = SpotBallAndCupStickyTableEnv()
    perceiver = SpotPerceiver()
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))

    # This should have the robot spin around and locate all objects.
    task = env.get_test_tasks()[0]
    obs = env.reset("test", 0)
    perceiver.reset(task)
    objects_in_view = {o.name: o for o in obs.objects_in_view}
    cup = objects_in_view["cup"]
    drafting_table = objects_in_view["drafting_table"]
    state = perceiver.step(obs)
    spot = next(o for o in state if o.type.name == "robot")
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    MoveToReachObject = nsrt_name_to_nsrt["MoveToReachObject"]
    PlaceObjectOnTop = nsrt_name_to_nsrt["PlaceObjectOnTop"]
    ground_nsrts: List[_GroundNSRT] = []
    for nsrt in sorted(nsrts):
        ground_nsrts.extend(utils.all_ground_nsrts(nsrt, set(state)))

    # First, move to the drafting table.
    move_to_drafting_table_nsrt = MoveToReachObject.ground(
        (spot, drafting_table))
    # Sample and run an option to move to the surface.
    option = move_to_drafting_table_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    for _ in range(10):  # should terminate much earlier
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        if option.terminal(state):
            break

    # Now, sample a placement option multiple times and run it.
    place_on_table_nsrt = PlaceObjectOnTop.ground([spot, cup, drafting_table])
    for _ in range(10):
        option = place_on_table_nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        for _ in range(100):  # should terminate much earlier
            action = option.policy(state)
            obs = env.step(action)
            perceiver.update_perceiver_with_action(action)
            state = perceiver.step(obs)
            if option.terminal(state):
                break


def real_robot_sweeping_nsrt_test() -> None:
    """Test for running the sweeping skill and base sampler on the real robot.

    This is similar to the test in spot_sweep.py, but it uses the whole NSRT.

    Run this test by running the file directly, i.e.,

    python tests/envs/test_spot_envs.py --spot_robot_ip <ip address>
    """
    args = utils.parse_args(env_required=False,
                            seed_required=False,
                            approach_required=False)
    utils.reset_config({
        "env": "spot_main_sweep_env",
        "approach": "spot_wrapper[oracle]",
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "seed": 123,
        "spot_robot_ip": args["spot_robot_ip"],
    })
    rng = np.random.default_rng(123)
    env = SpotMainSweepEnv()
    robot = env._robot  # pylint: disable=protected-access
    perceiver = SpotPerceiver()
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))

    # This should have the robot spin around and locate all objects.
    task = env.get_test_tasks()[0]
    obs = env.reset("test", 0)
    perceiver.reset(task)
    objects_in_view = {o.name: o for o in obs.objects_in_view}
    train_toy = objects_in_view["train_toy"]
    football = objects_in_view["football"]
    table = objects_in_view["black_table"]
    container = objects_in_view["bucket"]
    brush = objects_in_view["brush"]
    floor = objects_in_view["floor"]
    state = perceiver.step(obs)
    spot = next(o for o in state if o.type.name == "robot")
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    MoveToReadySweep = nsrt_name_to_nsrt["MoveToReadySweep"]
    SweepTwoObjectsIntoContainer = \
        nsrt_name_to_nsrt["SweepTwoObjectsIntoContainer"]
    MoveToReachObject = nsrt_name_to_nsrt["MoveToReachObject"]
    PlaceObjectOnTop = nsrt_name_to_nsrt["PlaceObjectOnTop"]
    move_to_container_nsrt = MoveToReadySweep.ground(
        (spot, container, train_toy))
    sweep_nsrt = SweepTwoObjectsIntoContainer.ground(
        (spot, brush, train_toy, football, table, container))
    move_to_floor_nsrt = MoveToReachObject.ground((spot, floor))
    place_nsrt = PlaceObjectOnTop.ground((spot, brush, floor))

    # Ask for the brush.
    hand_side_pose = math_helpers.SE3Pose(x=0.80,
                                          y=0.0,
                                          z=0.25,
                                          rot=math_helpers.Quat.from_yaw(
                                              -np.pi / 2))
    move_hand_to_relative_pose(robot, hand_side_pose)
    open_gripper(robot)
    # Press any key, instead of just enter. Useful for remote control.
    msg = "Put the brush in the robot's gripper, then press any key"
    utils.wait_for_any_button_press(msg)
    close_gripper(robot)
    stow_arm(robot)

    nsrt_sequence: List[_GroundNSRT] = []
    # Sweep 10 times.
    nsrt_sequence += [move_to_container_nsrt, sweep_nsrt] * 10
    # Drop the sweeper on the floor.
    nsrt_sequence += [move_to_floor_nsrt, place_nsrt]

    for nsrt in nsrt_sequence:
        option = nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        action = option.policy(state)
        obs = env.step(action)
        perceiver.update_perceiver_with_action(action)
        state = perceiver.step(obs)
        assert option.terminal(state)


if __name__ == "__main__":
    real_robot_cube_env_test()
    # real_robot_drafting_table_placement_test()
    # real_robot_sweeping_nsrt_test()
