"""Test cases for the Ball and Cup Sticky Table environment."""

import numpy as np

from predicators import utils
from predicators.envs.ball_and_cup_sticky_table import BallAndCupStickyTableEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.structs import Action


def test_sticky_table():
    """Tests for the Ball and Cup Sticky Table environment."""
    utils.reset_config({
        "env": "ball_and_cup_sticky_table",
        "num_train_tasks": 1,
        "num_test_tasks": 2,
        "sticky_table_place_smooth_fall_prob": 1.0,
        "sticky_table_place_sticky_fall_prob": 0.0,
        "sticky_table_pick_success_prob": 1.0,
    })
    env = BallAndCupStickyTableEnv()
    assert env.get_name() == "ball_and_cup_sticky_table"
    for env_task in env.get_train_tasks():
        task = env_task.task
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for env_task in env.get_test_tasks():
        task = env_task.task
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.types) == 4
    type_name_to_type = {t.name: t for t in env.types}
    cup_type = type_name_to_type["cup"]
    table_type = type_name_to_type["table"]
    robot_type = type_name_to_type["robot"]
    ball_type = type_name_to_type["ball"]
    assert len(env.predicates) == 12
    pred_name_to_pred = {p.name: p for p in env.predicates}
    BallOnTable = pred_name_to_pred["BallOnTable"]
    CupOnTable = pred_name_to_pred["CupOnTable"]
    CupOnFloor = pred_name_to_pred["CupOnFloor"]
    assert env.goal_predicates == {BallOnTable}
    assert env.action_space.shape == (5, )
    options = get_gt_options(env.get_name())
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    NavigateToCup = nsrt_name_to_nsrt["NavigateToCup"]
    PickCupWithoutBallFromFloor = nsrt_name_to_nsrt["PickCupWithoutBallFromFloor"]
    NavigateToTable = nsrt_name_to_nsrt["NavigateToTable"]
    PlaceCupWithoutBallOnTable = nsrt_name_to_nsrt["PlaceCupWithoutBallOnTable"]
    PickCupWithoutBallFromTable = nsrt_name_to_nsrt["PickCupWithoutBallFromTable"]
    PlaceCupWithoutBallOnFloor = nsrt_name_to_nsrt["PlaceCupWithoutBallOnFloor"]

    assert len(options) == len(nsrts) == 17
    env_train_tasks = env.get_train_tasks()
    assert len(env_train_tasks) == 1
    env_test_tasks = env.get_test_tasks()
    assert len(env_test_tasks) == 2
    env_task = env_test_tasks[1]

    # Extract objects for NSRT testing.
    init_state = env_test_tasks[0].task.init
    rng = np.random.default_rng(123)

    robot, = init_state.get_objects(robot_type)
    ball, = init_state.get_objects(ball_type)
    cup, = init_state.get_objects(cup_type)
    tables = init_state.get_objects(table_type)
    sticky_tables = [t for t in tables if init_state.get(t, "sticky") > 0.5]
    assert len(sticky_tables) == 1
    sticky_table = sticky_tables[0]
    normal_tables = [t for t in tables if t != sticky_table]
    # The cup starts out on the floor.
    assert CupOnFloor([cup]).holds(init_state)
    assert not any(CupOnTable([cup, t]).holds(init_state) for t in tables)
    
    # Test noise-free cup picking and placing on the floor and normal tables.
    first_table = normal_tables[0]
    ground_nsrt_plan = [
        NavigateToCup.ground([robot, cup]),
        PickCupWithoutBallFromFloor.ground([robot, cup, ball]),
        NavigateToTable.ground([robot, first_table]),
        PlaceCupWithoutBallOnTable.ground([robot, ball, cup, first_table]),
    ]
    for table, next_table in zip(normal_tables[:-1], normal_tables[1:]):
        ground_nsrt_plan.append(PickCupWithoutBallFromTable.ground([robot, cup, ball, table]))
        ground_nsrt_plan.append(NavigateToTable.ground([robot, next_table]))
        ground_nsrt_plan.append(PlaceCupWithoutBallOnTable.ground([robot, ball, cup, next_table]))
        ground_nsrt_plan.append(PickCupWithoutBallFromTable.ground([robot, cup, ball, normal_tables[-1]]))
    ground_nsrt_plan.append(PlaceCupWithoutBallOnFloor.ground([robot, ball, cup]))
    state = env.reset("test", 0)
    for ground_nsrt in ground_nsrt_plan:
        state = utils.run_ground_nsrt_with_assertions(ground_nsrt, state, env, rng)


