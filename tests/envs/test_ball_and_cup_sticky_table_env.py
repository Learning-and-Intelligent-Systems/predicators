"""Test cases for the Ball and Cup Sticky Table environment."""

import numpy as np
import pytest

from predicators import utils
from predicators.envs.ball_and_cup_sticky_table import BallAndCupStickyTableEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options


def test_sticky_table():
    """Tests for the Ball and Cup Sticky Table environment."""
    utils.reset_config({
        "env": "ball_and_cup_sticky_table",
        "num_train_tasks": 1,
        "num_test_tasks": 2,
        "sticky_table_place_smooth_fall_prob": 1.0,
        "sticky_table_place_sticky_fall_prob": 0.0,
        "sticky_table_pick_success_prob": 1.0,
        "sticky_table_place_ball_fall_prob": 0.0,
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
    BallOnFloor = pred_name_to_pred["BallOnFloor"]
    CupOnTable = pred_name_to_pred["CupOnTable"]
    CupOnFloor = pred_name_to_pred["CupOnFloor"]
    assert env.goal_predicates == {BallOnTable}
    assert env.action_space.shape == (5, )
    options = get_gt_options(env.get_name())
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    NavigateToCup = nsrt_name_to_nsrt["NavigateToCup"]
    PickCupWithoutBallFromFloor = nsrt_name_to_nsrt[
        "PickCupWithoutBallFromFloor"]
    NavigateToTable = nsrt_name_to_nsrt["NavigateToTable"]
    PlaceCupWithoutBallOnTable = nsrt_name_to_nsrt[
        "PlaceCupWithoutBallOnTable"]
    PickCupWithoutBallFromTable = nsrt_name_to_nsrt[
        "PickCupWithoutBallFromTable"]
    PlaceCupWithoutBallOnFloor = nsrt_name_to_nsrt[
        "PlaceCupWithoutBallOnFloor"]
    PickBallFromTable = nsrt_name_to_nsrt["PickBallFromTable"]
    PlaceBallOnTable = nsrt_name_to_nsrt["PlaceBallOnTable"]
    PlaceBallOnFloor = nsrt_name_to_nsrt["PlaceBallOnFloor"]
    PickBallFromFloor = nsrt_name_to_nsrt["PickBallFromFloor"]
    NavigateToBall = nsrt_name_to_nsrt["NavigateToBall"]
    PlaceBallInCupOnFloor = nsrt_name_to_nsrt["PlaceBallInCupOnFloor"]
    PlaceBallInCupOnTable = nsrt_name_to_nsrt["PlaceBallInCupOnTable"]
    PickCupWithBallFromFloor = nsrt_name_to_nsrt["PickCupWithBallFromFloor"]
    # PlaceCupWithBallOnTable = nsrt_name_to_nsrt["PlaceCupWithBallOnTable"]
    PlaceCupWithBallOnFloor = nsrt_name_to_nsrt["PlaceCupWithBallOnFloor"]
    # PickCupWithBallFromTable = nsrt_name_to_nsrt["PickCupWithBallFromTable"]

    assert len(options) == len(nsrts) == 16
    env_train_tasks = env.get_train_tasks()
    assert len(env_train_tasks) == 1
    env_test_tasks = env.get_test_tasks()
    assert len(env_test_tasks) == 2
    env_task = env_test_tasks[1]

    # Test rendering.
    env.reset("test", 1)
    with pytest.raises(NotImplementedError):
        env.render(caption="Test")

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
    # The ball starts out on some table.
    ball_init_tables = [
        t for t in tables if BallOnTable([ball, t]).holds(init_state)
    ]
    assert len(ball_init_tables) == 1
    ball_init_table = ball_init_tables[0]

    # Test noise-free CUP picking and placing on the floor and normal tables.
    # Also test placing the ball into the cup on the floor.
    first_table = normal_tables[0]
    ground_nsrt_plan = [
        NavigateToCup.ground([robot, cup]),
        PickCupWithoutBallFromFloor.ground([robot, cup, ball]),
        NavigateToTable.ground([robot, first_table]),
        PlaceCupWithoutBallOnTable.ground([robot, ball, cup, first_table]),
    ]
    for table, next_table in zip(normal_tables[:-1], normal_tables[1:]):
        ground_nsrt_plan.append(
            PickCupWithoutBallFromTable.ground([robot, cup, ball, table]))
        ground_nsrt_plan.append(NavigateToTable.ground([robot, next_table]))
        ground_nsrt_plan.append(
            PlaceCupWithoutBallOnTable.ground([robot, ball, cup, next_table]))
    ground_nsrt_plan.append(
        PickCupWithoutBallFromTable.ground(
            [robot, cup, ball, normal_tables[-1]]))
    ground_nsrt_plan.append(
        PlaceCupWithoutBallOnFloor.ground([robot, ball, cup]))
    ground_nsrt_plan.append(NavigateToTable.ground([robot, ball_init_table]))
    ground_nsrt_plan.append(
        PickBallFromTable.ground([robot, ball, cup, ball_init_table]))
    ground_nsrt_plan.append(NavigateToCup.ground([robot, cup]))
    ground_nsrt_plan.append(PlaceBallInCupOnFloor.ground([robot, ball, cup]))
    state = env.reset("test", 0)
    for ground_nsrt in ground_nsrt_plan:
        state = utils.run_ground_nsrt_with_assertions(ground_nsrt, state, env,
                                                      rng)

    # Test noise-free BALL picking and placing on the floor and normal tables.
    table_order = [ball_init_table
                   ] + [t for t in normal_tables if t != ball_init_table]
    ground_nsrt_plan = [NavigateToTable.ground([robot, ball_init_table])]
    for table, next_table in zip(table_order[:-1], table_order[1:]):
        ground_nsrt_plan.append(
            PickBallFromTable.ground([robot, ball, cup, table]))
        ground_nsrt_plan.append(NavigateToTable.ground([robot, next_table]))
        ground_nsrt_plan.append(
            PlaceBallOnTable.ground([robot, ball, cup, next_table]))
    ground_nsrt_plan.append(
        PickBallFromTable.ground([robot, ball, cup, normal_tables[-1]]))
    ground_nsrt_plan.append(PlaceBallOnFloor.ground([robot, cup, ball]))
    ground_nsrt_plan.append(NavigateToBall.ground([robot, ball]))
    ground_nsrt_plan.append(PickBallFromFloor.ground([robot, ball, cup]))
    state = env.reset("test", 0)
    for ground_nsrt in ground_nsrt_plan:
        state = utils.run_ground_nsrt_with_assertions(ground_nsrt, state, env,
                                                      rng)

    # Test putting the cup on the table first and then the ball in the cup.
    table = ball_init_table
    ground_nsrt_plan = [
        NavigateToCup.ground([robot, cup]),
        PickCupWithoutBallFromFloor.ground([robot, cup, ball]),
        NavigateToTable.ground([robot, table]),
        PlaceCupWithoutBallOnTable.ground([robot, ball, cup, table]),
        PickBallFromTable.ground([robot, ball, cup, table]),
        PlaceBallInCupOnTable.ground([robot, ball, cup, table]),
    ]
    state = env.reset("test", 0)
    for ground_nsrt in ground_nsrt_plan:
        state = utils.run_ground_nsrt_with_assertions(ground_nsrt, state, env,
                                                      rng)

    # Test putting the ball in the cup first and then going to the table,
    # then placing back onto the floor.
    table = ball_init_table
    ground_nsrt_plan = [
        NavigateToTable.ground([robot, table]),
        PickBallFromTable.ground([robot, ball, cup, table]),
        NavigateToCup.ground([robot, cup]),
        PlaceBallInCupOnFloor.ground([robot, ball, cup]),
        PickCupWithBallFromFloor.ground([robot, cup, ball]),
        NavigateToTable.ground([robot, table]),
        PlaceCupWithBallOnFloor.ground([robot, ball, cup]),
    ]
    state = env.reset("test", 0)
    for ground_nsrt in ground_nsrt_plan:
        state = utils.run_ground_nsrt_with_assertions(ground_nsrt, state, env,
                                                      rng)

    # Test picking the ball from inside the cup on the floor.
    table = ball_init_table
    ground_nsrt_plan = [
        NavigateToTable.ground([robot, table]),
        PickBallFromTable.ground([robot, ball, cup, table]),
        NavigateToCup.ground([robot, cup]),
        PlaceBallInCupOnFloor.ground([robot, ball, cup]),
        PickBallFromFloor.ground([robot, ball, cup]),
    ]
    state = env.reset("test", 0)
    for ground_nsrt in ground_nsrt_plan:
        state = utils.run_ground_nsrt_with_assertions(ground_nsrt, state, env,
                                                      rng)

    # Test placing the ball on the sticky table, which should always fail.
    utils.reset_config({
        "env": "ball_and_cup_sticky_table",
        "num_train_tasks": 1,
        "num_test_tasks": 2,
    })
    ground_nsrt_plan = [
        NavigateToTable.ground([robot, ball_init_table]),
        PickBallFromTable.ground([robot, ball, cup, ball_init_table]),
        NavigateToTable.ground([robot, sticky_table]),
        PlaceBallOnTable.ground([robot, ball, cup, sticky_table]),
    ]
    # Test 10 times, with different samples per time.
    for _ in range(10):
        state = env.reset("test", 0)
        for i, ground_nsrt in enumerate(ground_nsrt_plan):
            if i == len(ground_nsrt_plan) - 1:
                state = utils.run_ground_nsrt_with_assertions(
                    ground_nsrt,
                    state,
                    env,
                    rng,
                    assert_add_effects=False,
                    assert_delete_effects=False)
                assert BallOnFloor([ball]).holds(state)
            else:
                state = utils.run_ground_nsrt_with_assertions(
                    ground_nsrt, state, env, rng)

    # Test placing the cup without the ball on the sticky table, which should
    # SOMETIMES fail.
    ground_nsrt_plan = [
        NavigateToCup.ground([robot, cup]),
        PickCupWithoutBallFromFloor.ground([robot, cup, ball]),
        NavigateToTable.ground([robot, sticky_table]),
        PlaceCupWithoutBallOnTable.ground([robot, ball, cup, sticky_table]),
    ]
    # Test 10 times, with different samples per time.
    num_success_places = 0
    for _ in range(10):
        state = env.reset("test", 0)
        for i, ground_nsrt in enumerate(ground_nsrt_plan):
            if i == len(ground_nsrt_plan) - 1:
                state = utils.run_ground_nsrt_with_assertions(
                    ground_nsrt,
                    state,
                    env,
                    rng,
                    assert_add_effects=False,
                    assert_delete_effects=False)
                if CupOnTable([cup, sticky_table]).holds(state):
                    num_success_places += 1
            else:
                state = utils.run_ground_nsrt_with_assertions(
                    ground_nsrt, state, env, rng)
    assert 0 < num_success_places < 10
