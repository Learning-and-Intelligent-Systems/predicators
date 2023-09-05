"""Test cases for the Sticky Table environment."""

import numpy as np

from predicators import utils
from predicators.envs.sticky_table import StickyTableEnv, \
    StickyTableTrickyFloorEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.structs import Action


def test_sticky_table():
    """Tests for the Sticky Table environment."""
    utils.reset_config({
        "env": "sticky_table",
        "num_train_tasks": 1,
        "num_test_tasks": 2,
        "sticky_table_place_smooth_fall_prob": 1.0,
        "sticky_table_place_sticky_fall_prob": 0.0,
        "sticky_table_pick_success_prob": 1.0,
    })
    env = StickyTableEnv()
    assert env.get_name() == "sticky_table"
    for env_task in env.get_train_tasks():
        task = env_task.task
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for env_task in env.get_test_tasks():
        task = env_task.task
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 4
    HandEmpty, Holding, OnFloor, OnTable = sorted(env.predicates)
    assert HandEmpty.name == "HandEmpty"
    assert Holding.name == "Holding"
    assert OnFloor.name == "OnFloor"
    assert OnTable.name == "OnTable"
    assert env.goal_predicates == {OnTable}
    options = get_gt_options(env.get_name())
    assert len(env.types) == 2
    cube_type, table_type = sorted(env.types)
    assert cube_type.name == "cube"
    assert table_type.name == "table"
    assert env.action_space.shape == (2, )
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    assert len(nsrts) == 4
    assert len(options) == len(nsrts)
    env_train_tasks = env.get_train_tasks()
    assert len(env_train_tasks) == 1
    env_test_tasks = env.get_test_tasks()
    assert len(env_test_tasks) == 2
    env_task = env_test_tasks[1]
    env.reset("test", 1)
    imgs = env.render(caption="Test")
    assert len(imgs) == 1

    # Test NSRTs.
    PickFromFloor, PickFromTable, PlaceOnFloor, PlaceOnTable = sorted(nsrts)
    assert PickFromFloor.name == "PickFromFloor"
    assert PickFromTable.name == "PickFromTable"
    assert PlaceOnFloor.name == "PlaceOnFloor"
    assert PlaceOnTable.name == "PlaceOnTable"

    init_state = env_test_tasks[0].task.init
    rng = np.random.default_rng(123)

    cube, = init_state.get_objects(cube_type)
    tables = init_state.get_objects(table_type)
    sticky_tables = [t for t in tables if init_state.get(t, "sticky") > 0.5]
    assert len(sticky_tables) == 1
    sticky_table = sticky_tables[0]
    normal_tables = [t for t in tables if t != sticky_table]
    init_table = [t for t in tables if OnTable([cube, t]).holds(init_state)][0]

    assert not OnFloor([cube]).holds(init_state)

    # Test noise-free picking and placing on the floor and normal tables.
    table_order = [init_table] + [t for t in normal_tables if t != init_table]
    ground_nsrt_plan = []
    for table, next_table in zip(table_order[:-1], table_order[1:]):
        ground_nsrt_plan.append(PickFromTable.ground([cube, table]))
        ground_nsrt_plan.append(PlaceOnTable.ground([cube, next_table]))
    ground_nsrt_plan.append(PickFromTable.ground([cube, table_order[-1]]))
    ground_nsrt_plan.append(PlaceOnFloor.ground([cube]))
    ground_nsrt_plan.append(PickFromFloor.ground([cube]))
    state = init_state.copy()
    rng = np.random.default_rng(123)
    for ground_nsrt in ground_nsrt_plan:
        assert all(a.holds(state) for a in ground_nsrt.preconditions)
        option = ground_nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        action = option.policy(state)
        state = env.simulate(state, action)
        assert option.terminal(state)
        assert all(a.holds(state) for a in ground_nsrt.add_effects)
        assert not any(a.holds(state) for a in ground_nsrt.delete_effects)

    # Test noisy picking.
    utils.reset_config({
        "env": "sticky_table",
        "num_train_tasks": 1,
        "num_test_tasks": 2,
        "sticky_table_place_smooth_fall_prob": 1.0,
        "sticky_table_place_sticky_fall_prob": 0.0,
        "sticky_table_pick_success_prob": 0.0,
    })
    state = init_state.copy()
    ground_nsrt = PickFromTable.ground([cube, init_table])
    option = ground_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    action = option.policy(state)
    next_state = env.simulate(state, action)
    assert state.allclose(next_state)

    # Test noisy placing on a normal table.
    utils.reset_config({
        "env": "sticky_table",
        "num_train_tasks": 1,
        "num_test_tasks": 2,
        "sticky_table_place_smooth_fall_prob": 1.0,
        "sticky_table_place_sticky_fall_prob": 1.0,
        "sticky_table_pick_success_prob": 1.0,
    })
    ground_nsrt = PickFromTable.ground([cube, init_table])
    option = ground_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    action = option.policy(state)
    state = env.simulate(state, action)
    ground_nsrt = PlaceOnTable.ground([cube, init_table])
    option = ground_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    action = option.policy(state)
    state = env.simulate(state, action)
    assert OnFloor([cube]).holds(state)

    # Test placing on the the sticky table.
    utils.reset_config({
        "env": "sticky_table",
        "num_train_tasks": 1,
        "num_test_tasks": 2,
        "sticky_table_place_smooth_fall_prob": 0.0,
        "sticky_table_place_sticky_fall_prob": 1.0,
        "sticky_table_pick_success_prob": 1.0,
    })
    ground_nsrt = PickFromTable.ground([cube, init_table])
    option = ground_nsrt.sample_option(state, set(), rng)
    assert option.initiable(state)
    action = option.policy(state)
    state = env.simulate(state, action)
    ground_nsrt = PlaceOnTable.ground([cube, sticky_table])
    num_on_table = 0
    for _ in range(10):
        option = ground_nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        action = option.policy(state)
        next_state = env.simulate(state, action)
        if OnTable([cube, sticky_table]).holds(next_state):
            num_on_table += 1
        else:
            assert OnFloor([cube]).holds(next_state)
    assert 0 < num_on_table < 10

    # Test the tricky floor variation.
    utils.reset_config({
        "env": "sticky_table_tricky_floor",
        "num_train_tasks": 1,
        "num_test_tasks": 2,
        "sticky_table_place_smooth_fall_prob": 1.0,
        "sticky_table_tricky_floor_place_sticky_fall_prob": 0.0,
        "sticky_table_pick_success_prob": 1.0,
    })

    env = StickyTableTrickyFloorEnv()
    assert env.get_name() == "sticky_table_tricky_floor"

    # Test noise-free picking and placing on the tables, and place on floor.
    table_order = [init_table] + [t for t in normal_tables if t != init_table]
    ground_nsrt_plan = []
    for table, next_table in zip(table_order[:-1], table_order[1:]):
        ground_nsrt_plan.append(PickFromTable.ground([cube, table]))
        ground_nsrt_plan.append(PlaceOnTable.ground([cube, next_table]))
    ground_nsrt_plan.append(PickFromTable.ground([cube, table_order[-1]]))
    ground_nsrt_plan.append(PlaceOnFloor.ground([cube]))
    state = init_state.copy()
    rng = np.random.default_rng(123)
    for ground_nsrt in ground_nsrt_plan:
        assert all(a.holds(state) for a in ground_nsrt.preconditions)
        option = ground_nsrt.sample_option(state, set(), rng)
        assert option.initiable(state)
        action = option.policy(state)
        state = env.simulate(state, action)
        assert option.terminal(state)
        assert all(a.holds(state) for a in ground_nsrt.add_effects)
        assert not any(a.holds(state) for a in ground_nsrt.delete_effects)

    # Test picking from the floor.
    assert OnFloor([cube]).holds(state)
    # Picks that should fail.
    cube_size = state.get(cube, "size")
    cube_x = state.get(cube, "x") + cube_size / 2
    cube_y = state.get(cube, "y") + cube_size / 2
    next_state = env.simulate(
        state, Action(np.array([cube_x - 1e-5, cube_y], dtype=np.float32)))
    assert OnFloor([cube]).holds(next_state)
    next_state = env.simulate(
        state, Action(np.array([cube_x, cube_y - 1e-5], dtype=np.float32)))
    assert OnFloor([cube]).holds(next_state)
    # Pick that should succeed.
    next_state = env.simulate(
        state,
        Action(np.array([cube_x + 1e-5, cube_y + 1e-5], dtype=np.float32)))
    assert not OnFloor([cube]).holds(next_state)
