"""Test cases for the repeated NextTo environment."""

import numpy as np
import pytest

from predicators import utils
from predicators.envs.spot_env import SpotEnv
from predicators.structs import Action


def test_spot_env():
    """Tests for SpotEnv class."""
    utils.reset_config({"env": "realworld_spot", "num_train_tasks": 1, "num_test_tasks": 1})
    env = SpotEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert {pred.name for pred in env.predicates} == \
        {"ReachableSurface", "HoldingCan", "On", "HandEmpty", "ReachableCan"}
    assert {pred.name for pred in env.goal_predicates} == {"On"}
    assert len(env.options) == 4
    assert len(env.types) == 3
    assert len(env.strips_operators) == 4
    with pytest.raises(NotImplementedError):
        env.render_state_plt(task.init, task)


def test_spot_env_simulate():
    """Tests for the simulate() function."""
    utils.reset_config({
        "env": "repeated_nextto",
        "approach": "nsrt_learning",
    })
    env = SpotEnv()
    MoveToSurface = [o for o in env.options if o.name == "MoveToSurface"][0]
    MoveToCan = [o for o in env.options if o.name == "MoveToCan"][0]
    GraspCan = [o for o in env.options if o.name == "GraspCan"][0]
    PlaceCanOnTop = [o for o in env.options if o.name == "PlaceCanOntop"][0]    
    ReachableSurface = [o for o in env.predicates if o.name == "ReachableSurface"][0]
    HoldingCan = [o for o in env.predicates if o.name == "HoldingCan"][0]
    On = [o for o in env.predicates if o.name == "On"][0]
    HandEmpty = [o for o in env.predicates if o.name == "HandEmpty"][0]
    ReachableCan = [o for o in env.predicates if o.name == "ReachableCan"][0]
    robot_type = [t for t in env.types if t.name == "robot"][0]
    can_type = [t for t in env.types if t.name == "soda_can"][0]
    surface_type = [t for t in env.types if t.name == "flat_surface"][0]
    
    task = env.get_train_tasks()[0]
    state = task.init
    assert len(state.simulator_state) == 2
    assert "On(soda_can:soda_can, counter:flat_surface)" in str(state)
    assert "HandEmpty(spot:robot)" in str(state)
    objs = [obj for obj in task.init]
    counter = [obj for obj in objs if obj.name == "counter"][0]
    snack_table = [obj for obj in objs if obj.name == "snack_table"][0]
    soda_can = [obj for obj in objs if obj.name == "soda_can"][0]
    spot = [obj for obj in objs if obj.name == "spot"][0]
    # Try moving to the counter.
    act = MoveToSurface.ground([spot, counter], []).policy(task.init)
    counter_state = env.simulate(task.init, act)
    assert "ReachableSurface(spot:robot, counter:flat_surface)" in str(counter_state.simulator_state)
    # Try moving to the can.
    act = MoveToCan.ground([spot, soda_can], []).policy(task.init)
    can_state = env.simulate(task.init, act)
    assert "ReachableCan(spot:robot, soda_can:soda_can)" in str(can_state.simulator_state)
    # Try grasping the can when it is reachable.
    act = GraspCan.ground([spot, soda_can, counter], []).policy(can_state)
    can_held_state = env.simulate(can_state, act)
    assert "HoldingCan(spot:robot, soda_can:soda_can)}" in str(can_held_state.simulator_state)
    # Try grasping the can when it's not reachable.
    act = GraspCan.ground([spot, soda_can, counter], []).policy(task.init)
    init_state = env.simulate(task.init, act)
    assert task.init.allclose(init_state)
    # Try placing the can after it has been held.
    act = MoveToSurface.ground([spot, counter], []).policy(can_held_state)
    counter_holding_state = env.simulate(can_held_state, act)
    act = PlaceCanOnTop.ground([spot, soda_can, counter],[]).policy(counter_holding_state)
    can_putdown_state = env.simulate(counter_holding_state, act)
    assert can_putdown_state.allclose(counter_state)
    # Try placing the can when it isn't held.
    act = PlaceCanOnTop.ground([spot, soda_can, counter], []).policy(task.init)
    init_state = env.simulate(task.init, act)
    assert task.init.allclose(init_state)
    