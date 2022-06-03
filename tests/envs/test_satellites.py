"""Test cases for the satellites environment."""

import matplotlib.pyplot as plt
import numpy as np

from predicators.src import utils
from predicators.src.envs.satellites import SatellitesEnv, SatellitesSimpleEnv
from predicators.src.structs import Action


def test_satellites():
    """Tests for SatellitesEnv() and SatellitesSimpleEnv()."""
    utils.reset_config({
        "env": "satellites",
        "satellites_num_sat_train": [1],
        "satellites_num_obj_train": [1],
        "satellites_num_sat_test": [3],
        "satellites_num_obj_test": [3],
    })
    env = SatellitesEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 13
    assert len(env.goal_predicates) == 3
    assert len(env.options) == 5
    assert len(env.types) == 2
    obj_type, sat_type = sorted(env.types)
    assert obj_type.name == "object"
    assert sat_type.name == "satellite"
    assert env.action_space.shape == (10, )
    task = env.get_test_tasks()[0]
    state = task.init
    sat0, sat1, sat2 = state.get_objects(sat_type)
    obj0, obj1, obj2 = state.get_objects(obj_type)
    assert sat0.name == "sat0"
    assert sat1.name == "sat1"
    assert sat2.name == "sat2"
    assert obj0.name == "obj0"
    assert obj1.name == "obj1"
    assert obj2.name == "obj2"
    state.set(sat1, "is_calibrated", 1.0)
    state.set(sat2, "read_obj_id", 0.0)
    state.set(obj0, "has_chem_x", 1.0)
    state.set(obj1, "has_chem_y", 1.0)
    state.set(obj2, "has_chem_x", 1.0)
    state.set(obj2, "has_chem_y", 1.0)
    # Some arcane errors arise with calling fig2data() during unit testing,
    # so we'll just call render_state_plt() instead of render_state() here.
    env.render_state_plt(state, task)
    plt.close()
    # Make sure that simple version of env works as expected.
    utils.reset_config({
        "env": "satellites_simple",
        "satellites_num_sat_train": [1],
        "satellites_num_obj_train": [100],  # should be unused
        "satellites_num_sat_test": [1],
        "satellites_num_obj_test": [100],  # should be unused
    })
    env = SatellitesSimpleEnv()
    for task in env.get_train_tasks():
        assert len(task.init.get_objects(obj_type)) == 1
    for task in env.get_test_tasks():
        assert len(task.init.get_objects(obj_type)) == 1


def test_satellites_simulate_failures():
    """Tests for the failure cases of simulate()."""
    utils.reset_config({
        "env": "satellites",
        "num_train_tasks": 1,
        "satellites_num_sat_train": [1],
        "satellites_num_obj_train": [1],
    })
    env = SatellitesEnv()
    Calibrate, MoveTo, ShootChemX, ShootChemY, UseInstrument = \
        sorted(env.options)
    assert Calibrate.name == "Calibrate"
    assert MoveTo.name == "MoveTo"
    assert ShootChemX.name == "ShootChemX"
    assert ShootChemY.name == "ShootChemY"
    assert UseInstrument.name == "UseInstrument"
    obj_type, sat_type = sorted(env.types)
    state = env.get_train_tasks()[0].init
    sat, = state.get_objects(sat_type)
    obj, = state.get_objects(obj_type)
    assert sat.name == "sat0"
    assert obj.name == "obj0"
    act = Action(np.zeros(env.action_space.shape, dtype=np.float32))
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot move into collision.
    act = MoveTo.ground(
        [sat, obj],
        [state.get(obj, "x"), state.get(obj, "y")]).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot calibrate without seeing object.
    act = Calibrate.ground([sat, obj], []).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot shoot Chemical X without seeing object.
    act = ShootChemX.ground([sat, obj], []).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot shoot Chemical Y without seeing object.
    act = ShootChemY.ground([sat, obj], []).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot use any instrument without seeing object.
    act = UseInstrument.ground([sat, obj], []).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Successful move.
    act = MoveTo.ground(
        [sat, obj],
        [state.get(obj, "x") - SatellitesEnv.radius * 5,
         state.get(obj, "y")]).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    state = next_state
    # Cannot use any instrument without calibration.
    act = UseInstrument.ground([sat, obj], []).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Need to have Chemical X to shoot it.
    state.set(sat, "shoots_chem_x", 0.0)
    act = ShootChemX.ground([sat, obj], []).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Need to have Chemical Y to shoot it.
    state.set(sat, "shoots_chem_y", 0.0)
    act = ShootChemY.ground([sat, obj], []).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot calibrate the wrong object.
    assert state.get(sat, "calibration_obj_id") == 0.0
    state.set(sat, "calibration_obj_id", -1.0)
    act = Calibrate.ground([sat, obj], []).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Successful calibration.
    state.set(sat, "calibration_obj_id", 0.0)
    act = Calibrate.ground([sat, obj], []).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    state = next_state
    # Cannot use camera without Chemical X.
    state.set(sat, "instrument", 0.16)  # camera
    act = UseInstrument.ground([sat, obj], []).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Cannot use infrared without Chemical Y.
    state.set(sat, "instrument", 0.5)  # infrared
    act = UseInstrument.ground([sat, obj], []).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Geiger works fine.
    state.set(sat, "instrument", 0.83)  # Geiger
    act = UseInstrument.ground([sat, obj], []).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    # Make sure the read worked fine.
    assert state.get(sat, "read_obj_id") == -1.0
    assert next_state.get(sat, "read_obj_id") == 0.0
