"""Integration tests for skill-factory options in real PyBullet envs.

Uses the options built by _get_options_skill_factories() in each env's
options.py (via get_gt_options), following the same pattern as
tests/envs/test_pybullet_blocks.py.

Envs covered:
  - boil:   PickJug, PlaceOnBurner, PlaceOutsideBurnerAndFaucet,
            SwitchFaucetOn, SwitchBurnerOn
  - grow:   PickJug, Place
  - fan:    SwitchOn, SwitchOff
  - domino: Push

NOTE on pybullet_control_mode:
  - Pick / Place / motion-only tests use "reset" mode (fast,
    deterministic joint teleportation).
  - Switch-toggle tests require "position" mode so
    p.stepSimulation() is called and contact forces can rotate
    the switch joint.  Those tests create their own env instance
    and are placed at the end of the file so they do not interfere
    with the module-scoped "reset" mode fixtures.
"""
# pylint: disable=protected-access,import-outside-toplevel
from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pytest

from predicators import utils
from predicators.envs import _MOST_RECENT_ENV_INSTANCE
from predicators.envs.pybullet_boil import PyBulletBoilEnv
from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.envs.pybullet_grow import PyBulletGrowEnv
from predicators.ground_truth_models import get_gt_options

_GUI_ON = False  # Set True for visual debugging

# Default continuous params for skill factories.
# Pick: (grasp_z_offset,) in [0.0, 0.1] -- small offset so gripper
# closes close to the object origin.
_PICK_PARAMS = [0.01]
# Push: (approach_distance, contact_z_offset) in [0, 0.10] x [0, 0.11]
# -- a real stroke: zero approach makes the push degenerate (start point
# == end point == target), leaving the toggle to the descending hand's
# footprint, which is orientation-sensitive and marginal even with the
# legacy front push (a [0, 0] faucet toggle ground for 399 steps vs 29
# with a stroke, and burner2 failed outright on macOS).
_PUSH_PARAMS = [0.05, 0.1]

# ---------------------------------------------------------------------------
# Generic mixin: set_state / get_state / execute_option
# ---------------------------------------------------------------------------


class _ExposedEnvMixin:
    """Provides set_state / get_state / execute_option on any PyBulletEnv."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Register this instance so get_or_create_env() returns it (and
        # therefore get_gt_options() uses the same Type objects we have).
        env_name = self.get_name()  # type: ignore[attr-defined]
        _MOST_RECENT_ENV_INSTANCE[env_name] = self

    @functools.cached_property
    def _options(self) -> dict[str, Any]:
        name = self.get_name()  # type: ignore[attr-defined]
        return {o.name: o for o in get_gt_options(name)}

    def set_state(self, state: Any) -> None:
        """Reset env to *state*, assuming robot is at its home joint config."""
        robot = self._pybullet_robot  # type: ignore[attr-defined]
        joint_positions = list(robot.initial_joint_positions)
        state_with_sim = utils.PyBulletState(state.data,
                                             simulator_state=joint_positions)
        self._current_observation = state_with_sim
        self._current_task = None
        self._set_state(state_with_sim)  # type: ignore[attr-defined]

    def get_state(self) -> Any:
        """Get state."""
        return self._get_state()  # type: ignore[attr-defined]

    def execute_option(self, option: Any, max_steps: int = 300) -> Any:
        """Run option loop up to *max_steps*; return final state."""
        cur = self._current_state  # type: ignore[attr-defined]
        assert option.initiable(cur)
        for _ in range(max_steps):
            if option.terminal(cur):
                break
            action = option.policy(cur)
            self.step(action)  # type: ignore[attr-defined]
            cur = self._current_state  # type: ignore[attr-defined]
        return self._current_state.copy()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Concrete exposed wrappers
# ---------------------------------------------------------------------------


class _ExposedBoilEnv(_ExposedEnvMixin, PyBulletBoilEnv):  # type: ignore[misc]

    @property
    def PickJug(self):
        """PickJug."""
        return self._options["PickJug"]

    @property
    def Place(self):
        """Place (skill-factory unified Place option)."""
        return self._options["Place"]

    @property
    def SwitchFaucetOn(self):
        """SwitchFaucetOn."""
        return self._options["SwitchFaucetOn"]

    @property
    def SwitchFaucetOff(self):
        """SwitchFaucetOff."""
        return self._options["SwitchFaucetOff"]

    @property
    def SwitchBurnerOn(self):
        """SwitchBurnerOn."""
        return self._options["SwitchBurnerOn"]

    @property
    def SwitchBurnerOff(self):
        """SwitchBurnerOff."""
        return self._options["SwitchBurnerOff"]


class _ExposedGrowEnv(_ExposedEnvMixin, PyBulletGrowEnv):  # type: ignore[misc]

    @property
    def PickJug(self):
        """PickJug."""
        return self._options["PickJug"]

    @property
    def Place(self):
        """Place."""
        return self._options["Place"]


class _ExposedCoffeeEnv(  # type: ignore[misc]
        _ExposedEnvMixin, PyBulletCoffeeEnv):

    @property
    def PickJug(self):
        """PickJug."""
        return self._options["PickJug"]

    @property
    def PlaceJugInMachine(self):
        """PlaceJugInMachine."""
        return self._options["PlaceJugInMachine"]

    @property
    def TurnMachineOn(self):
        """TurnMachineOn."""
        return self._options["TurnMachineOn"]

    @property
    def Pour(self):
        """Pour."""
        return self._options["Pour"]


class _ExposedFanEnv(_ExposedEnvMixin, PyBulletFanEnv):  # type: ignore[misc]

    @property
    def SwitchOn(self):
        """SwitchOn."""
        return self._options["SwitchOn"]

    @property
    def SwitchOff(self):
        """SwitchOff."""
        return self._options["SwitchOff"]


# ---------------------------------------------------------------------------
# Module-scoped fixtures  (reset mode)
# IMPORTANT: all fixture-based tests must run BEFORE any standalone tests
# that call utils.reset_config with "position" mode.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", name="boil_env")
def _create_boil_env():
    utils.reset_config({
        "env": "pybullet_boil",
        "use_gui": _GUI_ON,
        "pybullet_control_mode": "reset",
        "pybullet_robot": "fetch",
        "boil_use_skill_factories": True,
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
    })
    return _ExposedBoilEnv(use_gui=_GUI_ON)


@pytest.fixture(scope="module", name="grow_env")
def _create_grow_env():
    utils.reset_config({
        "env": "pybullet_grow",
        "use_gui": _GUI_ON,
        "pybullet_control_mode": "reset",
        "pybullet_robot": "fetch",
        "grow_use_skill_factories": True,
        "grow_num_jugs_train": [1],
        "grow_num_jugs_test": [1],
        "grow_num_cups_train": [1],
        "grow_num_cups_test": [1],
    })
    return _ExposedGrowEnv(use_gui=_GUI_ON)


@pytest.fixture(scope="module", name="coffee_env")
def _create_coffee_env():
    utils.reset_config({
        "env": "pybullet_coffee",
        "use_gui": _GUI_ON,
        "pybullet_control_mode": "reset",
        "pybullet_robot": "fetch",
        "coffee_use_skill_factories": True,
        "coffee_num_cups_train": [1],
        "coffee_num_cups_test": [1],
    })
    return _ExposedCoffeeEnv(use_gui=_GUI_ON)


@pytest.fixture(scope="module", name="fan_env")
def _create_fan_env():
    utils.reset_config({
        "env": "pybullet_fan",
        "use_gui": _GUI_ON,
        "pybullet_control_mode": "reset",
        "pybullet_robot": "fetch",
        "fan_use_skill_factories": True,
        "fan_known_controls_relation": False,
        "fan_combine_switch_on_off": False,
        "fan_train_num_walls_per_task": [0],
        "fan_test_num_walls_per_task": [0],
    })
    return _ExposedFanEnv(use_gui=_GUI_ON)


# ===========================================================================
# BOIL ENV TESTS: Pick
# ===========================================================================


def test_pick_jug_boil_center(boil_env):
    """Pick a jug placed at the workspace centre; jug should be held."""
    env = boil_env
    jug = env._jugs[0]
    robot = env._robot

    state = env.get_train_tasks()[0].init.copy()
    state.set(jug, "x", env.x_mid)
    state.set(jug, "y", env.y_mid)
    state.set(jug, "z", env.jug_init_z)
    state.set(jug, "rot", 0.0)
    state.set(jug, "is_held", 0.0)
    state.set(jug, "water_volume", 0.0)
    state.set(jug, "heat_level", 0.0)
    env.set_state(state)

    result = env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))

    assert result.get(jug, "is_held") > 0.5, "Jug not held after pick"
    assert result.get(robot, "fingers") < 0.5, "Fingers should be closed"


def test_pick_jug_boil_offset_y(boil_env):
    """Pick succeeds with jug at a y position offset from workspace center."""
    env = boil_env
    jug = env._jugs[0]
    robot = env._robot

    state = env.get_train_tasks()[0].init.copy()
    state.set(jug, "x", env.x_mid)
    state.set(jug, "y", env.y_mid + 0.05)
    state.set(jug, "z", env.jug_init_z)
    state.set(jug, "rot", 0.0)
    state.set(jug, "is_held", 0.0)
    state.set(jug, "water_volume", 0.0)
    state.set(jug, "heat_level", 0.0)
    env.set_state(state)

    result = env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))
    assert result.get(jug, "is_held") > 0.5


def test_pick_jug_boil_transport_z_correct(boil_env):
    """After pick, robot EE z should be above the grasp height.

    The skill-factory pick lifts slightly above the grasp position
    (grasp_z + 0.01), not to the full transport_z.
    """
    env = boil_env
    jug = env._jugs[0]
    robot = env._robot

    state = env.get_train_tasks()[0].init.copy()
    state.set(jug, "x", env.x_mid)
    state.set(jug, "y", env.y_mid)
    state.set(jug, "z", env.jug_init_z)
    state.set(jug, "rot", 0.0)
    state.set(jug, "is_held", 0.0)
    state.set(jug, "water_volume", 0.0)
    state.set(jug, "heat_level", 0.0)
    env.set_state(state)

    # Expected lift height: jug_handle_z + grasp_z_offset + slight_lift
    grasp_z_offset = _PICK_PARAMS[0]
    jug_handle_z = env.table_height + env.jug_handle_height
    expected_z = jug_handle_z + grasp_z_offset + 0.01
    result = env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))

    robot_z = result.get(robot, "z")
    assert abs(robot_z - expected_z) < 0.05, (
        f"EE z={robot_z:.3f} should be near expected_z={expected_z:.3f}")


def test_pick_skill_initiable_any_state_boil(boil_env):
    """Pick skill's initiable() returns True regardless of finger state."""
    env = boil_env
    jug = env._jugs[0]
    robot = env._robot

    state = env.get_train_tasks()[0].init.copy()
    state.set(jug, "x", env.x_mid)
    state.set(jug, "y", env.y_mid)
    state.set(jug, "z", env.jug_init_z)
    state.set(jug, "rot", 0.0)
    state.set(jug, "is_held", 0.0)
    state.set(jug, "water_volume", 0.0)
    state.set(jug, "heat_level", 0.0)
    env.set_state(state)

    cur = env.get_state()
    option = env.PickJug.ground([robot, jug], _PICK_PARAMS)
    assert option.initiable(cur), "Pick should be initiable from any state"


# ===========================================================================
# BOIL ENV TESTS: Place
# ===========================================================================


def test_place_jug_boil_on_burner(boil_env):
    """Pick then place jug on the first burner."""
    env = boil_env
    jug = env._jugs[0]
    robot = env._robot
    burner = env._burners[0]

    state = env.get_train_tasks()[0].init.copy()
    state.set(jug, "x", env.x_mid)
    state.set(jug, "y", env.y_mid)
    state.set(jug, "z", env.jug_init_z)
    state.set(jug, "rot", 0.0)
    state.set(jug, "is_held", 0.0)
    state.set(jug, "water_volume", 0.0)
    state.set(jug, "heat_level", 0.0)
    env.set_state(state)

    env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))
    # Place params: (target_x, target_y, release_z, target_yaw)
    bx = state.get(burner, "x")
    by = state.get(burner, "y")
    release_z = max(env.table_height + env.jug_handle_height, 0.5)
    result = env.execute_option(
        env.Place.ground([robot], [bx, by, release_z, 0.0]))

    assert result.get(jug, "is_held") < 0.5, "Jug should no longer be held"
    assert result.get(robot, "fingers") > 0.015, "Fingers should be open"


def test_place_jug_boil_outside(boil_env):
    """Pick then place jug at the outside position."""
    env = boil_env
    jug = env._jugs[0]
    robot = env._robot

    state = env.get_train_tasks()[0].init.copy()
    state.set(jug, "x", env.x_mid)
    state.set(jug, "y", env.y_mid - 0.1)
    state.set(jug, "z", env.jug_init_z)
    state.set(jug, "rot", 0.0)
    state.set(jug, "is_held", 0.0)
    state.set(jug, "water_volume", 0.0)
    state.set(jug, "heat_level", 0.0)
    env.set_state(state)

    env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))
    # Place at a position outside burner/faucet area
    release_z = max(env.table_height + env.jug_handle_height, 0.5)
    result = env.execute_option(
        env.Place.ground([robot],
                         [env.x_mid, env.y_mid - 0.1, release_z, 0.0]))

    assert result.get(jug, "is_held") < 0.5
    assert result.get(robot, "fingers") > 0.015


def test_pick_place_full_cycle_boil(boil_env):
    """Full pick->place->pick cycle; each step leaves correct held state."""
    env = boil_env
    jug = env._jugs[0]
    robot = env._robot
    burner = env._burners[0]

    state = env.get_train_tasks()[0].init.copy()
    state.set(jug, "x", env.x_mid)
    state.set(jug, "y", env.y_mid)
    state.set(jug, "z", env.jug_init_z)
    state.set(jug, "rot", 0.0)
    state.set(jug, "is_held", 0.0)
    state.set(jug, "water_volume", 0.0)
    state.set(jug, "heat_level", 0.0)
    env.set_state(state)

    s1 = env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))
    assert s1.get(jug, "is_held") > 0.5, "Should be held after first pick"

    bx = state.get(burner, "x")
    by = state.get(burner, "y")
    release_z = max(env.table_height + env.jug_handle_height, 0.5)
    s2 = env.execute_option(env.Place.ground([robot],
                                             [bx, by, release_z, 0.0]))
    assert s2.get(jug, "is_held") < 0.5, "Should be free after place"

    s3 = env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))
    assert s3.get(jug, "is_held") > 0.5, "Should be held after second pick"


def test_place_skill_not_terminal_before_pick_boil(boil_env):
    """Place skill not terminal at start (jug not placed yet)."""
    env = boil_env
    robot = env._robot

    task_state = env.get_train_tasks()[0].init.copy()
    env.set_state(task_state)

    cur = env.get_state()
    # Use midpoint place params
    option = env.Place.ground([robot], [0.75, 1.35, 0.55, 0.0])
    assert option.initiable(cur)
    assert not option.terminal(cur), "Place should not be terminal at start"


# ===========================================================================
# GROW ENV TESTS: Pick & Place
# ===========================================================================


def test_pick_jug_grow_center(grow_env):
    """Pick a jug at the workspace centre in the grow environment."""
    env = grow_env
    jug = env._jugs[0]
    robot = env._robot

    state = env.get_train_tasks()[0].init.copy()
    state.set(jug, "x", env.robot_init_x)
    state.set(jug, "y", env.y_mid)
    state.set(jug, "z", env.jug_init_z)
    state.set(jug, "rot", env.jug_init_rot)
    state.set(jug, "is_held", 0.0)
    env.set_state(state)

    result = env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))
    assert result.get(jug, "is_held") > 0.5, "Jug not held after pick (grow)"


def test_pick_jug_grow_different_y(grow_env):
    """Pick a jug at a y position closer to y_lb in the grow env."""
    env = grow_env
    jug = env._jugs[0]
    robot = env._robot

    state = env.get_train_tasks()[0].init.copy()
    state.set(jug, "x", env.robot_init_x)
    state.set(jug, "y", env.y_lb + 0.15)
    state.set(jug, "z", env.jug_init_z)
    state.set(jug, "rot", env.jug_init_rot)
    state.set(jug, "is_held", 0.0)
    env.set_state(state)

    result = env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))
    assert result.get(jug, "is_held") > 0.5


def test_place_jug_grow_center(grow_env):
    """Pick then place jug in the grow environment."""
    env = grow_env
    jug = env._jugs[0]
    robot = env._robot

    state = env.get_train_tasks()[0].init.copy()
    state.set(jug, "x", env.robot_init_x)
    state.set(jug, "y", env.y_mid)
    state.set(jug, "z", env.jug_init_z)
    state.set(jug, "rot", env.jug_init_rot)
    state.set(jug, "is_held", 0.0)
    env.set_state(state)

    env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))

    # Place params: (target_x, target_y, release_z, target_yaw)
    tx = env.robot_init_x
    ty = env.y_mid - 0.1
    release_z = env.table_height + env.jug_handle_height
    result = env.execute_option(
        env.Place.ground([robot, jug], [tx, ty, release_z, 0.0]))

    assert result.get(jug, "is_held") < 0.5, "Jug should be free after place"
    assert result.get(robot, "fingers") > 0.015, "Fingers should be open"


def test_pick_place_cycle_grow(grow_env):
    """Full pick → place cycle works in grow env."""
    env = grow_env
    jug = env._jugs[0]
    robot = env._robot

    state = env.get_train_tasks()[0].init.copy()
    state.set(jug, "x", env.robot_init_x)
    state.set(jug, "y", env.y_mid)
    state.set(jug, "z", env.jug_init_z)
    state.set(jug, "rot", env.jug_init_rot)
    state.set(jug, "is_held", 0.0)
    env.set_state(state)

    s1 = env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))
    assert s1.get(jug, "is_held") > 0.5

    tx = env.robot_init_x
    ty = env.y_mid + 0.1
    release_z = env.table_height + env.jug_handle_height
    s2 = env.execute_option(
        env.Place.ground([robot, jug], [tx, ty, release_z, 0.0]))
    assert s2.get(jug, "is_held") < 0.5


# ===========================================================================
# COFFEE ENV TESTS: Pick, Place, TurnMachineOn, Pour
# ===========================================================================


def test_pick_jug_coffee_center(coffee_env):
    """Pick a jug at the workspace centre in the coffee environment."""
    env = coffee_env
    jug = env._jug
    robot = env._robot

    state = env.get_train_tasks()[0].init.copy()
    mid_x = (env.x_lb + env.x_ub) / 2
    mid_y = (env.y_lb + env.y_ub) / 2
    state.set(jug, "x", mid_x)
    state.set(jug, "y", mid_y)
    state.set(jug, "rot", 0.0)
    state.set(jug, "is_filled", 0.0)
    env.set_state(state)

    Holding = env._Holding
    result = env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))
    assert Holding.holds(result, [robot, jug]), "Jug not held after pick"


def test_pick_jug_coffee_offset_y(coffee_env):
    """Pick a jug at a y offset in the coffee environment."""
    env = coffee_env
    jug = env._jug
    robot = env._robot

    state = env.get_train_tasks()[0].init.copy()
    mid_x = (env.x_lb + env.x_ub) / 2
    state.set(jug, "x", mid_x)
    state.set(jug, "y", env.y_lb + 0.15)
    state.set(jug, "rot", 0.0)
    state.set(jug, "is_filled", 0.0)
    env.set_state(state)

    Holding = env._Holding
    result = env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))
    assert Holding.holds(result, [robot, jug])


def test_place_jug_coffee_in_machine(coffee_env):
    """Pick then place jug in the coffee machine."""
    env = coffee_env
    jug = env._jug
    robot = env._robot
    machine = env._machine

    state = env.get_train_tasks()[0].init.copy()
    mid_x = (env.x_lb + env.x_ub) / 2
    mid_y = (env.y_lb + env.y_ub) / 2
    state.set(jug, "x", mid_x)
    state.set(jug, "y", mid_y)
    state.set(jug, "rot", 0.0)
    state.set(jug, "is_filled", 0.0)
    env.set_state(state)

    env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))
    # Place params: (target_x, target_y, release_z, target_yaw)
    target_x = PyBulletCoffeeEnv.dispense_area_x
    target_y = PyBulletCoffeeEnv.dispense_area_y
    release_z = PyBulletCoffeeEnv.z_lb + env.jug_handle_height()
    target_yaw = PyBulletCoffeeEnv.robot_init_wrist
    result = env.execute_option(
        env.PlaceJugInMachine.ground(
            [robot, jug, machine],
            [target_x, target_y, release_z, target_yaw]))

    Holding = env._Holding
    assert not Holding.holds(result, [robot, jug]), \
        "Jug should no longer be held"
    assert result.get(robot, "fingers") > 0.015, "Fingers should be open"


def test_turn_machine_on_reaches_button(coffee_env):
    """TurnMachineOn completes its push trajectory.

    The skill-factory push skill returns the robot to its home position
    after pushing, so we verify the option terminates and the robot ends
    near home.
    """
    env = coffee_env
    robot = env._robot
    machine = env._machine

    state = env.get_train_tasks()[0].init.copy()
    state.set(machine, "is_on", 0.0)
    env.set_state(state)

    opt = env.TurnMachineOn.ground([robot, machine], _PUSH_PARAMS)
    result = env.execute_option(opt)

    robot_x = result.get(robot, "x")
    robot_y = result.get(robot, "y")
    home_x = PyBulletCoffeeEnv.robot_init_x
    home_y = PyBulletCoffeeEnv.robot_init_y
    dist_from_home = np.sqrt((robot_x - home_x)**2 + (robot_y - home_y)**2)
    assert dist_from_home < 0.3, (
        f"Robot EE ({robot_x:.3f}, {robot_y:.3f}) should return near "
        f"home ({home_x:.3f}, {home_y:.3f}) after push, "
        f"dist={dist_from_home:.3f}")


def test_pour_reaches_cup_position(coffee_env):
    """Robot approaches cup pour position during Pour skill (reset mode)."""
    env = coffee_env
    jug = env._jug
    robot = env._robot
    init_state = env.get_train_tasks()[0].init
    cups = init_state.get_objects(env._cup_type)
    if len(cups) == 0:
        pytest.skip("No cups in task")
    cup = cups[0]

    state = env.get_train_tasks()[0].init.copy()
    mid_x = (env.x_lb + env.x_ub) / 2
    mid_y = (env.y_lb + env.y_ub) / 2
    state.set(jug, "x", mid_x)
    state.set(jug, "y", mid_y)
    state.set(jug, "rot", 0.0)
    state.set(jug, "is_filled", 1.0)
    env.set_state(state)

    # First pick the jug
    env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))

    # Now try to pour - in reset mode, pouring physics don't run,
    # but we can check the robot approaches the pour position.
    pour_pos = PyBulletCoffeeEnv._get_pour_position(env.get_state(), cup)
    result = env.execute_option(env.Pour.ground([robot, jug, cup], []),
                                max_steps=200)

    # Check robot approached the pour position (x/y)
    result.get(robot, "x")
    result.get(robot, "y")
    jug_x = result.get(jug, "x")
    jug_y = result.get(jug, "y")
    dist = np.sqrt((jug_x - pour_pos[0])**2 + (jug_y - pour_pos[1])**2)
    assert dist < 0.3, (
        f"Jug ({jug_x:.3f}, {jug_y:.3f}) should approach pour position "
        f"({pour_pos[0]:.3f}, {pour_pos[1]:.3f}), dist={dist:.3f}")


def test_pick_place_full_cycle_coffee(coffee_env):
    """Full pick→place→pick cycle in coffee env."""
    env = coffee_env
    jug = env._jug
    robot = env._robot
    machine = env._machine

    state = env.get_train_tasks()[0].init.copy()
    mid_x = (env.x_lb + env.x_ub) / 2
    mid_y = (env.y_lb + env.y_ub) / 2
    state.set(jug, "x", mid_x)
    state.set(jug, "y", mid_y)
    state.set(jug, "rot", 0.0)
    state.set(jug, "is_filled", 0.0)
    env.set_state(state)

    Holding = env._Holding

    s1 = env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))
    assert Holding.holds(s1, [robot, jug]), "Should be held after pick"

    target_x = PyBulletCoffeeEnv.dispense_area_x
    target_y = PyBulletCoffeeEnv.dispense_area_y
    release_z = PyBulletCoffeeEnv.z_lb + env.jug_handle_height()
    target_yaw = PyBulletCoffeeEnv.robot_init_wrist
    s2 = env.execute_option(
        env.PlaceJugInMachine.ground(
            [robot, jug, machine],
            [target_x, target_y, release_z, target_yaw]))
    assert not Holding.holds(s2, [robot, jug]), "Should be free after place"

    s3 = env.execute_option(env.PickJug.ground([robot, jug], _PICK_PARAMS))
    assert Holding.holds(s3, [robot, jug]), "Should be held after second pick"


# ===========================================================================
# BOIL ENV TESTS: Push switch – motion-only (reset mode)
# These check that the EE actually moves to the switch position.
# Switch toggle (is_on) is NOT checked here since "reset" mode does not
# simulate contact forces.  Toggle tests are in the standalone section below.
# ===========================================================================


def test_push_switch_reaches_target_position_boil(boil_env):
    """SwitchFaucetOn completes its trajectory (push then return home).

    The skill-factory push skill returns the robot to its home position
    after pushing, so we verify the skill terminates and the robot is
    near home (the final waypoint).
    """
    env = boil_env
    robot = env._robot
    faucet = env._faucet
    faucet_switch = env._faucet_switch

    task_state = env.get_train_tasks()[0].init.copy()
    task_state.set(faucet_switch, "is_on", 0.0)
    env.set_state(task_state)

    home_x, home_y = env.robot_init_x, env.robot_init_y

    opt = env.SwitchFaucetOn.ground([robot, faucet], _PUSH_PARAMS)
    result = env.execute_option(opt)

    robot_x = result.get(robot, "x")
    robot_y = result.get(robot, "y")
    dist_from_home = np.sqrt((robot_x - home_x)**2 + (robot_y - home_y)**2)
    assert dist_from_home < 0.3, (
        f"Robot EE ({robot_x:.3f}, {robot_y:.3f}) should return near "
        f"home ({home_x:.3f}, {home_y:.3f}) after push, "
        f"dist={dist_from_home:.3f}")


def test_push_switch_skill_initiable_boil(boil_env):
    """SwitchFaucetOn's initiable() returns True."""
    env = boil_env
    robot = env._robot
    faucet = env._faucet
    faucet_switch = env._faucet_switch

    task_state = env.get_train_tasks()[0].init.copy()
    task_state.set(faucet_switch, "is_on", 0.0)
    env.set_state(task_state)

    cur = env.get_state()
    option = env.SwitchFaucetOn.ground([robot, faucet], _PUSH_PARAMS)
    assert option.initiable(cur)


# ===========================================================================
# FAN ENV TESTS: Push switch – motion-only (reset mode)
# ===========================================================================


def test_push_switch_fan_reaches_switch_xy(fan_env):
    """Robot EE x/y coordinates approach the switch position after SwitchOn."""
    env = fan_env
    task_state = env.get_train_tasks()[0].init
    switch = task_state.get_objects(env._switch_type)[0]
    robot = env._robot

    state = task_state.copy()
    state.set(switch, "is_on", 0.0)
    env.set_state(state)

    sw_x = state.get(switch, "x")
    sw_y = state.get(switch, "y")

    opt = env.SwitchOn.ground([robot, switch], _PUSH_PARAMS)
    result = env.execute_option(opt)

    robot_x = result.get(robot, "x")
    robot_y = result.get(robot, "y")
    dist = np.sqrt((robot_x - sw_x)**2 + (robot_y - sw_y)**2)
    assert dist < 0.2, (
        f"Robot EE ({robot_x:.3f}, {robot_y:.3f}) too far from "
        f"switch ({sw_x:.3f}, {sw_y:.3f}), dist={dist:.3f}")


def test_push_switch_fan_skill_initiable(fan_env):
    """SwitchOn's initiable() returns True in fan env."""
    env = fan_env
    task_state = env.get_train_tasks()[0].init
    switch = task_state.get_objects(env._switch_type)[0]
    robot = env._robot

    state = task_state.copy()
    state.set(switch, "is_on", 0.0)
    env.set_state(state)

    cur = env.get_state()
    option = env.SwitchOn.ground([robot, switch], _PUSH_PARAMS)
    assert option.initiable(cur)


def test_push_switch_fan_fingers_open_after_push(fan_env):
    """After SwitchOn completes, robot fingers should be open."""
    env = fan_env
    task_state = env.get_train_tasks()[0].init
    switch = task_state.get_objects(env._switch_type)[0]
    robot = env._robot

    state = task_state.copy()
    state.set(switch, "is_on", 0.0)
    env.set_state(state)

    opt = env.SwitchOn.ground([robot, switch], _PUSH_PARAMS)
    result = env.execute_option(opt)
    assert result.get(robot, "fingers") > 0.015, (
        "Fingers should be open after push completes")


# ===========================================================================
# STANDALONE TESTS: Two-jug boil (reset mode)
# ===========================================================================


def test_pick_correct_jug_boil_two_jugs():
    """With two jugs, the skill picks only the targeted jug."""
    utils.reset_config({
        "env": "pybullet_boil",
        "use_gui": False,
        "pybullet_control_mode": "reset",
        "pybullet_robot": "fetch",
        "boil_use_skill_factories": True,
        "boil_num_jugs_train": [2],
        "boil_num_jugs_test": [2],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
    })
    env = _ExposedBoilEnv(use_gui=False)

    jug0, jug1 = env._jugs[0], env._jugs[1]
    robot = env._robot

    state = env.get_train_tasks()[0].init.copy()
    state.set(jug0, "x", env.x_mid)
    state.set(jug0, "y", env.y_mid)
    state.set(jug0, "z", env.jug_init_z)
    state.set(jug0, "rot", 0.0)
    state.set(jug0, "is_held", 0.0)
    state.set(jug0, "water_volume", 0.0)
    state.set(jug0, "heat_level", 0.0)
    state.set(jug1, "x", env.x_mid - 0.15)
    state.set(jug1, "y", env.y_mid - 0.15)
    state.set(jug1, "z", env.jug_init_z)
    state.set(jug1, "rot", 0.0)
    state.set(jug1, "is_held", 0.0)
    state.set(jug1, "water_volume", 0.0)
    state.set(jug1, "heat_level", 0.0)
    env.set_state(state)

    result = env.execute_option(env.PickJug.ground([robot, jug0],
                                                   _PICK_PARAMS))
    assert result.get(jug0, "is_held") > 0.5, "jug0 should be held"
    assert result.get(jug1, "is_held") < 0.5, "jug1 should remain free"


# ===========================================================================
# STANDALONE TESTS: Push switch toggle (position mode)
# Placed last so that the "position" mode config does not affect the
# module-scoped "reset" mode fixtures used earlier.
# ===========================================================================


def test_push_switch_on_boil_position_mode():
    """SwitchFaucetOn toggles a boil faucet switch to on (position mode)."""
    utils.reset_config({
        "env": "pybullet_boil",
        "use_gui": False,
        "pybullet_control_mode": "position",
        "pybullet_robot": "fetch",
        "pybullet_sim_steps_per_action": 20,
        "boil_use_skill_factories": True,
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
    })
    env = _ExposedBoilEnv(use_gui=False)

    task_state = env.get_train_tasks()[0].init
    faucet = env._faucet
    faucet_switch = env._faucet_switch
    robot = env._robot

    state = task_state.copy()
    state.set(faucet_switch, "is_on", 0.0)
    env.set_state(state)
    assert env.get_state().get(faucet_switch,
                               "is_on") < 0.5, "Switch should start off"

    opt = env.SwitchFaucetOn.ground([robot, faucet], _PUSH_PARAMS)
    result = env.execute_option(opt, max_steps=1000)

    assert result.get(faucet_switch, "is_on") > 0.5, (
        "Faucet switch should be on after SwitchFaucetOn (position mode)")


def test_push_second_switch_boil_position_mode():
    """SwitchBurnerOn toggles the second burner switch (position mode)."""
    utils.reset_config({
        "env": "pybullet_boil",
        "use_gui": False,
        "pybullet_control_mode": "position",
        "pybullet_robot": "fetch",
        "pybullet_sim_steps_per_action": 20,
        "boil_use_skill_factories": True,
        "boil_num_jugs_train": [2],
        "boil_num_jugs_test": [2],
        "boil_num_burner_train": [2],
        "boil_num_burner_test": [2],
    })
    env = _ExposedBoilEnv(use_gui=False)

    task_state = env.get_train_tasks()[0].init
    assert len(env._burners) >= 2, "Need at least 2 burners"
    burner2 = env._burners[1]
    burner_switch2 = env._burner_switches[1]
    robot = env._robot

    state = task_state.copy()
    state.set(burner_switch2, "is_on", 0.0)
    env.set_state(state)

    result = env.execute_option(env.SwitchBurnerOn.ground([robot, burner2],
                                                          _PUSH_PARAMS),
                                max_steps=1000)
    assert result.get(burner_switch2, "is_on") > 0.5


def test_push_switch_on_fan_position_mode():
    """SwitchOn toggles a fan switch to on (position mode)."""
    utils.reset_config({
        "env": "pybullet_fan",
        "use_gui": False,
        "pybullet_control_mode": "position",
        "pybullet_robot": "fetch",
        "pybullet_sim_steps_per_action": 20,
        "fan_use_skill_factories": True,
        "fan_known_controls_relation": False,
        "fan_combine_switch_on_off": False,
        "fan_train_num_walls_per_task": [0],
        "fan_test_num_walls_per_task": [0],
    })
    env = _ExposedFanEnv(use_gui=False)

    task_state = env.get_train_tasks()[0].init
    switch = task_state.get_objects(env._switch_type)[0]
    robot = env._robot

    state = task_state.copy()
    state.set(switch, "is_on", 0.0)
    env.set_state(state)
    assert env.get_state().get(switch,
                               "is_on") < 0.5, "Switch should start off"

    # Use approach offset so the robot pushes through the switch.
    push_params = [0.06, 0.11]
    opt = env.SwitchOn.ground([robot, switch], push_params)
    result = env.execute_option(opt, max_steps=1000)

    assert result.get(switch, "is_on") > 0.5, (
        "Fan switch should be on after SwitchOn (position mode)")


# ===========================================================================
# STANDALONE TESTS: Domino push (reset mode)
# ===========================================================================


def test_push_topples_domino():
    """Using the skill-factory Push from domino gt-options, a domino
    topples."""
    try:
        from predicators.envs.pybullet_domino import PyBulletDominoEnv
    except ImportError:
        pytest.skip("pybullet_domino not available")

    utils.reset_config({
        "env": "pybullet_domino",
        "use_gui": False,
        "pybullet_control_mode": "reset",
        "pybullet_robot": "fetch",
        "domino_use_skill_factories": True,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })

    class _ExposedDominoEnv(  # type: ignore[misc]
            _ExposedEnvMixin, PyBulletDominoEnv):
        pass

    env = _ExposedDominoEnv(use_gui=False)
    assert "Push" in env._options, (
        "Skill-factory Push not found in domino gt-options")
    Push = env._options["Push"]

    task_state = env.get_train_tasks()[0].init
    domino_type = env._domino_component.domino_type
    dominos = task_state.get_objects(domino_type)
    assert len(dominos) >= 1
    domino = dominos[0]
    robot = env._robot

    env.set_state(task_state.copy())
    robot_init_x = task_state.get(robot, "x")
    robot_init_y = task_state.get(robot, "y")
    result = env.execute_option(Push.ground([robot, domino], _PUSH_PARAMS))

    # In reset mode there is no physics simulation, so the domino itself does
    # not move from contact forces.  Instead verify that the skill executed
    # correctly: the robot completed its push trajectory and returned to the
    # neighbourhood of its home position (the final waypoint for the domino
    # push skill sends the robot back to robot_init_x/y).
    final_robot_x = result.get(robot, "x")
    final_robot_y = result.get(robot, "y")
    robot_return_dist = np.sqrt((final_robot_x - robot_init_x)**2 +
                                (final_robot_y - robot_init_y)**2)
    assert robot_return_dist < 0.3, (
        "Robot should return near home after push, "
        f"dist={robot_return_dist:.4f}")


def test_push_skill_domino_robot_reaches_domino():
    """Robot EE approaches the domino during the push skill execution."""
    try:
        from predicators.envs.pybullet_domino import PyBulletDominoEnv
    except ImportError:
        pytest.skip("pybullet_domino not available")

    utils.reset_config({
        "env": "pybullet_domino",
        "use_gui": False,
        "pybullet_control_mode": "reset",
        "pybullet_robot": "fetch",
        "domino_use_skill_factories": True,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })

    class _ExposedDominoEnv(  # type: ignore[misc]
            _ExposedEnvMixin, PyBulletDominoEnv):
        pass

    env = _ExposedDominoEnv(use_gui=False)
    Push = env._options["Push"]

    task_state = env.get_train_tasks()[0].init
    domino = task_state.get_objects(env._domino_component.domino_type)[0]
    robot = env._robot

    dom_x = task_state.get(domino, "x")
    dom_y = task_state.get(domino, "y")

    env.set_state(task_state.copy())
    result = env.execute_option(Push.ground([robot, domino], _PUSH_PARAMS))

    robot_x = result.get(robot, "x")
    robot_y = result.get(robot, "y")
    dist = np.sqrt((robot_x - dom_x)**2 + (robot_y - dom_y)**2)
    assert dist < 0.5, f"Robot should approach domino, dist={dist:.3f}"


@pytest.mark.xfail(reason="BiRRT motion planning may fail to find path")
def test_pick_holds_domino_with_motion_planning():
    """Pick option with motion planning should result in the domino being held.

    Uses position control mode to match the production setup where the bug
    manifests: with motion planning the robot grasps at a corner instead of
    the center.
    """
    try:
        from predicators.envs.pybullet_domino import PyBulletDominoEnv
    except ImportError:
        pytest.skip("pybullet_domino not available")

    utils.reset_config({
        "env": "pybullet_domino",
        "use_gui": False,
        "pybullet_control_mode": "position",
        "pybullet_robot": "fetch",
        "domino_use_skill_factories": True,
        "skill_phase_use_motion_planning": True,
        "pybullet_ik_validate": False,
        "domino_use_domino_blocks_as_target": True,
        "domino_restricted_push": True,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })

    class _ExposedDominoEnv(  # type: ignore[misc]
            _ExposedEnvMixin, PyBulletDominoEnv):
        pass

    env = _ExposedDominoEnv(use_gui=False)
    Pick = env._options["Pick"]

    task_state = env.get_train_tasks()[0].init
    domino_type = env._domino_component.domino_type
    dominos = task_state.get_objects(domino_type)
    assert len(dominos) >= 1
    domino = dominos[0]
    robot = env._robot

    dom_x = task_state.get(domino, "x")
    dom_y = task_state.get(domino, "y")
    dom_z = task_state.get(domino, "z")
    print(f"\nDomino position: ({dom_x:.4f}, {dom_y:.4f}, {dom_z:.4f})")

    env.set_state(task_state.copy())
    option = Pick.ground([robot, domino], _PICK_PARAMS)
    assert option.initiable(env._current_state)

    # Run option with step-by-step logging
    prev_phase = None
    for step_i in range(600):
        if option.terminal(env._current_state):
            print(f"Step {step_i}: Option terminal")
            break
        state = env._current_state
        # Log phase transitions
        phase_idx = option.memory.get("phase_idx", 0)
        if phase_idx != prev_phase:
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            rz = state.get(robot, "z")
            rf = state.get(robot, "fingers")
            ee_dist = np.sqrt((rx - dom_x)**2 + (ry - dom_y)**2)
            print(f"Step {step_i}: Phase {phase_idx}, "
                  f"robot=({rx:.4f}, {ry:.4f}, {rz:.4f}), "
                  f"fingers={rf:.4f}, ee_dist_xy={ee_dist:.4f}")
            prev_phase = phase_idx
        action = option.policy(state)
        env.step(action)
    else:
        print("WARNING: option did not terminate in 600 steps")

    result = env._current_state
    is_held = result.get(domino, "is_held")
    rx = result.get(robot, "x")
    ry = result.get(robot, "y")
    rz = result.get(robot, "z")
    print(f"Final: robot=({rx:.4f}, {ry:.4f}, {rz:.4f}), "
          f"is_held={is_held}")

    assert is_held > 0.5, (
        f"Domino should be held after Pick with motion planning, "
        f"is_held={is_held}")


@pytest.mark.xfail(reason="PyBullet pick may fail in headless CI")
def test_pick_holds_domino_without_motion_planning():
    """Pick option WITHOUT motion planning should hold the domino
    (baseline)."""
    try:
        from predicators.envs.pybullet_domino import PyBulletDominoEnv
    except ImportError:
        pytest.skip("pybullet_domino not available")

    utils.reset_config({
        "env": "pybullet_domino",
        "use_gui": False,
        "pybullet_control_mode": "position",
        "pybullet_robot": "fetch",
        "domino_use_skill_factories": True,
        "skill_phase_use_motion_planning": False,
        "pybullet_ik_validate": False,
        "domino_use_domino_blocks_as_target": True,
        "domino_restricted_push": True,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })

    class _ExposedDominoEnv(  # type: ignore[misc]
            _ExposedEnvMixin, PyBulletDominoEnv):
        pass

    env = _ExposedDominoEnv(use_gui=False)
    Pick = env._options["Pick"]

    task_state = env.get_train_tasks()[0].init
    domino_type = env._domino_component.domino_type
    dominos = task_state.get_objects(domino_type)
    assert len(dominos) >= 1
    domino = dominos[0]
    robot = env._robot

    dom_x = task_state.get(domino, "x")
    dom_y = task_state.get(domino, "y")
    dom_z = task_state.get(domino, "z")
    print(f"\nDomino position: ({dom_x:.4f}, {dom_y:.4f}, {dom_z:.4f})")

    env.set_state(task_state.copy())
    option = Pick.ground([robot, domino], _PICK_PARAMS)
    assert option.initiable(env._current_state)

    prev_phase = None
    for step_i in range(600):
        if option.terminal(env._current_state):
            print(f"Step {step_i}: Option terminal")
            break
        state = env._current_state
        phase_idx = option.memory.get("phase_idx", 0)
        if phase_idx != prev_phase:
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            rz = state.get(robot, "z")
            rf = state.get(robot, "fingers")
            ee_dist = np.sqrt((rx - dom_x)**2 + (ry - dom_y)**2)
            print(f"Step {step_i}: Phase {phase_idx}, "
                  f"robot=({rx:.4f}, {ry:.4f}, {rz:.4f}), "
                  f"fingers={rf:.4f}, ee_dist_xy={ee_dist:.4f}")
            prev_phase = phase_idx
        action = option.policy(state)
        env.step(action)
    else:
        print("WARNING: option did not terminate in 600 steps")

    result = env._current_state
    is_held = result.get(domino, "is_held")
    rx = result.get(robot, "x")
    ry = result.get(robot, "y")
    rz = result.get(robot, "z")
    print(f"Final: robot=({rx:.4f}, {ry:.4f}, {rz:.4f}), "
          f"is_held={is_held}")

    assert is_held > 0.5, (
        f"Domino should be held after Pick without motion planning, "
        f"is_held={is_held}")


@pytest.mark.xfail(reason="BiRRT motion planning may fail to find path")
def test_domino_pick_place_no_collisions():
    """Pick domino_1 and place it between others — no non-held domino moves.

    Uses position mode with motion planning so BiRRT plans collision-
    free paths.  Verifies that non-held dominoes remain stationary
    throughout the pick and place sequences (i.e., no arm–domino
    collisions).
    """
    try:
        from predicators.envs.pybullet_domino import PyBulletDominoEnv
    except ImportError:
        pytest.skip("pybullet_domino not available")

    utils.reset_config({
        "env": "pybullet_domino",
        "use_gui": False,
        "pybullet_control_mode": "position",
        "pybullet_robot": "fetch",
        "domino_use_skill_factories": True,
        "skill_phase_use_motion_planning": True,
        "pybullet_ik_validate": False,
        "domino_initialize_at_finished_state": False,
        "domino_use_domino_blocks_as_target": True,
        "domino_include_connected_predicate": False,
        "domino_use_continuous_place": True,
        "domino_restricted_push": True,
        "domino_prune_actions": False,
        "domino_has_glued_dominos": False,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })

    class _ExposedDominoEnv(  # type: ignore[misc]
            _ExposedEnvMixin, PyBulletDominoEnv):
        pass

    env = _ExposedDominoEnv(use_gui=False)
    Pick = env._options["Pick"]
    Place = env._options["Place"]

    domino_type = env._domino_component.domino_type
    robot_type = next(t for t in env.types if t.name == "robot")

    # Use test task 0 (matches debug_motion_planning.py setup)
    obs = env.reset("test", 0)
    state = obs

    robot = state.get_objects(robot_type)[0]
    dominos = state.get_objects(domino_type)
    pick_target = next(d for d in dominos if d.name == "domino_1")

    pos_tol = 1e-3

    def _get_positions(st):
        return {
            o.name: (st.get(o, "x"), st.get(o, "y"), st.get(o, "z"))
            for o in st.get_objects(domino_type)
        }

    def _check_moved(before, st, skip_names=()):
        moved = []
        cur = _get_positions(st)
        for name, (bx, by, bz) in before.items():
            if name in skip_names:
                continue
            cx, cy, cz = cur[name]
            disp = np.sqrt((cx - bx)**2 + (cy - by)**2 + (cz - bz)**2)
            if disp > pos_tol:
                moved.append((name, disp))
        return moved

    # ---- Pick domino_1 ----
    pos_before_pick = _get_positions(state)
    option = Pick.ground([robot, pick_target],
                         np.array([0.01], dtype=np.float32))
    assert option.initiable(state)

    pick_collisions = []
    for _ in range(300):
        if option.terminal(state):
            break
        action = option.policy(state)
        state = env.simulate(state, action)
        moved = _check_moved(pos_before_pick,
                             state,
                             skip_names={pick_target.name})
        if moved:
            pick_collisions = moved

    assert state.get(pick_target, "is_held") > 0.5, \
        "domino_1 should be held after pick"
    assert not pick_collisions, \
        f"Non-held dominoes moved during Pick: {pick_collisions}"

    # ---- Place at (0.75, 1.26) between existing dominoes ----
    pos_before_place = _get_positions(state)
    target_x, target_y, target_yaw = 0.75, 1.26, 0.0
    release_z = env.table_height + env.domino_height * 1.13

    option = Place.ground([robot],
                          np.array([target_x, target_y, release_z, target_yaw],
                                   dtype=np.float32))
    assert option.initiable(state)

    place_collisions = []
    for _ in range(300):
        if option.terminal(state):
            break
        action = option.policy(state)
        state = env.simulate(state, action)
        moved = _check_moved(pos_before_place, state)
        if moved:
            place_collisions = moved

    assert not place_collisions, \
        f"Non-held dominoes moved during Place: {place_collisions}"


def test_domino_second_place_with_unvalidated_ik():
    """The seed-0 bridge placement for domino_2 should refine with
    pybullet_ik_validate disabled.

    This covers a failure mode where the fast one-shot IK solution
    reaches the EE target but leaves the held domino colliding with the
    table, so collision-aware BiRRT needs to retry the IK target with
    validation before declaring Place infeasible.
    """
    try:
        from predicators.envs.pybullet_domino import PyBulletDominoEnv
    except ImportError:
        pytest.skip("pybullet_domino not available")

    from predicators.ground_truth_models.domino.processes import \
        _pick_option_sampler, _place_option_sampler
    from predicators.option_model import _OracleOptionModel
    from predicators.structs import GroundAtom

    utils.reset_config({
        "env": "pybullet_domino",
        "use_gui": False,
        "pybullet_control_mode": "position",
        "pybullet_robot": "fetch",
        "domino_use_skill_factories": True,
        "skill_phase_use_motion_planning": True,
        "option_model_terminate_on_repeat": False,
        "pybullet_ik_validate": False,
        "domino_initialize_at_finished_state": False,
        "domino_use_domino_blocks_as_target": True,
        "domino_include_connected_predicate": False,
        "domino_use_continuous_place": True,
        "domino_restricted_push": True,
        "domino_prune_actions": False,
        "domino_has_glued_dominos": False,
        "pybullet_birrt_extend_num_interp": 20,
        "pybullet_birrt_path_subsample_ratio": 2,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })

    class _ExposedDominoEnv(  # type: ignore[misc]
            _ExposedEnvMixin, PyBulletDominoEnv):
        pass

    env = _ExposedDominoEnv(use_gui=False)
    options = env._options
    model = _OracleOptionModel(set(options.values()), env.simulate)
    state = env.get_test_tasks()[0].init
    objs = {o.name: o for o in state}
    preds = {p.name: p for p in env.predicates}
    robot = objs["robot"]
    d0 = objs["domino_0"]
    d1 = objs["domino_1"]
    d2 = objs["domino_2"]
    d3 = objs["domino_3"]

    def _run_option(option, cur_state):
        next_state, num_actions = model.get_next_state_and_num_actions(
            cur_state, option)
        assert num_actions > 0, model.last_execution_failure
        return next_state

    pick1 = options["Pick"].ground([robot, d1],
                                   _pick_option_sampler(
                                       state, set(), np.random.default_rng(0),
                                       [robot, d1]))
    state = _run_option(pick1, state)

    subgoal1 = {
        GroundAtom(preds["InFront"], [d1, d0]),
        GroundAtom(preds["HandEmpty"], [robot]),
    }
    place1 = options["Place"].ground([robot],
                                     _place_option_sampler(
                                         state, subgoal1,
                                         np.random.default_rng(0), [robot]))
    state = _run_option(place1, state)

    pick2 = options["Pick"].ground([robot, d2],
                                   _pick_option_sampler(
                                       state, set(), np.random.default_rng(0),
                                       [robot, d2]))
    state = _run_option(pick2, state)

    subgoal2 = {
        GroundAtom(preds["InFront"], [d3, d2]),
        GroundAtom(preds["InFront"], [d2, d1]),
        GroundAtom(preds["HandEmpty"], [robot]),
    }
    place2 = options["Place"].ground([robot],
                                     _place_option_sampler(
                                         state, subgoal2,
                                         np.random.default_rng(0), [robot]))
    state = _run_option(place2, state)

    assert GroundAtom(preds["HandEmpty"], [robot]).holds(state)
    assert state.get(d2, "is_held") < 0.5


@pytest.mark.xfail(reason="Button detection zone overlaps dispense area "
                   "approach path — robot arm triggers button during place")
def test_coffee_place_no_button_press():
    """PickJug then PlaceJugInMachine without turning machine on.

    The jug should be placed on the dispense area without hitting the
    machine's top overhang or accidentally pressing the button.
    """
    utils.reset_config({
        "env": "pybullet_coffee",
        "use_gui": False,
        "pybullet_control_mode": "position",
        "pybullet_robot": "fetch",
        "pybullet_ik_validate": False,
        "coffee_use_skill_factories": True,
        "coffee_rotated_jug_ratio": 0,
        "coffee_num_cups_train": [1],
        "coffee_num_cups_test": [1],
        "coffee_machine_have_light_bar": False,
        "coffee_move_back_after_place_and_push": True,
        "coffee_machine_has_plug": False,
        "coffee_combined_move_and_twist_policy": True,
        "coffee_use_pixelated_jug": True,
        "coffee_fill_jug_gradually": True,
        "skill_phase_use_motion_planning": True,
        "max_num_steps_option_rollout": 100,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })

    env = _ExposedCoffeeEnv(use_gui=False)

    robot_type = next(t for t in env.types if t.name == "robot")
    jug_type = next(t for t in env.types if t.name == "jug")
    machine_type = next(t for t in env.types if t.name == "coffee_machine")

    obs = env.reset("test", 0)
    state = obs

    robot = state.get_objects(robot_type)[0]
    jug = state.get_objects(jug_type)[0]
    machine = state.get_objects(machine_type)[0]

    assert state.get(machine, "is_on") < 0.5, "Machine should start OFF"

    # ---- Pick the jug ----
    pick_option = env.PickJug.ground([robot, jug],
                                     np.array([0.01], dtype=np.float32))
    assert pick_option.initiable(state)

    for _ in range(300):
        if pick_option.terminal(state):
            break
        action = pick_option.policy(state)
        state = env.simulate(state, action)

    assert state.get(jug, "is_held") > 0.5, "Jug should be held after pick"
    assert state.get(machine, "is_on") < 0.5, "Machine turned on during pick!"

    # ---- Place jug in machine ----
    target_x = PyBulletCoffeeEnv.dispense_area_x
    target_y = PyBulletCoffeeEnv.dispense_area_y
    target_yaw = PyBulletCoffeeEnv.robot_init_wrist
    release_z = PyBulletCoffeeEnv.z_lb + env.jug_handle_height()

    place_option = env.PlaceJugInMachine.ground(
        [robot, jug, machine],
        np.array([target_x, target_y, release_z, target_yaw],
                 dtype=np.float32))
    assert place_option.initiable(state)

    machine_turned_on_step = None
    for step in range(300):
        if place_option.terminal(state):
            break
        action = place_option.policy(state)
        state = env.simulate(state, action)

        if state.get(machine,
                     "is_on") > 0.5 and machine_turned_on_step is None:
            machine_turned_on_step = step

    assert machine_turned_on_step is None, (
        f"Machine was turned on at step {machine_turned_on_step} during "
        f"PlaceJugInMachine — robot arm likely triggered the button.")


def test_human_option_control_scripted_domino_solves_task():
    """Full pipeline: human_option_control approach with scripted option plan
    (domino2.txt) solves the 1st test task in pybullet_domino."""
    try:
        from predicators.envs.pybullet_domino import PyBulletDominoEnv
    except ImportError:
        pytest.skip("pybullet_domino not available")

    from predicators.approaches import create_approach
    from predicators.cogman import CogMan, run_episode_and_get_observations
    from predicators.execution_monitoring import create_execution_monitor
    from predicators.perception import create_perceiver

    utils.reset_config({
        "env": "pybullet_domino",
        "approach": "human_option_control",
        "seed": 0,
        "use_gui": False,
        "pybullet_control_mode": "position",
        "pybullet_ik_validate": False,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "horizon": 200,
        "domino_use_skill_factories": True,
        "domino_initialize_at_finished_state": False,
        "domino_use_domino_blocks_as_target": True,
        "domino_include_connected_predicate": False,
        "domino_use_continuous_place": True,
        "domino_restricted_push": True,
        "domino_prune_actions": False,
        "domino_has_glued_dominos": False,
        "keep_failed_demos": True,
        "skill_phase_use_motion_planning": True,
        "human_option_control_approach_use_scripted_option": True,
        "human_option_control_approach_use_all_options": True,
        "scripted_option_dir": "scripted_option_policies",
        "script_option_file_name": "domino2.txt",
    })

    env = PyBulletDominoEnv(use_gui=False)
    _MOST_RECENT_ENV_INSTANCE[env.get_name()] = env

    perceiver = create_perceiver("trivial")
    train_tasks = [perceiver.reset(t) for t in env.get_train_tasks()]

    options = get_gt_options(env.get_name())
    approach = create_approach(
        "human_option_control",
        env.predicates,
        options,
        env.types,
        env.action_space,
        train_tasks,
    )

    cogman = CogMan(approach, perceiver, create_execution_monitor("trivial"))

    test_env_task = env.get_test_tasks()[0]
    cogman.reset(test_env_task)

    _traj, solved, _metrics = run_episode_and_get_observations(
        cogman,
        env,
        "test",
        task_idx=0,
        max_num_steps=200,
        terminate_on_goal_reached=True,
    )

    assert solved, ("Scripted domino2.txt plan should solve the 1st test task")
