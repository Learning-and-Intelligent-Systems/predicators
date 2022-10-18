"""Test cases for PyBulletCoverEnv."""

import numpy as np
import pytest

from predicators import utils
from predicators.envs.pybullet_cover import PyBulletCoverEnv
from predicators.settings import CFG
from predicators.structs import Object, State

_GUI_ON = False  # toggle for debugging


class _ExposedPyBulletCoverEnv(PyBulletCoverEnv):

    disable_hand_regions = False

    @property
    def workspace_dimensions(self):
        """Expose the workspace dimensions."""
        return (self._workspace_x, self._workspace_z)

    @property
    def block_type(self):
        """Expose the block type."""
        return self._block_type

    @property
    def target_type(self):
        """Expose the target type."""
        return self._target_type

    @property
    def robot(self):
        """Expose the robot, which is a static object."""
        return self._robot

    @property
    def PickPlace(self):
        """Expose the PickPlace parameterized option."""
        return self._PickPlace

    def set_state(self, state):
        """Forcibly reset the state."""
        joint_positions = list(self._pybullet_robot.initial_joint_positions)
        state_with_sim = utils.PyBulletState(state.data,
                                             simulator_state=joint_positions)
        self._current_state = state_with_sim
        self._current_task = None
        self._reset_state(state_with_sim)

    def get_state(self):
        """Expose get_state()."""
        return self._get_state()

    def execute_option(self, option):
        """Helper for executing a single option, updating the env state."""
        # Note that since we want to use self._current_state, it's convenient
        # to make this an environment method, rather than a helper function.
        assert option.initiable(self._current_state)
        # Execute the option.
        for _ in range(100):
            if option.terminal(self._current_state):
                break
            action = option.policy(self._current_state)
            self.step(action)
        return self._current_state.copy()

    def _get_hand_regions(self, state):
        if self.disable_hand_regions:
            return []
        return super()._get_hand_regions(state)


@pytest.fixture(scope="module", name="env", params=("fetch", "panda"))
def _create_exposed_pybullet_cover_env(request):
    """Only create once and share among all tests, for efficiency."""
    utils.reset_config({
        "env": "pybullet_cover",
        "pybullet_use_gui": _GUI_ON,
        "cover_initial_holding_prob": 0.0,
        # We run this test using the POSITION control mode.
        "pybullet_control_mode": "position",
        # Which robot we're using
        "pybullet_robot": request.param,
    })
    return _ExposedPyBulletCoverEnv()


def test_pybullet_cover_reset(env):
    """Tests for PyBulletCoverEnv.reset()."""
    for idx, task in enumerate(env.get_train_tasks()):
        assert isinstance(task.init, utils.PyBulletState)
        state = env.reset("train", idx)
        assert state.allclose(task.init)
    for idx, task in enumerate(env.get_test_tasks()):
        state = env.reset("test", idx)
        assert state.allclose(task.init)
    # Simulate and render state should be not implemented.
    action = env.action_space.sample()
    with pytest.raises(NotImplementedError):
        env.simulate(state, action)
    task = env.get_train_tasks()[0]
    with pytest.raises(NotImplementedError):
        env.render_state(state, task, action)


def test_pybullet_cover_step(env):
    """Tests for taking actions in PyBulletCoverEnv."""
    block = Object("block0", env.block_type)
    target = Object("target0", env.target_type)
    robot = env.robot
    workspace_x, workspace_z = env.workspace_dimensions
    # Create a simple custom state with one block for testing.
    init_state = State({
        robot:
        np.array([0.5, workspace_x, workspace_z]),
        block:
        np.array([1.0, 0.0, CFG.cover_block_widths[0], 0.25, -1]),
        target:
        np.array([0.0, 1.0, CFG.cover_target_widths[0], 0.75]),
    })
    env.set_state(init_state)
    recovered_state = env.get_state()
    assert recovered_state.allclose(init_state)
    # Create an option for picking the block.
    option = env.PickPlace.ground([], [0.25])
    # Execute the option.
    state = env.execute_option(option)
    # The block should now be held.
    assert state.get(block, "grasp") != -1
    # Create an option for placing the block.
    option = env.PickPlace.ground([], [0.75])
    # Execute the option.
    state = env.execute_option(option)
    # The block should now be placed.
    assert state.get(block, "grasp") == -1
    assert abs(state.get(block, "pose") - 0.75) < 0.01
    # Create a second simple custom state with the block starting off held.
    init_state = State({
        robot:
        np.array([0.4, workspace_x, workspace_z]),
        block:
        np.array([1.0, 0.0, CFG.cover_block_widths[0], 0.4, 0]),
        target:
        np.array([0.0, 1.0, CFG.cover_target_widths[0], 0.75]),
    })
    env.set_state(init_state)
    recovered_state = env.get_state()
    assert abs(recovered_state.get(block, "grasp")) < 0.01
    # Try placing outside the hand regions, which should fail.
    env.disable_hand_regions = True
    option = env.PickPlace.ground([], [0.75])
    state = env.execute_option(option)
    assert abs(state.get(block, "grasp")) < 0.01
    # Now correctly place.
    env.disable_hand_regions = False
    option = env.PickPlace.ground([], [0.75])
    state = env.execute_option(option)
    assert state.get(block, "grasp") == -1
    assert abs(state.get(block, "pose") - 0.75) < 0.01


def test_pybullet_cover_pick_workspace_bounds(env):
    """Tests for picking at workspace bounds in PyBulletCoverEnv."""
    block = Object("block0", env.block_type)
    robot = env.robot
    workspace_x, workspace_z = env.workspace_dimensions
    for pose in [0.0, 1.0]:
        # Create a simple custom state with one block for testing.
        init_state = State({
            robot:
            np.array([0.5, workspace_x, workspace_z]),
            block:
            np.array([1.0, 0.0, CFG.cover_block_widths[0], pose, -1]),
        })
        env.set_state(init_state)
        recovered_state = env.get_state()
        assert recovered_state.allclose(init_state)
        # Create an option for picking the block.
        option = env.PickPlace.ground([], [pose])
        # Execute the option.
        state = env.execute_option(option)
        # The block should now be held.
        assert state.get(block, "grasp") != -1
