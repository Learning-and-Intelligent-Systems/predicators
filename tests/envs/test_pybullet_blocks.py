"""Test cases for PyBulletBlocksEnv."""

import numpy as np
from predicators.src.envs import PyBulletBlocksEnv
from predicators.src.structs import State, Object, Type
from predicators.src import utils


class _ExposedPyBulletBlocksEnv(PyBulletBlocksEnv):

    @property
    def block_type(self) -> Type:
        """Expose the block type."""
        return self._block_type

    @property
    def robot(self) -> Object:
        """Expose the robot, which is a static object."""
        return self._robot

    def set_state(self, state: State) -> None:
        """Forcibly reset the state."""
        self._current_state = state
        self._current_task = None
        self._reset_state(state)

    def get_state(self) -> State:
        """Expose get state."""
        return self._get_state()


def test_pybullet_blocks_reset():
    """Tests for PyBulletBlocksEnv.reset()."""
    utils.reset_config({"env": "pybullet_blocks"})
    env = PyBulletBlocksEnv()
    env.seed(123)
    for idx, task in enumerate(env.get_train_tasks()):
        state = env.reset("train", idx)
        assert state.allclose(task.init)
    for idx, task in enumerate(env.get_test_tasks()):
        state = env.reset("test", idx)
        assert state.allclose(task.init)


def test_pybullet_blocks_picking():
    """Tests cases for picking blocks in PyBulletBlocksEnv."""
    utils.reset_config({"env": "pybullet_blocks"})
    env = _ExposedPyBulletBlocksEnv()
    env.seed(123)
    block = Object("block0", env.block_type)
    robot = env.robot
    bx = (env.x_lb + env.x_ub) / 2
    by = (env.y_lb + env.y_ub) / 2
    bz = env.table_height
    rx, ry, rz = env.robot_init_x, env.robot_init_y, env.robot_init_z
    rf = env.open_fingers
    # Create a simple custom state with one block for testing.
    state = State({
        robot: np.array([rx, ry, rz, rf]),
        block: np.array([bx, by, bz, 0.0]),
    })
    env.set_state(state)
    assert env.get_state().allclose(state)
