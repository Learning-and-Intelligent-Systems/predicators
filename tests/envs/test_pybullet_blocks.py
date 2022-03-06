"""Test cases for PyBulletBlocksEnv."""

import numpy as np
from predicators.src.envs import PyBulletBlocksEnv
from predicators.src.structs import Object, State
from predicators.src import utils

GUI_ON = True  # toggle for debugging


class _ExposedPyBulletBlocksEnv(PyBulletBlocksEnv):

    @property
    def block_type(self):
        """Expose the block type."""
        return self._block_type

    @property
    def robot(self):
        """Expose the robot, which is a static object."""
        return self._robot

    @property
    def Pick(self):
        """Expose the Pick parameterized option."""
        return self._Pick

    def set_state(self, state) -> None:
        """Forcibly reset the state."""
        self._current_state = state
        self._current_task = None
        self._reset_state(state)

    def get_state(self):
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
    utils.reset_config({"env": "pybullet_blocks", "pybullet_use_gui": GUI_ON})
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
    init_state = State({
        robot: np.array([rx, ry, rz, rf]),
        block: np.array([bx, by, bz, 0.0]),
    })
    env.set_state(init_state)
    assert env.get_state().allclose(init_state)
    # Create an option for picking the block.
    option = env.Pick.ground([robot, block], [])
    state = init_state.copy()
    assert option.initiable(state)
    # Execute the option.
    for _ in range(100):
        if option.terminal(state):
            break
        action = option.policy(state)
        state = env.step(action)
    else:
        assert False, "Option failed to terminate."
    # The block should now be held.
    assert state.get(block, "held") == 1.0
