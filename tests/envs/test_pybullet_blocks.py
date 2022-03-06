"""Test cases for PyBulletBlocksEnv."""

import numpy as np
from predicators.src.envs import PyBulletBlocksEnv
from predicators.src.structs import Object, State, Action
from predicators.src import utils

GUI_ON = False  # toggle for debugging


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
    # Execute the option. Also record the actions for use in the next test.
    pick_actions = []
    for _ in range(100):
        if option.terminal(state):
            break
        action = option.policy(state)
        pick_actions.append(action)
        state = env.step(action)
    else:
        assert False, "Option failed to terminate."
    # The block should now be held.
    assert state.get(block, "held") == 1.0
    # Test the case where the right finger is on the left side of the block,
    # but within the grasp tolerance. The contact normal check should prevent
    # a holding constraint from being created.
    env.set_state(init_state)
    state = init_state.copy()
    move_left_actions = [
        # Move to the left of the block.
        Action(np.array([0.0, env.block_size, 0.0, 0.0], dtype=np.float32)),
        # Make room for the finger.
        Action(np.array([0.0, 0.04, 0.0, 0.0], dtype=np.float32)),
    ]
    actions = move_left_actions + pick_actions
    for action in actions:
        state = env.step(action)
    # The block should NOT be held.
    assert state.get(block, "held") == 0.0
    # Test that the block can be picked at the extremes of the workspace.
    half_size = env.block_size / 2
    corners = [
        (env.x_lb + half_size, env.y_lb + half_size),
        (env.x_ub - half_size, env.y_lb + half_size),
        (env.x_lb + half_size, env.y_ub - half_size),
        (env.x_ub - half_size, env.y_ub - half_size),
    ]
    for (bx, by) in corners:
        state = init_state.copy()
        state.set(block, "pose_x", bx)
        state.set(block, "pose_y", by)
        env.set_state(state)
        assert env.get_state().allclose(state)
        # Create an option for picking the block.
        option = env.Pick.ground([robot, block], [])
        assert option.initiable(state)
        # Execute the option. Also record the actions for use in the next test.
        pick_actions = []
        for _ in range(100):
            if option.terminal(state):
                break
            action = option.policy(state)
            pick_actions.append(action)
            state = env.step(action)
        else:
            assert False, "Option failed to terminate."
        # The block should now be held.
        assert state.get(block, "held") == 1.0
