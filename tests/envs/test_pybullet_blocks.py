"""Test cases for PyBulletBlocksEnv."""

import numpy as np
import pytest
from predicators.src.envs import PyBulletBlocksEnv
from predicators.src.structs import Object, State, Action
from predicators.src import utils
from predicators.src.settings import CFG

_GUI_ON = False  # toggle for debugging
_EXPOSED_PYBULLET_ENV = None  # only create once, since init is expensive


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

    @property
    def Stack(self):
        """Expose the Stack parameterized option."""
        return self._Stack

    @property
    def PutOnTable(self):
        """Expose the PutOnTable parameterized option."""
        return self._PutOnTable

    @property
    def On(self):
        """Expose the On predicate."""
        return self._On

    @property
    def OnTable(self):
        """Expose the OnTable predicate."""
        return self._OnTable

    @property
    def GripperOpen(self):
        """Expose the GripperOpen predicate."""
        return self._GripperOpen

    @property
    def Clear(self):
        """Expose the Clear predicate."""
        return self._Clear

    @property
    def Holding(self):
        """Expose the Holding predicate."""
        return self._Holding

    def set_state(self, state):
        """Forcibly reset the state."""
        self._current_state = state
        self._current_task = None
        self._reset_state(state)

    def get_state(self):
        """Expose get state."""
        return self._get_state()

    def execute_pick(self, block):
        """Convenient for tests that want a state where block is held.

        The reason we do it this way instead of setting the state directly
        is that a holding constraint needs to be created.

        Returns a copy of the state upon completion.
        """
        option = self._Pick.ground([self._robot, block], [])
        state = self._execute_option(option)
        assert state.get(block, "held") == 1.0
        return state

    def execute_stack(self, block):
        """Helper for tests involving stacking."""
        option = self._Stack.ground([self._robot, block], [])
        return self._execute_option(option)

    def execute_putontable(self, norm_x, norm_y):
        """Helper for tests involving putting on table."""
        option = self._PutOnTable.ground([self._robot], [norm_x, norm_y])
        return self._execute_option(option)

    def _execute_option(self, option):
        """Helper for execution methods."""
        assert option.initiable(self._current_state)
        # Execute the pick option.
        while True:
            if option.terminal(self._current_state):
                break
            action = option.policy(self._current_state)
            self.step(action)
        return self._current_state.copy()


def _get__EXPOSED_PYBULLET_ENV():
    global _EXPOSED_PYBULLET_ENV  # pylint:disable=global-statement
    if _EXPOSED_PYBULLET_ENV is None:
        _EXPOSED_PYBULLET_ENV = _ExposedPyBulletBlocksEnv()
    return _EXPOSED_PYBULLET_ENV


def test_pybullet_blocks_reset():
    """Tests for PyBulletBlocksEnv.reset()."""
    utils.reset_config({"env": "pybullet_blocks", "pybullet_use_gui": _GUI_ON})
    env = _get__EXPOSED_PYBULLET_ENV()
    env.seed(123)
    for idx, task in enumerate(env.get_train_tasks()):
        state = env.reset("train", idx)
        assert state.allclose(task.init)
    for idx, task in enumerate(env.get_test_tasks()):
        state = env.reset("test", idx)
        assert state.allclose(task.init)
    # Test that resetting raises an error if an unreachable state is given.
    state = env.get_train_tasks()[0].init.copy()
    block = state.get_objects(env.block_type)[0]
    # Make the state impossible.
    state.set(block, "held", -10000)
    with pytest.raises(ValueError) as e:
        env.set_state(state)
    assert "Could not reconstruct state." in str(e)


def test_pybullet_blocks_picking():
    """Tests cases for picking blocks in PyBulletBlocksEnv."""
    utils.reset_config({"env": "pybullet_blocks", "pybullet_use_gui": _GUI_ON})
    env = _get__EXPOSED_PYBULLET_ENV()
    env.seed(123)
    block = Object("block0", env.block_type)
    robot = env.robot
    bx = (env.x_lb + env.x_ub) / 2
    by = (env.y_lb + env.y_ub) / 2
    bz = env.table_height + 0.5 * env.block_size
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
    assert env.GripperOpen([robot]).holds(state)
    # Execute the option. Also record the actions for use in the next test.
    pick_actions = []
    while True:
        if option.terminal(state):
            break
        action = option.policy(state)
        pick_actions.append(action)
        state = env.step(action)
    # The block should now be held.
    assert state.get(block, "held") == 1.0
    assert not env.GripperOpen([robot]).holds(state)
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
    corners = [
        (env.x_lb, env.y_lb),
        (env.x_ub, env.y_lb),
        (env.x_lb, env.y_ub),
        (env.x_ub, env.y_ub),
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
        assert env.GripperOpen([robot]).holds(state)
        # Execute the option.
        while True:
            if option.terminal(state):
                break
            action = option.policy(state)
            state = env.step(action)
        # The block should now be held.
        assert state.get(block, "held") == 1.0
        assert not env.GripperOpen([robot]).holds(state)


def test_pybullet_blocks_stacking():
    """Tests cases for stacking blocks in PyBulletBlocksEnv."""
    utils.reset_config({"env": "pybullet_blocks", "pybullet_use_gui": _GUI_ON})
    env = _get__EXPOSED_PYBULLET_ENV()
    env.seed(123)
    block0 = Object("block0", env.block_type)
    block1 = Object("block1", env.block_type)
    robot = env.robot
    bx0 = (env.x_lb + env.x_ub) / 2
    by0 = (env.y_lb + env.y_ub) / 2 - env.block_size
    by1 = (env.y_lb + env.y_ub) / 2 + env.block_size
    bz0 = env.table_height + 0.5 * env.block_size
    rx, ry, rz = env.robot_init_x, env.robot_init_y, env.robot_init_z
    rf = env.open_fingers
    # Create a state with two blocks.
    init_state = State({
        robot: np.array([rx, ry, rz, rf]),
        block0: np.array([bx0, by0, bz0, 0.0]),
        block1: np.array([bx0, by1, bz0, 0.0]),
    })
    env.set_state(init_state)
    assert env.get_state().allclose(init_state)
    # Pick block0 to get to a state where we are prepared to stack.
    state = env.execute_pick(block0)
    # Create a stack option.
    option = env.Stack.ground([robot, block1], [])
    assert option.initiable(state)
    assert not env.GripperOpen([robot]).holds(state)
    # Execute the stack option.
    while True:
        if option.terminal(state):
            break
        action = option.policy(state)
        state = env.step(action)
    # The block should now NOT be held.
    assert state.get(block0, "held") == 0.0
    # And block0 should be on block1.
    assert env.On([block0, block1]).holds(state)
    assert env.GripperOpen([robot]).holds(state)
    # Test extremes: stacking a block on the tallest possible tower, at each
    # of the possible corners.
    corners = [
        (env.x_lb, env.y_lb),
        (env.x_ub, env.y_lb),
        (env.x_lb, env.y_ub),
        (env.x_ub, env.y_ub),
    ]
    max_num_blocks = max(max(CFG.blocks_num_blocks_train),
                         max(CFG.blocks_num_blocks_test))
    block_to_z = {
        Object(f"block{i+1}", env.block_type): bz0 + i * env.block_size
        for i in range(max_num_blocks - 1)
    }
    top_block = max(block_to_z, key=block_to_z.get)
    for (bx, by) in corners:
        state = State({
            robot: np.array([rx, ry, rz, rf]),
            block0: np.array([bx0, by0, bz0, 0.0]),
            **{b: np.array([bx, by, bz, 0.0])
               for b, bz in block_to_z.items()}
        })
        env.set_state(state)
        assert env.get_state().allclose(state)
        # Pick block0 to get to a state where we are prepared to stack.
        state = env.execute_pick(block0)
        # Create a stack option.
        option = env.Stack.ground([robot, top_block], [])
        assert option.initiable(state)
        assert not env.GripperOpen([robot]).holds(state)
        # Execute the stack option.
        while True:
            if option.terminal(state):
                break
            action = option.policy(state)
            state = env.step(action)
        # The block should now NOT be held.
        assert state.get(block0, "held") == 0.0
        # And block0 should be on top_block.
        assert env.On([block0, top_block]).holds(state)
        assert env.GripperOpen([robot]).holds(state)


def test_pybullet_blocks_putontable():
    """Tests cases for putting blocks on the table in PyBulletBlocksEnv."""
    utils.reset_config({"env": "pybullet_blocks", "pybullet_use_gui": _GUI_ON})
    env = _get__EXPOSED_PYBULLET_ENV()
    env.seed(123)
    block = Object("block0", env.block_type)
    robot = env.robot
    bx = (env.x_lb + env.x_ub) / 2
    by = (env.y_lb + env.y_ub) / 2
    bz = env.table_height + 0.5 * env.block_size
    rx, ry, rz = env.robot_init_x, env.robot_init_y, env.robot_init_z
    rf = env.open_fingers
    # Create a simple custom state with one block for testing.
    init_state = State({
        robot: np.array([rx, ry, rz, rf]),
        block: np.array([bx, by, bz, 0.0]),
    })
    env.set_state(init_state)
    assert env.get_state().allclose(init_state)
    # Pick block to get to a state where we are prepared to place.
    state = env.execute_pick(block)
    # Create a PutOnTable option.
    # The params space is relative, so this should put the block at the center
    # of the work space.
    option = env.PutOnTable.ground([robot], [0.5, 0.5])
    assert option.initiable(state)
    assert not env.GripperOpen([robot]).holds(state)
    # Execute the option.
    while True:
        if option.terminal(state):
            break
        action = option.policy(state)
        state = env.step(action)
    # The block should now NOT be held.
    assert state.get(block, "held") == 0.0
    # And block should be on the table.
    assert env.OnTable([block]).holds(state)
    assert env.GripperOpen([robot]).holds(state)
    # Specifically, it should be at the center of the workspace.
    # Note: setting this threshold to 1e-3 causes the check to fail.
    # If this is not precise enough in practice, we will need to revisit
    # and try to improve the PutOnTable controller.
    assert abs(state.get(block, "pose_x") - (env.x_lb + env.x_ub) / 2.) < 1e-2
    assert abs(state.get(block, "pose_y") - (env.y_lb + env.y_ub) / 2.) < 1e-2
    # Test that the block can be placed at the extremes of the workspace.
    corners = [
        (env.x_lb, env.y_lb),
        (env.x_ub, env.y_lb),
        (env.x_lb, env.y_ub),
        (env.x_ub, env.y_ub),
    ]
    corner_params = [(0., 0.), (1., 0.), (0., 1.), (1., 1.)]
    for (bx, by), (px, py) in zip(corners, corner_params):
        state = init_state.copy()
        # Pick block to get to a state where we are prepared to place.
        state = env.execute_pick(block)
        # Create a PutOnTable option.
        option = env.PutOnTable.ground([robot], [px, py])
        assert option.initiable(state)
        assert not env.GripperOpen([robot]).holds(state)
        # Execute the option.
        while True:
            if option.terminal(state):
                break
            action = option.policy(state)
            state = env.step(action)
        # The block should now NOT be held.
        assert state.get(block, "held") == 0.0
        # And block should be on the table.
        assert env.OnTable([block]).holds(state)
        assert env.GripperOpen([robot]).holds(state)
        # Specifically, it should be at the given corner of the workspace.
        # Note: setting this threshold to 1e-3 causes the check to fail.
        # If this is not precise enough in practice, we will need to revisit
        # and try to improve the PutOnTable controller.
        assert abs(state.get(block, "pose_x") - bx) < 1e-2
        assert abs(state.get(block, "pose_y") - by) < 1e-2


def test_pybullet_blocks_close_pick_place():
    """Test a tricky case where we attempt to pick and place immediately next
    to a pile of blocks.

    Make sure that the pile is not disturbed.
    """
    utils.reset_config({"env": "pybullet_blocks", "pybullet_use_gui": _GUI_ON})
    env = _get__EXPOSED_PYBULLET_ENV()
    env.seed(123)
    block = Object("block0", env.block_type)
    robot = env.robot
    bx = (env.x_lb + env.x_ub) / 2
    by = (env.y_lb + env.y_ub) / 2
    # Start the block out on the left side of the pile.
    by0 = by + env.collision_padding * env.block_size
    bz0 = env.table_height + 0.5 * env.block_size
    rx, ry, rz = env.robot_init_x, env.robot_init_y, env.robot_init_z
    rf = env.open_fingers
    max_num_blocks = max(max(CFG.blocks_num_blocks_train),
                         max(CFG.blocks_num_blocks_test))
    block_to_z = {
        Object(f"block{i+1}", env.block_type): bz0 + i * env.block_size
        for i in range(max_num_blocks - 1)
    }
    state = State({
        robot: np.array([rx, ry, rz, rf]),
        block: np.array([bx, by0, bz0, 0.0]),
        **{b: np.array([bx, by, bz, 0.0])
           for b, bz in block_to_z.items()}
    })
    env.set_state(state)
    assert env.get_state().allclose(state)
    initial_pile_state = State({b: state[b] for b in block_to_z})
    state = env.execute_pick(block)
    # The main block should now be held.
    assert state.get(block, "held") == 1.0
    # The other block states should be the same.
    pile_state = State({b: state[b] for b in block_to_z})
    assert initial_pile_state.allclose(pile_state)
    # Now place on the other side.
    by0 = by - env.collision_padding * env.block_size
    # Normalize to get a parameter for PutOnTable.
    py = (by0 - (env.y_lb + env.block_size / 2)) / (env.y_ub - env.y_lb -
                                                    env.block_size)
    option = env.PutOnTable.ground([robot], [0.5, py])
    assert option.initiable(state)
    assert not env.GripperOpen([robot]).holds(state)
    # Execute the option.
    while True:
        if option.terminal(state):
            break
        action = option.policy(state)
        state = env.step(action)
    # The block should now NOT be held.
    assert state.get(block, "held") == 0.0
    # And block should be on the table.
    assert env.OnTable([block]).holds(state)
    assert env.GripperOpen([robot]).holds(state)
    assert abs(state.get(block, "pose_x") - bx) < 1e-2
    assert abs(state.get(block, "pose_y") - by0) < 1e-2
    # The other block states should be the same.
    pile_state = State({b: state[b] for b in block_to_z})
    assert initial_pile_state.allclose(pile_state)


def test_pybullet_blocks_abstract_states():
    """Tests abstract states during option execution in PyBulletBlocksEnv."""
    utils.reset_config({"env": "pybullet_blocks", "pybullet_use_gui": _GUI_ON})
    env = _get__EXPOSED_PYBULLET_ENV()
    On = env.On
    OnTable = env.OnTable
    GripperOpen = env.GripperOpen
    Holding = env.Holding
    Clear = env.Clear
    env.seed(123)
    block0 = Object("block0", env.block_type)
    block1 = Object("block1", env.block_type)
    robot = env.robot
    bx0 = (env.x_lb + env.x_ub) / 2
    by0 = (env.y_lb + env.y_ub) / 2 - env.block_size
    by1 = (env.y_lb + env.y_ub) / 2 + env.block_size
    bz0 = env.table_height + 0.5 * env.block_size
    rx, ry, rz = env.robot_init_x, env.robot_init_y, env.robot_init_z
    rf = env.open_fingers
    # Create a state with two blocks on the table.
    state = State({
        robot: np.array([rx, ry, rz, rf]),
        block0: np.array([bx0, by0, bz0, 0.0]),
        block1: np.array([bx0, by1, bz0, 0.0]),
    })
    env.set_state(state)
    assert env.get_state().allclose(state)
    expected_abstract_state = {
        OnTable([block0]),
        OnTable([block1]),
        GripperOpen([robot]),
        Clear([block0]),
        Clear([block1]),
    }
    assert utils.abstract(state, env.predicates) == expected_abstract_state
    # Execute a pick of block0.
    state = env.execute_pick(block0)
    expected_abstract_state = {
        Holding([block0]),
        OnTable([block1]),
        Clear([block1]),
    }
    assert utils.abstract(state, env.predicates) == expected_abstract_state
    # Stack block0 on block1.
    state = env.execute_stack(block1)
    expected_abstract_state = {
        On([block0, block1]),
        OnTable([block1]),
        Clear([block0]),
        GripperOpen([robot]),
    }
    assert utils.abstract(state, env.predicates) == expected_abstract_state
    # Unstack block0.
    state = env.execute_pick(block0)
    expected_abstract_state = {
        Holding([block0]),
        OnTable([block1]),
        Clear([block1]),
    }
    assert utils.abstract(state, env.predicates) == expected_abstract_state
    # Put block0 on the table.
    state = env.execute_putontable(0.5, 0.9)
    expected_abstract_state = {
        OnTable([block0]),
        OnTable([block1]),
        GripperOpen([robot]),
        Clear([block0]),
        Clear([block1]),
    }
    assert utils.abstract(state, env.predicates) == expected_abstract_state
