"""Test cases for PyBulletBlocksEnv."""

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.settings import CFG
from predicators.structs import Object, ParameterizedOption, State

_GUI_ON = False  # toggle for debugging


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

    def set_state(self, state):
        """Forcibly reset the state.

        Note that for convenience, we do not expect that the simulator
        state will be present in the input state (because that simulator
        state needs to include the robot joint states). For even further
        convenience, we assume that the state that we are setting to has
        the robot in the initial position, so we can just use the
        robot's initial joint values rather than rerunning inverse
        kinematics here.
        """
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


@pytest.fixture(scope="module", name="env", params=("fetch", "panda"))
def _create_exposed_pybullet_blocks_env(request):
    """Only create once and share among all tests, for efficiency."""
    utils.reset_config({
        "env": "pybullet_blocks",
        "pybullet_use_gui": _GUI_ON,
        # We run this test using the RESET control mode.
        "pybullet_control_mode": "reset",
        # Which robot we're using
        "pybullet_robot": request.param,
    })
    return _ExposedPyBulletBlocksEnv()


def _get_predicates_by_names(env, names):
    # Note that we do this, rather than use the function of the same name
    # defined in ground_truth_nsrts, because that function calls
    # get_or_create_env. We want to use the _ExposedPyBulletBlocksEnv only,
    # not create a new instance of the PyBulletEnv. This is especially
    # important when _GUI_ON = True.
    name_to_pred = {p.name: p for p in env.predicates}
    return [name_to_pred[n] for n in names]


def test_pybullet_blocks_reset(env):
    """Tests for PyBulletBlocksEnv.reset()."""
    for idx, task in enumerate(env.get_train_tasks()):
        assert isinstance(task.init, utils.PyBulletState)
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
    # Simulate and render state should be not implemented.
    action = env.action_space.sample()
    with pytest.raises(NotImplementedError):
        env.simulate(state, action)
    task = env.get_train_tasks()[0]
    with pytest.raises(NotImplementedError):
        env.render_state(state, task, action)
    with pytest.raises(NotImplementedError) as e:
        env.render_state_plt(state, task, action)
    assert "This env does not use Matplotlib" in str(e)


def test_pybullet_blocks_picking(env):
    """Tests for picking blocks in PyBulletBlocksEnv."""
    block = Object("block0", env.block_type)
    robot = env.robot
    bx = (env.x_lb + env.x_ub) / 2
    by = (env.y_lb + env.y_ub) / 2
    bz = env.table_height + 0.5 * env.block_size
    rx, ry, rz = env.robot_init_x, env.robot_init_y, env.robot_init_z
    rf = 1.0
    # Create a simple custom state with one block for testing.
    init_state = State({
        robot: np.array([rx, ry, rz, rf]),
        block: np.array([bx, by, bz, 0.0, 1.0, 0.0, 0.0]),
    })
    env.set_state(init_state)
    recovered_state = env.get_state()
    assert recovered_state.allclose(init_state)
    # Create an option for picking the block.
    option = env.Pick.ground([robot, block], [])
    # Execute the option.
    state = env.execute_option(option)
    # The block should now be held.
    assert state.get(block, "held") == 1.0
    assert state.get(robot, "fingers") == 0.0


def test_pybullet_blocks_picking_corners(env):
    """Test that the block can be picked at the extremes of the workspace."""
    block = Object("block0", env.block_type)
    robot = env.robot
    bx = (env.x_lb + env.x_ub) / 2
    by = (env.y_lb + env.y_ub) / 2
    bz = env.table_height + 0.5 * env.block_size
    rx, ry, rz = env.robot_init_x, env.robot_init_y, env.robot_init_z
    rf = 1.0
    # Create a simple custom state with one block for testing.
    init_state = State({
        robot: np.array([rx, ry, rz, rf]),
        block: np.array([bx, by, bz, 0.0, 1.0, 0.0, 0.0]),
    })
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
        recovered_state = env.get_state()
        assert recovered_state.allclose(state)
        # Create an option for picking the block.
        option = env.Pick.ground([robot, block], [])
        # Execute the option.
        state = env.execute_option(option)
        # The block should now be held.
        assert state.get(block, "held") == 1.0
        assert state.get(robot, "fingers") == 0.0


def test_pybullet_blocks_stacking(env):
    """Tests for stacking blocks in PyBulletBlocksEnv."""
    block0 = Object("block0", env.block_type)
    block1 = Object("block1", env.block_type)
    robot = env.robot
    bx0 = (env.x_lb + env.x_ub) / 2
    by0 = (env.y_lb + env.y_ub) / 2 - env.block_size
    by1 = (env.y_lb + env.y_ub) / 2 + env.block_size
    bz0 = env.table_height + 0.5 * env.block_size
    rx, ry, rz = env.robot_init_x, env.robot_init_y, env.robot_init_z
    rf = 1.0
    # Create a state with two blocks.
    init_state = State({
        robot: np.array([rx, ry, rz, rf]),
        block0: np.array([bx0, by0, bz0, 0.0, 1.0, 0.0, 0.0]),
        block1: np.array([bx0, by1, bz0, 0.0, 0.0, 1.0, 0.0]),
    })
    env.set_state(init_state)
    assert env.get_state().allclose(init_state)
    # Pick block0 to get to a state where we are prepared to stack.
    option = env.Pick.ground([robot, block0], [])
    state = env.execute_option(option)
    # Create a stack option.
    option = env.Stack.ground([robot, block1], [])
    # Execute the stack option.
    state = env.execute_option(option)
    # The block should now NOT be held.
    assert state.get(block0, "held") == 0.0
    assert state.get(robot, "fingers") == 1.0
    # And block0 should be on block1.
    On, = _get_predicates_by_names(env, ["On"])
    assert On([block0, block1]).holds(state)


def test_pybullet_blocks_stacking_corners(env):
    """Test stacking a block on the tallest possible tower at each of the
    possible corners."""
    On, = _get_predicates_by_names(env, ["On"])
    corners = [
        (env.x_lb, env.y_lb),
        (env.x_ub, env.y_lb),
        (env.x_lb, env.y_ub),
        (env.x_ub, env.y_ub),
    ]
    block0 = Object("block0", env.block_type)
    robot = env.robot
    bx0 = (env.x_lb + env.x_ub) / 2
    by0 = (env.y_lb + env.y_ub) / 2 - env.block_size
    bz0 = env.table_height + 0.5 * env.block_size
    rx, ry, rz = env.robot_init_x, env.robot_init_y, env.robot_init_z
    rf = 1.0
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
            block0: np.array([bx0, by0, bz0, 0.0, 1.0, 0.0, 0.0]),
            **{
                b: np.array([bx, by, bz, 0.0, 0.0, 1.0, 0.0])
                for b, bz in block_to_z.items()
            }
        })
        env.set_state(state)
        assert env.get_state().allclose(state)
        # Pick block0 to get to a state where we are prepared to stack.
        option = env.Pick.ground([robot, block0], [])
        state = env.execute_option(option)
        # Create a stack option.
        option = env.Stack.ground([robot, top_block], [])
        state = env.execute_option(option)
        # The block should now NOT be held.
        assert state.get(block0, "held") == 0.0
        assert state.get(robot, "fingers") == 1.0
        # And block0 should be on top_block.
        assert On([block0, top_block]).holds(state)


def test_pybullet_blocks_putontable(env):
    """Tests for putting blocks on the table in PyBulletBlocksEnv."""
    OnTable, = _get_predicates_by_names(env, ["OnTable"])
    block = Object("block0", env.block_type)
    robot = env.robot
    bx = (env.x_lb + env.x_ub) / 2
    by = (env.y_lb + env.y_ub) / 2
    bz = env.table_height + 0.5 * env.block_size
    rx, ry, rz = env.robot_init_x, env.robot_init_y, env.robot_init_z
    rf = 1.0
    # Create a simple custom state with one block for testing.
    init_state = State({
        robot: np.array([rx, ry, rz, rf]),
        block: np.array([bx, by, bz, 0.0, 1.0, 0.0, 0.0]),
    })
    env.set_state(init_state)
    assert env.get_state().allclose(init_state)
    # Pick block to get to a state where we are prepared to place.
    option = env.Pick.ground([robot, block], [])
    state = env.execute_option(option)
    # Create a PutOnTable option.
    # The params space is relative, so this should put the block at the center
    # of the workspace.
    option = env.PutOnTable.ground([robot], [0.5, 0.5])
    state = env.execute_option(option)
    # The block should now NOT be held.
    assert state.get(block, "held") == 0.0
    assert state.get(robot, "fingers") == 1.0
    # And block should be on the table.
    assert OnTable([block]).holds(state)
    # Specifically, it should be at the center of the workspace.
    assert abs(state.get(block, "pose_x") - (env.x_lb + env.x_ub) / 2.) < 1e-3
    assert abs(state.get(block, "pose_y") - (env.y_lb + env.y_ub) / 2.) < 1e-3
    # Test that when we attempt to put the block outside the workspace, an
    # OptionExecutionFailure is raised. This is for the panda only because
    # the fetch uses pybullet IK, which is not even smart enough to realize its
    # own ineptitude.
    if CFG.pybullet_robot == "panda":
        env.set_state(init_state)
        option = env.Pick.ground([robot, block], [])
        state = env.execute_option(option)
        # Make a copy of the PutOnTable option with larger params space because
        # the other option's param space doesn't allow out-of-bounds puts.
        PutOnTable_unrestricted = ParameterizedOption(
            "PutOnTable_unrestricted",
            types=env.PutOnTable.types,
            params_space=Box(-np.inf, np.inf, (2, ), dtype=np.float32),
            policy=env.PutOnTable.policy,
            initiable=env.PutOnTable.initiable,
            terminal=env.PutOnTable.terminal)
        option = PutOnTable_unrestricted.ground([robot], [25.0, 25.0])
        with pytest.raises(utils.OptionExecutionFailure) as e:
            state = env.execute_option(option)
        assert "Inverse kinematics failed" in str(e)


def test_pybullet_blocks_putontable_corners(env):
    """Test that the block can be placed at the extremes of the workspace."""
    OnTable, = _get_predicates_by_names(env, ["OnTable"])
    block = Object("block0", env.block_type)
    robot = env.robot
    bx = (env.x_lb + env.x_ub) / 2
    by = (env.y_lb + env.y_ub) / 2
    bz = env.table_height + 0.5 * env.block_size
    rx, ry, rz = env.robot_init_x, env.robot_init_y, env.robot_init_z
    rf = 1.0
    # Create a simple custom state with one block for testing.
    init_state = State({
        robot: np.array([rx, ry, rz, rf]),
        block: np.array([bx, by, bz, 0.0, 1.0, 0.0, 0.0]),
    })
    corners = [
        (env.x_lb, env.y_lb),
        (env.x_ub, env.y_lb),
        (env.x_lb, env.y_ub),
        (env.x_ub, env.y_ub),
    ]
    corner_params = [(0., 0.), (1., 0.), (0., 1.), (1., 1.)]
    for (bx, by), (px, py) in zip(corners, corner_params):
        state = init_state.copy()
        env.set_state(state)
        assert env.get_state().allclose(state)
        # Pick block to get to a state where we are prepared to place.
        option = env.Pick.ground([robot, block], [])
        state = env.execute_option(option)
        # Create a PutOnTable option.
        option = env.PutOnTable.ground([robot], [px, py])
        state = env.execute_option(option)
        # The block should now NOT be held.
        assert state.get(block, "held") == 0.0
        assert state.get(robot, "fingers") == 1.0
        # And block should be on the table.
        assert OnTable([block]).holds(state)
        # Specifically, it should be at the given corner of the workspace.
        # Note: setting this threshold to 1e-3 causes the check to fail.
        # If this is not precise enough in practice, we will need to revisit
        # and try to improve the PutOnTable controller.
        assert abs(state.get(block, "pose_x") - bx) < 1e-2
        assert abs(state.get(block, "pose_y") - by) < 1e-2


def test_pybullet_blocks_close_pick_place(env):
    """Test a tricky case where we attempt to pick and place immediately next
    to a pile of blocks.

    Make sure that the pile is not disturbed.
    """
    OnTable, = _get_predicates_by_names(env, ["OnTable"])
    block = Object("block0", env.block_type)
    robot = env.robot
    bx = (env.x_lb + env.x_ub) / 2
    by = (env.y_lb + env.y_ub) / 2
    # Start the block out on the left side of the pile.
    by0 = by + env.collision_padding * env.block_size
    bz0 = env.table_height + 0.5 * env.block_size
    rx, ry, rz = env.robot_init_x, env.robot_init_y, env.robot_init_z
    rf = 1.0
    max_num_blocks = max(max(CFG.blocks_num_blocks_train),
                         max(CFG.blocks_num_blocks_test))
    block_to_z = {
        Object(f"block{i+1}", env.block_type): bz0 + i * env.block_size
        for i in range(max_num_blocks - 1)
    }
    state = State({
        robot: np.array([rx, ry, rz, rf]),
        block: np.array([bx, by0, bz0, 0.0, 1.0, 0.0, 0.0]),
        **{
            b: np.array([bx, by, bz, 0.0, 0.0, 1.0, 0.0])
            for b, bz in block_to_z.items()
        }
    })
    env.set_state(state)
    assert env.get_state().allclose(state)
    initial_pile_state = State({b: state[b] for b in block_to_z})
    option = env.Pick.ground([robot, block], [])
    state = env.execute_option(option)
    # The main block should now be held.
    assert state.get(block, "held") == 1.0
    assert state.get(robot, "fingers") == 0.0
    # The other block states should be the same.
    pile_state = State({b: state[b] for b in block_to_z})
    assert initial_pile_state.allclose(pile_state)
    # Now place on the other side.
    by0 = by - env.collision_padding * env.block_size
    # Normalize to get a parameter for PutOnTable.
    py = (by0 - (env.y_lb + env.block_size / 2)) / (env.y_ub - env.y_lb -
                                                    env.block_size)
    option = env.PutOnTable.ground([robot], [0.5, py])
    state = env.execute_option(option)
    # The block should now NOT be held.
    assert state.get(block, "held") == 0.0
    assert state.get(robot, "fingers") == 1.0
    # And block should be on the table.
    assert OnTable([block]).holds(state)
    assert abs(state.get(block, "pose_x") - bx) < 1e-2
    assert abs(state.get(block, "pose_y") - by0) < 1e-2
    # The other block states should be the same.
    pile_state = State({b: state[b] for b in block_to_z})
    assert initial_pile_state.allclose(pile_state)


def test_pybullet_blocks_abstract_states(env):
    """Tests abstract states during option execution in PyBulletBlocksEnv."""
    On, OnTable, GripperOpen, Holding, Clear = _get_predicates_by_names(
        env, ["On", "OnTable", "GripperOpen", "Holding", "Clear"])
    block0 = Object("block0", env.block_type)
    block1 = Object("block1", env.block_type)
    robot = env.robot
    bx0 = (env.x_lb + env.x_ub) / 2
    by0 = (env.y_lb + env.y_ub) / 2 - env.block_size
    by1 = (env.y_lb + env.y_ub) / 2 + env.block_size
    bz0 = env.table_height + 0.5 * env.block_size
    rx, ry, rz = env.robot_init_x, env.robot_init_y, env.robot_init_z
    rf = 1.0
    # Create a state with two blocks on the table.
    state = State({
        robot: np.array([rx, ry, rz, rf]),
        block0: np.array([bx0, by0, bz0, 0.0, 1.0, 0.0, 0.0]),
        block1: np.array([bx0, by1, bz0, 0.0, 0.0, 1.0, 0.0]),
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
    option = env.Pick.ground([robot, block0], [])
    state = env.execute_option(option)
    expected_abstract_state = {
        Holding([block0]),
        OnTable([block1]),
        Clear([block1]),
    }
    assert utils.abstract(state, env.predicates) == expected_abstract_state
    # Stack block0 on block1.
    option = env.Stack.ground([robot, block1], [])
    state = env.execute_option(option)
    expected_abstract_state = {
        On([block0, block1]),
        OnTable([block1]),
        Clear([block0]),
        GripperOpen([robot]),
    }
    assert utils.abstract(state, env.predicates) == expected_abstract_state
    # Unstack block0.
    option = env.Pick.ground([robot, block0], [])
    state = env.execute_option(option)
    expected_abstract_state = {
        Holding([block0]),
        OnTable([block1]),
        Clear([block1]),
    }
    assert utils.abstract(state, env.predicates) == expected_abstract_state
    # Put block0 on the table.
    option = env.PutOnTable.ground([robot], [0.5, 0.9])
    state = env.execute_option(option)
    expected_abstract_state = {
        OnTable([block0]),
        OnTable([block1]),
        GripperOpen([robot]),
        Clear([block0]),
        Clear([block1]),
    }
    assert utils.abstract(state, env.predicates) == expected_abstract_state
