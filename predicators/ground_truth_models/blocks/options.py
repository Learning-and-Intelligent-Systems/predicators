"""Ground-truth options for the (non-pybullet) blocks environment."""

from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.blocks import BlocksEnv
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, create_move_end_effector_to_pose_option
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class BlocksGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the (non-pybullet) blocks environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"blocks", "blocks_clear"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        block_type = types["block"]
        block_size = CFG.blocks_block_size

        Holding = predicates['Holding']
        On = predicates['On']
        OnTable = predicates['OnTable']

        # def _Pick_terminal(s: State, m: Dict, o: Sequence[Object],
        #                    p: Array) -> bool:
        #     robot, block = o
        #     return Holding.holds(s, [block])

        Pick = utils.SingletonParameterizedOption(
            # variables: [robot, object to pick]
            # params: []
            "Pick",
            cls._create_pick_policy(action_space),
            types=[robot_type, block_type],
            # terminal=_Pick_terminal,
        )

        # Probably will need to change the option parameters for this to work
        # def _Stack_terminal(s: State, m: Dict, o: Sequence[Object],
        #                     p: Array) -> bool:
        #     block, otherblock, _ = o
        #     return On.holds(s, [block, otherblock])

        Stack = utils.SingletonParameterizedOption(
            # variables: [robot, object on which to stack currently-held-object]
            # params: []
            "Stack",
            cls._create_stack_policy(action_space, block_size),
            types=[robot_type, block_type],
            # types = [block_type, block_type, robot_type],
            # terminal=_Stack_terminal,
        )

        # def _PutOnTable_terminal(s: State, m: Dict, o: Sequence[Object],
        #                          p: Array) -> bool:
        #     block, _ = o
        #     return OnTable.holds(s, [block])

        PutOnTable = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: [x, y] (normalized coordinates on the table surface)
            "PutOnTable",
            cls._create_putontable_policy(action_space, block_size),
            types=[robot_type],
            # types=[block_type, robot_type],
            params_space=Box(0, 1, (2, )),
            # terminal=_PutOnTable_terminal,
        )

        return {Pick, Stack, PutOnTable}

    @classmethod
    def _create_pick_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, block = objects
            block_pose = np.array([
                state.get(block, "pose_x"),
                state.get(block, "pose_y"),
                state.get(block, "pose_z")
            ])
            arr = np.r_[block_pose, 0.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_stack_policy(cls, action_space: Box,
                             block_size: float) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, block = objects
            # _, block, _ = objects

            block_pose = np.array([
                state.get(block, "pose_x"),
                state.get(block, "pose_y"),
                state.get(block, "pose_z")
            ])
            relative_grasp = np.array([
                0.,
                0.,
                block_size,
            ])
            arr = np.r_[block_pose + relative_grasp, 1.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_putontable_policy(cls, action_space: Box,
                                  block_size: float) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects  # unused
            # De-normalize parameters to actual table coordinates.
            x_norm, y_norm = params
            x = BlocksEnv.x_lb + (BlocksEnv.x_ub - BlocksEnv.x_lb) * x_norm
            y = BlocksEnv.y_lb + (BlocksEnv.y_ub - BlocksEnv.y_lb) * y_norm
            z = BlocksEnv.table_height + 0.5 * block_size
            arr = np.array([x, y, z, 1.0], dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy


class PyBulletBlocksGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the pybullet_blocks environment."""

    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _offset_z: ClassVar[float] = 0.01

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_blocks"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        _, pybullet_robot, _ = \
            PyBulletBlocksEnv.initialize_pybullet(using_gui=False)

        robot_type = types["robot"]
        block_type = types["block"]
        block_size = CFG.blocks_block_size

        def get_current_fingers(state: State) -> float:
            robot, = state.get_objects(robot_type)
            return PyBulletBlocksEnv.fingers_state_to_joint(
                pybullet_robot, state.get(robot, "fingers"))

        def open_fingers_func(state: State, objects: Sequence[Object],
                              params: Array) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = pybullet_robot.open_fingers
            return current, target

        def close_fingers_func(state: State, objects: Sequence[Object],
                               params: Array) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = pybullet_robot.closed_fingers
            return current, target

        # Pick
        option_types = [robot_type, block_type]
        params_space = Box(0, 1, (0, ))
        Pick = utils.LinearChainParameterizedOption(
            "Pick",
            [
                # Move to far above the block which we will grasp.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToPreGrasp",
                    z_func=lambda _: PyBulletBlocksEnv.pick_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBlocksEnv.grasp_tol),
                # Move down to grasp.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToGrasp",
                    z_func=lambda block_z: (block_z + cls._offset_z),
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Close fingers.
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_types, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBlocksEnv.grasp_tol),
                # Move back up.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: PyBulletBlocksEnv.pick_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
            ],
            "Pick up block ?block")

        # Stack
        option_types = [robot_type, block_type]
        params_space = Box(0, 1, (0, ))
        Stack = utils.LinearChainParameterizedOption(
            "Stack",
            [
                # Move to above the block on which we will stack.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToPreStack",
                    z_func=lambda _: PyBulletBlocksEnv.pick_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Move down to place.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToStack",
                    z_func=lambda block_z:
                    (block_z + block_size + cls._offset_z),
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBlocksEnv.grasp_tol),
                # Move back up.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: PyBulletBlocksEnv.pick_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
            ],
            annotation="Stack the block in hand onto block ?otherblock")

        # PutOnTable
        option_types = [robot_type]
        params_space = Box(0, 1, (2, ))
        place_z = PyBulletBlocksEnv.table_height + \
            block_size / 2 + cls._offset_z
        PutOnTable = utils.LinearChainParameterizedOption(
            "PutOnTable",
            [
                # Move to above the table at the (x, y) where we will place.
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorToPrePutOnTable",
                    z=PyBulletBlocksEnv.pick_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Move down to place.
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorToPutOnTable",
                    z=place_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBlocksEnv.grasp_tol),
                # Move back up.
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorBackUp",
                    z=PyBulletBlocksEnv.pick_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
            ],
            annotation="Put block on table")

        return {Pick, Stack, PutOnTable}

    @classmethod
    def _create_blocks_move_to_above_block_option(
            cls, name: str, z_func: Callable[[float],
                                             float], finger_status: str,
            pybullet_robot: SingleArmPyBulletRobot, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        block argument.

        The parameter z_func maps the block's z position to the target z
        position.
        """
        home_orn = PyBulletBlocksEnv.get_robot_ee_home_orn()

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not params
            robot, block = objects
            current_position = (state.get(robot, "pose_x"),
                                state.get(robot, "pose_y"),
                                state.get(robot, "pose_z"))
            current_pose = Pose(current_position, home_orn)
            target_position = (state.get(block,
                                         "pose_x"), state.get(block, "pose_y"),
                               z_func(state.get(block, "pose_z")))
            target_pose = Pose(target_position, home_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude)

    @classmethod
    def _create_blocks_move_to_above_table_option(
            cls, name: str, z: float, finger_status: str,
            pybullet_robot: SingleArmPyBulletRobot, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        table.

        The z position of the target pose must be provided.
        """
        home_orn = PyBulletBlocksEnv.get_robot_ee_home_orn()

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            robot, = objects
            current_position = (state.get(robot, "pose_x"),
                                state.get(robot, "pose_y"),
                                state.get(robot, "pose_z"))
            current_pose = Pose(current_position, home_orn)
            # De-normalize parameters to actual table coordinates.
            x_norm, y_norm = params
            target_position = (
                PyBulletBlocksEnv.x_lb +
                (PyBulletBlocksEnv.x_ub - PyBulletBlocksEnv.x_lb) * x_norm,
                PyBulletBlocksEnv.y_lb +
                (PyBulletBlocksEnv.y_ub - PyBulletBlocksEnv.y_lb) * y_norm, z)
            target_pose = Pose(target_position, home_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude)
