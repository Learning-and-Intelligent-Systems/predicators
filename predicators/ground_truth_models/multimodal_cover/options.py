"""Ground-truth options for the (non-pybullet) blocks environment."""
import logging
from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.multimodal_cover import MultiModalCoverEnv
from predicators.envs.pybullet_multimodal_cover import PyBulletMultiModalCoverEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, MotionPlanController, create_move_end_effector_to_pose_option
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class MultiModalCoverGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the (non-pybullet) block-stack environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"multimodal_cover"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        robot_type = types["robot"]
        block_type = types["block"]
        block_size = CFG.blocks_block_size
        table_height = PyBulletMultiModalCoverEnv.table_height

        Pick = utils.SingletonParameterizedOption(
            # variables: [robot, object to pick]
            "Pick",
            cls._create_pick_policy(action_space),
            types=[robot_type, block_type])

        PutOnTable = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: [x, y] (normalized coordinates on the table surface)
            "PutOnTable",
            cls._create_putontable_policy(action_space, block_size),
            types=[robot_type],
            params_space=Box(0, 1, (2,)))

        logging.info(f"Pick pspace: {Pick.params_space}")
        logging.info(f"PutOnTable pspace: {PutOnTable.params_space}")

        return {Pick, PutOnTable}

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
    def _create_putontable_policy(cls, action_space: Box,
                                  block_size: float) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects  # unused
            # De-normalize parameters to actual table coordinates.
            x_norm, y_norm = params
            x = MultiModalCoverEnv.x_lb + (MultiModalCoverEnv.x_ub - MultiModalCoverEnv.x_lb) * x_norm
            y = MultiModalCoverEnv.y_lb + (MultiModalCoverEnv.y_ub - MultiModalCoverEnv.y_lb) * y_norm
            z = MultiModalCoverEnv.table_height + 0.5 * block_size

            logging.info(f"policy out: {(x,y,z)}")
            file_path = "output_dist.txt"

            # Open the file in append mode
            with open(file_path, "a") as file:
                # Iterate through the array and write each element to the file
                file.write(f"({x}, {y})\n")  # Each item on a new line

            arr = np.array([x, y, z, 1.0], dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)

            return Action(arr)

        return policy


class PyBulletMultiModalCoverGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the pybullet_block_stack environment."""

    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _offset_z: ClassVar[float] = 0.01

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_multimodal_cover"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        client_id, pybullet_robot, bodies = \
            PyBulletMultiModalCoverEnv.initialize_pybullet(using_gui=False)

        robot_type = types["robot"]
        block_type = types["block"]
        block_size = CFG.blocks_block_size
        table_height = PyBulletMultiModalCoverEnv.table_height

        def get_current_fingers(state: State) -> float:
            robot, = state.get_objects(robot_type)
            return PyBulletMultiModalCoverEnv.fingers_state_to_joint(
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
        params_space = Box(0, 1, (0,))
        Pick = utils.LinearChainParameterizedOption(
            "Pick",
            [
                # Move to far above the block which we will grasp.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToPreGrasp",
                    z_func=lambda _: PyBulletMultiModalCoverEnv.pick_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletMultiModalCoverEnv.grasp_tol),
                # Move down to grasp.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToPreGraspFinal",
                    z_func=lambda block_z: (table_height + block_size*1.5),
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToGraspFinal",
                    z_func=lambda block_z: (block_z + cls._offset_z),
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
                # Close fingers.
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_types, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletMultiModalCoverEnv.grasp_tol),
                # Move back up.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorBackUpPick",
                    z_func=lambda _: PyBulletMultiModalCoverEnv.pick_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
            ])

        # PutOnTable
        option_types = [robot_type]
        params_space = Box(0, 1, (2,))
        place_z = PyBulletMultiModalCoverEnv.table_height + \
                  block_size / 2 + cls._offset_z
        PutOnTable = utils.LinearChainParameterizedOption(
            "PutOnTable",
            [
                # Move to above the table at the (x, y) where we will place.
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorToPrePutOnTable",
                    z=PyBulletMultiModalCoverEnv.pick_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
                # Move down to place.
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorToPutOnTable",
                    z=place_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletMultiModalCoverEnv.grasp_tol),
                # Move back up.
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorBackUp",
                    z=PyBulletMultiModalCoverEnv.pick_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
            ])

        return {Pick, PutOnTable}


    @classmethod
    def _create_blocks_move_to_above_block_option(
            cls, name: str, z_func: Callable[[float],
            float], finger_status: str,
            pybullet_robot: SingleArmPyBulletRobot, option_types: List[Type],
            params_space: Box,
            physics_client_id,
            bodies
    ) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        block argument.

        The parameter z_func maps the block's z position to the target z
        position.
        """
        home_orn = PyBulletMultiModalCoverEnv.get_robot_ee_home_orn()
        motion_planner = MotionPlanController(bodies, physics_client_id)

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

        return motion_planner.create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude, physics_client_id, bodies)


    @classmethod
    def _create_blocks_move_to_above_table_option(
            cls, name: str, z: float, finger_status: str,
            pybullet_robot: SingleArmPyBulletRobot, option_types: List[Type],
            params_space: Box,
            physics_client_id,
            bodies
    ) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        table.

        The z position of the target pose must be provided.
        """
        home_orn = PyBulletMultiModalCoverEnv.get_robot_ee_home_orn()
        motion_planner = MotionPlanController(bodies, physics_client_id)

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
                PyBulletMultiModalCoverEnv.x_lb +
                (PyBulletMultiModalCoverEnv.x_ub - PyBulletMultiModalCoverEnv.x_lb) * x_norm,
                PyBulletMultiModalCoverEnv.y_lb +
                (PyBulletMultiModalCoverEnv.y_ub - PyBulletMultiModalCoverEnv.y_lb) * y_norm, z)
            target_pose = Pose(target_position, home_orn)

            return current_pose, target_pose, finger_status

        return motion_planner.create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude, physics_client_id, bodies)
