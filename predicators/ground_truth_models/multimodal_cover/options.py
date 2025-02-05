"""Ground-truth options for the (non-pybullet) blocks environment."""
import logging
from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
from gym.spaces import Box
from lisdf.utils.transformations import euler_from_quaternion, quaternion_from_matrix, quaternion_from_euler

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
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion, matrix_from_quat, multiply_poses
from scipy.spatial.transform import Rotation as R


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
        gripper_max_depth = 0.03
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
            cls._create_putontable_policy(action_space, block_type, gripper_max_depth = 0.03),
            types=[robot_type, block_type],
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
    def _create_putontable_policy(cls, action_space: Box, block_type, gripper_max_depth) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, objects  # unused

            held_block = None

            for block in state:
                if not block.is_instance(block_type):
                    continue
                if state.get(block, "held") >= MultiModalCoverEnv.held_tol:
                    held_block = block

            block_dim = (state.get(held_block, "depth"),
                         state.get(held_block, "width"),
                         state.get(held_block, "height"))

            if CFG.multi_modal_cover_real_robot:
                block_orn = (state.get(held_block, "qx"),
                             state.get(held_block, "qy"),
                             state.get(held_block, "qz"),
                             state.get(held_block, "qw"))
                x_norm, y_norm, theta_norm = params

                angle = 2 * np.pi * theta_norm
            else:
                block_orn = (0,0,0,1)
                x_norm, y_norm = params

            rotated_height = PyBulletMultiModalCoverEnv.get_rotated_height(block_dim, block_orn)

            # De-normalize parameters to actual table coordinates.


            x = MultiModalCoverEnv.x_lb + (MultiModalCoverEnv.x_ub - MultiModalCoverEnv.x_lb) * x_norm
            y = MultiModalCoverEnv.y_lb + (MultiModalCoverEnv.y_ub - MultiModalCoverEnv.y_lb) * y_norm
            z = MultiModalCoverEnv.table_height + rotated_height * 0.5

            arr = np.array([x, y, z, 1.0], dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)

            if CFG.multi_modal_cover_real_robot:
                arr = np.array([arr[0],arr[1],arr[2],angle, arr[3]])

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
        gripper_max_depth = 0.03

        Pick = utils.LinearChainParameterizedOption(
            "Pick",
            [
                # Move to far above the block which we will grasp.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToPreGrasp",
                    z_func=lambda _: PyBulletMultiModalCoverEnv.pick_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    rotate=True,
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
                    z_func=lambda top_z: (top_z + cls._offset_z),
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    rotate=False,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToGraspFinal",
                    z_func=lambda top_z: top_z - gripper_max_depth,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    rotate=False,
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
                    rotate=False,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
            ])

        # PutOnTable
        option_types = [robot_type, block_type]
        params_space = Box(0, 1, (2,))
        place_z = PyBulletMultiModalCoverEnv.table_height + \
                  block_size / 2 + cls._offset_z
        PutOnTable = utils.LinearChainParameterizedOption(
            "PutOnTable",
            [
                # Move to above the table at the (x, y) where we will place.
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorToPrePutOnTable",
                    z_func = lambda _: PyBulletMultiModalCoverEnv.pick_z,
                    finger_status="closed",
                    rotate=True,
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorToPutOnTableFinal",
                    z_func = lambda height: height + table_height,
                    finger_status="closed",
                    rotate=False,
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
                    z_func=lambda _: PyBulletMultiModalCoverEnv.pick_z,
                    finger_status="open",
                    rotate=False,
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
            float], rotate: bool, finger_status: str,
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

            current_orn = (state.get(robot, "orn_x"),
                           state.get(robot, "orn_y"),
                           state.get(robot, "orn_z"),
                           state.get(robot, "orn_w"),)

            world_robot = Pose(current_position,current_orn)


            block_dim = (state.get(block,"depth"),
                         state.get(block,"width"),
                         state.get(block,"height"))

            if CFG.multi_modal_cover_real_robot:
                block_orn = (state.get(block,"qx"),
                             state.get(block,"qy"),
                             state.get(block,"qz"),
                             state.get(block,"qw"))
            else:
                block_orn = (0,0,0,1)

            block_pos = (state.get(block, "pose_x"),
                         state.get(block, "pose_y"),
                         state.get(block, "pose_z"))

            rotated_block_height = PyBulletMultiModalCoverEnv.get_rotated_height(block_dim, block_orn)

            # Calculate the gripper's target position above the block
            target_position = (
                state.get(block, "pose_x"),
                state.get(block, "pose_y"),
                z_func(state.get(block, "pose_z") + rotated_block_height / 2.0)
            )

            # Get the rotation matrix from the block's orientation quaternion
            block_rotation = R.from_quat(block_orn)
            rotation_matrix = block_rotation.as_matrix()

            # Extract the z-components of the rotation matrix to determine the upward-facing axis
            z_components = [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2]]
            parallel_axis = max(range(3), key=lambda i: abs(z_components[i]))  # Most aligned with global z-axis

            # Identify the shortest axis of the block dimensions excluding the parallel axis
            remaining_axes = [i for i in range(3) if i != parallel_axis]
            shortest_axis = min(remaining_axes, key=lambda axis: block_dim[axis])

            # Determine the remaining axis for gripper alignment
            remaining_axis = [i for i in range(3) if i != parallel_axis and i != shortest_axis][0]

            # Extract the block's axes from the rotation matrix
            block_axes = rotation_matrix

            # Determine the gripper's axes
            gripper_x = block_axes[:, remaining_axis]  # Align with the remaining axis
            gripper_z = block_axes[:, parallel_axis]  # Align with the block's parallel axis
            gripper_y = np.cross(gripper_z, gripper_x)  # Perpendicular to x and z

            # Normalize gripper axes
            gripper_x = gripper_x / np.linalg.norm(gripper_x)
            gripper_y = gripper_y / np.linalg.norm(gripper_y)
            gripper_z = gripper_z / np.linalg.norm(gripper_z)

            # Ensure the gripper faces downward in the world frame
            if gripper_z[2] > 0:  # If pointing upward, invert it
                gripper_z = -gripper_z
                gripper_y = -gripper_y  # Adjust y-axis for proper handedness

            # Construct the gripper's rotation matrix
            gripper_rotation = np.column_stack((gripper_x, gripper_y, gripper_z))

            # Convert the rotation matrix to a quaternion
            gripper_rotation = R.from_matrix(gripper_rotation).as_quat()

            # Target orientation and pose
            target_orn = gripper_rotation
            target_pose = Pose(target_position, target_orn)

            logging.info(f"Target Orientation (Quaternion): {target_orn}")
            logging.info(f"Target Orientation (RPY): {euler_from_quaternion(target_orn)}")

            # Return the results
            return world_robot, target_pose, finger_status

        return motion_planner.create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude, physics_client_id, bodies)


    @classmethod
    def _create_blocks_move_to_above_table_option(
            cls, name: str, z_func: Callable[[float], float], finger_status: str, rotate: bool,
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
            robot, block = objects

            if CFG.multi_modal_cover_real_robot:
                x_norm, y_norm, theta_norm = params
            else:
                x_norm, y_norm = params

            home_orn = PyBulletMultiModalCoverEnv.get_robot_ee_home_orn()

            current_position = (state.get(robot, "pose_x"),
                                state.get(robot, "pose_y"),
                                state.get(robot, "pose_z"))

            current_orn = (state.get(robot, "orn_x"),
                           state.get(robot, "orn_y"),
                           state.get(robot, "orn_z"),
                           state.get(robot, "orn_w"),)

            current_pose = Pose(current_position, current_orn)
            # De-normalize parameters to actual table coordinates.

            block_dim = (state.get(block, "depth"),
                         state.get(block, "width"),
                         state.get(block, "height"))

            target_position = (
                PyBulletMultiModalCoverEnv.x_lb +
                (PyBulletMultiModalCoverEnv.x_ub - PyBulletMultiModalCoverEnv.x_lb) * x_norm,
                PyBulletMultiModalCoverEnv.y_lb +
                (PyBulletMultiModalCoverEnv.y_ub - PyBulletMultiModalCoverEnv.y_lb) * y_norm, z_func(block_dim[2]))

            if CFG.multi_modal_cover_real_robot:
                angle = 2 * theta_norm * np.pi
                logging.info(f"Pybullet block theta: {angle}")

                gripper_pose = Pose(current_position, current_orn)

                incremental_rotation = R.from_rotvec(angle * np.array([0,0,1]))
                # Apply the incremental rotation to the block's current rotation
                updated_gripper_rotation = incremental_rotation * R.from_quat(current_orn)

                # Extract the updated quaternion
                target_orn = updated_gripper_rotation.as_quat() if rotate else current_orn
            else:
                target_orn = home_orn

            target_pose = Pose(target_position, target_orn)

            return current_pose, target_pose, finger_status

        return motion_planner.create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude, physics_client_id, bodies)
