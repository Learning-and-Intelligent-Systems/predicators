"""Ground-truth options for the (non-pybullet) rings environment."""
import logging
from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.ring_stack import RingStackEnv
from predicators.envs.pybullet_ring_stack import PyBulletRingEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, MotionPlanController
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type

class RingsGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the (non-pybullet) ring-stack environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"ring_stack"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        robot_type = types["robot"]
        ring_type = types["ring"]
        pole_type = types["pole"]
        ring_size = CFG.ring_size
        ring_height = CFG.ring_height
        pole_height = CFG.pole_height
        Pick = utils.SingletonParameterizedOption(
            # variables: [robot, object to pick]
            # params: []
            "Pick",
            cls._create_pick_policy(action_space),
            types=[robot_type, ring_type])

        PutAroundPole = utils.SingletonParameterizedOption(
            # variables: [robot, object on which to stack currently-held-object]
            # params: []
            "PutOnTableAroundPole",
            cls._create_putaroundpole_policy(action_space, pole_height),
            types=[robot_type, pole_type],
        )
        PutOnTable = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: [x, y] (normalized coordinates on the table surface)
            "PutOnTable",
            cls._create_putontable_policy(action_space, ring_height),
            types=[robot_type],
            params_space=Box(0, 1, (2,)))

        logging.info(f"Pick pspace: {Pick.params_space}")
        logging.info(f"PutAroundPole pspace: {PutAroundPole.params_space}")
        logging.info(f"PutOnTable pspace: {PutOnTable.params_space}")

        return {Pick, PutOnTable, PutAroundPole}

    @classmethod
    def _create_pick_policy(cls, action_space: Box) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, ring = objects
            ring = np.array([
                state.get(ring, "pose_x"),
                state.get(ring, "pose_y"),
                state.get(ring, "pose_z")
            ])
            arr = np.r_[ring, 0.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_putaroundpole_policy(cls, action_space: Box,
                             pole_height: float) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, pole = objects
            pole_pose = np.array([
                state.get(pole, "pose_x"),
                state.get(pole, "pose_y"),
                state.get(pole, "pose_z")
            ])
            relative_grasp = np.array([
                0.,
                0.,
                pole_height,
            ])
            arr = np.r_[pole_pose + relative_grasp, 1.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_putontable_policy(cls, action_space: Box,
                                  ring_height: float) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects  # unused
            # De-normalize parameters to actual table coordinates.
            x_norm, y_norm = params
            x = RingStackEnv.x_lb + (RingStackEnv.x_ub - RingStackEnv.x_lb) * x_norm
            y = RingStackEnv.y_lb + (RingStackEnv.y_ub - RingStackEnv.y_lb) * y_norm
            z = RingStackEnv.table_height + 0.5 * ring_height
            arr = np.array([x, y, z, 1.0], dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

class PyBulletRingsGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the pybullet_ring_stack environment."""

    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _offset_z: ClassVar[float] = 0.01

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_ring_stack"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        client_id, pybullet_robot, bodies = \
            PyBulletRingEnv.initialize_pybullet(using_gui=False)

        robot_type = types["robot"]
        ring_type = types["ring"]
        pole_type = types["pole"]
        ring_size = CFG.ring_size
        ring_height = CFG.ring_height
        pole_height = CFG.pole_height


        def get_current_fingers(state: State) -> float:
            robot, = state.get_objects(robot_type)
            return PyBulletRingEnv.fingers_state_to_joint(
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
        option_types = [robot_type, ring_type]
        params_space = Box(0, 1, (0, ))
        Pick = utils.LinearChainParameterizedOption(
            "Pick",
            [
                # Move to far above the ring which we will grasp.
                cls._create_rings_move_to_above_ring_option(
                    name="MoveEndEffectorToPreGrasp",
                    z_func=lambda _: PyBulletRingEnv.pick_z,
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
                    PyBulletRingEnv.grasp_tol),
                # Move down to grasp.
                cls._create_rings_move_to_above_ring_option(
                    name="MoveEndEffectorToGrasp",
                    z_func=lambda ring_z: (ring_z + cls._offset_z),
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
                    PyBulletRingEnv.grasp_tol),
                # Move back up.
                cls._create_rings_move_to_above_ring_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: PyBulletRingEnv.pick_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
            ])

        # Stack
        option_types = [robot_type, ring_type]
        params_space = Box(0, 1, (0, ))
        Stack = utils.LinearChainParameterizedOption(
            "Stack",
            [
                # Move to above the ring on which we will stack.
                cls._create_rings_move_to_above_ring_option(
                    name="MoveEndEffectorToPrePole",
                    z_func=lambda _: PyBulletRingEnv.pick_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
                # Move down to place.
                cls._create_rings_move_to_above_ring_option(
                    name="MoveEndEffectorToPole",
                    z_func=lambda ring_z:
                    (ring_z + ring_height + cls._offset_z),
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
                    PyBulletRingEnv.grasp_tol),
                # Move back up.
                cls._create_rings_move_to_above_ring_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: PyBulletRingEnv.pick_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
            ])

        # PutOnTable
        option_types = [robot_type]
        params_space = Box(0, 1, (2, ))
        place_z = PyBulletRingEnv.table_height + \
            ring_height / 2 + cls._offset_z
        PutOnTable = utils.LinearChainParameterizedOption(
            "PutOnTable",
            [
                # Move to above the table at the (x, y) where we will place.
                cls._create_rings_move_to_above_table_option(
                    name="MoveEndEffectorToPrePutOnTable",
                    z=PyBulletRingEnv.pick_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
                # Move down to place.
                cls._create_rings_move_to_above_table_option(
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
                    PyBulletRingEnv.grasp_tol),
                # Move back up.
                cls._create_rings_move_to_above_table_option(
                    name="MoveEndEffectorBackUp",
                    z=PyBulletRingEnv.pick_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
            ])
        params_space = Box(0, 1, (0, ))
        option_types = [robot_type, pole_type]
        PutOnTableAroundPole = utils.LinearChainParameterizedOption(
            "PutOnTableAroundPole",
            [
                # Move to above the pole at the (x, y) where we will place.
                cls._create_rings_move_to_above_pole_option(
                    name="MoveEndEffectorToPrePutAroundPole",
                    z=PyBulletRingEnv.pick_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
                # Move down to place.
                cls._create_rings_move_to_above_pole_option(
                    name="MoveEndEffectorToPutAroundPole",
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
                    PyBulletRingEnv.grasp_tol),
                # Move back up.
                cls._create_rings_move_to_above_pole_option(
                    name="MoveEndEffectorBackUp",
                    z=PyBulletRingEnv.pick_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    physics_client_id=client_id,
                    bodies=bodies
                ),
            ])

        return {Pick, Stack, PutOnTable, PutOnTableAroundPole}

    @classmethod
    def _create_rings_move_to_above_ring_option(
            cls, name: str, z_func: Callable[[float],
                                             float], finger_status: str,
            pybullet_robot: SingleArmPyBulletRobot, option_types: List[Type],
            params_space: Box,
            physics_client_id,
            bodies
    ) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        ring argument.

        The parameter z_func maps the ring's z position to the target z
        position.
        """
        home_orn = PyBulletRingEnv.get_robot_ee_home_orn()
        motion_planner = MotionPlanController(bodies, physics_client_id)
        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not params
            robot, ring = objects
            current_position = (state.get(robot, "pose_x"),
                                state.get(robot, "pose_y"),
                                state.get(robot, "pose_z"))
            current_pose = Pose(current_position, home_orn)
            target_position = (state.get(ring,
                                         "pose_x"), state.get(ring, "pose_y"),
                               z_func(state.get(ring, "pose_z")))
            target_pose = Pose(target_position, home_orn)
            return current_pose, target_pose, finger_status

        return motion_planner.create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            physics_client_id,
            bodies
        )

    @classmethod
    def _create_rings_move_to_above_pole_option(
            cls, name: str, z: float, finger_status: str,
            pybullet_robot: SingleArmPyBulletRobot, option_types: List[Type],
            params_space: Box,
            physics_client_id,
            bodies
    ) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        table at the x,y pos of the pole.

        The z position of the target pose must be provided.
        """
        home_orn = PyBulletRingEnv.get_robot_ee_home_orn()
        motion_planner = MotionPlanController(bodies, physics_client_id)

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not params
            robot, pole = objects
            current_position = (state.get(robot, "pose_x"),
                                state.get(robot, "pose_y"),
                                state.get(robot, "pose_z"))
            current_pose = Pose(current_position, home_orn)
            target_position = (state.get(pole,
                                         "pose_x"), state.get(pole, "pose_y"),
                              z)
            target_pose = Pose(target_position, home_orn)
            return current_pose, target_pose, finger_status

        return motion_planner.create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            physics_client_id,
            bodies
        )

    @classmethod
    def _create_rings_move_to_above_table_option(
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
        home_orn = PyBulletRingEnv.get_robot_ee_home_orn()
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
                PyBulletRingEnv.x_lb +
                (PyBulletRingEnv.x_ub - PyBulletRingEnv.x_lb) * x_norm,
                PyBulletRingEnv.y_lb +
                (PyBulletRingEnv.y_ub - PyBulletRingEnv.y_lb) * y_norm, z)
            target_pose = Pose(target_position, home_orn)
            return current_pose, target_pose, finger_status

        return motion_planner.create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            physics_client_id,
            bodies
        )
