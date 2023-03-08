"""Ground-truth options for the pybullet shelf environment."""

from typing import Callable, ClassVar, Dict, Sequence, Set, Tuple

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_shelf import PyBulletShelfEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, create_move_end_effector_to_pose_option
from predicators.pybullet_helpers.geometry import Pose, Pose3D
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, \
    ParameterizedInitiable, ParameterizedOption, ParameterizedPolicy, \
    Predicate, State, Type


class PyBulletShelfGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the pybullet shelf environment."""

    _offset_z: ClassVar[float] = 0.01
    _pick_z: ClassVar[float] = 0.75
    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_shelf"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        _, pybullet_robot, bodies = \
            PyBulletShelfEnv.initialize_pybullet(using_gui=False)
        collision_bodies = {bodies["table_id"], bodies["shelf_id"]}
        block_id = bodies["block_id"]

        robot_type = types["robot"]
        block_type = types["block"]

        def get_current_fingers(state: State) -> float:
            robot, = state.get_objects(robot_type)
            return PyBulletShelfEnv.fingers_state_to_joint(
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

        ## PickPlace option
        option_types = [robot_type, block_type]
        params_space = Box(0, 1, (0, ))
        PickPlace = utils.LinearChainParameterizedOption(
            "PickPlace",
            [
                # Move to far above the object that we will grasp.
                cls._create_move_to_above_object_option(
                    name="MoveEndEffectorToPreGrasp",
                    z_func=lambda _: cls._pick_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletShelfEnv.grasp_tol),
                # Move down to grasp.
                cls._create_move_to_above_object_option(
                    name="MoveEndEffectorToGrasp",
                    z_func=lambda ing_z: (ing_z + cls._offset_z),
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Close fingers.
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_types, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletShelfEnv.grasp_tol),
                # Move back up.
                cls._create_move_to_above_object_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: cls._pick_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Use motion planning to move to shelf pre-place pose.
                cls._create_move_to_shelf_place_option(
                    pybullet_robot=pybullet_robot,
                    block_id=block_id,
                    collision_bodies=collision_bodies,
                    option_types=option_types,
                    params_space=params_space)
            ])

        return {PickPlace}

    @classmethod
    def _create_move_to_above_object_option(
            cls, name: str, z_func: Callable[[float], float],
            finger_status: str, pybullet_robot: SingleArmPyBulletRobot,
            option_types: Sequence[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        argument.

        The parameter z_func maps the object's z position to the target
        z position.
        """

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose3D, Pose3D, str]:
            assert not params
            robot, obj = objects
            current_position = (state.get(robot, "pose_x"),
                                state.get(robot, "pose_y"),
                                state.get(robot, "pose_z"))
            current_orn = (
                state.get(robot, "pose_q0"),
                state.get(robot, "pose_q1"),
                state.get(robot, "pose_q2"),
                state.get(robot, "pose_q3"),
            )
            current_pose = Pose(current_position, current_orn)
            target_position = (state.get(obj,
                                         "pose_x"), state.get(obj, "pose_y"),
                               z_func(state.get(obj, "pose_z")))
            target_orn = PyBulletShelfEnv.get_robot_ee_home_orn()
            target_pose = Pose(target_position, target_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude)

    @classmethod
    def _create_move_to_shelf_place_option(
            cls, pybullet_robot: SingleArmPyBulletRobot, block_id: int,
            collision_bodies: Set[int], option_types: Sequence[Type],
            params_space: Box):
        name = "MoveToShelfPrePlace"
        finger_status = "closed"

        tx = PyBulletShelfEnv.shelf_x
        ty = PyBulletShelfEnv.shelf_y
        tz = PyBulletShelfEnv.table_height + \
            PyBulletShelfEnv.shelf_base_height + \
            PyBulletShelfEnv.block_size / 2 + cls._offset_z
        tquat = (0.7071, 0.0, 0.7071, 0.0)
        target_pose = Pose((tx, ty, tz), (tquat))

        # import time
        # target_joints = self._pybullet_robot.inverse_kinematics(target_pose, validate=True)
        # self._pybullet_robot.set_joints(target_joints)
        # while True:
        #     p.stepSimulation(physicsClientId=self._pybullet_robot.physics_client_id)
        #     time.sleep(0.001)

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose3D, Pose3D, str]:
            assert not params
            robot, _ = objects
            current_pose = Pose(
                (state.get(robot, "pose_x"), state.get(
                    robot, "pose_y"), state.get(robot, "pose_z")),
                (state.get(robot, "pose_q0"), state.get(robot, "pose_q1"),
                 state.get(robot, "pose_q2"), state.get(robot, "pose_q3")),
            )
            return current_pose, target_pose, finger_status

        def _get_held_body_transform(
                state: State,
                objects: Sequence[Object]) -> Tuple[Pose3D, Pose3D, str]:
            robot, block = objects
            world_to_base_link = Pose(
                (state.get(robot, "pose_x"), state.get(
                    robot, "pose_y"), state.get(robot, "pose_z")),
                (state.get(robot, "pose_q0"), state.get(robot, "pose_q1"),
                 state.get(robot, "pose_q2"), state.get(robot, "pose_q3")),
            )
            base_link_to_world = np.r_[p.invertTransform(
                world_to_base_link.position, world_to_base_link.orientation)]
            world_to_obj = Pose(
                (state.get(block, "pose_x"), state.get(
                    block, "pose_y"), state.get(block, "pose_z")),
                (state.get(block, "pose_q0"), state.get(block, "pose_q1"),
                 state.get(block, "pose_q2"), state.get(block, "pose_q3")),
            )
            held_obj_to_base_link = p.invertTransform(*p.multiplyTransforms(
                base_link_to_world[:3], base_link_to_world[3:],
                world_to_obj.position, world_to_obj.orientation))
            base_link_to_held_obj = p.invertTransform(*held_obj_to_base_link)
            return base_link_to_held_obj

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            mode="motion_planning",
            get_collision_bodies=lambda _1, _2: collision_bodies,
            get_held_object_id=lambda _1, _2: block_id,
            get_held_body_transform=_get_held_body_transform)
