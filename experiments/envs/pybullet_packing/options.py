import logging
import os
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple
import gym
from experiments.envs.pybullet_packing.env import PyBulletPackingEnv, PyBulletPackingState
from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import create_change_fingers_option, create_move_end_effector_to_pose_option
from predicators.pybullet_helpers.geometry import Pose, multiply_poses
from predicators.pybullet_helpers.inverse_kinematics import InverseKinematicsError
from predicators.pybullet_helpers.motion_planning import run_motion_planning
from pybullet_utils.transformations import quaternion_from_euler
from predicators.pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, Predicate, State, Type
import pybullet as p
import numpy as np
from gym.spaces import Box

class OptionHelper:
    def __init__(self, pybullet_robot: SingleArmPyBulletRobot, robot_type: Type):
        self.pybullet_robot = pybullet_robot
        self.robot_type = robot_type

    def get_current_fingers(self, state: State) -> float:
            robot, = state.get_objects(self.robot_type)
            return PyBulletPackingEnv.fingers_state_to_joint(self.pybullet_robot, state.get(robot, "fingers"))

    def open_fingers_func(self, state: State, objects: Sequence[Object],
                            params: Array) -> Tuple[float, float]:
        del objects, params  # unused
        current = self.get_current_fingers(state)
        target = self.pybullet_robot.open_fingers
        return current, target

    def close_fingers_func(self, state: State, objects: Sequence[Object],
                            params: Array) -> Tuple[float, float]:
        del objects, params  # unused
        current = self.get_current_fingers(state)
        target = self.pybullet_robot.closed_fingers
        return current, target

    def pregrasp_block_guide(
        self,
        state: State,
        objects: Sequence[Object],
        params: Array
    ) -> Tuple[Pose, Pose, str]:
        grip_height, _, _ = params
        robot, _, block = objects

        block_from_block_center = Pose((0, 0, grip_height * state.get(block, "h")))

        bx, by, bz = state[block][:3]
        bqx, bqy, bqz, bqw = state[block][6:10]
        world_from_block_rot: Pose = Pose((bx, by, bz), (bqx, bqy, bqz, bqw))

        for rot_times in range(4):
            block_rot_from_block = Pose((0, 0, 0), quaternion_from_euler(0, 0, np.pi/2 * rot_times)) # type: ignore
            world_from_gripper: Pose = multiply_poses(
                world_from_block_rot,
                block_rot_from_block,
                block_from_block_center,
                PyBulletPackingGroundTruthOptionFactory.block_center_from_gripper_offset,
            )
            world_from_up_gripper: Pose = multiply_poses(
                Pose((0, 0, 0), world_from_gripper.orientation),
                PyBulletPackingGroundTruthOptionFactory.gripper_rot_from_up_gripper
            )
            if world_from_up_gripper.position[2] >= PyBulletPackingGroundTruthOptionFactory.up_gripper_thresh:
                world_from_target_gripper = world_from_gripper
                break
        else:
            logging.info(f"BLOCK {block} NOT ON ITS SIDE; BLOCK STATE {state[block]}")
            raise RuntimeError("Block not on its side")

        rx, ry, rz, rqx, rqy, rqz, rqw = state[robot][:7]
        world_from_current_gripper: Pose = Pose((rx, ry, rz), (rqx, rqy, rqz, rqw))

        return world_from_current_gripper, world_from_target_gripper, "open"
    
    def grasp_block_guide(
        self,
        state: State,
        objects: Sequence[Object],
        params: Array
    ) -> Tuple[Pose, Pose, str]:
        grip_height, _, _ = params
        robot, _, block = objects

        block_from_block_center = Pose((0, 0, grip_height * state.get(block, "h")))

        bx, by, bz = state[block][:3]
        bqx, bqy, bqz, bqw = state[block][6:10]
        world_from_block_rot: Pose = Pose((bx, by, bz), (bqx, bqy, bqz, bqw))

        for rot_times in range(4):
            block_rot_from_block = Pose((0, 0, 0), quaternion_from_euler(0, 0, np.pi/2 * rot_times)) # type: ignore
            world_from_gripper: Pose = multiply_poses(
                world_from_block_rot,
                block_rot_from_block,
                block_from_block_center,
                PyBulletPackingGroundTruthOptionFactory.block_center_from_gripper,
            )
            world_from_up_gripper: Pose = multiply_poses(
                Pose((0, 0, 0), world_from_gripper.orientation),
                PyBulletPackingGroundTruthOptionFactory.gripper_rot_from_up_gripper
            )
            if world_from_up_gripper.position[2] >= PyBulletPackingGroundTruthOptionFactory.up_gripper_thresh:
                world_from_target_gripper = world_from_gripper
                break
        else:
            logging.info(f"BLOCK {block} NOT ON ITS SIDE; BLOCK STATE {state[block]}")
            raise RuntimeError("Block not on its side")

        rx, ry, rz, rqx, rqy, rqz, rqw = state[robot][:7]
        world_from_current_gripper: Pose = Pose((rx, ry, rz), (rqx, rqy, rqz, rqw))

        return world_from_current_gripper, world_from_target_gripper, "open"

    def put_block_guide(
        self,
        state: State,
        objects: Sequence[Object],
        params: Array
    ) -> Tuple[Pose, Pose, str]:
        grip_height, offset_x, offset_y = params
        robot, box, block = objects

        block_from_block_center = Pose((0, 0, grip_height * state.get(block, "h")))
        block_bottom_from_block = Pose((0, 0, state.get(block, "h")/2))
        box_bottom_from_block_bottom = Pose((offset_x, offset_y, 0))
        box_from_box_bottom_offset = Pose((0, 0, -state.get(box, "h")/2))

        rx, ry, rz = state[box][:3]
        world_from_box: Pose = Pose((rx, ry, rz))

        rx, ry, rz, rqx, rqy, rqz, rqw = state[robot][:7]
        world_from_current_gripper: Pose = Pose((rx, ry, rz), (rqx, rqy, rqz, rqw))

        world_from_target_gripper: Pose = multiply_poses(
            world_from_box,
            box_from_box_bottom_offset,
            PyBulletPackingGroundTruthOptionFactory.box_bottom_offset_from_box_bottom,
            box_bottom_from_block_bottom,
            block_bottom_from_block,
            block_from_block_center,
            PyBulletPackingGroundTruthOptionFactory.block_center_from_gripper,
        )

        return world_from_current_gripper, world_from_target_gripper, "closed"
    
    def post_put_block_guide(
        self,
        state: State,
        objects: Sequence[Object],
        params: Array
    ) -> Tuple[Pose, Pose, str]:
        grip_height, offset_x, offset_y = params
        robot, box, block = objects

        block_from_block_center = Pose((0, 0, grip_height * state.get(block, "h")))
        block_bottom_from_block = Pose((0, 0, state.get(block, "h")/2))
        box_bottom_from_block_bottom = Pose((offset_x, offset_y, 0))
        box_from_box_bottom_offset = Pose((0, 0, -state.get(box, "h")/2))

        rx, ry, rz = state[box][:3]
        world_from_box: Pose = Pose((rx, ry, rz))

        rx, ry, rz, rqx, rqy, rqz, rqw = state[robot][:7]
        world_from_current_gripper: Pose = Pose((rx, ry, rz), (rqx, rqy, rqz, rqw))

        world_from_target_gripper: Pose = multiply_poses(
            world_from_box,
            box_from_box_bottom_offset,
            PyBulletPackingGroundTruthOptionFactory.box_bottom_offset_from_box_bottom,
            box_bottom_from_block_bottom,
            block_bottom_from_block,
            block_from_block_center,
            PyBulletPackingGroundTruthOptionFactory.block_center_from_gripper_offset,
        )

        return world_from_current_gripper, world_from_target_gripper, "open"

class PyBulletPackingOptionRobot:
    def __getattr__(self, name: str) -> Any:
        _, pybullet_robot, _ = PyBulletPackingEnv.initialize_pybullet(False)
        return pybullet_robot.__getattribute__(name)

class PyBulletPackingGroundTruthOptionFactory(GroundTruthOptionFactory):
    block_center_from_gripper_offset: ClassVar[Pose] = Pose((-0.12, 0, 0), quaternion_from_euler(0, np.pi*1/2, 0)) # type: ignore
    block_center_from_gripper: ClassVar[Pose] = Pose((-0.015, 0, 0), quaternion_from_euler(0, np.pi*1/2, 0)) # type: ignore
    box_bottom_offset_from_box_bottom: ClassVar[Pose] = Pose((0, 0, 0.011))
    gripper_rot_from_up_gripper: ClassVar[Pose] = Pose((0, 0, 0), quaternion_from_euler(0, np.pi*1/2, 0)).invert().multiply(Pose((-1, 0, 0))) # type: ignore

    up_gripper_thresh: ClassVar[float] = 0.9

    finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    move_to_pose_tol: ClassVar[float] = 1e-4

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_packing"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        pybullet_robot = PyBulletPackingOptionRobot()

        # Types
        robot_type = types["robot"]
        block_type = types["block"]
        box_type = types["box"]

        # Option calculation helper
        option_helper = OptionHelper(pybullet_robot, robot_type)

        # Option funcs
        box_depths, box_widths, _ = zip(*(PyBulletPackingEnv._get_box_dims(box_id) for box_id in PyBulletPackingEnv.box_col_counts))
        box_max_depth, box_max_width = max(box_depths), max(box_widths)

        option_types = [robot_type, box_type, block_type]
        params_space = Box(
            np.array([-0.15, -box_max_depth/2, -box_max_width/2]),
            np.array([0.35, box_max_depth/2, box_max_width/2])
        )
        close_fingers = create_change_fingers_option(
            robot = pybullet_robot,
            name = "CloseFingers",
            types = option_types,
            params_space = params_space,
            get_current_and_target_val = option_helper.close_fingers_func,
            max_vel_norm = CFG.pybullet_max_vel_norm,
            grasp_tol = PyBulletPackingEnv.grasp_tol,
        )
        open_fingers = create_change_fingers_option(
            robot = pybullet_robot,
            name = "OpenFingers",
            types = option_types,
            params_space = params_space,
            get_current_and_target_val = option_helper.open_fingers_func,
            max_vel_norm = CFG.pybullet_max_vel_norm,
            grasp_tol = PyBulletPackingEnv.grasp_tol,
        )
        option = (
            cls._create_jumping_options #if CFG.option_model_terminate_on_repeat else cls._create_movement_options
        )(
            pybullet_robot, robot_type, option_types, params_space, option_helper, close_fingers, open_fingers
        )
        return {option}

    @classmethod
    def _create_movement_options(
        cls,
        pybullet_robot: PyBulletPackingOptionRobot,
        robot_type: Type, option_types: List[Type],
        params_space: Box, option_helper: OptionHelper,
        close_fingers: ParameterizedOption, open_fingers: ParameterizedOption,
    ) -> ParameterizedOption:
        # assert CFG.pybullet_control_mode == "position"

        def grab_initiable(
            state: State, memory: Dict,
            objects: Sequence[Object],
            params: Array
        ) -> bool:
            assert isinstance(state, PyBulletPackingState)
            memory["finger_nudge"] = cls.finger_action_nudge_magnitude
            _, target_pose, _ = option_helper.grasp_block_guide(state, objects, params)
            try:
                target_joint_positions = pybullet_robot.inverse_kinematics(target_pose, True)
            except InverseKinematicsError:
                memory["motion_plan"] = None
            else:
                memory["motion_plan"] = PyBulletPackingEnv.run_motion_planning(state, target_joint_positions)
            return True

        def put_initiable(
            state: State, memory: Dict,
            objects: Sequence[Object],
            params: Array
        ) -> bool:
            assert isinstance(state, PyBulletPackingState)
            memory["finger_nudge"] = -cls.finger_action_nudge_magnitude
            _, target_pose, _ = option_helper.put_block_guide(state, objects, params)
            try:
                target_joint_positions = pybullet_robot.inverse_kinematics(target_pose, True)
            except InverseKinematicsError:
                memory["motion_plan"] = None
            else:
                memory["motion_plan"] = PyBulletPackingEnv.run_motion_planning(state, target_joint_positions)
            return True

        def motion_plan_policy(
            state: State, memory: Dict,
            objects: Sequence[Object],
            params: Array
        ) -> Action:
            if memory["motion_plan"] is None:
                raise utils.OptionExecutionFailure("Motion planning failed.")
            action_arr = np.array(memory["motion_plan"][0], dtype=np.float32)
            action_arr[pybullet_robot.left_finger_joint_idx] += memory["finger_nudge"]
            action_arr[pybullet_robot.right_finger_joint_idx] += memory["finger_nudge"]
            action_arr = np.clip(action_arr, pybullet_robot.action_space.low,
                             pybullet_robot.action_space.high)
            memory["motion_plan"] = memory["motion_plan"][1:]
            return Action(action_arr)

        def motion_plan_terminal(
            state: State, memory: Dict,
            objects: Sequence[Object],
            params: Array
        ) -> bool:
            if memory["motion_plan"] is None:
                raise utils.OptionExecutionFailure("Motion planning failed.")
            return not memory["motion_plan"]

        return utils.LinearChainParameterizedOption("Move",[
            ParameterizedOption(
                "MoveEndEffectorToGrasp",
                types = option_types,
                params_space = params_space,
                policy = motion_plan_policy,
                initiable = grab_initiable,
                terminal = motion_plan_terminal
            ),
            close_fingers,
            ParameterizedOption(
                "MoveBlockToBox",
                types = option_types,
                params_space = params_space,
                policy = motion_plan_policy,
                initiable = put_initiable,
                terminal = motion_plan_terminal
            ),
            open_fingers,
        ])

    @classmethod
    def _create_jumping_options(
        cls,
        pybullet_robot: PyBulletPackingOptionRobot,
        robot_type: Type, option_types: List[Type],
        params_space: Box, option_helper: OptionHelper,
        close_fingers: ParameterizedOption, open_fingers: ParameterizedOption,
    ) -> ParameterizedOption:
        assert CFG.pybullet_control_mode == "reset"

        return utils.LinearChainParameterizedOption("Move",[
            create_move_end_effector_to_pose_option(
                robot = pybullet_robot,
                name = "MoveEndEffectorToPreGrasp",
                types = option_types,
                params_space = params_space,
                get_current_and_target_pose_and_finger_status = option_helper.pregrasp_block_guide,
                move_to_pos_tol = cls.move_to_pose_tol,
                max_vel_norm = CFG.pybullet_max_vel_norm,
                finger_action_nudge_magnitude = cls.finger_action_nudge_magnitude,
            ),
            create_move_end_effector_to_pose_option(
                robot = pybullet_robot,
                name = "MoveEndEffectorToGrasp",
                types = option_types,
                params_space = params_space,
                get_current_and_target_pose_and_finger_status = option_helper.grasp_block_guide,
                move_to_pos_tol = cls.move_to_pose_tol,
                max_vel_norm = CFG.pybullet_max_vel_norm,
                finger_action_nudge_magnitude = cls.finger_action_nudge_magnitude,
            ),
            close_fingers,
            create_move_end_effector_to_pose_option(
                robot = pybullet_robot,
                name = "MoveBlockToBox",
                types = option_types,
                params_space = params_space,
                get_current_and_target_pose_and_finger_status = option_helper.put_block_guide,
                move_to_pos_tol = cls.move_to_pose_tol,
                max_vel_norm = CFG.pybullet_max_vel_norm,
                finger_action_nudge_magnitude = cls.finger_action_nudge_magnitude,
            ),
            open_fingers,
            create_move_end_effector_to_pose_option(
                robot = pybullet_robot,
                name = "RetractEEFromBox",
                types = option_types,
                params_space = params_space,
                get_current_and_target_pose_and_finger_status = option_helper.post_put_block_guide,
                move_to_pos_tol = cls.move_to_pose_tol,
                max_vel_norm = CFG.pybullet_max_vel_norm,
                finger_action_nudge_magnitude = cls.finger_action_nudge_magnitude,
            ),
        ])
