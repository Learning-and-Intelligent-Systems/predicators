import logging
import os
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple
import gym
from experiments.envs.pybullet_scale.env import PyBulletScaleEnv, PyBulletScaleState
from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import create_change_fingers_option, create_move_end_effector_to_pose_option
from predicators.pybullet_helpers.geometry import Pose, multiply_poses
from predicators.pybullet_helpers.inverse_kinematics import InverseKinematicsError
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, Predicate, State, Type
import numpy as np
from gym.spaces import Box

class OptionHelper:
    block_top_from_ee_rot: ClassVar[Pose] = Pose((0, 0, -0.01), (0, 1, 0, 0))
    block_top_from_ee_rot_offset: ClassVar[Pose] = Pose((0, 0, 0.15)).multiply(block_top_from_ee_rot)
    left_side_thresh: ClassVar[float] = 0.5
    scale_offset = 0.01

    def __init__(self, pybullet_robot: SingleArmPyBulletRobot, robot_type: Type):
        self.pybullet_robot = pybullet_robot
        self.robot_type = robot_type

    def get_current_fingers(self, state: State) -> float:
            robot, = state.get_objects(self.robot_type)
            return PyBulletScaleEnv.fingers_state_to_joint(self.pybullet_robot, state.get(robot, "fingers"))

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
        relative_block_angle, _, _, _ = params
        robot, _, block = objects

        bx, by, bz = state[block][:3]
        bqx, bqy, bqz, bqw = state[block][6:10]
        world_from_block: Pose = Pose((bx, by, bz), (bqx, bqy, bqz, bqw))
        bh = state.get(block, "h")

        rx, ry, rz, rqx, rqy, rqz, rqw = state[robot][:7]
        world_from_current_gripper: Pose = Pose((rx, ry, rz), (rqx, rqy, rqz, rqw))

        return world_from_current_gripper, self._compute_world_from_ee_grasp(world_from_block, bh, relative_block_angle, True), "open"

    def grasp_block_guide(
        self,
        state: State,
        objects: Sequence[Object],
        params: Array
    ) -> Tuple[Pose, Pose, str]:
        relative_block_angle, _, _, _ = params
        robot, _, block = objects

        bx, by, bz = state[block][:3]
        bqx, bqy, bqz, bqw = state[block][6:10]
        world_from_block: Pose = Pose((bx, by, bz), (bqx, bqy, bqz, bqw))
        bh = state.get(block, "h")

        rx, ry, rz, rqx, rqy, rqz, rqw = state[robot][:7]
        world_from_current_gripper: Pose = Pose((rx, ry, rz), (rqx, rqy, rqz, rqw))

        return world_from_current_gripper, self._compute_world_from_ee_grasp(world_from_block, bh, relative_block_angle, False), "open"

    def _compute_world_from_ee_grasp(
        self,
        world_from_block: Pose,
        block_h: float,
        relative_block_angle: float,
        gripper_offset: bool,
    ) -> Pose:
        relative_block_angle = (np.arctan2(world_from_block.position[1], world_from_block.position[0]) - world_from_block.rpy[2]) % (np.pi * 2)
        rot = round(relative_block_angle / (np.pi/2)) * np.pi/2
        world_from_ee_rot = multiply_poses(
            world_from_block,
            Pose((0, 0, block_h/2)),
            self.block_top_from_ee_rot_offset if gripper_offset else self.block_top_from_ee_rot,
        )
        ee_rot_from_ee = Pose.from_rpy((0, 0, 0), (0, 0, rot))
        return world_from_ee_rot.multiply(ee_rot_from_ee)

    def put_block_guide(
        self,
        state: State,
        objects: Sequence[Object],
        params: Array
    ) -> Tuple[Pose, Pose, str]:
        _, offset_x, offset_y, left_side_val = params
        robot, _, block = objects

        rx, ry, rz, rqx, rqy, rqz, rqw = state[robot][:7]
        world_from_current_gripper: Pose = Pose((rx, ry, rz), (rqx, rqy, rqz, rqw))

        return world_from_current_gripper, self._compute_world_from_ee_put(
            (offset_x, offset_y), left_side_val, state.get(block, "h"), False
        ), "closed"

    def post_put_block_guide(
        self,
        state: State,
        objects: Sequence[Object],
        params: Array
    ) -> Tuple[Pose, Pose, str]:
        _, offset_x, offset_y, left_side_val = params
        robot, _, block = objects

        rx, ry, rz, rqx, rqy, rqz, rqw = state[robot][:7]
        world_from_current_gripper: Pose = Pose((rx, ry, rz), (rqx, rqy, rqz, rqw))

        return world_from_current_gripper, self._compute_world_from_ee_put(
            (offset_x, offset_y), left_side_val, state.get(block, "h"), True
        ), "open"

    def _compute_world_from_ee_put(
        self,
        block_offset: Tuple[float, float],
        left_side_val: float,
        block_h: float,
        gripper_offset: bool,
    ) -> Pose:
        world_from_scale_bottom = PyBulletScaleEnv.scale_poses[1 if left_side_val > self.left_side_thresh else 0]
        scale_bottom_from_block_bottom = Pose((*block_offset, PyBulletScaleEnv.scale_size[2]))
        block_bottom_from_block_top = Pose((0, 0, block_h + self.scale_offset))
        block_from_gripper = (self.block_top_from_ee_rot_offset if gripper_offset else self.block_top_from_ee_rot).multiply(Pose.identity())
        return multiply_poses(
            world_from_scale_bottom,
            scale_bottom_from_block_bottom,
            block_bottom_from_block_top,
            block_from_gripper
        )

class PyBulletScaleOptionRobot:
    def __getattr__(self, name: str) -> Any:
        _, pybullet_robot, _ = PyBulletScaleEnv.initialize_pybullet(False)
        return pybullet_robot.__getattribute__(name)

class CheckScalePolicy():
    def __init__(self, action_space: Box):
        self.output = np.add(action_space.low, action_space.high)/2

    def __call__(self, *args, **kwargs) -> Action:
        return Action(self.output, extra_info=True)

class PyBulletScaleGroundTruthOptionFactory(GroundTruthOptionFactory):
    finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    move_to_pose_tol: ClassVar[float] = 1e-4

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_scale"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        pybullet_robot = PyBulletScaleOptionRobot()

        # Types
        robot_type = types["robot"]
        block_type = types["block"]
        scale_type = types["scale"]

        # Option calculation helper
        option_helper = OptionHelper(pybullet_robot, robot_type)

        # Option funcs
        params_space = Box(
            np.array([0.0, -PyBulletScaleEnv.scale_size[0], -PyBulletScaleEnv.scale_size[1], 0.0]),
            np.array([np.pi * 2, PyBulletScaleEnv.scale_size[0], PyBulletScaleEnv.scale_size[1], 1.0])
        )
        option_types = [robot_type, scale_type, block_type]
        close_fingers = create_change_fingers_option(
            robot = pybullet_robot,
            name = "CloseFingers",
            types = option_types,
            params_space = params_space,
            get_current_and_target_val = option_helper.close_fingers_func,
            max_vel_norm = CFG.pybullet_max_vel_norm,
            grasp_tol = PyBulletScaleEnv.grasp_tol,
        )
        open_fingers = create_change_fingers_option(
            robot = pybullet_robot,
            name = "OpenFingers",
            types = option_types,
            params_space = params_space,
            get_current_and_target_val = option_helper.open_fingers_func,
            max_vel_norm = CFG.pybullet_max_vel_norm,
            grasp_tol = PyBulletScaleEnv.grasp_tol,
        )

        # Options
        options = set()
        ## Move
        options.add(cls._create_jumping_options(
            pybullet_robot,
            robot_type,
            option_types,
            params_space,
            option_helper,
            close_fingers,
            open_fingers
        ))

        # CheckScale
        options.add(ParameterizedOption(
            "CheckScale",
            [scale_type],
            Box(np.zeros(0), np.zeros(0)),
            CheckScalePolicy(action_space),
            cls._check_scale_initiable,
            cls._check_scale_terminal,
        ))

        return options

    @classmethod
    def _check_scale_initiable(
        cls,
        state: State,
        data: Dict,
        objects: Sequence[Object],
        arr: Array
    ) -> bool:
        return True

    @classmethod
    def _check_scale_terminal(
        cls,
        state: State,
        data: Dict,
        objects: Sequence[Object],
        arr: Array
    ) -> bool:
        terminal_executed = data.get("terminal_executed", False)
        data["terminal_executed"] = True
        return terminal_executed

    @classmethod
    def _create_jumping_options(
        cls,
        pybullet_robot: PyBulletScaleOptionRobot,
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
            # create_move_end_effector_to_pose_option(
            #     robot = pybullet_robot,
            #     name = "RetractEEFromBox",
            #     types = option_types,
            #     params_space = params_space,
            #     get_current_and_target_pose_and_finger_status = option_helper.post_put_block_guide,
            #     move_to_pos_tol = cls.move_to_pose_tol,
            #     max_vel_norm = CFG.pybullet_max_vel_norm,
            #     finger_action_nudge_magnitude = cls.finger_action_nudge_magnitude,
            # ),
        ])
