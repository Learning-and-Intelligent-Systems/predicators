"""Ground-truth options for the coffee environment."""

import logging
from functools import lru_cache
from typing import ClassVar, Dict, List, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.coffee import CoffeeEnv
from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import \
    get_change_fingers_action, get_move_end_effector_to_pose_action
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type, _TypedEntity
from predicators.ground_truth_models.coffee.options import PyBulletCoffeeGroundTruthOptionFactory

@lru_cache
def _get_pybullet_robot() -> SingleArmPyBulletRobot:
    _, pybullet_robot, _ = \
        PyBulletCoffeeEnv.initialize_pybullet(using_gui=False)
    return pybullet_robot

class PyBulletGrowGroundTruthOptionFactory(GroundTruthOptionFactory):

    env_cls: ClassVar[TypingType[CoffeeEnv]] = PyBulletCoffeeEnv
    pick_policy_tol: ClassVar[float] = 1e-3
    pour_policy_tol: ClassVar[float] = 1e-3
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_grow"}
    
    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        # Types
        robot_type = types["robot"]
        jug_type = types["jug"]
        cup_type = types["cup"]
        # Predicates
        Holding = predicates["Holding"]
        Grown = predicates["Grown"]

        # PickJug
        def _PickJug_terminal(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> bool:
            del memory, params  # unused
            robot, jug = objects
            holds = Holding.holds(state, [robot, jug])
            return holds
            
        PickJug = ParameterizedOption(
            name="PickJug",
            types=[robot_type, jug_type],
            params_space=Box(0, 1, (0,)),
            policy=PyBulletCoffeeGroundTruthOptionFactory._create_pick_jug_policy(),
            # policy=cls._create_pick_jug_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_PickJug_terminal)
        
        # Pour
        def _Pour_terminal(state: State, memory: Dict,
                           objects: Sequence[Object],
                           params: Array) -> bool:
            del memory, params
            _, _, cup = objects
            return Grown.holds(state, [cup])

        Pour = ParameterizedOption(
            "Pour",
            [robot_type, jug_type, cup_type],
            params_space=Box(0, 1, (0,)),
            policy=PyBulletCoffeeGroundTruthOptionFactory._create_pour_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_Pour_terminal)

        return {PickJug, Pour}
    
    # @classmethod
    # def _create_pick_jug_policy(cls) -> ParameterizedPolicy:

    #     def policy(state: State, memory: Dict, objects: Sequence[Object],
    #                params: Array) -> Action:
    #         # This policy moves the robot to a safe height, then moves to behind
    #         # the handle in the y direction, then moves down in the z direction,
    #         # then moves forward in the y direction before finally grasping.
    #         del memory, params  # unused
    #         robot, jug = objects
    #         x = state.get(robot, "x")
    #         y = state.get(robot, "y")
    #         z = state.get(robot, "z")
    #         robot_pos = (x, y, z)
    #         handle_pos = cls._get_jug_handle_grasp(state, jug)
    #         # If close enough, pick.
    #         sq_dist_to_handle = np.sum(np.subtract(handle_pos, robot_pos)**2)
    #         if sq_dist_to_handle < cls.pick_policy_tol:
    #             return cls._get_pick_action(state)
    #         target_x, target_y, target_z = handle_pos
    #         # Distance to the handle in the x/z plane.
    #         xz_handle_sq_dist = (target_x - x)**2 + (target_z - z)**2
    #         # Distance to the penultimate waypoint in the x/y plane.
    #         waypoint_y = target_y - cls.env_cls.pick_jug_y_padding
    #         # Distance in the z direction to a safe move distance.
    #         safe_z_sq_dist = (cls.env_cls.robot_init_z - z)**2
    #         xy_waypoint_sq_dist = (target_x - x)**2 + (waypoint_y - y)**2
    #         dwrist = cls.env_cls.robot_init_wrist - state.get(robot, "wrist")
    #         dtilt = cls.env_cls.robot_init_tilt - state.get(robot, "tilt")
    #         # If at the correct x and z position and behind in the y direction,
    #         # move directly toward the target.
    #         if target_y > y and xz_handle_sq_dist < cls.pick_policy_tol:
    #             return cls._get_move_action(state, handle_pos, robot_pos,
    #                                         dtilt, dwrist)
    #         # If close enough to the penultimate waypoint in the x/y plane,
    #         # move to the waypoint (in the z direction).
    #         if xy_waypoint_sq_dist < cls.pick_policy_tol:
    #             return cls._get_move_action(state,
    #                                         (target_x, waypoint_y, target_z),
    #                                         robot_pos)
    #         # If at a safe height, move to the position above the penultimate
    #         # waypoint, still at a safe height.
    #         if safe_z_sq_dist < cls.env_cls.safe_z_tol:
    #             return cls._get_move_action(
    #                 state, (target_x, waypoint_y, cls.env_cls.robot_init_z),
    #                 robot_pos)
    #         # Move up to a safe height.
    #         return cls._get_move_action(state,
    #                                     (x, y, cls.env_cls.robot_init_z),
    #                                     robot_pos)

    #     return policy

    # @classmethod
    # def _get_jug_handle_grasp(cls, state: State,
    #                           jug: Object) -> Tuple[float, float, float]:
    #     # Hack to avoid duplicate code.
    #     return cls.env_cls._get_jug_handle_grasp(state, jug)  # pylint: disable=protected-access

    # @classmethod
    # def _get_finger_action(cls, state: State,
    #                        target_pybullet_fingers: float) -> Action:
    #     pybullet_robot = _get_pybullet_robot()
    #     robots = [r for r in state if r.type.name == "robot"]
    #     assert len(robots) == 1
    #     robot = robots[0]
    #     current_finger_state = state.get(robot, "fingers")
    #     current_finger_joint = PyBulletCoffeeEnv.fingers_state_to_joint(
    #         pybullet_robot, current_finger_state)
    #     assert isinstance(state, utils.PyBulletState)
    #     current_joint_positions = state.joint_positions

    #     return get_change_fingers_action(
    #         pybullet_robot,
    #         current_joint_positions,
    #         current_finger_joint,
    #         target_pybullet_fingers,
    #         CFG.pybullet_max_vel_norm,
    #     )

    # @classmethod
    # def _get_pick_action(cls, state: State) -> Action:
    #     pybullet_robot = _get_pybullet_robot()
    #     return cls._get_finger_action(state, pybullet_robot.closed_fingers)

    # @classmethod
    # def _get_finger_action(cls, state: State,
    #                        target_pybullet_fingers: float) -> Action:
    #     pybullet_robot = _get_pybullet_robot()
    #     robots = [r for r in state if r.type.name == "robot"]
    #     assert len(robots) == 1
    #     robot = robots[0]
    #     current_finger_state = state.get(robot, "fingers")
    #     current_finger_joint = PyBulletCoffeeEnv.fingers_state_to_joint(
    #         pybullet_robot, current_finger_state)
    #     assert isinstance(state, utils.PyBulletState)
    #     current_joint_positions = state.joint_positions

    #     return get_change_fingers_action(
    #         pybullet_robot,
    #         current_joint_positions,
    #         current_finger_joint,
    #         target_pybullet_fingers,
    #         CFG.pybullet_max_vel_norm,
    #     )

    # @classmethod
    # def _get_pick_action(cls, state: State) -> Action:
    #     pybullet_robot = _get_pybullet_robot()
    #     return cls._get_finger_action(state, pybullet_robot.closed_fingers)

    # @classmethod
    # def _get_move_action(cls,
    #                      state: State,
    #                      target_pos: Tuple[float, float, float],
    #                      robot_pos: Tuple[float, float, float],
    #                      dtilt: float = 0.0,
    #                      dwrist: float = 0.0,
    #                      finger_status: str = "open") -> Action:
    #     # Determine orientations.
    #     robots = [r for r in state if r.type.name == "robot"]
    #     assert len(robots) == 1
    #     robot = robots[0]
    #     current_joint_positions = state.joint_positions
    #     pybullet_robot = _get_pybullet_robot()

    #     # Early stop
    #     if target_pos == robot_pos and dtilt == 0 and dwrist == 0:
    #         pybullet_robot.set_joints(current_joint_positions)
    #         action_arr = np.array(current_joint_positions, dtype=np.float32)
    #         # action_arr = np.clip(action_arr, pybullet_robot.action_space.low,
    #         #              pybullet_robot.action_space.high)
    #         try:
    #             assert pybullet_robot.action_space.contains(action_arr)
    #         except:
    #             logging.debug(f"action_space: {pybullet_robot.action_space}\n")
    #             logging.debug(f"action arr type: {type(action_arr)}")
    #             logging.debug(f"action arr: {action_arr}")
    #         return Action(action_arr)

    #     current_tilt = state.get(robot, "tilt")
    #     current_wrist = state.get(robot, "wrist")
    #     current_quat = PyBulletCoffeeEnv.tilt_wrist_to_gripper_orn(
    #         current_tilt, current_wrist)
    #     target_quat = PyBulletCoffeeEnv.tilt_wrist_to_gripper_orn(
    #         current_tilt + dtilt, current_wrist + dwrist)
    #     # assert dwrist == 0.0  # temp
    #     current_pose = Pose(robot_pos, current_quat)
    #     target_pose = Pose(target_pos, target_quat)
    #     assert isinstance(state, utils.PyBulletState)

    #     # import pybullet as p
    #     # p.addUserDebugText("+", robot_pos,
    #     #                        [0.0, 1.0, 0.0],
    #     #                      physicsClientId=pybullet_robot.physics_client_id)
    #     # p.addUserDebugText("*", target_pos,
    #     #                        [1.0, 0.0, 0.0],
    #     #                      physicsClientId=pybullet_robot.physics_client_id)

    #     # import time
    #     # time.sleep(1.0)

    #     return get_move_end_effector_to_pose_action(
    #         pybullet_robot, current_joint_positions, current_pose, target_pose,
    #         finger_status, CFG.pybullet_max_vel_norm,
    #         cls._finger_action_nudge_magnitude)