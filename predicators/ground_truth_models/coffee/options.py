"""Ground-truth options for the coffee environment."""

from typing import ClassVar, Dict, Sequence, Set, Tuple
from typing import Type as TypingType
from functools import lru_cache

import numpy as np
from gym.spaces import Box

from predicators.envs.coffee import CoffeeEnv
from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.pybullet_helpers.controllers import get_move_end_effector_to_pose_action, get_change_fingers_action
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.pybullet_helpers.geometry import Pose
from predicators import utils


class CoffeeGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the coffee environment."""

    # Hyperparameters
    twist_policy_tol: ClassVar[float] = 1e-1
    pick_policy_tol: ClassVar[float] = 1e-1
    pour_policy_tol: ClassVar[float] = 1e-1
    env_cls: ClassVar[TypingType[CoffeeEnv]] = CoffeeEnv

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"coffee"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # Types
        robot_type = types["robot"]
        jug_type = types["jug"]
        machine_type = types["machine"]
        cup_type = types["cup"]

        # Predicates
        Twisting = predicates["Twisting"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        JugInMachine = predicates["JugInMachine"]
        MachineOn = predicates["MachineOn"]
        CupFilled = predicates["CupFilled"]

        # MoveToTwistJug
        def _MoveToTwistJug_terminal(state: State, memory: Dict,
                                     objects: Sequence[Object],
                                     params: Array) -> bool:
            del memory, params  # unused
            robot, jug = objects
            return Twisting.holds(state, [robot, jug])

        MoveToTwistJug = ParameterizedOption(
            "MoveToTwistJug",
            types=[robot_type, jug_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_move_to_twist_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_MoveToTwistJug_terminal,
        )

        # TwistJug
        def _TwistJug_terminal(state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> bool:
            del memory, params  # unused
            robot, _ = objects
            return HandEmpty.holds(state, [robot])

        TwistJug = ParameterizedOption(
            "TwistJug",
            types=[robot_type, jug_type],
            # The parameter is a normalized amount to twist by.
            params_space=Box(-1, 1, (1, )),
            policy=cls._create_twist_jug_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_TwistJug_terminal,
        )

        # PickJug
        def _PickJug_terminal(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> bool:
            del memory, params  # unused
            robot, jug = objects
            return Holding.holds(state, [robot, jug])

        PickJug = ParameterizedOption(
            "PickJug",
            types=[robot_type, jug_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_pick_jug_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_PickJug_terminal,
        )

        # PlaceJugInMachine
        def _PlaceJugInMachine_terminal(state: State, memory: Dict,
                                        objects: Sequence[Object],
                                        params: Array) -> bool:
            del memory, params  # unused
            robot, jug, machine = objects
            return not Holding.holds(state, [robot, jug]) and \
                JugInMachine.holds(state, [jug, machine])

        PlaceJugInMachine = ParameterizedOption(
            "PlaceJugInMachine",
            types=[robot_type, jug_type, machine_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_place_jug_in_machine_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_PlaceJugInMachine_terminal,
        )

        # TurnMachineOn
        def _TurnMachineOn_terminal(state: State, memory: Dict,
                                    objects: Sequence[Object],
                                    params: Array) -> bool:
            del memory, params  # unused
            _, machine = objects
            return MachineOn.holds(state, [machine])

        TurnMachineOn = ParameterizedOption(
            "TurnMachineOn",
            types=[robot_type, machine_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_turn_machine_on_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_TurnMachineOn_terminal,
        )

        # Pour
        def _Pour_terminal(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            _, _, cup = objects
            return CupFilled.holds(state, [cup])

        Pour = ParameterizedOption(
            "Pour",
            types=[robot_type, jug_type, cup_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_pour_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_Pour_terminal,
        )

        return {
            TwistJug, PickJug, PlaceJugInMachine, TurnMachineOn, Pour,
            MoveToTwistJug
        }

    @classmethod
    def _create_move_to_twist_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # This policy moves the robot to above the jug, then moves down.
            del memory, params  # unused
            robot, jug = objects
            x = state.get(robot, "x")
            y = state.get(robot, "y")
            z = state.get(robot, "z")
            robot_pos = (x, y, z)
            jug_x = state.get(jug, "x")
            jug_y = state.get(jug, "y")
            jug_z = cls.env_cls.jug_height
            jug_top = (jug_x, jug_y, jug_z)
            xy_sq_dist = (jug_x - x)**2 + (jug_y - y)**2
            # If at the correct x and y position, move directly toward the
            # target.
            if xy_sq_dist < cls.twist_policy_tol:
                return cls._get_move_action(state, jug_top, robot_pos)
            # Move to the position above the jug.
            return cls._get_move_action(state, (jug_x, jug_y, cls.env_cls.robot_init_z),
                                        robot_pos)

        return policy

    @classmethod
    def _create_twist_jug_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # This policy twists until the jug is in the desired rotation, and
            # then moves up to break contact with the jug.
            del memory  # unused
            robot, jug = objects
            current_rot = state.get(jug, "rot")
            norm_desired_rot, = params
            desired_rot = norm_desired_rot * CFG.coffee_jug_init_rot_amt
            delta_rot = np.clip(desired_rot - current_rot,
                                -cls.env_cls.max_angular_vel,
                                cls.env_cls.max_angular_vel)
            if abs(delta_rot) < cls.twist_policy_tol:
                # Move up to stop twisting.
                x = state.get(robot, "x")
                y = state.get(robot, "y")
                z = state.get(robot, "z")
                robot_pos = (x, y, z)
                return cls._get_move_action(state, (x, y, cls.env_cls.robot_init_z),
                                            robot_pos)
            dtwist = delta_rot / cls.env_cls.max_angular_vel
            return cls._get_twist_action(dtwist)

        return policy

    @classmethod
    def _create_pick_jug_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # This policy moves the robot to a safe height, then moves to behind
            # the handle in the y direction, then moves down in the z direction,
            # then moves forward in the y direction before finally grasping.
            del memory, params  # unused
            robot, jug = objects
            x = state.get(robot, "x")
            y = state.get(robot, "y")
            z = state.get(robot, "z")
            robot_pos = (x, y, z)
            handle_pos = cls._get_jug_handle_grasp(state, jug)
            # If close enough, pick.
            sq_dist_to_handle = np.sum(np.subtract(handle_pos, robot_pos)**2)
            if sq_dist_to_handle < cls.pick_policy_tol:
                return cls._get_pick_action(state)
            target_x, target_y, target_z = handle_pos
            # Distance to the handle in the x/z plane.
            xz_handle_sq_dist = (target_x - x)**2 + (target_z - z)**2
            # Distance to the penultimate waypoint in the x/y plane.
            waypoint_y = target_y - cls.env_cls.pick_jug_y_padding
            # Distance in the z direction to a safe move distance.
            safe_z_sq_dist = (cls.env_cls.robot_init_z - z)**2
            xy_waypoint_sq_dist = (target_x - x)**2 + (waypoint_y - y)**2
            # If at the correct x and z position and behind in the y direction,
            # move directly toward the target.
            if target_y > y and xz_handle_sq_dist < cls.pick_policy_tol:
                print("MOVE DIRECTLY TOWARD THE TARGET")
                return cls._get_move_action(state, handle_pos, robot_pos)
            # If close enough to the penultimate waypoint in the x/y plane,
            # move to the waypoint (in the z direction).
            if xy_waypoint_sq_dist < cls.pick_policy_tol:
                print("MOVE TO PENULTIMATE WAYPOINT")
                return cls._get_move_action(state, (target_x, waypoint_y, target_z),
                                            robot_pos)
            # If at a safe height, move to the position above the penultimate
            # waypoint, still at a safe height.
            if safe_z_sq_dist < cls.env_cls.safe_z_tol:
                print("MOVE ABOVE PENULTIMATE WAYPOINT")
                return cls._get_move_action(state, 
                    (target_x, waypoint_y, cls.env_cls.robot_init_z), robot_pos)
            # Move up to a safe height.
            print("MOVE TO SAFE HEIGHT")
            return cls._get_move_action(state, (x, y, cls.env_cls.robot_init_z),
                                        robot_pos)

        return policy

    @classmethod
    def _create_place_jug_in_machine_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # This policy picks the jug up slightly above the table to avoid
            # worrying about friction, then moves directly to the place
            # position, then places the jug.
            del memory, params  # unused
            robot, jug, _ = objects
            # Use the jug position as the origin.
            x = state.get(jug, "x")
            y = state.get(jug, "y")
            z = state.get(robot, "z") - cls.env_cls.jug_handle_height
            jug_pos = (x, y, z)
            place_pos = (cls.env_cls.dispense_area_x, cls.env_cls.dispense_area_y,
                         cls.env_cls.z_lb)
            # If close enough, place.
            sq_dist_to_place = np.sum(np.subtract(jug_pos, place_pos)**2)
            if sq_dist_to_place < cls.env_cls.place_jug_in_machine_tol:
                return cls._get_place_action()
            # If already above the table, move directly toward the place pos.
            if z > cls.env_cls.z_lb:
                return cls._get_move_action(state, place_pos, jug_pos)
            # Move up.
            return cls._get_move_action(state, (x, y, z + cls.env_cls.max_position_vel),
                                        jug_pos)

        return policy

    @classmethod
    def _create_turn_machine_on_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # This policy moves the robot up to be level with the button in the
            # z direction and then moves forward in the y direction to press it.
            del memory, params  # unused
            robot, _ = objects
            x = state.get(robot, "x")
            y = state.get(robot, "y")
            z = state.get(robot, "z")
            robot_pos = (x, y, z)
            button_pos = (cls.env_cls.button_x, cls.env_cls.button_y,
                          cls.env_cls.button_z)
            if (cls.env_cls.button_z - z)**2 < cls.env_cls.button_radius**2:
                # Move directly toward the button.
                return cls._get_move_action(state, button_pos, robot_pos)
            # Move only in the z direction.
            return cls._get_move_action(state, (x, y, cls.env_cls.button_z), robot_pos)

        return policy

    @classmethod
    def _create_pour_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # This policy moves the robot next to the cup and then pours until
            # the cup is filled. Note that if starting out at the end of another
            # pour, we need to start by rotating the cup to prevent any further
            # pouring until we've moved over the next cup.
            del memory, params  # unused
            move_tilt = cls.env_cls.tilt_lb
            pour_tilt = cls.env_cls.tilt_ub
            robot, jug, cup = objects
            robot_x = state.get(robot, "x")
            robot_y = state.get(robot, "y")
            robot_z = state.get(robot, "z")
            robot_pos = (robot_x, robot_y, robot_z)
            tilt = state.get(robot, "tilt")
            jug_x = state.get(jug, "x")
            jug_y = state.get(jug, "y")
            jug_z = cls._get_jug_z(state, robot, jug)
            jug_pos = (jug_x, jug_y, jug_z)
            pour_x, pour_y, _ = pour_pos = cls._get_pour_position(state, cup)
            # If we're close enough to the pour position, pour.
            sq_dist_to_pour = np.sum(np.subtract(jug_pos, pour_pos)**2)
            if sq_dist_to_pour < cls.pour_policy_tol:
                dtilt = pour_tilt - tilt
                return cls._get_move_action(state, jug_pos, jug_pos, dtilt=dtilt)
            dtilt = move_tilt - tilt
            # If we're above the pour position, move down to pour.
            xy_pour_sq_dist = (jug_x - pour_x)**2 + (jug_y - pour_y)**2
            if xy_pour_sq_dist < cls.env_cls.safe_z_tol:
                return cls._get_move_action(state, pour_pos, jug_pos, dtilt=dtilt)
            # If we're at a safe height, move toward above the pour position.
            if (robot_z - cls.env_cls.robot_init_z)**2 < cls.env_cls.safe_z_tol:
                return cls._get_move_action(state, (pour_x, pour_y, jug_z),
                                            jug_pos,
                                            dtilt=dtilt)
            # Move to a safe moving height.
            return cls._get_move_action(state, 
                (robot_x, robot_y, cls.env_cls.robot_init_z),
                robot_pos,
                dtilt=dtilt)

        return policy

    ############################ Utility functions ############################

    @classmethod
    def _get_move_action(cls,
                         state: State,
                         target_pos: Tuple[float, float, float],
                         robot_pos: Tuple[float, float, float],
                         dtilt: float = 0.0,
                         dwrist: float = 0.0) -> Action:
        del state  # used in PyBullet subclass
        # We want to move in this direction.
        delta = np.subtract(target_pos, robot_pos)
        # But we can only move at most max_position_vel in one step.
        # Get the norm full move delta.
        pos_norm = float(np.linalg.norm(delta))
        # If the norm is more than max_position_vel, rescale the delta so
        # that its norm is max_position_vel.
        if pos_norm > cls.env_cls.max_position_vel:
            delta = cls.env_cls.max_position_vel * (delta / pos_norm)
            pos_norm = cls.env_cls.max_position_vel
        # Now normalize so that the action values are between -1 and 1, as
        # expected by simulate and the action space.
        if pos_norm > 0:
            delta = delta / cls.env_cls.max_position_vel
        dx, dy, dz = delta
        dtilt = np.clip(dtilt, -cls.env_cls.max_angular_vel,
                        cls.env_cls.max_angular_vel)
        dtilt = dtilt / cls.env_cls.max_angular_vel
        return Action(
            np.array([dx, dy, dz, dtilt, dwrist, 0.0], dtype=np.float32))
    
    @classmethod
    def _get_twist_action(cls, dtwist: float) -> Action:
        return Action(
                np.array([0.0, 0.0, 0.0, 0.0, dtwist, 0.0], dtype=np.float32))
    
    @classmethod
    def _get_pick_action(cls, state: State) -> Action:
        del state  # used by PyBullet subclass
        return Action(
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                             dtype=np.float32))

    @classmethod
    def _get_place_action(cls) -> Action:
        return Action(
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    @classmethod
    def _get_jug_handle_grasp(cls, state: State,
                              jug: Object) -> Tuple[float, float, float]:
        # Hack to avoid duplicate code.
        return cls.env_cls._get_jug_handle_grasp(state, jug)  # pylint: disable=protected-access

    @classmethod
    def _get_jug_z(cls, state: State, robot: Object, jug: Object) -> float:
        assert state.get(jug, "is_held") > 0.5
        # Offset to account for handle.
        return state.get(robot, "z") - cls.env_cls.jug_handle_height

    @classmethod
    def _get_pour_position(cls, state: State,
                           cup: Object) -> Tuple[float, float, float]:
        target_x = state.get(cup, "x") + cls.env_cls.pour_x_offset
        target_y = state.get(cup, "y") + cls.env_cls.pour_y_offset
        target_z = cls.env_cls.pour_z_offset
        return (target_x, target_y, target_z)


@lru_cache
def _get_pybullet_robot() -> SingleArmPyBulletRobot:
    _, pybullet_robot, _ = \
        PyBulletCoffeeEnv.initialize_pybullet(using_gui=False)  # TODO turn off
    return pybullet_robot


class PyBulletCoffeeGroundTruthOptionFactory(CoffeeGroundTruthOptionFactory):
    """Ground-truth options for the pybullet_coffee environment."""

    env_cls: ClassVar[TypingType[CoffeeEnv]] = PyBulletCoffeeEnv
    twist_policy_tol: ClassVar[float] = 1e-3
    pick_policy_tol: ClassVar[float] = 1e-3
    pour_policy_tol: ClassVar[float] = 1e-3
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_coffee"}
    
    @classmethod
    def _get_move_action(cls,
                         state: State,
                         target_pos: Tuple[float, float, float],
                         robot_pos: Tuple[float, float, float],
                         dtilt: float = 0.0,
                         dwrist: float = 0.0) -> Action:
        pybullet_robot = _get_pybullet_robot()

        # Determine orientations.
        robots = [r for r in state if r.type.name == "robot"]
        assert len(robots) == 1
        robot = robots[0]
        current_tilt = state.get(robot, "tilt")
        current_wrist = state.get(robot, "wrist")
        current_quat = PyBulletCoffeeEnv.tilt_wrist_to_gripper_orn(
            current_tilt, current_wrist
        )
        target_quat = PyBulletCoffeeEnv.tilt_wrist_to_gripper_orn(
            current_tilt + dtilt, current_wrist + dwrist
        )
        assert dtilt == 0.0 # temp
        assert dwrist == 0.0  # temp
        current_pose = Pose(robot_pos, current_quat)
        target_pose = Pose(target_pos, target_quat)
        finger_status = "open"  # TODO
        assert isinstance(state, utils.PyBulletState)
        current_joint_positions = state.joint_positions

        # import pybullet as p
        # p.addUserDebugText("+", robot_pos,
        #                        [0.0, 1.0, 0.0],
        #                        physicsClientId=pybullet_robot.physics_client_id)
        # p.addUserDebugText("*", target_pos,
        #                        [1.0, 0.0, 0.0],
        #                        physicsClientId=pybullet_robot.physics_client_id)
        
        # import time
        # time.sleep(1.0)

        return get_move_end_effector_to_pose_action(pybullet_robot, current_joint_positions, current_pose, target_pose,
                                                    finger_status,
                                                    CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude)

    @classmethod
    def _get_twist_action(cls, dtwist: float) -> Action:
        import ipdb; ipdb.set_trace()
    
    @classmethod
    def _get_pick_action(cls, state: State) -> Action:
        pybullet_robot = _get_pybullet_robot()
        robots = [r for r in state if r.type.name == "robot"]
        assert len(robots) == 1
        robot = robots[0]
        current_finger_state = state.get(robot, "fingers")
        current_finger_joint = PyBulletCoffeeEnv.fingers_state_to_joint(
            pybullet_robot, current_finger_state
        )
        assert isinstance(state, utils.PyBulletState)
        current_joint_positions = state.joint_positions

        return get_change_fingers_action(
            pybullet_robot,
            current_joint_positions,
            current_finger_joint,
            pybullet_robot.closed_fingers,
            CFG.pybullet_max_vel_norm,
        )

    @classmethod
    def _get_place_action(cls) -> Action:
        import ipdb; ipdb.set_trace()


# class PyBulletCoffeeGroundTruthOptionFactory(GroundTruthOptionFactory):
#     """Ground-truth options for the pybullet_coffee environment."""

#     @classmethod
#     def get_env_names(cls) -> Set[str]:
#         return {"pybullet_coffee"}

#     @classmethod
#     def get_options(cls, env_name: str, types: Dict[str, Type],
#                     predicates: Dict[str, Predicate],
#                     action_space: Box) -> Set[ParameterizedOption]:

#         # The options are the same as in the regular coffee environment, except
#         # for the policies.
#         coffee_options = CoffeeGroundTruthOptionFactory.get_options(
#             "coffee", types, predicates, action_space)
#         coffee_option_name_to_option = {o.name: o for o in coffee_options}

#         # MoveToTwistJug
#         coffee_MoveToTwistJug = coffee_option_name_to_option["MoveToTwistJug"]
#         MoveToTwistJug = ParameterizedOption(
#             coffee_MoveToTwistJug.name,
#             types=coffee_MoveToTwistJug.types,
#             params_space=coffee_MoveToTwistJug.params_space,
#             policy=cls._create_move_to_twist_policy(),
#             initiable=coffee_MoveToTwistJug.initiable,
#             terminal=coffee_MoveToTwistJug.terminal)

#         # TwistJug
#         coffee_TwistJug = coffee_option_name_to_option["TwistJug"]
#         TwistJug = ParameterizedOption(
#             coffee_TwistJug.name,
#             types=coffee_TwistJug.types,
#             params_space=coffee_TwistJug.params_space,
#             policy=cls._create_twist_jug_policy(),
#             initiable=coffee_TwistJug.initiable,
#             terminal=coffee_TwistJug.terminal,
#         )

#         # PickJug
#         coffee_PickJug = coffee_option_name_to_option["PickJug"]
#         PickJug = ParameterizedOption(
#             coffee_PickJug.name,
#             types=coffee_PickJug.types,
#             params_space=coffee_PickJug.params_space,
#             policy=cls._create_pick_jug_policy(),
#             initiable=coffee_PickJug.initiable,
#             terminal=coffee_PickJug.terminal,
#         )

#         # PlaceJugInMachine
#         coffee_PlaceJugInMachine = \
#             coffee_option_name_to_option["PlaceJugInMachine"]
#         PlaceJugInMachine = ParameterizedOption(
#             coffee_PlaceJugInMachine.name,
#             types=coffee_PlaceJugInMachine.types,
#             params_space=coffee_PlaceJugInMachine.params_space,
#             policy=cls._create_place_jug_in_machine_policy(),
#             initiable=coffee_PlaceJugInMachine.initiable,
#             terminal=coffee_PlaceJugInMachine.terminal,
#         )

#         # TurnMachineOn
#         coffee_TurnMachineOn = coffee_option_name_to_option["TurnMachineOn"]
#         TurnMachineOn = ParameterizedOption(
#             coffee_TurnMachineOn.name,
#             types=coffee_TurnMachineOn.types,
#             params_space=coffee_TurnMachineOn.params_space,
#             policy=cls._create_turn_machine_on_policy(),
#             initiable=coffee_TurnMachineOn.initiable,
#             terminal=coffee_TurnMachineOn.terminal,
#         )

#         # Pour
#         coffee_Pour = coffee_option_name_to_option["Pour"]
#         Pour = ParameterizedOption(
#             coffee_Pour.name,
#             types=coffee_Pour.types,
#             params_space=coffee_Pour.params_space,
#             policy=cls._create_pour_policy(),
#             initiable=coffee_Pour.initiable,
#             terminal=coffee_Pour.terminal,
#         )

#         return {
#             TwistJug, PickJug, PlaceJugInMachine, TurnMachineOn, Pour,
#             MoveToTwistJug
#         }

#     @classmethod
#     def _create_move_to_twist_policy(cls) -> ParameterizedPolicy:

#         def policy(state: State, memory: Dict, objects: Sequence[Object],
#                    params: Array) -> Action:
#             import ipdb
#             ipdb.set_trace()

#         return policy

#     @classmethod
#     def _create_twist_jug_policy(cls) -> ParameterizedPolicy:

#         def policy(state: State, memory: Dict, objects: Sequence[Object],
#                    params: Array) -> Action:

#             import ipdb
#             ipdb.set_trace()

#         return policy

#     @classmethod
#     def _create_pick_jug_policy(cls) -> ParameterizedPolicy:

#         def policy(state: State, memory: Dict, objects: Sequence[Object],
#                    params: Array) -> Action:
#             import ipdb
#             ipdb.set_trace()

#         return policy

#     @classmethod
#     def _create_place_jug_in_machine_policy(cls) -> ParameterizedPolicy:

#         def policy(state: State, memory: Dict, objects: Sequence[Object],
#                    params: Array) -> Action:
#             import ipdb
#             ipdb.set_trace()

#         return policy

#     @classmethod
#     def _create_turn_machine_on_policy(cls) -> ParameterizedPolicy:

#         def policy(state: State, memory: Dict, objects: Sequence[Object],
#                    params: Array) -> Action:
#             import ipdb
#             ipdb.set_trace()

#         return policy

#     @classmethod
#     def _create_pour_policy(cls) -> ParameterizedPolicy:

#         def policy(state: State, memory: Dict, objects: Sequence[Object],
#                    params: Array) -> Action:
#             import ipdb
#             ipdb.set_trace()

#         return policy
