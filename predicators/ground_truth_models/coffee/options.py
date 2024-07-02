"""Ground-truth options for the coffee environment."""

import logging
from functools import lru_cache
from typing import ClassVar, Dict, Sequence, Set, Tuple
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
    ParameterizedPolicy, Predicate, State, Type


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
            params_space=Box(-1, 1, (1 if CFG.coffee_twist_sampler else 0, )),
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
            holds = Holding.holds(state, [robot, jug])
            return holds

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
        def _Pour_initiable(state: State, memory: Dict,
                            objects: Sequence[Object], params: Array) -> bool:
            del memory, params
            robot, jug, _ = objects
            return Holding.holds(state, [robot, jug])

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
            initiable=_Pour_initiable,
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
            return cls._get_move_action(
                state, (jug_x, jug_y, cls.env_cls.robot_init_z), robot_pos)

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
            norm_desired_rot = params[0] if params.shape[0] == 1 else 0.0
            desired_rot = norm_desired_rot * CFG.coffee_jug_init_rot_amt
            delta_rot = np.clip(desired_rot - current_rot,
                                -cls.env_cls.max_angular_vel,
                                cls.env_cls.max_angular_vel)
            x = state.get(robot, "x")
            y = state.get(robot, "y")
            z = state.get(robot, "z")
            robot_pos = (x, y, z)
            if abs(delta_rot) < cls.twist_policy_tol:
                # Move up to stop twisting.
                return cls._get_move_action(state,
                                            (x, y, cls.env_cls.robot_init_z),
                                            robot_pos)
            dtwist = delta_rot / cls.env_cls.max_angular_vel
            return cls._get_twist_action(state, robot_pos, dtwist)

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
                return cls._get_move_action(state, handle_pos, robot_pos)
            # If close enough to the penultimate waypoint in the x/y plane,
            # move to the waypoint (in the z direction).
            if xy_waypoint_sq_dist < cls.pick_policy_tol:
                return cls._get_move_action(state,
                                            (target_x, waypoint_y, target_z),
                                            robot_pos)
            # If at a safe height, move to the position above the penultimate
            # waypoint, still at a safe height.
            if safe_z_sq_dist < cls.env_cls.safe_z_tol:
                return cls._get_move_action(
                    state, (target_x, waypoint_y, cls.env_cls.robot_init_z),
                    robot_pos)
            # Move up to a safe height.
            return cls._get_move_action(state,
                                        (x, y, cls.env_cls.robot_init_z),
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
            place_pos = (cls.env_cls.dispense_area_x,
                         cls.env_cls.dispense_area_y, cls.env_cls.z_lb)
            # If close enough, place.
            sq_dist_to_place = np.sum(np.subtract(jug_pos, place_pos)**2)
            if sq_dist_to_place < cls.env_cls.place_jug_in_machine_tol:
                return Action(
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            # If already above the table, move directly toward the place pos.
            if z > cls.env_cls.z_lb:
                return cls._get_move_action(state, place_pos, jug_pos)
            # Move up.
            return cls._get_move_action(
                state, (x, y, z + cls.env_cls.max_position_vel), jug_pos)

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
            return cls._get_move_action(state, (x, y, cls.env_cls.button_z),
                                        robot_pos)

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
                return cls._get_move_action(state,
                                            jug_pos,
                                            jug_pos,
                                            dtilt=dtilt)
            dtilt = move_tilt - tilt
            # If we're above the pour position, move down to pour.
            xy_pour_sq_dist = (jug_x - pour_x)**2 + (jug_y - pour_y)**2
            if xy_pour_sq_dist < cls.env_cls.safe_z_tol:
                return cls._get_move_action(state,
                                            pour_pos,
                                            jug_pos,
                                            dtilt=dtilt)
            # If we're at a safe height, move toward above the pour position.
            if (robot_z -
                    cls.env_cls.robot_init_z)**2 < cls.env_cls.safe_z_tol:
                return cls._get_move_action(state, (pour_x, pour_y, jug_z),
                                            jug_pos,
                                            dtilt=dtilt)
            # Move to a safe moving height.
            return cls._get_move_action(
                state, (robot_x, robot_y, cls.env_cls.robot_init_z),
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
                         dwrist: float = 0.0,
                         finger_status: str = "open") -> Action:
        del state, finger_status  # used in PyBullet subclass
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
    def _get_twist_action(cls, state: State, cur_robot_pos: Tuple[float, float,
                                                                  float],
                          dtwist: float) -> Action:
        del state, cur_robot_pos  # used by PyBullet subclass
        return Action(
            np.array([0.0, 0.0, 0.0, 0.0, dtwist, 0.0], dtype=np.float32))

    @classmethod
    def _get_pick_action(cls, state: State) -> Action:
        del state  # used by PyBullet subclass
        return Action(
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32))

    @classmethod
    def _get_place_action(cls, state: State) -> Action:
        del state  # used by PyBullet subclass
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
        target_z = cls.env_cls.z_lb + cls.env_cls.pour_z_offset
        return (target_x, target_y, target_z)


@lru_cache
def _get_pybullet_robot() -> SingleArmPyBulletRobot:
    _, pybullet_robot, _ = \
        PyBulletCoffeeEnv.initialize_pybullet(using_gui=False)
    return pybullet_robot


class PyBulletCoffeeGroundTruthOptionFactory(CoffeeGroundTruthOptionFactory):
    """Ground-truth options for the pybullet_coffee environment."""

    env_cls: ClassVar[TypingType[CoffeeEnv]] = PyBulletCoffeeEnv
    # twist_policy_tol: ClassVar[float] = 1e-3
    pick_policy_tol: ClassVar[float] = 1e-3
    pour_policy_tol: ClassVar[float] = 1e-3
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_coffee"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        options = super().get_options(env_name, types, predicates,
                                      action_space)

        robot_type = types["robot"]
        jug_type = types["jug"]

        # TwistJug
        def _TwistJug_terminal(state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> bool:
            del memory, params  # unused
            robot, jug = objects
            # return HandEmpty.holds(state, [robot])
            # modify to stop at the beginning state
            robot_pose = [
                state.get(robot, "x"),
                state.get(robot, "y"),
                state.get(robot, "z"),
            ]
            robot_wrist = state.get(robot, "wrist")
            robot_tilt = state.get(robot, "tilt")
            robot_finger = state.get(robot, "fingers")
            return np.allclose(robot_pose, [cls.env_cls.robot_init_x,
                                            cls.env_cls.robot_init_y,
                                            cls.env_cls.robot_init_z],
                                            atol=1e-2) and \
                   np.allclose([robot_wrist, robot_tilt, robot_finger],
                               [cls.env_cls.robot_init_wrist,
                                cls.env_cls.robot_init_tilt,
                                cls.env_cls.open_fingers],
                                            atol=1e-2)

        TwistJug = ParameterizedOption(
            "TwistJug",
            types=[robot_type, jug_type],
            # The parameter is a normalized amount to twist by.
            params_space=Box(-1, 1, (1 if CFG.coffee_twist_sampler else
                                     0, )),  # temp; originally 1
            policy=cls._create_twist_jug_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_TwistJug_terminal,
        )
        options.remove(TwistJug)
        options.add(TwistJug)
        return options

    @classmethod
    def _create_twist_jug_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # This policy twists until the jug is in the desired rotation, and
            # then moves up to break contact with the jug.
            del memory  # unused
            robot, jug = objects
            current_rot = state.get(jug, "rot")
            # norm_desired_rot, = params
            norm_desired_rot = params[0] if params.shape[0] == 1 else 0.0
            desired_rot = norm_desired_rot * CFG.coffee_jug_init_rot_amt
            delta_rot = np.clip(desired_rot - current_rot,
                                -cls.env_cls.max_angular_vel,
                                cls.env_cls.max_angular_vel)
            x = state.get(robot, "x")
            y = state.get(robot, "y")
            z = state.get(robot, "z")
            robot_pos = (x, y, z)
            jug_x = state.get(jug, "x")
            jug_y = state.get(jug, "y")
            jug_top = (jug_x, jug_y, cls.env_cls.jug_height)
            # print("[Taking a new twist action]")
            # print(f"[policy] desired jug rot {desired_rot:.3f}")
            # print(f"[policy] current jug rot {current_rot:.3f}")
            # print(f"[policy] delta jug rot {delta_rot:.3f} -- the policy wants "\
            #       "to move the jug by this amount")

            # current_ee_rpy = _get_pybullet_robot().forward_kinematics(
            #     state.joint_positions).rpy
            # print(f"[policy] current ee rpy {current_ee_rpy}")
            if abs(delta_rot) < cls.twist_policy_tol:
                # print(f"Moving up")
                # Rotate the ee back to init after not in the twisting position
                sq_dist_to_jug_top = np.sum(np.subtract(jug_top, (x, y, z))**2)
                if sq_dist_to_jug_top > cls.env_cls.grasp_position_tol:
                    dwrist = cls.env_cls.robot_init_wrist - state.get(
                        robot, "wrist")
                    dtilt = cls.env_cls.robot_init_tilt - state.get(
                        robot, "tilt")
                else:
                    dtilt = 0.0
                    dwrist = 0.0

                # Move up to stop twisting.
                return cls._get_move_action(
                    state, (cls.env_cls.robot_init_x, cls.env_cls.robot_init_y,
                            cls.env_cls.robot_init_z), robot_pos, dtilt,
                    dwrist)
            dtwist = delta_rot / cls.env_cls.max_angular_vel
            new_joint_pos = cls._get_twist_action(state, robot_pos, dtwist)
            # new_ee_rpy = _get_pybullet_robot().forward_kinematics(
            #     new_joint_pos.arr.tolist()).rpy
            # print(f"[policy] new ee rpy {new_ee_rpy}")
            # print(f"[policy] d_roll {-(new_ee_rpy[0] - current_ee_rpy[0]):.3f}")
            # breakpoint()
            return new_joint_pos

        return policy

    @classmethod
    def _create_place_jug_in_machine_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # This policy moves directly to place the jug.
            del memory, params  # unused
            robot, jug, _ = objects

            # Get the current robot position.
            x = state.get(robot, "x")
            y = state.get(robot, "y")
            z = state.get(robot, "z")
            robot_pos = (x, y, z)

            # Get the difference between the jug location and the target.
            # Use the jug position as the origin.
            jx = state.get(jug, "x")
            jy = state.get(jug, "y")
            jz = cls.env_cls.z_lb + cls.env_cls.jug_height
            current_jug_pos = (jx, jy, jz)
            target_jug_pos = (cls.env_cls.dispense_area_x,
                              cls.env_cls.dispense_area_y,
                              cls.env_cls.z_lb + cls.env_cls.jug_handle_height)
            dx, dy, dz = np.subtract(target_jug_pos, current_jug_pos)

            # Get the target robot position.
            target_robot_pos = (x + dx, y + dy, z + dz)
            # If close enough, place.
            sq_dist_to_place = np.sum(
                np.subtract(robot_pos, target_robot_pos)**2)
            if sq_dist_to_place < cls.env_cls.place_jug_in_machine_tol:
                return cls._get_place_action(state)
            # If already above the table, move directly toward the place pos.
            return cls._get_move_action(state,
                                        target_robot_pos,
                                        robot_pos,
                                        finger_status="closed")

        return policy

    @classmethod
    def _create_pour_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # This policy moves the robot next to the cup and then pours until
            # the cup is filled. Note that if starting out at the end of another
            # pour, we need to start by rotating the jug to prevent any further
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
            dx, dy, dz = np.subtract(pour_pos, jug_pos)
            # Get the target robot position.
            robot_pour_pos = (robot_x + dx, robot_y + dy, robot_z + dz)
            # If we're close enough to the pour position, pour.
            sq_dist_to_pour = np.sum(np.subtract(jug_pos, pour_pos)**2)
            if sq_dist_to_pour < cls.pour_policy_tol:
                dtilt = pour_tilt - tilt
                if abs(dtilt) < cls.env_cls.pour_angle_tol * 0.1:
                    # make pouring more stable
                    dtilt = 0
                # logging.debug(f"Pour: dtils {dtilt}")
                # current_ee_rpy = _get_pybullet_robot().forward_kinematics(
                #     state.joint_positions).rpy
                # cur_formated_jp = np.array(
                #     [round(jp, 3) for jp in state.joint_positions])
                # current_ee_rpy = tuple(round(v, 3) for v in current_ee_rpy)
                new_joint_pos = cls._get_move_action(state,
                                                     robot_pos,
                                                     robot_pos,
                                                     dtilt=dtilt,
                                                     finger_status="closed")
                # new_ee_rpy = _get_pybullet_robot().forward_kinematics(
                #     new_joint_pos.arr.tolist()).rpy
                # new_ee_rpy = tuple(round(v, 3) for v in new_ee_rpy)
                # new_formated_jp = np.array(
                #     [round(jp, 3) for jp in new_joint_pos.arr])
                # logging.debug(f"[pour] cur joint position {cur_formated_jp}")
                # logging.debug(f"[pour] new joint position {new_formated_jp}")
                # logging.debug(f"[pour] joint delta {np.round(new_formated_jp - cur_formated_jp, 2)}")
                # logging.debug(f"[pour] cur ee rpy {current_ee_rpy}")
                # logging.debug(f"[pour] new ee rpy {new_ee_rpy}")
                # logging.debug(f"[pour] d_roll {new_ee_rpy[0] - current_ee_rpy[0]}")
                return new_joint_pos
            dtilt = move_tilt - tilt
            # If we're above the pour position, move down to pour.
            xy_pour_sq_dist = (jug_x - pour_x)**2 + (jug_y - pour_y)**2
            if xy_pour_sq_dist < cls.env_cls.safe_z_tol * 1e-2:
                # logging.debug("Move down to pour")
                # current_ee_rpy = _get_pybullet_robot().forward_kinematics(
                #     state.joint_positions).rpy
                # current_ee_rpy = tuple(round(v, 3) for v in current_ee_rpy)
                # cur_formated_jp = np.array(
                #     [round(jp, 3) for jp in state.joint_positions])
                new_joint_pos = cls._get_move_action(
                    state,
                    # robot_pour_pos,
                    (robot_x, robot_y, robot_pour_pos[2]),
                    robot_pos,
                    dtilt=0.0,
                    finger_status="closed")
                # new_ee_rpy = _get_pybullet_robot().forward_kinematics(
                #     new_joint_pos.arr.tolist()).rpy
                # new_ee_rpy = tuple(round(v, 3) for v in new_ee_rpy)
                # new_formated_jp = np.array(
                #     [round(jp, 3) for jp in new_joint_pos.arr])
                # logging.debug(f"[move down] cur joint position {cur_formated_jp}")
                # logging.debug(f"[move down] new joint position {new_formated_jp}")
                # logging.debug(f"[move down] joint delta {np.round(new_formated_jp - cur_formated_jp, 2)}")
                # logging.debug(f"[move down] cur ee rpy {current_ee_rpy}"\
                #                 " -- roll should be the same as new")
                # logging.debug(f"[move down] new ee rpy {new_ee_rpy}"\
                #                     " -- roll should be the same as current")
                # logging.debug(f"[move down] d_roll {new_ee_rpy[0] - current_ee_rpy[0]}")
                return new_joint_pos
            # If we're at a safe height, move toward above the pour position.
            if (robot_z -
                    cls.env_cls.robot_init_z)**2 < cls.env_cls.safe_z_tol:
                # logging.debug("At a safe height, move towards the pour position")
                return cls._get_move_action(
                    state, (robot_pour_pos[0], robot_pour_pos[1], robot_z),
                    robot_pos,
                    dtilt=0.0,
                    finger_status="closed")

            # Move backward and to a safe moving height.
            # logging.debug("Move backward and to a safe moving height")
            return cls._get_move_action(
                state, (robot_x, robot_y - 1e-1, cls.env_cls.robot_init_z),
                robot_pos,
                dtilt=0.0,
                finger_status="closed")

        return policy

    @classmethod
    def _get_move_action(cls,
                         state: State,
                         target_pos: Tuple[float, float, float],
                         robot_pos: Tuple[float, float, float],
                         dtilt: float = 0.0,
                         dwrist: float = 0.0,
                         finger_status: str = "open") -> Action:
        # Determine orientations.
        robots = [r for r in state if r.type.name == "robot"]
        assert len(robots) == 1
        robot = robots[0]
        current_joint_positions = state.joint_positions
        pybullet_robot = _get_pybullet_robot()

        # Early stop
        if target_pos == robot_pos and dtilt == 0 and dwrist == 0:
            pybullet_robot.set_joints(current_joint_positions)
            action_arr = np.array(current_joint_positions, dtype=np.float32)
            # action_arr = np.clip(action_arr, pybullet_robot.action_space.low,
            #              pybullet_robot.action_space.high)
            try:
                assert pybullet_robot.action_space.contains(action_arr)
            except:
                logging.debug(f"action_space: {pybullet_robot.action_space}\n")
                logging.debug(f"action arr type: {type(action_arr)}")
                logging.debug(f"action arr: {action_arr}")
            return Action(action_arr)


        current_tilt = state.get(robot, "tilt")
        current_wrist = state.get(robot, "wrist")
        current_quat = PyBulletCoffeeEnv.tilt_wrist_to_gripper_orn(
            current_tilt, current_wrist)
        target_quat = PyBulletCoffeeEnv.tilt_wrist_to_gripper_orn(
            current_tilt + dtilt, current_wrist + dwrist)
        # assert dwrist == 0.0  # temp
        current_pose = Pose(robot_pos, current_quat)
        target_pose = Pose(target_pos, target_quat)
        assert isinstance(state, utils.PyBulletState)

        # import pybullet as p
        # p.addUserDebugText("+", robot_pos,
        #                        [0.0, 1.0, 0.0],
        #                      physicsClientId=pybullet_robot.physics_client_id)
        # p.addUserDebugText("*", target_pos,
        #                        [1.0, 0.0, 0.0],
        #                      physicsClientId=pybullet_robot.physics_client_id)

        # import time
        # time.sleep(1.0)

        return get_move_end_effector_to_pose_action(
            pybullet_robot, current_joint_positions, current_pose, target_pose,
            finger_status, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude)

    @classmethod
    def _get_twist_action(cls, state: State, cur_robot_pos: Tuple[float, float,
                                                                  float],
                          dtwist: float) -> Action:
        delta_rot = dtwist * cls.env_cls.max_angular_vel
        return cls._get_move_action(state, cur_robot_pos, cur_robot_pos, 0.0,
                                    delta_rot)

    @classmethod
    def _get_finger_action(cls, state: State,
                           target_pybullet_fingers: float) -> Action:
        pybullet_robot = _get_pybullet_robot()
        robots = [r for r in state if r.type.name == "robot"]
        assert len(robots) == 1
        robot = robots[0]
        current_finger_state = state.get(robot, "fingers")
        current_finger_joint = PyBulletCoffeeEnv.fingers_state_to_joint(
            pybullet_robot, current_finger_state)
        assert isinstance(state, utils.PyBulletState)
        current_joint_positions = state.joint_positions

        return get_change_fingers_action(
            pybullet_robot,
            current_joint_positions,
            current_finger_joint,
            target_pybullet_fingers,
            CFG.pybullet_max_vel_norm,
        )

    @classmethod
    def _get_pick_action(cls, state: State) -> Action:
        pybullet_robot = _get_pybullet_robot()
        return cls._get_finger_action(state, pybullet_robot.closed_fingers)

    @classmethod
    def _get_place_action(cls, state: State) -> Action:
        pybullet_robot = _get_pybullet_robot()
        return cls._get_finger_action(state, pybullet_robot.open_fingers)
