"""Ground-truth options for the coffee environment."""

from typing import ClassVar, Dict, Sequence, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators.envs.coffee import CoffeeEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type
from predicators import utils


class CoffeeGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the coffee environment."""

    # Hyperparameters
    twist_policy_tol: ClassVar[float] = 1e-1
    pick_policy_tol: ClassVar[float] = 1e-1
    pour_policy_tol: ClassVar[float] = 1e-1

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
            jug_z = CoffeeEnv.jug_height
            jug_top = (jug_x, jug_y, jug_z)
            xy_sq_dist = (jug_x - x)**2 + (jug_y - y)**2
            # If at the correct x and y position, move directly toward the
            # target.
            if xy_sq_dist < cls.twist_policy_tol:
                return cls._get_move_action(jug_top, robot_pos)
            # Move to the position above the jug.
            return cls._get_move_action((jug_x, jug_y, CoffeeEnv.robot_init_z),
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
                                -CoffeeEnv.max_angular_vel,
                                CoffeeEnv.max_angular_vel)
            if abs(delta_rot) < cls.twist_policy_tol:
                # Move up to stop twisting.
                x = state.get(robot, "x")
                y = state.get(robot, "y")
                z = state.get(robot, "z")
                robot_pos = (x, y, z)
                return cls._get_move_action((x, y, CoffeeEnv.robot_init_z),
                                            robot_pos)
            dtwist = delta_rot / CoffeeEnv.max_angular_vel
            return Action(
                np.array([0.0, 0.0, 0.0, 0.0, dtwist, 0.0], dtype=np.float32))

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
                return Action(
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                             dtype=np.float32))
            target_x, target_y, target_z = handle_pos
            # Distance to the handle in the x/z plane.
            xz_handle_sq_dist = (target_x - x)**2 + (target_z - z)**2
            # Distance to the penultimate waypoint in the x/y plane.
            waypoint_y = target_y - CoffeeEnv.pick_jug_y_padding
            # Distance in the z direction to a safe move distance.
            safe_z_sq_dist = (CoffeeEnv.robot_init_z - z)**2
            xy_waypoint_sq_dist = (target_x - x)**2 + (waypoint_y - y)**2
            # If at the correct x and z position and behind in the y direction,
            # move directly toward the target.
            if target_y > y and xz_handle_sq_dist < cls.pick_policy_tol:
                return cls._get_move_action(handle_pos, robot_pos)
            # If close enough to the penultimate waypoint in the x/y plane,
            # move to the waypoint (in the z direction).
            if xy_waypoint_sq_dist < cls.pick_policy_tol:
                return cls._get_move_action((target_x, waypoint_y, target_z),
                                            robot_pos)
            # If at a safe height, move to the position above the penultimate
            # waypoint, still at a safe height.
            if safe_z_sq_dist < CoffeeEnv.safe_z_tol:
                return cls._get_move_action(
                    (target_x, waypoint_y, CoffeeEnv.robot_init_z), robot_pos)
            # Move up to a safe height.
            return cls._get_move_action((x, y, CoffeeEnv.robot_init_z),
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
            z = state.get(robot, "z") - CoffeeEnv.jug_handle_height
            jug_pos = (x, y, z)
            place_pos = (CoffeeEnv.dispense_area_x, CoffeeEnv.dispense_area_y,
                         CoffeeEnv.z_lb)
            # If close enough, place.
            sq_dist_to_place = np.sum(np.subtract(jug_pos, place_pos)**2)
            if sq_dist_to_place < CoffeeEnv.place_jug_in_machine_tol:
                return Action(
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            # If already above the table, move directly toward the place pos.
            if z > CoffeeEnv.z_lb:
                return cls._get_move_action(place_pos, jug_pos)
            # Move up.
            return cls._get_move_action((x, y, z + CoffeeEnv.max_position_vel),
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
            button_pos = (CoffeeEnv.button_x, CoffeeEnv.button_y,
                          CoffeeEnv.button_z)
            if (CoffeeEnv.button_z - z)**2 < CoffeeEnv.button_radius**2:
                # Move directly toward the button.
                return cls._get_move_action(button_pos, robot_pos)
            # Move only in the z direction.
            return cls._get_move_action((x, y, CoffeeEnv.button_z), robot_pos)

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
            move_tilt = CoffeeEnv.tilt_lb
            pour_tilt = CoffeeEnv.tilt_ub
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
                return cls._get_move_action(jug_pos, jug_pos, dtilt=dtilt)
            dtilt = move_tilt - tilt
            # If we're above the pour position, move down to pour.
            xy_pour_sq_dist = (jug_x - pour_x)**2 + (jug_y - pour_y)**2
            if xy_pour_sq_dist < CoffeeEnv.safe_z_tol:
                return cls._get_move_action(pour_pos, jug_pos, dtilt=dtilt)
            # If we're at a safe height, move toward above the pour position.
            if (robot_z - CoffeeEnv.robot_init_z)**2 < CoffeeEnv.safe_z_tol:
                return cls._get_move_action((pour_x, pour_y, jug_z),
                                            jug_pos,
                                            dtilt=dtilt)
            # Move to a safe moving height.
            return cls._get_move_action(
                (robot_x, robot_y, CoffeeEnv.robot_init_z),
                robot_pos,
                dtilt=dtilt)

        return policy

    ############################ Utility functions ############################

    @classmethod
    def _get_move_action(cls,
                         target_pos: Tuple[float, float, float],
                         robot_pos: Tuple[float, float, float],
                         dtilt: float = 0.0,
                         dwrist: float = 0.0) -> Action:
        # We want to move in this direction.
        delta = np.subtract(target_pos, robot_pos)
        # But we can only move at most max_position_vel in one step.
        # Get the norm full move delta.
        pos_norm = float(np.linalg.norm(delta))
        # If the norm is more than max_position_vel, rescale the delta so
        # that its norm is max_position_vel.
        if pos_norm > CoffeeEnv.max_position_vel:
            delta = CoffeeEnv.max_position_vel * (delta / pos_norm)
            pos_norm = CoffeeEnv.max_position_vel
        # Now normalize so that the action values are between -1 and 1, as
        # expected by simulate and the action space.
        if pos_norm > 0:
            delta = delta / CoffeeEnv.max_position_vel
        dx, dy, dz = delta
        dtilt = np.clip(dtilt, -CoffeeEnv.max_angular_vel,
                        CoffeeEnv.max_angular_vel)
        dtilt = dtilt / CoffeeEnv.max_angular_vel
        return Action(
            np.array([dx, dy, dz, dtilt, dwrist, 0.0], dtype=np.float32))

    @classmethod
    def _get_jug_handle_grasp(cls, state: State,
                              jug: Object) -> Tuple[float, float, float]:
        # Hack to avoid duplicate code.
        return CoffeeEnv._get_jug_handle_grasp(state, jug)  # pylint: disable=protected-access

    @classmethod
    def _get_jug_z(cls, state: State, robot: Object, jug: Object) -> float:
        try:
            assert state.get(jug, "is_held") > 0.5
        except:
            raise utils.OptionExecutionFailure("Jug is not held.")
        # Offset to account for handle.
        return state.get(robot, "z") - CoffeeEnv.jug_handle_height

    @staticmethod
    def _get_pour_position(state: State,
                           cup: Object) -> Tuple[float, float, float]:
        target_x = state.get(cup, "x") + CoffeeEnv.pour_x_offset
        target_y = state.get(cup, "y") + CoffeeEnv.pour_y_offset
        target_z = CoffeeEnv.pour_z_offset
        return (target_x, target_y, target_z)
