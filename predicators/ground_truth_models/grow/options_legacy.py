"""Legacy option implementations for the grow environment."""

from functools import lru_cache
from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.envs.pybullet_grow import PyBulletGrowEnv
from predicators.ground_truth_models.coffee.options import \
    PyBulletCoffeeGroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, create_move_end_effector_to_pose_option
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


@lru_cache
def _get_pybullet_robot() -> SingleArmPyBulletRobot:
    _, pybullet_robot, _ = \
        PyBulletCoffeeEnv.initialize_pybullet(using_gui=False)
    return pybullet_robot


class _GrowLegacyOptionsMixin:
    # Declare attributes provided by the concrete class that uses this mixin.
    env_cls: ClassVar[TypingType[PyBulletGrowEnv]]
    pour_policy_tol: ClassVar[float]
    _finger_action_nudge_magnitude: ClassVar[float]
    """Legacy option implementations, mixed into the main factory class."""

    @classmethod
    def _get_options_legacy(cls, _env_name: str, types: Dict[str, Type],
                            predicates: Dict[str, Predicate],
                            action_space: Box) -> Set[ParameterizedOption]:
        """Legacy option implementations."""
        _, pybullet_robot, _ = \
            PyBulletGrowEnv.initialize_pybullet(using_gui=False)

        # Types
        robot_type = types["robot"]
        jug_type = types["jug"]
        cup_type = types["cup"]
        # Predicates
        Holding = predicates["Holding"]
        Grown = predicates["Grown"]
        _JugAboveCup = predicates["JugAboveCup"]
        HandTilted = predicates["HandTilted"]

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
            params_space=Box(0, 1, (0, )),
            policy=PyBulletCoffeeGroundTruthOptionFactory.  # pylint: disable=protected-access
            _create_pick_jug_policy(),
            # policy=cls._create_pick_jug_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_PickJug_terminal)

        # Pour
        def _Pour_terminal(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            robot, jug, cup = objects
            if CFG.grow_weak_pour_terminate_condition:
                if not Holding.holds(state, [robot, jug]):
                    return False
                jug_x = state.get(jug, "x")
                jug_y = state.get(jug, "y")
                jug_z = state.get(robot, "z") -\
                    PyBulletCoffeeEnv.jug_handle_height()
                jug_pos = (jug_x, jug_y, jug_z)
                pour_pos = PyBulletCoffeeEnv._get_pour_position(state, cup)  # pylint: disable=protected-access
                sq_dist_to_pour = np.sum(np.subtract(jug_pos, pour_pos)**2)
                jug_above_cup = sq_dist_to_pour < cls.env_cls.pour_pos_tol/\
                                            (cls.env_cls.pour_pos_tol_factor*2)

                cond = jug_above_cup and HandTilted.holds(state, [robot])
            else:
                cond = Grown.holds(state, [cup])
            return cond

        Pour = ParameterizedOption(
            "Pour",
            [robot_type, jug_type, cup_type],
            params_space=Box(0, 1, (0, )),
            policy=PyBulletCoffeeGroundTruthOptionFactory.  # pylint: disable=protected-access
            _create_pour_policy(pour_policy_tol=cls.pour_policy_tol),
            initiable=lambda s, m, o, p: True,
            terminal=_Pour_terminal)

        # Place
        def _Place_terminal(state: State, memory: Dict,
                            objects: Sequence[Object], params: Array) -> bool:
            del memory, params
            robot, jug = objects
            return not Holding.holds(state, [robot, jug])

        if CFG.grow_place_option_no_sampler:
            params_space = Box(0, 1, (0, ))
        else:
            params_space = Box(0, 1, (2, ))

        Place = utils.LinearChainParameterizedOption(
            "Place",
            [
                # Move to above the target location
                cls._create_move_to_place_location_option(
                    name="MoveToAbovePlaceLocation",
                    z_func=lambda _: PyBulletCoffeeEnv.z_ub - 0.3,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=[robot_type, jug_type],
                    params_space=params_space),
                # Move down to place
                cls._create_move_to_place_location_option(
                    name="MoveToPlaceLocation",
                    # z_func=lambda _: cls.env_cls.z_lb + cls.env_cls.jug_height
                    # / 2,
                    z_func=lambda z: z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=[robot_type, jug_type],
                    params_space=params_space),
                # Open fingers to release
                create_change_fingers_option(
                    pybullet_robot,
                    "OpenFingers",
                    [robot_type, jug_type],
                    params_space,
                    lambda state, objects, params: (
                        PyBulletCoffeeEnv._fingers_state_to_joint(  # pylint: disable=protected-access
                            pybullet_robot, state.get(objects[0], "fingers")),
                        pybullet_robot.open_fingers),
                    CFG.pybullet_max_vel_norm,
                    cls.env_cls.place_jug_tol,
                    terminal=_Place_terminal),
            ])

        # Wait
        params_space = Box(0, 1, (0, ))

        def _create_wait_policy() -> ParameterizedPolicy:
            nonlocal action_space

            def _policy(state: State, memory: Dict, objects: Sequence[Object],
                        params: Array) -> Action:
                del memory, params
                robot = objects[0]
                nonlocal action_space
                # check finger open or closed
                finger = state.get(robot, "fingers")
                mid_point = (pybullet_robot.open_fingers +
                             pybullet_robot.closed_fingers) / 2
                if finger > mid_point:
                    # currently open
                    finger_delta = cls._finger_action_nudge_magnitude
                else:
                    finger_delta = -cls._finger_action_nudge_magnitude

                # nudge finger to the direction of the current state to counter
                assert isinstance(state, utils.PyBulletState)
                joint_positions = state.joint_positions.copy()
                finger_position = joint_positions[
                    pybullet_robot.left_finger_joint_idx]
                # The finger action is an absolute joint position for the
                # fingers.
                f_action = finger_position + finger_delta
                # Override the meaningless finger values in joint_action.
                joint_positions[
                    pybullet_robot.left_finger_joint_idx] = f_action
                joint_positions[
                    pybullet_robot.right_finger_joint_idx] = f_action
                # slide
                action = np.array(joint_positions, dtype=np.float32)
                action = action.clip(action_space.low,
                                     action_space.high).astype(np.float32)
                return Action(action)

            return _policy

        Wait = ParameterizedOption(
            "Wait",
            types=[robot_type],
            params_space=params_space,
            policy=_create_wait_policy(),
            initiable=lambda _1, _2, _3, _4: True,
            terminal=lambda _1, _2, _3, _4: False,
        )

        return {PickJug, Pour, Place, Wait}

    @classmethod
    def _create_move_to_place_location_option(
            cls, name: str, z_func: Callable[[float],
                                             float], finger_status: str,
            pybullet_robot: SingleArmPyBulletRobot, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to the target place
        location.

        The parameter z_func maps the current z position to the target z
        position.
        """

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            robot, jug = objects

            # Current pose
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)

            # Target pose - determine target jug position
            if CFG.grow_place_option_no_sampler:
                target_jug_pos = (jug.init_x, jug.init_y, jug.init_z)
            else:
                x_norm, y_norm = params
                target_jug_pos = (
                    cls.env_cls.x_lb +
                    (cls.env_cls.x_ub - cls.env_cls.x_lb) * x_norm,
                    cls.env_cls.y_lb +
                    (cls.env_cls.y_ub - cls.env_cls.y_lb) * y_norm,
                    cls.env_cls.z_lb + cls.env_cls.jug_height / 2)

            # Calculate robot target position based on jug displacement
            current_jug_pos = (state.get(jug, "x"), state.get(jug, "y"),
                               state.get(jug, "z"))
            dx, dy, dz = np.subtract(target_jug_pos, current_jug_pos)
            target_position = (current_position[0] + dx,
                               current_position[1] + dy,
                               z_func(current_position[2] + dz))

            target_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt, cls.env_cls.robot_init_wrist])
            target_pose = Pose(target_position, target_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls.env_cls.place_jug_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            validate=CFG.pybullet_ik_validate if hasattr(
                CFG, 'pybullet_ik_validate') else False)
