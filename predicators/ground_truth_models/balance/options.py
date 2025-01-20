"""Ground-truth options for the (non-pybullet) blocks environment."""

from functools import lru_cache
from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_balance import PyBulletBalanceEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, create_move_end_effector_to_pose_option, \
    get_move_end_effector_to_pose_action
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


@lru_cache
def _get_pybullet_robot() -> SingleArmPyBulletRobot:
    _, pybullet_robot, _ = \
            PyBulletBalanceEnv.initialize_pybullet(using_gui=False)
    return pybullet_robot


class PyBulletBalanceGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the pybullet_balance environment."""

    env_cls = PyBulletBalanceEnv
    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _offset_z: ClassVar[float] = 0.01
    _transport_z: ClassVar[float] = env_cls.z_ub - 0.2

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_balance"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        _, pybullet_robot, _ = \
            PyBulletBalanceEnv.initialize_pybullet(using_gui=False)

        robot_type = types["robot"]
        block_type = types["block"]
        machine_type = types["machine"]
        plate_type = types["plate"]
        block_size = CFG.blocks_block_size

        GripperOpen = predicates['GripperOpen']
        MachineOn = predicates['MachineOn']
        Balanced = predicates['Balanced'].untransformed_predicate

        def get_current_fingers(state: State) -> float:
            robot, = state.get_objects(robot_type)
            return PyBulletBalanceEnv._fingers_state_to_joint(
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
        params_space = Box(0, 1, (0, ))
        Pick = utils.LinearChainParameterizedOption(
            "Pick",
            [
                # Move to far above the block which we will grasp.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToPreGrasp",
                    z_func=lambda _: cls._transport_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Move down to grasp.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToGrasp",
                    z_func=lambda block_z: (block_z + cls._offset_z),
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Close fingers.
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_types, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBalanceEnv.grasp_tol),
                # Move back up.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: cls._transport_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    move_to_pose_tol=cls._move_to_pose_tol * 100),
            ],
            # "Pick up block ?block"
        )

        # Stack
        option_types = [robot_type, block_type]
        params_space = Box(0, 1, (0, ))
        Stack = utils.LinearChainParameterizedOption(
            "Stack",
            [
                # Move to above the block on which we will stack.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToPreStack",
                    z_func=lambda _: cls._transport_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Move down to place.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToStack",
                    z_func=lambda block_z: (block_z + block_size * 1.3),
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBalanceEnv.grasp_tol),
                # Move back up.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: cls._transport_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    move_to_pose_tol=cls._move_to_pose_tol * 100),
            ],
            # annotation="Stack the block in hand onto block ?otherblock"
        )

        # PutOnPlate
        option_types = [robot_type, plate_type]
        params_space = Box(0, 1, (2, ))
        PutOnPlate = utils.LinearChainParameterizedOption(
            "PutOnPlate",
            [
                # Move to above the table at the (x, y) where we will place.
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorToPrePutOnPlate",
                    z=cls.env_cls.z_ub,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Move down to place.
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorToPutOnPlate",
                    z=cls.env_cls.z_ub - 0.2,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBalanceEnv.grasp_tol),
                # Move back up.
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorBackUp",
                    z=cls.env_cls.z_ub,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
            ],
            # annotation="Put block on plate"
        )

        # TurnMachineOn
        def _TurnMachineOn_initiable(state: State, memory: Dict,
                                     objects: Sequence[Object],
                                     params: Array) -> bool:
            return True
            del memory, params  # unused
            plate1, plate2 = objects
            robot = state.get_objects(robot_type)[0]
            # robot = [r for r in state if r.type.name == "robot"][0]
            return GripperOpen.holds(state, [robot]) and\
                    Balanced.holds(state, [plate1, plate2])

        def _TurnMachineOn_terminal(state: State, memory: Dict,
                                    objects: Sequence[Object],
                                    params: Array) -> bool:
            del memory, params  # unused
            machine = state.get_objects(machine_type)[0]
            robot = state.get_objects(robot_type)[0]
            # machine = objects[1]
            return MachineOn.holds(state, [machine, robot])

        TurnMachineOn = ParameterizedOption(
            "TurnMachineOn",
            types=[plate_type, plate_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_turn_machine_on_policy(),
            initiable=_TurnMachineOn_initiable,
            terminal=_TurnMachineOn_terminal,
            # annotation="Turn the machine on."
        )

        return {Pick, Stack, PutOnPlate, TurnMachineOn}

    @classmethod
    def _create_blocks_move_to_above_block_option(
        cls,
        name: str,
        z_func: Callable[[float], float],
        finger_status: str,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
        move_to_pose_tol: float = _move_to_pose_tol,
    ) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        block argument.

        The parameter z_func maps the block's z position to the target z
        position.
        """
        home_orn = PyBulletBalanceEnv.get_robot_ee_home_orn()

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not params
            robot, block = objects
            current_position = (state.get(robot, "x"),
                                state.get(robot, "y"),
                                state.get(robot, "z"))
            current_pose = Pose(current_position, home_orn)
            target_position = (state.get(block,
                                         "x"), state.get(block, "y"),
                               z_func(state.get(block, "z")))
            target_pose = Pose(target_position, home_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            validate=CFG.pybullet_ik_validate)

    @classmethod
    def _create_blocks_move_to_above_table_option(
            cls, name: str, z: float, finger_status: str,
            pybullet_robot: SingleArmPyBulletRobot, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        table.

        The z position of the target pose must be provided.
        """
        home_orn = PyBulletBalanceEnv.get_robot_ee_home_orn()

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            robot, _ = objects
            current_position = (state.get(robot, "x"),
                                state.get(robot, "y"),
                                state.get(robot, "z"))
            current_pose = Pose(current_position, home_orn)
            # De-normalize parameters to actual table coordinates.
            x_norm, y_norm = params
            target_position = (
                PyBulletBalanceEnv.x_lb +
                (PyBulletBalanceEnv.x_ub - PyBulletBalanceEnv.x_lb) * x_norm,
                PyBulletBalanceEnv.y_lb +
                (PyBulletBalanceEnv.y_ub - PyBulletBalanceEnv.y_lb) * y_norm,
                z)
            target_pose = Pose(target_position, home_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude)

    @classmethod
    def _create_turn_machine_on_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # This policy moves the robot up to be level with the button in the
            # z direction and then moves forward in the y direction to press it.
            del memory, params  # unused
            # robot = objects[0]
            # robot = state.get_objects(cls.env_cls._robot_type)[0]
            robot = [r for r in state if r.type.name == "robot"][0]
            x = state.get(robot, "x")
            y = state.get(robot, "y")
            z = state.get(robot, "z")
            robot_pos = (x, y, z)
            button_pos = (cls.env_cls.button_x, cls.env_cls.button_y,
                          cls.env_cls.button_z + cls.env_cls._button_radius)
            # arr = np.r_[button_pos, 1.0].astype(np.float32)
            # # arr = np.clip(arr, cls.env_cls.action_space.low,
            # #               cls.env_cls.action_space.high)
            # return Action(arr)
            if (cls.env_cls.button_x - x)**2 < \
                    cls.env_cls._button_radius**2 and\
                (cls.env_cls.button_y - y)**2 < \
                    cls.env_cls._button_radius**2:
                # Move directly toward the button.
                return cls._get_move_action(state,
                                            button_pos,
                                            robot_pos,
                                            finger_status="closed")
            # Move only in the z direction.
            return cls._get_move_action(
                state, (cls.env_cls.button_x, cls.env_cls.button_y, z),
                robot_pos,
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

        home_orn = PyBulletBalanceEnv.get_robot_ee_home_orn()
        current_pose = Pose(robot_pos, home_orn)
        target_pose = Pose(target_pos, home_orn)
        assert isinstance(state, utils.PyBulletState)

        return get_move_end_effector_to_pose_action(
            pybullet_robot,
            current_joint_positions,
            current_pose,
            target_pose,
            finger_status,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            validate=CFG.pybullet_ik_validate)
