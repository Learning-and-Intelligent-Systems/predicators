"""Ground-truth options for the playroom environment."""

from typing import ClassVar, Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.playroom import PlayroomEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, \
    ParameterizedInitiable, ParameterizedOption, ParameterizedPolicy, \
    Predicate, State, Type


class PlayroomSimpleGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the playroom environment."""

    _block_size: ClassVar[float] = 0.5

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"playroom_simple", "playroom_simple_clear"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        block_type = types["block"]
        dial_type = types["dial"]

        Pick = utils.SingletonParameterizedOption(
            # variables: [robot, object to pick]
            # params: [rotation]
            "Pick",
            cls._create_pick_policy(action_space),
            types=[robot_type, block_type],
            params_space=Box(-1, 1, (1, )),
            initiable=cls._create_next_to_table_initiable(predicates))

        Stack = utils.SingletonParameterizedOption(
            # variables: [robot, object on which to stack currently-held-object]
            # params: [rotation]
            "Stack",
            cls._create_stack_policy(action_space),
            types=[robot_type, block_type],
            params_space=Box(-1, 1, (1, )),
            initiable=cls._create_next_to_table_initiable(predicates))

        PutOnTable = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: [x, y, rotation] (normalized coords on table surface)
            "PutOnTable",
            cls._create_put_on_table_policy(action_space),
            types=[robot_type],
            params_space=Box(low=np.array([0.0, 0.0, -1.0]),
                             high=np.array([1.0, 1.0, 1.0])),
            initiable=cls._create_next_to_table_initiable(predicates))

        MoveTableToDial = utils.SingletonParameterizedOption(
            # variables: [robot, dial]
            # params: [dx, dy, rotation]
            "MoveTableToDial",
            cls._create_move_table_to_dial_policy(
                action_space),  # uses robot, dial
            types=[robot_type, dial_type],
            params_space=Box(low=np.array([-4.0, -4.0, -1.0]),
                             high=np.array([4.0, 4.0, 1.0])),
            initiable=cls._create_next_to_table_initiable(
                predicates))  # uses robot

        TurnOnDial = utils.SingletonParameterizedOption(
            # variables: [robot, dial]
            # params: [dx, dy, dz, rotation]
            "TurnOnDial",
            cls._create_toggle_dial_policy(action_space),
            types=[robot_type, dial_type],
            params_space=Box(low=np.array([-5.0, -5.0, -5.0, -1.0]),
                             high=np.array([5.0, 5.0, 5.0, 1.0])),
            initiable=cls._create_toggle_dial_initiable(predicates))

        TurnOffDial = utils.SingletonParameterizedOption(
            # variables: [robot, dial]
            # params: [dx, dy, dz, rotation]
            "TurnOffDial",
            cls._create_toggle_dial_policy(action_space),
            types=[robot_type, dial_type],
            params_space=Box(low=np.array([-5.0, -5.0, -5.0, -1.0]),
                             high=np.array([5.0, 5.0, 5.0, 1.0])),
            initiable=cls._create_toggle_dial_initiable(predicates))

        return {
            Pick, Stack, PutOnTable, MoveTableToDial, TurnOnDial, TurnOffDial
        }

    @classmethod
    def _create_next_to_table_initiable(
            cls, predicates: Dict[str, Predicate]) -> ParameterizedInitiable:

        NextToTable = predicates["NextToTable"]

        def initiable(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del memory, params  # unused
            robot = objects[0]
            return NextToTable.holds(state, (robot, ))

        return initiable

    @classmethod
    def _create_toggle_dial_initiable(
            cls, predicates: Dict[str, Predicate]) -> ParameterizedInitiable:

        NextToDial = predicates["NextToDial"]

        def initiable(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del memory, params  # unused
            # objects: (robot, dial)
            return NextToDial.holds(state, objects)

        return initiable

    @classmethod
    def _create_pick_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # Differs from blocks because need robot rotation
            del memory  # unused
            _, block = objects
            block_pose = np.array([
                state.get(block, "pose_x"),
                state.get(block, "pose_y"),
                state.get(block, "pose_z")
            ])
            arr = np.r_[block_pose, params[-1], 0.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_stack_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # Differs from blocks because need robot rotation
            del memory  # unused
            _, block = objects
            block_pose = np.array([
                state.get(block, "pose_x"),
                state.get(block, "pose_y"),
                state.get(block, "pose_z")
            ])
            relative_grasp = np.array([
                0.,
                0.,
                cls._block_size,
            ])
            arr = np.r_[block_pose + relative_grasp, params[-1],
                        1.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_put_on_table_policy(cls,
                                    action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # Differs from blocks because need robot rotation, table bounds
            del state, memory, objects  # unused
            # Un-normalize parameters to actual table coordinates
            x_norm, y_norm = params[:-1]
            x = PlayroomEnv.table_x_lb + (PlayroomEnv.table_x_ub -
                                          PlayroomEnv.table_x_lb) * x_norm
            y = PlayroomEnv.table_y_lb + (PlayroomEnv.table_y_ub -
                                          PlayroomEnv.table_y_lb) * y_norm
            z = PlayroomEnv.table_height + 0.5 * cls._block_size
            arr = np.array([x, y, z, params[-1], 1.0], dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_move_table_to_dial_policy(
            cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            # params: [dx, dy, rotation]
            robot, dial = objects
            fingers = state.get(robot, "fingers")
            dial_pose = np.array(
                [state.get(dial, "pose_x"),
                 state.get(dial, "pose_y")])
            arr = np.r_[dial_pose + params[:-1], 1.0, params[-1],
                        fingers].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_toggle_dial_policy(cls,
                                   action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            _, dial = objects
            dial_pose = np.array([
                state.get(dial, "pose_x"),
                state.get(dial, "pose_y"), PlayroomEnv.dial_button_z
            ])
            arr = np.r_[dial_pose + params[:-1], params[-1],
                        1.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy


class PlayroomGroundTruthOptionFactory(PlayroomSimpleGroundTruthOptionFactory):
    """Ground-truth options for the playroom environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"playroom", "playroom_hard"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        region_type = types["region"]
        door_type = types["door"]
        dial_type = types["dial"]

        # Additional options from parent class
        MoveToDoor = utils.SingletonParameterizedOption(
            # variables: [robot, region, door]
            # params: [dx, dy, rotation]
            "MoveToDoor",
            cls._create_move_to_door_policy(action_space),  # uses robot, door
            types=[robot_type, region_type, door_type],
            params_space=Box(-1, 1, (3, )),
            initiable=cls._create_move_from_region_initiable(
                predicates))  # uses robot, region

        MoveDoorToTable = utils.SingletonParameterizedOption(
            # variables: [robot, region]
            # params: [x, y, rotation] (x, y normalized)
            "MoveDoorToTable",
            cls._create_move_to_table_policy(action_space),  # uses robot
            types=[robot_type, region_type],
            params_space=Box(-1, 1, (3, )),
            initiable=cls._create_move_from_region_initiable(
                predicates))  # uses robot, region

        MoveDoorToDial = utils.SingletonParameterizedOption(
            # variables: [robot, region, dial]
            # params: [dx, dy, rotation]
            "MoveDoorToDial",
            cls._create_move_to_dial_policy(action_space),  # uses robot, dial
            types=[robot_type, region_type, dial_type],
            params_space=Box(low=np.array([-4.0, -4.0, -1.0]),
                             high=np.array([4.0, 4.0, 1.0])),
            initiable=cls._create_move_from_region_initiable(
                predicates))  # uses robot, region

        OpenDoor = utils.SingletonParameterizedOption(
            # variables: [robot, door]
            # params: [dx, dy, dz, rotation]
            "OpenDoor",
            cls._create_toggle_door_policy(action_space),
            types=[robot_type, door_type],
            params_space=Box(low=np.array([-5.0, -5.0, -5.0, -1.0]),
                             high=np.array([5.0, 5.0, 5.0, 1.0])),
            initiable=cls._create_toggle_door_initiable(predicates))

        CloseDoor = utils.SingletonParameterizedOption(
            # variables: [robot, door]
            # params: [dx, dy, dz, rotation]
            "CloseDoor",
            cls._create_toggle_door_policy(action_space),
            types=[robot_type, door_type],
            params_space=Box(low=np.array([-5.0, -5.0, -5.0, -1.0]),
                             high=np.array([5.0, 5.0, 5.0, 1.0])),
            initiable=cls._create_toggle_door_initiable(predicates))

        parent_options = super().get_options(env_name, types, predicates,
                                             action_space)

        # Remove MoveTableToDial
        option_name_to_option = {o.name: o for o in parent_options}
        parent_options.remove(option_name_to_option["MoveTableToDial"])

        return parent_options | {
            MoveToDoor, MoveDoorToTable, MoveDoorToDial, OpenDoor, CloseDoor
        }

    @classmethod
    def _create_move_to_door_policy(cls,
                                    action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:

            del memory  # unused
            # params: [dx, dy, rotation]
            robot, door = objects[0], objects[-1]
            fingers = state.get(robot, "fingers")
            door_pose = np.array([
                state.get(door, "pose_x"),
                state.get(door, "pose_y"),
            ])
            arr = np.r_[door_pose + params[:-1], 1.0, params[-1],
                        fingers].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_move_to_table_policy(cls,
                                     action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:

            del memory  # unused
            # params: [x, y, rotation] (x, y in normalized coords)
            robot = objects[0]
            fingers = state.get(robot, "fingers")
            x_norm, y_norm = params[:-1]
            x = PlayroomEnv.table_x_lb + (PlayroomEnv.table_x_ub -
                                          PlayroomEnv.table_x_lb) * x_norm
            y = PlayroomEnv.table_y_lb + (PlayroomEnv.table_y_ub -
                                          PlayroomEnv.table_y_lb) * y_norm
            arr = np.array([x, y, 1.0, params[-1], fingers], dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_move_to_dial_policy(cls,
                                    action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            # params: [dx, dy, rotation]
            robot, _, dial = objects
            fingers = state.get(robot, "fingers")
            dial_pose = np.array(
                [state.get(dial, "pose_x"),
                 state.get(dial, "pose_y")])
            arr = np.r_[dial_pose + params[:-1], 1.0, params[-1],
                        fingers].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_toggle_door_policy(cls,
                                   action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:

            del memory  # unused
            _, door = objects
            door_pose = np.array([
                state.get(door, "pose_x"),
                state.get(door, "pose_y"), PlayroomEnv.door_button_z
            ])
            arr = np.r_[door_pose + params[:-1], params[-1],
                        1.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_move_from_region_initiable(
            cls, predicates: Dict[str, Predicate]) -> ParameterizedInitiable:

        InRegion = predicates["InRegion"]

        def initiable(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del memory, params  # unused
            # objects: robot, region, ...
            return InRegion.holds(state, objects[:2])

        return initiable

    @classmethod
    def _create_toggle_door_initiable(
            cls, predicates: Dict[str, Predicate]) -> ParameterizedInitiable:

        NextToDoor = predicates["NextToDoor"]

        def initiable(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del memory, params  # unused
            # objects: (robot, door)
            return NextToDoor.holds(state, objects)

        return initiable
