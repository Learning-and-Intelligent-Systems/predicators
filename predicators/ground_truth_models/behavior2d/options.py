"""Ground-truth options for the Behavior2D environment."""

import itertools
from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.behavior2d import Behavior2DEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class Behavior2DGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the Behavior2D environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"behavior2d"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        # name, policy, param_dim, arity, (ub,lb)
        option_specs = [
            ("NavigateTo", cls.create_NavigateTo_policy(action_space), 1, (np.array([-5.0, -5.0]), np.array([5.0, 5.0]))),
            ("Grasp", cls.create_Grasp_policy(action_space), 1, (np.array([0.0, -np.pi]), np.array([1.0, np.pi]))),
            ("PlaceInside", cls.create_PlaceInside_policy(action_space), 1, (np.array([0.0]), np.array([1.0]))),
            ("PlaceNextTo", cls.create_PlaceNextto_policy(action_space), 1, (np.array([0.0]), np.array([2.0]))),
            ("PlaceTouching", cls.create_PlaceTouching_policy(action_space), 1, (np.array([0.0]), np.array([2.0]))),
            ("Open", cls.create_Open_policy(action_space), 1, (np.array([]), np.array([]))),
            ("Close", cls.create_Close_policy(action_space), 1, (np.array([]), np.array([]))),
            ("ToggleOn", cls.create_ToggleOn_policy(action_space), 1, (np.array([]), np.array([]))),
            ("Cook", cls.create_Cook_policy(action_space), 1, (np.array([]), np.array([]))),
            ("Freeze", cls.create_Freeze_policy(action_space), 1, (np.array([]), np.array([]))),
            ("Soak", cls.create_Soak_policy(action_space), 1, (np.array([]), np.array([]))),
            ("CleanDusty", cls.create_CleanDusty_policy(action_space), 1, (np.array([]), np.array([]))),
            ("CleanStained", cls.create_CleanStained_policy(action_space), 1, (np.array([]), np.array([]))),
            ("Slice", cls.create_Sliced_policy(action_space), 1, (np.array([]), np.array([])))
        ]

        options = set()
        robot_type = types["robot"]
        other_types = list(set(named_type.parent for name, named_type in types.items() if name != "robot" and name != "object"))
        # other_types = [named_type for name, named_type in types.items() if name != "robot"]
        assert len(other_types) == 1
        object_type = other_types[0]
        for name, policy, arity, bounds in option_specs:
            opt = utils.SingletonParameterizedOption(
                name, policy, types=[robot_type] + [object_type] * arity,
                params_space=Box(bounds[0], bounds[1], dtype=np.float32)
            )
            options.add(opt)
        return options


    @classmethod
    def create_NavigateTo_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            del memory
            robby, obj = objects
            obj_x = state.get(obj, "pose_x")
            obj_y = state.get(obj, "pose_y")
            obj_w = state.get(obj, "width")
            obj_h = state.get(obj, "height")
            obj_yaw = state.get(obj, "yaw")

            offset_x, offset_y = params
            pos_x = obj_x + offset_x * np.cos(obj_yaw) - \
                    offset_y * np.sin(obj_yaw)
            pos_y = obj_y + offset_x * np.sin(obj_yaw) + \
                    offset_y * np.cos(obj_yaw)

            if offset_x / obj_w < 0 and 0 <= offset_y / obj_h <= 1:
                yaw = 0
            elif offset_x / obj_w > 1 and 0 <= offset_y / obj_h <= 1:
                yaw = -np.pi
            elif 0 <= offset_x / obj_w <= 1 and offset_y / obj_h < 0:
                yaw = np.pi / 2
            elif 0 <= offset_x / obj_w <= 1 and offset_y / obj_h > 1:
                yaw = -np.pi / 2
            elif 0 <= offset_x / obj_w <= 1 and 0 <= offset_y / obj_h <= 1:
                # Collision with object; will fail, so set any value
                yaw = 0
            else:
                x = offset_x / obj_w - 1 if offset_x / obj_w > 1 else offset_x / obj_w
                y = offset_y / obj_h - 1 if offset_y / obj_h > 1 else offset_y / obj_h
                yaw = np.arctan2(-y * obj_h, -x * obj_w)
            yaw += obj_yaw
            while yaw > np.pi:
                yaw -= (2 * np.pi)
            while yaw < -np.pi:
                yaw += (2 * np.pi)
            obj_id = Behavior2DEnv.get_id_by_object(state, obj)
            action_id = 0.0
            # dimensions: [x, y, yaw, offset_gripper, target_obj_id, action_id]
            arr = np.array([pos_x, pos_y, yaw, 0.0, obj_id, action_id], 
                           dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def create_Grasp_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            del memory
            robby, obj = objects
            obj_yaw = state.get(obj, "yaw")

            robby_x = state.get(robby, "pose_x")
            robby_y = state.get(robby, "pose_y")

            offset_gripper, obj_yaw = params
            obj_id = Behavior2DEnv.get_id_by_object(state, obj)
            action_id = 1.0
            # dimensions: [x, y, yaw, offset_gripper, target_obj_id, action_id]
            arr = np.array([robby_x, robby_y, obj_yaw, offset_gripper, obj_id, action_id],
                           dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def create_PlaceInside_policy(cls, action_space: Box) -> ParameterizedPolicy:
        
        def policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            del memory
            robby, inside_obj = objects

            robby_x = state.get(robby, "pose_x")
            robby_y = state.get(robby, "pose_y")
            robby_yaw = state.get(robby, "yaw")

            offset_gripper, = params
            inside_obj_id = Behavior2DEnv.get_id_by_object(state, inside_obj)
            action_id = 2.0
            # dimensions: [x, y, yaw, offset_gripper, target_obj_id, action_id]
            arr = np.array([robby_x, robby_y, robby_yaw, offset_gripper, inside_obj_id, action_id],
                           dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def create_PlaceNextto_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            del memory
            robby, nextto_obj = objects

            robby_x = state.get(robby, "pose_x")
            robby_y = state.get(robby, "pose_y")
            robby_yaw = state.get(robby, "yaw")

            offset_gripper, = params
            nextto_obj_id = Behavior2DEnv.get_id_by_object(state, nextto_obj)
            action_id = 3.0
            # dimensions: [x, y, yaw, offset_gripper, target_obj_id, action_id]
            arr = np.array([robby_x, robby_y, robby_yaw, offset_gripper, nextto_obj_id, action_id],
                            dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def create_PlaceTouching_policy(cls, action_space: Box) -> ParameterizedPolicy:
        return cls.create_PlaceNextto_policy(action_space)

    @classmethod
    def create_Open_policy(cls, action_space: Box) -> ParameterizedPolicy:
        action_id = 4.0
        return cls.create_Dummy_policy(action_space, action_id)

    @classmethod
    def create_Close_policy(cls, action_space: Box) -> ParameterizedPolicy:
        action_id = 5.0
        return cls.create_Dummy_policy(action_space, action_id)

    @classmethod
    def create_ToggleOn_policy(cls, action_space: Box) -> ParameterizedPolicy:
        action_id = 6.0
        return cls.create_Dummy_policy(action_space, action_id)

    @classmethod
    def create_Cook_policy(cls, action_space: Box) -> ParameterizedPolicy:
        action_id = 7.0
        return cls.create_Dummy_policy(action_space, action_id)

    @classmethod
    def create_Freeze_policy(cls, action_space: Box) -> ParameterizedPolicy:
        action_id = 8.0
        return cls.create_Dummy_policy(action_space, action_id)

    @classmethod
    def create_Soak_policy(cls, action_space: Box) -> ParameterizedPolicy:
        action_id = 9.0
        return cls.create_Dummy_policy(action_space, action_id)

    @classmethod
    def create_CleanDusty_policy(cls, action_space: Box) -> ParameterizedPolicy:
        action_id = 10.0
        return cls.create_Dummy_policy(action_space, action_id)

    @classmethod
    def create_CleanStained_policy(cls, action_space: Box) -> ParameterizedPolicy:
        action_id = 11.0
        return cls.create_Dummy_policy(action_space, action_id)

    @classmethod
    def create_Sliced_policy(cls, action_space: Box) -> ParameterizedPolicy:
        action_id = 12.0
        return cls.create_Dummy_policy(action_space, action_id)

    @classmethod
    def create_Dummy_policy(cls, action_space: Box, action_id: float) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            del memory, params
            robby, target_obj = objects

            robby_x = state.get(robby, "pose_x")
            robby_y = state.get(robby, "pose_y")
            robby_yaw = state.get(robby, "yaw")

            target_obj_id = Behavior2DEnv.get_id_by_object(state, target_obj)
            # dimensions: [x, y, yaw, offset_gripper, target_obj_id, action_id]
            arr = np.array([robby_x, robby_y, robby_yaw, 0.0, target_obj_id, action_id],
                           dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy
