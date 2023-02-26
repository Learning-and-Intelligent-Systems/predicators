"""Ground-truth options for the painting environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.painting import PaintingEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class PaintingGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the painting environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"painting"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # Types
        robot_type = types["robot"]
        obj_type = types["obj"]
        lid_type = types["lid"]

        Pick = utils.SingletonParameterizedOption(
            # variables: [robot, object to pick]
            # params: [grasp]
            "Pick",
            cls._create_pick_policy(action_space),
            types=[robot_type, obj_type],
            params_space=Box(np.array([-0.01], dtype=np.float32),
                             np.array([1.01], dtype=np.float32)))

        Wash = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: []
            "Wash",
            cls._create_wash_policy(),
            types=[robot_type])

        Dry = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: []
            "Dry",
            cls._create_dry_policy(),
            types=[robot_type])

        Paint = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: [new color]
            "Paint",
            cls._create_paint_policy(),
            types=[robot_type],
            params_space=Box(-0.01, 1.01, (1, )))

        Place = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: [absolute x, absolute y, absolute z]
            "Place",
            cls._create_place_policy(),
            types=[robot_type],
            params_space=Box(
                np.array([
                    PaintingEnv.obj_x - 1e-2, PaintingEnv.env_lb,
                    PaintingEnv.obj_z - 1e-2
                ],
                         dtype=np.float32),
                np.array([
                    PaintingEnv.obj_x + 1e-2, PaintingEnv.env_ub,
                    PaintingEnv.obj_z + 1e-2
                ],
                         dtype=np.float32)))

        OpenLid = utils.SingletonParameterizedOption(
            # variables: [robot, lid]
            # params: []
            "OpenLid",
            cls._create_open_lid_policy(),
            types=[robot_type, lid_type])

        return {Pick, Wash, Dry, Paint, Place, OpenLid}

    @classmethod
    def _create_pick_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            _, obj = objects
            obj_x = state.get(obj, "pose_x")
            obj_y = state.get(obj, "pose_y")
            obj_z = state.get(obj, "pose_z")
            grasp, = params
            arr = np.array([obj_x, obj_y, obj_z, grasp, 1.0, 0.0, 0.0, 0.0],
                           dtype=np.float32)
            # The grasp could cause the action to go out of bounds, so we clip
            # it back into the bounds for safety.
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_wash_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params  # unused
            arr = np.array([
                PaintingEnv.obj_x, PaintingEnv.table_lb, PaintingEnv.obj_z,
                0.0, 0.0, 1.0, 0.0, 0.0
            ],
                           dtype=np.float32)
            return Action(arr)

        return policy

    @classmethod
    def _create_dry_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params  # unused
            arr = np.array([
                PaintingEnv.obj_x, PaintingEnv.table_lb, PaintingEnv.obj_z,
                0.0, 0.0, 0.0, 1.0, 0.0
            ],
                           dtype=np.float32)
            return Action(arr)

        return policy

    @classmethod
    def _create_paint_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects  # unused
            new_color, = params
            new_color = min(max(new_color, 0.0), 1.0)
            arr = np.array([
                PaintingEnv.obj_x, PaintingEnv.table_lb, PaintingEnv.obj_z,
                0.0, 0.0, 0.0, 0.0, new_color
            ],
                           dtype=np.float32)
            return Action(arr)

        return policy

    @classmethod
    def _create_place_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects  # unused
            x, y, z = params
            arr = np.array([x, y, z, 0.5, -1.0, 0.0, 0.0, 0.0],
                           dtype=np.float32)
            return Action(arr)

        return policy

    @classmethod
    def _create_open_lid_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params  # unused
            arr = np.array([
                PaintingEnv.obj_x,
                (PaintingEnv.box_lb + PaintingEnv.box_ub) / 2,
                PaintingEnv.obj_z, 0.0, 1.0, 0.0, 0.0, 0.0
            ],
                           dtype=np.float32)
            return Action(arr)

        return policy


class RNTPaintingGroundTruthOptionFactory(PaintingGroundTruthOptionFactory):
    """Ground-truth options for the painting environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"repeated_nextto_painting"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        obj_type = types["obj"]
        box_type = types["box"]
        shelf_type = types["shelf"]

        MoveToObj = utils.SingletonParameterizedOption(
            "MoveToObj",
            cls._create_move_policy(),
            types=[robot_type, obj_type],
            params_space=Box(PaintingEnv.env_lb, PaintingEnv.env_ub, (1, )))

        MoveToBox = utils.SingletonParameterizedOption(
            "MoveToBox",
            cls._create_move_policy(),
            types=[robot_type, box_type],
            params_space=Box(PaintingEnv.env_lb, PaintingEnv.env_ub, (1, )))

        MoveToShelf = utils.SingletonParameterizedOption(
            "MoveToShelf",
            cls._create_move_policy(),
            types=[robot_type, shelf_type],
            params_space=Box(PaintingEnv.env_lb, PaintingEnv.env_ub, (1, )))

        parent_options = super().get_options(env_name, types, predicates,
                                             action_space)
        return parent_options | {MoveToObj, MoveToBox, MoveToShelf}

    @classmethod
    def _create_move_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            _, obj = objects
            next_x = state.get(obj, "pose_x")
            next_y = params[0]
            return Action(
                np.array([next_x, next_y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         dtype=np.float32))

        return policy
