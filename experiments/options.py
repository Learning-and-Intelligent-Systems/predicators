"""Ground truth options for the shelves2d environment"""

import gym
from typing import Dict, Sequence, Set, Type, cast

import numpy as np
from experiments.shelves2d import Shelves2DEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, Predicate, State
from predicators.utils import SingletonParameterizedOption

__all__ = ["Shelves2DGroundTruthOptionFactory"]

class Shelves2DGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground truth options for the shelves2d environment"""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return set(["shelves2d"])

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: gym.spaces.Box) -> Set[ParameterizedOption]:

        # Types
        box_type = types["box"]
        shelf_type = types["shelf"]
        bundle_type = types["bundle"]
        cover_type = types["cover"]

        # Prediactes
        GoesInto = predicates["GoesInto"]
        CoverAtStart = predicates["CoverAtStart"]
        Bundles = predicates["Bundles"]
        In = predicates["In"]
        CoverFor = predicates["CoverFor"]
        CoversTop = predicates["CoversFront"]
        CoversBottom = predicates["CoversBack"]

        # Miscellaneous helper variables
        world_dist_x = Shelves2DEnv.range_world_x[1] - Shelves2DEnv.range_world_x[0]
        world_dist_y = Shelves2DEnv.range_world_y[1] - Shelves2DEnv.range_world_y[0]
        offset_space = gym.spaces.Box(
            np.array([-world_dist_x, -world_dist_y]),
            np.array([world_dist_x, world_dist_y]),
        )
        options = set()

        # MoveCoverToTop option
        options.add(ParameterizedOption(
            "MoveCoverToTop",
            [cover_type, bundle_type],
            offset_space,
            _MoveCoverTo_policy_helper(True),
            cls.initiable,
            cls.terminal
        ))

        # MoveCoverToBottom option
        options.add(ParameterizedOption(
            "MoveCoverToBottom",
            [cover_type, bundle_type],
            offset_space,
            _MoveCoverTo_policy_helper(False),
            cls.initiable,
            cls.terminal
        ))

        # MoveBox option
        def MoveBox_policy(state: State, data: Dict, objects: Sequence[Object], arr: Array) -> Action:
            box, shelf = objects
            offset_x, offset_y = arr

            shelf_x, shelf_y, shelf_w, shelf_h = Shelves2DEnv.get_shape_data(state, shelf)
            box_x, box_y, box_w, box_h = Shelves2DEnv.get_shape_data(state, box)

            action = [
                box_x + box_w/2, box_y + box_h/2,
                shelf_x + offset_x - box_x, shelf_y + offset_y - box_y
            ]

            bounds = Shelves2DEnv.action_space_bounds()
            return Action(np.clip(action, bounds.low, bounds.high, dtype=np.float32))

        options.add(ParameterizedOption(
            "MoveBox",
            [box_type, shelf_type],
            offset_space,
            MoveBox_policy,
            cls.initiable,
            cls.terminal
        ))

        return options

    @classmethod
    def initiable(cls, state: State, data: Dict, objects: Sequence[Object], arr: Array) -> bool:
        return True

    @classmethod
    def terminal(cls, state: State, data: Dict, objects: Sequence[Object], arr: Array) -> bool:
        terminal_executed = data.get("terminal_executed", False)
        data["terminal_executed"] = True
        return terminal_executed

    @classmethod
    def move(cls, state: State, data: Dict, objects: Sequence[Object], arr: Array) -> Action:
        return Action(arr)

def _MoveCoverTo_policy_helper(move_to_top: bool):
    def MoveCoverTo_policy(state: State, data: Dict, objects: Sequence[Object], arr: Array) -> Action:
        cover, bundle = objects
        offset_x, offset_y = arr

        bundle_x, bundle_y, _, bundle_h = Shelves2DEnv.get_shape_data(state, bundle)
        cover_x, cover_y, cover_w, cover_h = Shelves2DEnv.get_shape_data(state, cover)

        action = [
            cover_x + cover_w/2, cover_y + cover_h/2,
            bundle_x + offset_x - cover_x, bundle_y + offset_y - cover_y + (bundle_h if move_to_top else -cover_h)
        ]

        bounds = Shelves2DEnv.action_space_bounds()
        return Action(np.clip(action, bounds.low, bounds.high, dtype=np.float32))
    return MoveCoverTo_policy