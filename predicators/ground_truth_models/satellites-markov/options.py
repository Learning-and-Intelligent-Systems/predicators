"""Ground-truth options for the satellites environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class SatellitesMkGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the satellites environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"satellites-markov"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        sat_type = types["satellite"]
        obj_type = types["object"]

        MoveTo = utils.SingletonParameterizedOption(
            "MoveTo",
            types=[sat_type, obj_type],
            params_space=Box(0, 1, (2, )),  # target absolute x/y
            policy=cls._create_move_to_policy())
        
        MoveAway = utils.SingletonParameterizedOption(
            "MoveAway",
            types=[sat_type, obj_type],
            params_space=Box(0, 1, (2, )),  # target absolute x/y
            policy=cls._create_move_away_policy())

        Calibrate = utils.SingletonParameterizedOption(
            "Calibrate",
            types=[sat_type, obj_type],
            policy=cls._create_calibrate_policy())

        ShootChemX = utils.SingletonParameterizedOption(
            "ShootChemX",
            types=[sat_type, obj_type],
            policy=cls._create_shoot_chem_x_policy())

        ShootChemY = utils.SingletonParameterizedOption(
            "ShootChemY",
            types=[sat_type, obj_type],
            policy=cls._create_shoot_chem_y_policy())

        UseInstrument = utils.SingletonParameterizedOption(
            "UseInstrument",
            types=[sat_type, obj_type],
            policy=cls._create_use_instrument_policy())

        return {MoveTo, MoveAway, Calibrate, ShootChemX, ShootChemY, UseInstrument}

    @classmethod
    def _create_move_away_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            sat, obj = objects
            cur_sat_x = state.get(sat, "x")
            cur_sat_y = state.get(sat, "y")
            obj_x = state.get(obj, "x")
            obj_y = state.get(obj, "y")
            target_sat_x, target_sat_y = params
            arr = np.array([
                cur_sat_x, cur_sat_y, obj_x, obj_y, target_sat_x, target_sat_y,
                0.0, 0.0, 0.0, 0.0
            ],
                           dtype=np.float32)
            return Action(arr)

        return policy
    
    @classmethod
    def _create_move_to_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            sat, obj = objects
            cur_sat_x = state.get(sat, "x")
            cur_sat_y = state.get(sat, "y")
            obj_x = state.get(obj, "x")
            obj_y = state.get(obj, "y")
            target_sat_x, target_sat_y = params
            arr = np.array([
                cur_sat_x, cur_sat_y, obj_x, obj_y, target_sat_x, target_sat_y,
                0.0, 0.0, 0.0, 0.0
            ],
                           dtype=np.float32)
            return Action(arr)

        return policy

    @classmethod
    def _create_calibrate_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            sat, obj = objects
            cur_sat_x = state.get(sat, "x")
            cur_sat_y = state.get(sat, "y")
            obj_x = state.get(obj, "x")
            obj_y = state.get(obj, "y")
            target_sat_x = cur_sat_x
            target_sat_y = cur_sat_y
            arr = np.array([
                cur_sat_x, cur_sat_y, obj_x, obj_y, target_sat_x, target_sat_y,
                1.0, 0.0, 0.0, 0.0
            ],
                           dtype=np.float32)
            return Action(arr)

        return policy

    @classmethod
    def _create_shoot_chem_x_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:

            del memory, params  # unused
            sat, obj = objects
            cur_sat_x = state.get(sat, "x")
            cur_sat_y = state.get(sat, "y")
            obj_x = state.get(obj, "x")
            obj_y = state.get(obj, "y")
            target_sat_x = cur_sat_x
            target_sat_y = cur_sat_y
            arr = np.array([
                cur_sat_x, cur_sat_y, obj_x, obj_y, target_sat_x, target_sat_y,
                0.0, 1.0, 0.0, 0.0
            ],
                           dtype=np.float32)
            return Action(arr)

        return policy

    @classmethod
    def _create_shoot_chem_y_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:

            del memory, params  # unused
            sat, obj = objects
            cur_sat_x = state.get(sat, "x")
            cur_sat_y = state.get(sat, "y")
            obj_x = state.get(obj, "x")
            obj_y = state.get(obj, "y")
            target_sat_x = cur_sat_x
            target_sat_y = cur_sat_y
            arr = np.array([
                cur_sat_x, cur_sat_y, obj_x, obj_y, target_sat_x, target_sat_y,
                0.0, 0.0, 1.0, 0.0
            ],
                           dtype=np.float32)
            return Action(arr)

        return policy

    @classmethod
    def _create_use_instrument_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            sat, obj = objects
            cur_sat_x = state.get(sat, "x")
            cur_sat_y = state.get(sat, "y")
            obj_x = state.get(obj, "x")
            obj_y = state.get(obj, "y")
            target_sat_x = cur_sat_x
            target_sat_y = cur_sat_y
            arr = np.array([
                cur_sat_x, cur_sat_y, obj_x, obj_y, target_sat_x, target_sat_y,
                0.0, 0.0, 0.0, 1.0
            ],
                           dtype=np.float32)
            return Action(arr)

        return policy
