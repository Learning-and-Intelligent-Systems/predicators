"""Ground-truth NSRTs for the sticky table environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.sticky_table import StickyTableEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class StickyTableGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the sticky table environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"sticky_table"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        # Types
        cube_type = types["cube"]
        table_type = types["table"]

        # Predicates
        OnTable = predicates["OnTable"]
        OnFloor = predicates["OnFloor"]
        Holding = predicates["Holding"]
        HandEmpty = predicates["HandEmpty"]

        # Options
        PickFromTable = options["PickFromTable"]
        PickFromFloor = options["PickFromFloor"]
        PlaceOnTable = options["PlaceOnTable"]
        PlaceOnFloor = options["PlaceOnFloor"]

        nsrts = set()

        # PickFromTable
        cube = Variable("?cube", cube_type)
        table = Variable("?table", table_type)
        parameters = [cube, table]
        option_vars = parameters
        option = PickFromTable
        preconditions = {
            LiftedAtom(OnTable, [cube, table]),
            LiftedAtom(HandEmpty, []),
        }
        add_effects = {LiftedAtom(Holding, [cube])}
        delete_effects = {
            LiftedAtom(OnTable, [cube, table]),
            LiftedAtom(HandEmpty, []),
        }

        pickfromtable_nsrt = NSRT("PickFromTable", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars, null_sampler)
        nsrts.add(pickfromtable_nsrt)

        # PickFromFloor
        parameters = [cube]
        option_vars = parameters
        option = PickFromFloor
        preconditions = {
            LiftedAtom(OnFloor, [cube]),
            LiftedAtom(HandEmpty, []),
        }
        add_effects = {LiftedAtom(Holding, [cube])}
        delete_effects = {
            LiftedAtom(OnFloor, [cube]),
            LiftedAtom(HandEmpty, []),
        }

        pickfromfloor_nsrt = NSRT("PickFromFloor", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars, null_sampler)
        nsrts.add(pickfromfloor_nsrt)

        # PlaceOnTable
        parameters = [cube, table]
        option_vars = parameters
        option = PlaceOnTable
        preconditions = {LiftedAtom(Holding, [cube])}
        add_effects = {
            LiftedAtom(OnTable, [cube, table]),
            LiftedAtom(HandEmpty, []),
        }
        delete_effects = {LiftedAtom(Holding, [cube])}

        def place_on_table_sampler(state: State, goal: Set[GroundAtom],
                                   rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
            del goal  # unused
            cube, table = objs
            table_x = state.get(table, "x")
            table_y = state.get(table, "y")
            table_radius = state.get(table, "radius")
            cube_size = state.get(cube, "size")
            dist = rng.uniform(0, table_radius - cube_size)
            theta = rng.uniform(0, 2 * np.pi)
            x = table_x + dist * np.cos(theta)
            y = table_y + dist * np.sin(theta)
            return np.array([x, y], dtype=np.float32)

        placeontable_nsrt = NSRT("PlaceOnTable", parameters,
                                 preconditions, add_effects, delete_effects,
                                 set(), option, option_vars,
                                 place_on_table_sampler)
        nsrts.add(placeontable_nsrt)

        # PlaceOnFloor
        parameters = [cube]
        option_vars = parameters
        option = PlaceOnFloor
        preconditions = {LiftedAtom(Holding, [cube])}
        add_effects = {
            LiftedAtom(OnFloor, [cube]),
            LiftedAtom(HandEmpty, []),
        }
        delete_effects = {LiftedAtom(Holding, [cube])}

        def place_on_floor_nsrt(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
            del state, goal, objs  # not used
            # Just place in the center of the room.
            x = (StickyTableEnv.x_lb + StickyTableEnv.x_ub) / 2
            y = (StickyTableEnv.y_lb + StickyTableEnv.y_ub) / 2
            return np.array([x, y], dtype=np.float32)

        placeonfloor_nsrt = NSRT("PlaceOnFloor", parameters,
                                 preconditions, add_effects, delete_effects,
                                 set(), option, option_vars,
                                 place_on_floor_nsrt)
        nsrts.add(placeonfloor_nsrt)

        return nsrts
