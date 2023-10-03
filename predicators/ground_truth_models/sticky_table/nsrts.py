"""Ground-truth NSRTs for the sticky table environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.sticky_table import StickyTableEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable


class StickyTableGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the sticky table environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"sticky_table", "sticky_table_tricky_floor"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        # Types
        robot_type = types["robot"]
        cube_type = types["cube"]
        table_type = types["table"]

        # Predicates
        OnTable = predicates["OnTable"]
        OnFloor = predicates["OnFloor"]
        Holding = predicates["Holding"]
        HandEmpty = predicates["HandEmpty"]
        ReachableCube = predicates["IsReachableCube"]
        ReachableSurface = predicates["IsReachableSurface"]

        # Options
        PickFromTable = options["PickFromTable"]
        PickFromFloor = options["PickFromFloor"]
        PlaceOnTable = options["PlaceOnTable"]
        PlaceOnFloor = options["PlaceOnFloor"]
        NavigateToCube = options["NavigateToCube"]
        NavigateToTable = options["NavigateToTable"]

        nsrts = set()

        # PickFromTable
        robot = Variable("?robot", robot_type)
        cube = Variable("?cube", cube_type)
        table = Variable("?table", table_type)
        parameters = [robot, cube, table]
        option_vars = parameters
        option = PickFromTable
        preconditions = {
            LiftedAtom(ReachableSurface, [robot, table]),
            LiftedAtom(OnTable, [cube, table]),
            LiftedAtom(HandEmpty, []),
        }
        add_effects = {LiftedAtom(Holding, [cube])}
        delete_effects = {
            LiftedAtom(OnTable, [cube, table]),
            LiftedAtom(HandEmpty, []),
        }

        def pick_sampler(state: State, goal: Set[GroundAtom],
                         rng: np.random.Generator,
                         objs: Sequence[Object]) -> Array:
            # Sample within ball around center of the object.
            del goal  # unused
            cube = objs[1]
            size = state.get(cube, "size")
            cube_x = state.get(cube, "x") + size / 2
            cube_y = state.get(cube, "y") + size / 2
            dist = rng.uniform(0, size / 4)
            theta = rng.uniform(0, 2 * np.pi)
            x = cube_x + dist * np.cos(theta)
            y = cube_y + dist * np.sin(theta)
            return np.array([1.0, x, y], dtype=np.float32)

        pickfromtable_nsrt = NSRT("PickFromTable", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars, pick_sampler)
        nsrts.add(pickfromtable_nsrt)

        # PickFromFloor
        parameters = [robot, cube]
        option_vars = parameters
        option = PickFromFloor
        preconditions = {
            LiftedAtom(ReachableCube, [robot, cube]),
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
                                  set(), option, option_vars, pick_sampler)
        nsrts.add(pickfromfloor_nsrt)

        # PlaceOnTable
        parameters = [robot, cube, table]
        option_vars = parameters
        option = PlaceOnTable
        preconditions = {
            LiftedAtom(ReachableSurface, [robot, table]),
            LiftedAtom(Holding, [cube])
        }
        add_effects = {
            LiftedAtom(OnTable, [cube, table]),
            LiftedAtom(HandEmpty, []),
        }
        delete_effects = {LiftedAtom(Holding, [cube])}

        def place_on_table_sampler(state: State, goal: Set[GroundAtom],
                                   rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
            del goal  # unused
            _, cube, table = objs
            table_x = state.get(table, "x")
            table_y = state.get(table, "y")
            table_radius = state.get(table, "radius")
            cube_size = state.get(cube, "size")
            dist = rng.uniform(0, table_radius - cube_size)
            theta = rng.uniform(0, 2 * np.pi)
            x = table_x + dist * np.cos(theta)
            y = table_y + dist * np.sin(theta)
            return np.array([1.0, x, y], dtype=np.float32)

        placeontable_nsrt = NSRT("PlaceOnTable", parameters,
                                 preconditions, add_effects, delete_effects,
                                 set(), option, option_vars,
                                 place_on_table_sampler)
        nsrts.add(placeontable_nsrt)

        # PlaceOnFloor
        parameters = [robot, cube]
        option_vars = parameters
        option = PlaceOnFloor
        preconditions = {LiftedAtom(Holding, [cube])}
        add_effects = {
            LiftedAtom(OnFloor, [cube]),
            LiftedAtom(HandEmpty, []),
        }
        delete_effects = {LiftedAtom(Holding, [cube])}

        def place_on_floor_sampler(state: State, goal: Set[GroundAtom],
                                   rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
            del state, goal, rng, objs  # not used
            # Just place in the center of the room.
            x = (StickyTableEnv.x_lb + StickyTableEnv.x_ub) / 2
            y = (StickyTableEnv.y_lb + StickyTableEnv.y_ub) / 2
            return np.array([1.0, x, y], dtype=np.float32)

        placeonfloor_nsrt = NSRT("PlaceOnFloor", parameters,
                                 preconditions, add_effects, delete_effects,
                                 set(), option, option_vars,
                                 place_on_floor_sampler)
        nsrts.add(placeonfloor_nsrt)

        # NavigateToCube
        parameters = [robot, cube]
        option_vars = parameters
        option = NavigateToCube
        preconditions = set()
        add_effects = {ReachableCube([robot, cube])}
        ignore_effects = {ReachableSurface, ReachableCube}

        def navigate_to_obj_sampler(state: State, goal: Set[GroundAtom],
                                    rng: np.random.Generator,
                                    objs: Sequence[Object]) -> Array:
            del goal  # not used
            obj = objs[1]
            if obj.type.name == "cube":
                size = state.get(obj, "size")
            else:
                assert obj.type.name == "table"
                size = state.get(obj, "radius") * 2
            obj_x = state.get(obj, "x") + size / 2
            obj_y = state.get(obj, "y") + size / 2
            nav_dist = StickyTableEnv.reachable_thresh
            dist = rng.uniform(size / 2, nav_dist)
            theta = rng.uniform(0, 2 * np.pi)
            x = obj_x + dist * np.cos(theta)
            y = obj_y + dist * np.sin(theta)
            return np.array([0.0, x, y], dtype=np.float32)

        navigatetocube_nsrt = NSRT("NavigateToCube", parameters,
                                   preconditions, add_effects, set(),
                                   ignore_effects, option, option_vars,
                                   navigate_to_obj_sampler)
        nsrts.add(navigatetocube_nsrt)

        # NavigateToTable
        parameters = [robot, table]
        option_vars = parameters
        option = NavigateToTable
        preconditions = set()
        add_effects = {ReachableSurface([robot, table])}
        ignore_effects = {ReachableSurface, ReachableCube}

        navigatetotable_nsrt = NSRT("NavigateToTable", parameters,
                                    preconditions, add_effects, set(),
                                    ignore_effects, option, option_vars,
                                    navigate_to_obj_sampler)
        nsrts.add(navigatetotable_nsrt)

        return nsrts
