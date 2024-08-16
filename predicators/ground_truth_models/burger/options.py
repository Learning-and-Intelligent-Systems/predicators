"""Ground-truth options for the burger environment."""

from typing import Dict, Iterator, Optional, Sequence, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.burger import BurgerEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class BurgerGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the burger environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"burger"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # Types
        # top_bun_type = types["top_bun"]
        # bottom_bun_type = types["bottom_bun"]
        # cheese_type = types["cheese"]
        tomato_type = types["lettuce"]
        patty_type = types["patty"]

        grill_type = types["grill"]
        cutting_board_type = types["cutting_board"]
        robot_type = types["robot"]

        item_type = types["item"]
        # station_type = types["station"]
        object_type = types["object"]

        # Predicates
        # Adjacent = predicates["Adjacent"]
        # AdjacentToNothing = predicates["AdjacentToNothing"]
        Facing = predicates["Facing"]
        # AdjacentNotFacing = predicates["AdjacentNotFacing"]
        IsCooked = predicates["IsCooked"]
        IsSliced = predicates["IsSliced"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        On = predicates["On"]

        # Slice
        def _Slice_terminal(state: State, memory: Dict,
                            objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            _, tomato, _ = objects
            return IsSliced.holds(state, [tomato])

        Slice = ParameterizedOption(
            "Chop",
            types=[robot_type, tomato_type, cutting_board_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_slice_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_Slice_terminal)

        # Cook
        def _Cook_terminal(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            _, patty, _ = objects
            return IsCooked.holds(state, [patty])

        Cook = ParameterizedOption("Cook",
                                   types=[robot_type, patty_type, grill_type],
                                   params_space=Box(0, 1, (0, )),
                                   policy=cls._create_cook_policy(),
                                   initiable=lambda s, m, o, p: True,
                                   terminal=_Cook_terminal)

        # Move
        def _Move_terminal(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            robot, to_obj = objects
            return Facing.holds(state, [robot, to_obj])

        Move = ParameterizedOption("Move",
                                   types=[robot_type, object_type],
                                   params_space=Box(0, 1, (0, )),
                                   policy=cls._create_move_policy(),
                                   initiable=lambda s, m, o, p: True,
                                   terminal=_Move_terminal)

        # Pick
        def _Pick_terminal(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            robot, item = objects
            return Holding.holds(state, [robot, item])

        Pick = ParameterizedOption("Pick",
                                   types=[robot_type, item_type],
                                   params_space=Box(0, 1, (0, )),
                                   policy=cls._create_pickplace_policy(),
                                   initiable=lambda s, m, o, p: True,
                                   terminal=_Pick_terminal)

        # Place
        def _Place_terminal(state: State, memory: Dict,
                            objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            robot, item, obj = objects
            return HandEmpty.holds(state, [robot]) and On.holds(
                state, [item, obj])

        Place = ParameterizedOption("Place",
                                    types=[robot_type, item_type, object_type],
                                    params_space=Box(0, 1, (0, )),
                                    policy=cls._create_pickplace_policy(),
                                    initiable=lambda s, m, o, p: True,
                                    terminal=_Place_terminal)

        return {Move, Pick, Place, Cook, Slice}

    @classmethod
    def _create_slice_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params  # unused
            action = Action(np.array([0, 0, -1, 1, 0], dtype=np.float32))
            return action

        return policy

    @classmethod
    def _create_cook_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params  # unused
            action = Action(np.array([0, 0, -1, 1, 0], dtype=np.float32))
            return action

        return policy

    @classmethod
    def _move_subpolicy(cls, state: State, robot: Object,
                        to_obj: Object) -> Optional[Action]:
        rx, ry = BurgerEnv.get_position(robot, state)
        ox, oy = BurgerEnv.get_position(to_obj, state)

        if BurgerEnv.Facing_holds(state, [robot, to_obj]):
            return None

        # If we're adjacent to the object but not facing it, turn to face
        # it.
        if BurgerEnv.Adjacent_holds(state, [robot, to_obj]) and \
            not BurgerEnv.Facing_holds(state, [robot, to_obj]):
            if rx == ox:
                if ry > oy:
                    action = Action(np.array([0, 0, 2, 0, 0],
                                             dtype=np.float32))
                elif ry < oy:
                    action = Action(np.array([0, 0, 0, 0, 0],
                                             dtype=np.float32))
            elif ry == oy:
                if rx > ox:
                    action = Action(np.array([0, 0, 1, 0, 0],
                                             dtype=np.float32))
                elif rx < ox:
                    action = Action(np.array([0, 0, 3, 0, 0],
                                             dtype=np.float32))

        else:
            # Find the path we need to take to the object.
            init = BurgerEnv.get_position(robot, state)

            def _check_goal(s: Tuple[int, int]) -> bool:
                sx, sy = s
                if BurgerEnv.is_adjacent(sx, sy, ox, oy):
                    return True
                return False

            def _get_successors(s: Tuple[int, int]) -> \
                Iterator[Tuple[None, Tuple[int, int], float]]:
                # Find the adjacent cells that are empty.
                empty_cells = BurgerEnv.get_empty_cells(state)
                sx, sy = s
                adjacent_empty = []
                for cell in empty_cells:
                    cx, cy = cell
                    if BurgerEnv.is_adjacent(sx, sy, cx, cy):
                        adjacent_empty.append(cell)
                for cell in adjacent_empty:
                    yield (None, cell, 1.0)

            def heuristic(s: Tuple[int, int]) -> float:
                sx, sy = s
                return abs(sx - ox) + abs(sy - oy)

            path, _ = utils.run_astar(initial_state=init,
                                      check_goal=_check_goal,
                                      get_successors=_get_successors,
                                      heuristic=heuristic)

            # Now, compute the action to take based on the path we have
            # planned. Note that the path is a list of (x, y) tuples
            # starting from the location of the robot.
            nx, ny = path[1]
            dx = np.clip(nx - rx, -1, 1)
            dy = np.clip(ny - ry, -1, 1)
            action = Action(np.array([dx, dy, -1, 0, 0], dtype=np.float32))

        return action

    @classmethod
    def _create_move_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            robot, to_obj = objects
            # We put the move policy code in a different function so that
            # subclasses can use it.
            return cls._move_subpolicy(state, robot, to_obj)

        return policy

    @classmethod
    def _create_pickplace_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params  # unused
            action = Action(np.array([0, 0, -1, 0, 1], dtype=np.float32))
            return action

        return policy


class BurgerNoMoveGroundTruthOptionFactory(BurgerGroundTruthOptionFactory):
    """Ground-truth options for the Burger environment with no distinct
    movement options."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"burger_no_move"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # Types
        # top_bun_type = types["top_bun"]
        # bottom_bun_type = types["bottom_bun"]
        # cheese_type = types["cheese"]
        tomato_type = types["lettuce"]
        patty_type = types["patty"]

        grill_type = types["grill"]
        cutting_board_type = types["cutting_board"]
        robot_type = types["robot"]

        item_type = types["item"]
        # station_type = types["station"]
        object_type = types["object"]

        # Predicates
        # Adjacent = predicates["Adjacent"]
        # AdjacentToNothing = predicates["AdjacentToNothing"]
        # Facing = predicates["Facing"]
        # AdjacentNotFacing = predicates["AdjacentNotFacing"]
        IsCooked = predicates["IsCooked"]
        IsSliced = predicates["IsSliced"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        On = predicates["On"]

        # Slice
        def _Slice_terminal(state: State, memory: Dict,
                            objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            _, tomato, _ = objects
            return IsSliced.holds(state, [tomato])

        Slice = ParameterizedOption(
            "Chop",
            types=[robot_type, tomato_type, cutting_board_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_slice_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_Slice_terminal)

        # Cook
        def _Cook_terminal(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            _, patty, _ = objects
            return IsCooked.holds(state, [patty])

        Cook = ParameterizedOption("Cook",
                                   types=[robot_type, patty_type, grill_type],
                                   params_space=Box(0, 1, (0, )),
                                   policy=cls._create_cook_policy(),
                                   initiable=lambda s, m, o, p: True,
                                   terminal=_Cook_terminal)

        # Pick
        def _Pick_terminal(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            robot, item = objects
            return Holding.holds(state, [robot, item])

        Pick = ParameterizedOption("Pick",
                                   types=[robot_type, item_type],
                                   params_space=Box(0, 1, (0, )),
                                   policy=cls._create_pick_policy(),
                                   initiable=lambda s, m, o, p: True,
                                   terminal=_Pick_terminal)

        # Place
        def _Place_terminal(state: State, memory: Dict,
                            objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            robot, item, obj = objects
            return HandEmpty.holds(state, [robot]) and On.holds(
                state, [item, obj])

        Place = ParameterizedOption("Place",
                                    types=[robot_type, item_type, object_type],
                                    params_space=Box(0, 1, (0, )),
                                    policy=cls._create_place_policy(),
                                    initiable=lambda s, m, o, p: True,
                                    terminal=_Place_terminal)

        return {Pick, Place, Cook, Slice}

    @classmethod
    def _create_slice_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            robot, item, _ = objects
            move_action = cls._move_subpolicy(state, robot, item)
            slice_action = Action(np.array([0, 0, -1, 1, 0], dtype=np.float32))
            if move_action is not None:
                return move_action
            return slice_action

        return policy

    @classmethod
    def _create_cook_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            robot, item, _ = objects
            move_action = cls._move_subpolicy(state, robot, item)
            cook_action = Action(np.array([0, 0, -1, 1, 0], dtype=np.float32))
            if move_action is not None:
                return move_action
            return cook_action

        return policy

    @classmethod
    def _create_pick_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            robot, item = objects
            move_action = cls._move_subpolicy(state, robot, item)
            pick_action = Action(np.array([0, 0, -1, 0, 1], dtype=np.float32))
            if move_action is not None:
                return move_action
            return pick_action

        return policy

    @classmethod
    def _create_place_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            robot, _, to_obj = objects
            move_action = cls._move_subpolicy(state, robot, to_obj)
            place_action = Action(np.array([0, 0, -1, 0, 1], dtype=np.float32))
            if move_action is not None:
                return move_action
            return place_action

        return policy
