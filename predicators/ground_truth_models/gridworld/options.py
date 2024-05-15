"""Ground-truth options for the gridworld environment."""

from typing import Dict, Sequence, Set, Tuple, Iterator

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.gridworld import GridWorldEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class GridWorldGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the gridworld environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"gridworld"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # Types
        bottom_bun_type = types["bottom_bun"]
        top_bun_type = types["top_bun"]
        patty_type = types["patty"]
        cheese_type = types["cheese"]
        tomato_type = types["tomato"]
        grill_type = types["grill"]
        cutting_board_type = types["cutting_board"]
        robot_type = types["robot"]
        cell_type = types["cell"]
        item_type = types["item"]
        station_type = types["station"]
        object_type = types["object"]

        # Predicates
        Adjacent = predicates["Adjacent"]
        AdjacentToNothing = predicates["AdjacentToNothing"]
        Facing = predicates["Facing"]
        AdjacentNotFacing = predicates["AdjacentNotFacing"]
        IsCooked = predicates["IsCooked"]
        IsSliced = predicates["IsSliced"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        On = predicates["On"]
        # GoalHack = predicates["GoalHack"]

        # Cook
        def _Cook_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            _, patty, _ = objects
            return IsCooked.holds(state, [patty])

        Cook = ParameterizedOption(
            "Cook",
            types = [robot_type, patty_type, grill_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_cook_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_Cook_terminal
        )

        # Move
        def _Move_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            robot, to_obj = objects
            return Facing.holds(state, [robot, to_obj])

        Move = ParameterizedOption(
            "Move",
            types = [robot_type, object_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_move_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_Move_terminal
        )

        # Pick
        def _Pick_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            robot, item = objects
            return Holding.holds(state, [robot, item])

        Pick = ParameterizedOption(
            "Pick",
            types = [robot_type, item_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_pickplace_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_Pick_terminal
        )

        # Place
        def _Place_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            robot, item, station = objects
            return HandEmpty.holds(state, [robot]) and On.holds(state, [item, station])

        Place = ParameterizedOption(
            "Place",
            types = [robot_type, item_type, station_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_pickplace_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_Place_terminal
        )

        return {Move, Pick, Place, Cook}

    @classmethod
    def _create_cook_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
               params: Array) -> Action:
            del state, memory, objects, params  # unused
            action = Action(np.array([0, 0, -1, 1, 0], dtype=np.float32))
            return action

        return policy

    @classmethod
    def _create_move_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
               params: Array) -> Action:
            del memory, params  # unused
            robot, to_obj = objects
            rx, ry = GridWorldEnv.get_position(robot, state)
            ox, oy = GridWorldEnv.get_position(to_obj, state)

            if GridWorldEnv.Adjacent_holds(state, [robot, to_obj]) and not GridWorldEnv.Facing_holds(state, [robot, to_obj]):
                if rx == ox:
                    if ry > oy:
                        action = Action(np.array([0, 0, 2, 0, 0], dtype=np.float32))
                    elif ry < oy:
                        action = Action(np.array([0, 0, 0, 0, 0], dtype=np.float32))
                elif ry == oy:
                    if rx > ox:
                        action = Action(np.array([0, 0, 1, 0, 0], dtype=np.float32))
                    elif rx < ox:
                        action = Action(np.array([0, 0, 3, 0, 0], dtype=np.float32))

            else:
                # Find the path we need to take to become adjacent to the to_obj.
                init = [obj for obj in state if obj.is_instance(GridWorldEnv._cell_type) and (rx, ry) == GridWorldEnv.get_position(obj, state)][0]

                def _check_goal(s):
                    sx, sy = GridWorldEnv.get_position(s, state)
                    if GridWorldEnv.is_adjacent(sx, sy, ox, oy):
                        return True
                    return False

                def _get_successors(s) -> Iterator[Tuple[None, Object, float]]:
                    # find the adjacent cells that are empty
                    empty_cells = GridWorldEnv.get_empty_cells(state)
                    sx, sy = GridWorldEnv.get_position(s, state)
                    adjacent_empty = []
                    for c in empty_cells:
                        cx, cy = GridWorldEnv.get_position(c, state)
                        if GridWorldEnv.is_adjacent(sx, sy, cx, cy):
                            adjacent_empty.append(c)
                    for cell in adjacent_empty:
                        yield (None, cell, 1.0)

                def get_priority(node):
                    x, y = GridWorldEnv.get_position(node.state, state)
                    return abs(x - ox) + abs(y - oy)

                path, _ = utils._run_heuristic_search(
                    init,
                    _check_goal,
                    _get_successors,
                    get_priority,
                )

                # Now, compute the action to take based on the path we have
                # planned. Note that the path is a list of cell objects starting
                # from the cell that the robot is in.
                next = path[1]
                nx, ny = GridWorldEnv.get_position(next, state)
                dx = np.clip(nx - rx, -1, 1)
                dy = np.clip(ny - ry, -1, 1)
                action = Action(np.array([dx, dy, -1, 0, 0], dtype=np.float32))
            return action

        return policy

    @classmethod
    def _create_pickplace_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
               params: Array) -> Action:
            del state, memory, objects, params  # unused
            action = Action(np.array([0, 0, -1, 0, 1], dtype=np.float32))
            return action

        return policy
