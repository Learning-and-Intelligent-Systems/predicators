"""Ground-truth options for the gridworld environment."""

from typing import Dict, Sequence, Set

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
        patty = types["patty"]
        cheese_type = types["cheese"]
        tomato_type = types["tomato"]
        grill_type = types["grill"]
        cutting_board_type = types["cutting_board"]
        robot_type = types["robot"]
        item_type = types["item"]
        station_type = types["station"]

        # Predicates
        Adjacent = predicates["Adjacent"]
        Facing = predicates["Facing"]
        IsCooked = predicates["IsCooked"]
        IsSliced = predicates["IsSliced"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        On = predicates["On"]
        # GoalHack = predicates["GoalHack"]

        # Pick
        def _Pick_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            robot, item = objects
            return Holding.holds(state, [robot, item])

        Pick = ParameterizedOption(
            "Pick",
            types = [robot_type, item_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_pick_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_Pick_terminal
        )

        return {Pick}

    @classmethod
    def _create_pick_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
               params: Array) -> Action:
            import pdb; pdb.set_trace()
            robot, item = objects
            rx = state.get(robot, "col")
            ry = state.get(robot, "row")
            ix = state.get(item, "col")
            iy = state.get(item, "row")

            if GridWorldEnv.Facing_holds(state, [robot, item]):
                return Action[np.array([0, 0, -1, 1, 0], dtype=np.float32)]

            elif GridWorldEnv.Adjacent_holds(state, [robot, item]):
                if rx == ix:
                    if ry > iy:
                        return Action(np.array([0, 0, 2, 0, 0], dtype=np.float32))
                    elif ry < iy:
                        return Action(np.array([0, 0, 0, 0, 0], dtype=np.float32))
                elif ry == iy:
                    if rx > ix:
                        return Action(np.array([0, 0, 1, 0, 0], dtype=np.float32))
                    elif rx < ix:
                        return Action(np.array([0, 0, 3, 0, 0], dtype=np.float32))

            # Move until we are adjacent
            dx = np.clip(ix - rx, -1, 1)
            dy = np.clip(iy - ry, -1, 1)
            return Action(np.array([dx, dy, -1, 0, 0], dtype=np.float32))

        return policy










