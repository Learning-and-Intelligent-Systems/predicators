"""Helper predicates for the fan environment.

The grid predicates (BallAtLoc, ClearLoc, SideOf, FanFacingSide,
OppositeFan) reason over the ``loc``/``side`` helper objects that the
ground-truth models inject into a task at plan time (see
``PyBulletFanGroundTruthTypeFactory.augment_task_with_helper_objects``).
Only oracle / process-planning approaches consume these helpers; agent
approaches run grid-free.

The ``_holds`` logic here is env-free (it depends only on state features
and ``pos_gap``) so the factory can build the predicates without
instantiating the PyBullet env, which is expensive.
"""

import functools
from typing import Dict, Sequence, Set

from predicators.ground_truth_models import GroundTruthPredicateFactory
from predicators.structs import Object, Predicate, State, Type


def _is_close(bx: float, by: float, tx: float, ty: float,
              pos_gap: float) -> bool:
    """Whether (bx, by) is within half a grid cell of (tx, ty).

    Mirrors ``PyBulletFanEnv._is_ball_close_to_position``.
    """
    return abs(bx - tx) < pos_gap / 2 and abs(by - ty) < pos_gap / 2


def _ball_at_loc_holds(state: State, objects: Sequence[Object],
                       pos_gap: float) -> bool:
    ball, pos = objects
    return _is_close(state.get(ball, "x"), state.get(ball, "y"),
                     state.get(pos, "xx"), state.get(pos, "yy"), pos_gap)


def _clear_loc_holds(state: State, objects: Sequence[Object],
                     pos_gap: float) -> bool:
    """Whether the grid cell is clear of walls."""
    pos, = objects
    pos_x, pos_y = state.get(pos, "xx"), state.get(pos, "yy")
    for obj in state:
        if obj.type.name != "wall":
            continue
        if _is_close(pos_x, pos_y, state.get(obj, "x"), state.get(obj, "y"),
                     pos_gap):
            return False
    return True


def _fan_facing_side_holds(state: State, objects: Sequence[Object],
                           pos_gap: float) -> bool:
    del pos_gap  # unused
    fan, side = objects
    return state.get(fan, "facing_side") == state.get(side, "side_idx")


def _opposite_fan_holds(state: State, objects: Sequence[Object],
                        pos_gap: float) -> bool:
    del pos_gap  # unused
    fan1, fan2 = objects
    if fan1.name == fan2.name:
        return False
    side1 = state.get(fan1, "facing_side")
    side2 = state.get(fan2, "facing_side")
    # Sides 0,1 are opposite (left/right); sides 2,3 are opposite (down/up).
    return abs(side1 - side2) == 1 and (side1 // 2) == (side2 // 2)


def _side_of_holds(state: State, objects: Sequence[Object],
                   pos_gap: float) -> bool:
    """Whether pos1 is on the given side of pos2 (one grid cell away)."""
    pos1, pos2, side = objects
    side_val = state.get(side, "side_idx")
    p1x, p1y = state.get(pos1, "xx"), state.get(pos1, "yy")
    p2x, p2y = state.get(pos2, "xx"), state.get(pos2, "yy")
    if side_val == 1:  # left
        return _is_close(p1x + pos_gap, p1y, p2x, p2y, pos_gap)
    if side_val == 0:  # right
        return _is_close(p1x - pos_gap, p1y, p2x, p2y, pos_gap)
    if side_val == 2:  # down
        return _is_close(p1x, p1y - pos_gap, p2x, p2y, pos_gap)
    if side_val == 3:  # up
        return _is_close(p1x, p1y + pos_gap, p2x, p2y, pos_gap)
    return False


class PyBulletFanGroundTruthPredicateFactory(GroundTruthPredicateFactory):
    """Ground-truth helper predicates for the fan environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_fan"}

    @classmethod
    def get_helper_predicates(cls, env_name: str,
                              types: Dict[str, Type]) -> Set[Predicate]:
        """Get the grid helper predicates for the fan environment.

        Only oracle / process-planning approaches consume these; agent
        approaches run grid-free.
        """
        del env_name  # unused

        # pos_gap is a fixed class constant; grabbing it does not construct
        # the (expensive) PyBullet env.
        from predicators.envs.pybullet_fan import \
            PyBulletFanEnv  # pylint: disable=import-outside-toplevel
        pos_gap = PyBulletFanEnv.pos_gap

        ball_type = types["ball"]
        fan_type = types["fan"]
        loc_type = types["loc"]
        side_type = types["side"]

        BallAtLoc = Predicate("BallAtLoc", [ball_type, loc_type],
                              functools.partial(_ball_at_loc_holds,
                                                pos_gap=pos_gap),
                              natural_language_assertion=lambda os:
                              f"ball {os[0]} is at location {os[1]}")
        ClearLoc = Predicate("ClearLoc", [loc_type],
                             functools.partial(_clear_loc_holds,
                                               pos_gap=pos_gap),
                             natural_language_assertion=lambda os:
                             f"location {os[0]} is clear of objects")
        FanFacingSide = Predicate("FanFacingSide", [fan_type, side_type],
                                  functools.partial(_fan_facing_side_holds,
                                                    pos_gap=pos_gap),
                                  natural_language_assertion=lambda os:
                                  f"fan {os[0]} is facing the side {os[1]}")
        OppositeFan = Predicate(
            "OppositeFan", [fan_type, fan_type],
            functools.partial(_opposite_fan_holds, pos_gap=pos_gap),
            natural_language_assertion=lambda os:
            f"fan {os[0]} is facing the opposite side of fan {os[1]}")
        SideOf = Predicate(
            "SideOf", [loc_type, loc_type, side_type],
            functools.partial(_side_of_holds, pos_gap=pos_gap),
            natural_language_assertion=lambda os:
            f"location {os[0]} is to the {os[2]} side of location {os[1]}")

        return {BallAtLoc, ClearLoc, FanFacingSide, OppositeFan, SideOf}
