"""A hand-written refinement cost estimator."""

from typing import List

from predicators.refinement_estimators import BaseRefinementEstimator
from predicators.settings import CFG
from predicators.structs import _GroundNSRT


class OracleRefinementEstimator(BaseRefinementEstimator):
    """A refinement cost estimator that returns a hand-designed cost
    estimation."""

    @classmethod
    def get_name(cls) -> str:
        return "oracle"

    def get_cost(self, skeleton: List[_GroundNSRT]) -> float:
        env_name = CFG.env
        if env_name == "narrow_passage":
            return narrow_passage_oracle_estimator(skeleton)

        # Given environment doesn't have an implemented oracle estimator
        raise NotImplementedError(
            f"No oracle refinement cost estimator for env {env_name}")


def narrow_passage_oracle_estimator(skeleton: List[_GroundNSRT]) -> float:
    """Oracle refinement estimation function for narrow_passage env."""

    # Hard-coded estimated num_samples needed to refine different operators
    move_and_open_door = 1
    move_through_door = 1
    move_through_passage = 3

    # Sum metric of difficulty over skeleton
    cost = 0
    door_open = False
    for ground_nsrt in skeleton:
        if ground_nsrt.name == "MoveAndOpenDoor":
            cost += move_and_open_door
            door_open = True
        elif ground_nsrt.name == "MoveToTarget":
            cost += move_through_door if door_open else move_through_passage
    return cost
