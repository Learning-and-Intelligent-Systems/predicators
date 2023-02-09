"""A hand-written refinement cost estimator."""

from typing import List, Set

from predicators.refinement_estimators import BaseRefinementEstimator
from predicators.settings import CFG
from predicators.structs import GroundAtom, State, _GroundNSRT


class OracleRefinementEstimator(BaseRefinementEstimator):
    """A refinement cost estimator that returns a hand-designed cost
    estimation."""

    @classmethod
    def get_name(cls) -> str:
        return "oracle"

    @property
    def is_learning_based(self) -> bool:
        return False

    def get_cost(self, initial_state: State, skeleton: List[_GroundNSRT],
                 atoms_sequence: List[Set[GroundAtom]]) -> float:
        env_name = CFG.env
        if env_name == "narrow_passage":
            return narrow_passage_oracle_estimator(skeleton, atoms_sequence)

        # Given environment doesn't have an implemented oracle estimator
        raise NotImplementedError(
            f"No oracle refinement cost estimator for env {env_name}")


def narrow_passage_oracle_estimator(
    skeleton: List[_GroundNSRT],
    atoms_sequence: List[Set[GroundAtom]],
) -> float:
    """Oracle refinement estimation function for narrow_passage env."""
    del atoms_sequence  # unused

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
