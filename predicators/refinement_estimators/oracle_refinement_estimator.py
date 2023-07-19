"""A hand-written refinement cost estimator."""

from typing import List, Set

from predicators.envs import BaseEnv
from predicators.envs.exit_garage import ExitGarageEnv
from predicators.refinement_estimators import BaseRefinementEstimator
from predicators.settings import CFG
from predicators.structs import GroundAtom, State, Task, _GroundNSRT


class OracleRefinementEstimator(BaseRefinementEstimator):
    """A refinement cost estimator that returns a hand-designed cost
    estimation."""

    @classmethod
    def get_name(cls) -> str:
        return "oracle"

    @property
    def is_learning_based(self) -> bool:
        return False

    def get_cost(self, initial_task: Task, skeleton: List[_GroundNSRT],
                 atoms_sequence: List[Set[GroundAtom]]) -> float:
        env_name = CFG.env
        if env_name == "narrow_passage":
            return narrow_passage_oracle_estimator(self._env,
                                                   initial_task.init, skeleton,
                                                   atoms_sequence)
        if env_name == "exit_garage":
            return exit_garage_oracle_estimator(self._env, initial_task.init,
                                                skeleton, atoms_sequence)

        # Given environment doesn't have an implemented oracle estimator
        raise NotImplementedError(
            f"No oracle refinement cost estimator for env {env_name}")


def narrow_passage_oracle_estimator(
    env: BaseEnv,
    initial_state: State,
    skeleton: List[_GroundNSRT],
    atoms_sequence: List[Set[GroundAtom]],
) -> float:
    """Oracle refinement estimation function for narrow_passage env."""
    del atoms_sequence  # unused

    # Extract door and passage widths from the state
    door_type, _, _, _, wall_type = sorted(env.types)
    door, = initial_state.get_objects(door_type)
    door_width = initial_state.get(door, "width")
    _, middle_wall, right_wall = sorted(initial_state.get_objects(wall_type))
    passage_x = initial_state.get(middle_wall, "x") + \
                initial_state.get(middle_wall, "width")
    passage_width = initial_state.get(right_wall, "x") - passage_x

    # If the door is wider than the passage, then opening the door is
    # beneficial. Otherwise, opening the door should be costly.
    cost_of_open_door = -1 if door_width > passage_width else 1

    # Sum metric of difficulty over skeleton
    cost = 0
    for ground_nsrt in skeleton:
        if ground_nsrt.name == "MoveAndOpenDoor":
            cost += cost_of_open_door
        elif ground_nsrt.name == "MoveToTarget":
            cost += 1
    return cost


def exit_garage_oracle_estimator(
    env: BaseEnv,
    initial_state: State,
    skeleton: List[_GroundNSRT],
    atoms_sequence: List[Set[GroundAtom]],
) -> float:
    """Oracle refinement estimation function for exit_garage env."""
    del atoms_sequence  # unused

    assert isinstance(env, ExitGarageEnv)
    obstacle_radius = env.obstacle_radius
    obstruction_ub = env.exit_top + 2 * obstacle_radius
    obstruction_lb = env.exit_top - env.exit_height - 2 * obstacle_radius

    # Each picked-up obstacle decreases the refinement cost of DriveCarToExit
    # if it is in the direct path of the car to the exit, otherwise it has a
    # positive cost and should be avoided
    cost: float = 0
    for ground_nsrt in skeleton:
        if ground_nsrt.name == "ClearObstacle":
            obstacle = ground_nsrt.objects[1]
            obstacle_y = initial_state.get(obstacle, "y")
            if obstruction_lb < obstacle_y < obstruction_ub:
                cost -= 1
            else:
                cost += 0.5
    return cost
