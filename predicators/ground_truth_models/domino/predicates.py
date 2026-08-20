"""Helper predicates for the domino environment.

The grid predicates (DominoAtPos, DominoAtRot, PosClear,
InFrontDirection, InFront, AdjacentTo) are defined canonically by
``GridComponent``; this factory simply delegates to it so there is a
single source of truth.
"""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthPredicateFactory
from predicators.structs import Predicate, Type


class PyBulletDominoGroundTruthPredicateFactory(GroundTruthPredicateFactory):
    """Ground-truth helper predicates for the domino environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "pybullet_domino", "pybullet_domino_real",
            "pybullet_domino_real_geometry"
        }

    @classmethod
    def get_helper_predicates(cls, env_name: str,
                              types: Dict[str, Type]) -> Set[Predicate]:
        """Get helper predicates for the domino environment.

        Delegates to ``GridComponent``, the canonical definition of the
        grid predicates. Only oracle / process-planning approaches
        consume these helpers; agent approaches run grid-free.
        """
        del env_name  # unused

        from predicators.envs.pybullet_domino.components.grid_component import \
            GridComponent  # pylint: disable=import-outside-toplevel
        return GridComponent(domino_type=types["domino"]).get_predicates()
