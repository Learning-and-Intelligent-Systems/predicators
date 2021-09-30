"""Default imports for approaches folder.
"""

from predicators.src.approaches.base_approach import BaseApproach
from predicators.src.approaches.random_actions_approach import \
    RandomActionsApproach

__all__ = [
    "BaseApproach",
    "RandomActionsApproach",
]
