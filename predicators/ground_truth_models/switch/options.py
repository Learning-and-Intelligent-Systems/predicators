"""Ground-truth options for switch environment."""

from typing import Dict, Set

from gym.spaces import Box

from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import ParameterizedOption, Predicate, Type


class SwitchGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Placeholder ground-truth option factory for switch environments."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_switch"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        return set()
