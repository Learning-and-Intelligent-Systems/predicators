"""Ground-truth options for magic_bin environment."""

from typing import Dict, Set

from gym.spaces import Box

from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import ParameterizedOption, Predicate, Type


class MagicBinGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Placeholder ground-truth option factory for magic_bin environments."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_magic_bin"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        return set()
