from typing import Dict, Sequence, Set
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, Predicate, State, Type
from gym.spaces import Box

__all__ = ['IdentityGroundTruthOptionFactory']


class IdentityGroundTruthOptionFactory(GroundTruthOptionFactory):
    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"shelves2d", "donuts", "statue", "bokksu", "jigsaw", "jigsawrelative", "wbox"}

    @classmethod
    def get_options(
        cls, env_name: str, types: Dict[str, Type],
        predicates: Dict[str, Predicate],
        action_space: Box
    ) -> Set[ParameterizedOption]:
        return {ParameterizedOption(
            "Act",
            [],  # From, To
            action_space,
            cls.act,
            cls.initiable,
            cls.terminal
        )}

    @classmethod
    def initiable(cls, state: State, data: Dict, objects: Sequence[Object], arr: Array) -> bool:
        return True

    @classmethod
    def act(cls, state: State, data: Dict, objects: Sequence[Object], arr: Array) -> Action:
        assert not objects
        return Action(arr)

    @classmethod
    def terminal(cls, state: State, data: Dict, objects: Sequence[Object], arr: Array) -> bool:
        terminal_executed = data.get("terminal_executed", False)
        data["terminal_executed"] = True
        return terminal_executed
