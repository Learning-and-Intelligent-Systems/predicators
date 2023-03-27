"""A hand-written bridge policy."""

from typing import Callable, List, Set

from predicators.bridge_policies import BaseBridgePolicy
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import GroundAtom, State, Task, _GroundNSRT, _Option


class OracleBridgePolicy(BaseBridgePolicy):
    """A hand-written bridge policy."""

    def __init__(self) -> None:
        super().__init__()
        self._oracle_policy = self._create_oracle_policy()

    @classmethod
    def get_name(cls) -> str:
        return "oracle"

    @property
    def is_learning_based(self) -> bool:
        return False

    def __call__(self, state: State, atoms: Set[GroundAtom],
                 failed_nsrt: _GroundNSRT) -> _Option:
        return self._oracle_policy(state, atoms, failed_nsrt)

    def _create_oracle_policy(
            self) -> Callable[[State, Set[GroundAtom], _GroundNSRT], _Option]:
        env_name = CFG.env
        if env_name == "painting":
            return _painting_oracle_bridge_policy
        raise NotImplementedError(f"No oracle bridge policy for {env_name}")


def _painting_oracle_bridge_policy(state: State, atoms: Set[GroundAtom],
                                   failed_nsrt: _GroundNSRT) -> _Option:
    import ipdb
    ipdb.set_trace()
