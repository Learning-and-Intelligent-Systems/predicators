"""Behavior (iGibson) environment.
"""

from typing import List, Set, Sequence, Dict, Tuple
import numpy as np
from gym.spaces import Box
try:
    from igibson.envs.behavior_env import BehaviorEnv
    _BEHAVIOR_IMPORTED = True
except ModuleNotFoundError:
    _BEHAVIOR_IMPORTED = False
from predicators.src.envs import BaseEnv
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object, Action, GroundAtom, Image, Array
from predicators.src.settings import CFG
from predicators.src import utils


class BehaviorEnv(BaseEnv):
    """Behavior (iGibson) environment.
    """
    def __init__(self) -> None:
        if not _BEHAVIOR_IMPORTED:
            raise ModuleNotFoundError("Behavior is not installed.")
        super().__init__()
        # Types
        # TODO
        
        # Predicates
        # TODO
        
        # Options
        # TODO
        
        # Objects
        # TODO

    def simulate(self, state: State, action: Action) -> State:
        # TODO
        import ipdb; ipdb.set_trace()

    def get_train_tasks(self) -> List[Task]:
        # TODO
        import ipdb; ipdb.set_trace()

    def get_test_tasks(self) -> List[Task]:
        # TODO
        import ipdb; ipdb.set_trace()

    @property
    def predicates(self) -> Set[Predicate]:
        # TODO
        import ipdb; ipdb.set_trace()

    @property
    def types(self) -> Set[Type]:
        # TODO
        import ipdb; ipdb.set_trace()

    @property
    def options(self) -> Set[ParameterizedOption]:
        # TODO
        import ipdb; ipdb.set_trace()

    @property
    def action_space(self) -> Box:
        # TODO
        import ipdb; ipdb.set_trace()

    def render(self, state: State) -> Image:
        # TODO
        import ipdb; ipdb.set_trace()
