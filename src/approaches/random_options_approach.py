"""An approach that just takes random options.
"""

from collections import defaultdict
from typing import Callable, List
from predicators.src.approaches import BaseApproach
from predicators.src.structs import State, Task, Action
from predicators.src.settings import CFG
from predicators.src import utils


class RandomOptionsApproach(BaseApproach):
    """Samples random options (and random parameters for those options).
    """
    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        options = sorted(self._initial_options, key=lambda o: o.name)
        plan: List[Action] = []
        def _policy(state: State) -> Action:
            nonlocal plan
            if plan:
                return plan.pop(0)
            for _ in range(CFG.random_options_max_tries):
                param_opt = options[self._rng.choice(len(options))]
                types_to_objs = defaultdict(list)
                for obj in state:
                    types_to_objs[obj.type].append(obj)
                objs = [types_to_objs[type][self._rng.choice(
                    len(types_to_objs[type]))] for type in param_opt.types]
                params = param_opt.params_space.sample()
                opt = param_opt.ground(objs, params)
                if opt.initiable(state):
                    break
            else:  # fall back to a random action
                return Action(self._action_space.sample())
            _, plan = utils.option_to_trajectory(
                state, self._simulator, opt,
                CFG.max_num_steps_option_rollout)
            return plan.pop(0)
        return _policy
