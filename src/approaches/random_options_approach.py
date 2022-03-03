"""An approach that just executes random options."""

from typing import Callable
from predicators.src.approaches import BaseApproach
from predicators.src.structs import State, Task, Action, DummyOption
from predicators.src.settings import CFG
from predicators.src import utils


class RandomOptionsApproach(BaseApproach):
    """Samples random options (and random parameters for those options)."""

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        options = sorted(self._initial_options, key=lambda o: o.name)
        cur_option = DummyOption
        cur_option_ind = 0

        def policy(state: State) -> Action:
            nonlocal cur_option, cur_option_ind
            if cur_option is DummyOption or cur_option.terminal(state):
                for _ in range(CFG.random_options_max_tries):
                    param_opt = options[self._rng.choice(len(options))]
                    objs = utils.get_random_object_combination(
                        list(state), param_opt.types, self._rng)
                    if objs is None:
                        continue
                    params = param_opt.params_space.sample()
                    opt = param_opt.ground(objs, params)
                    if opt.initiable(state):
                        cur_option = opt
                        cur_option_ind = 0
                        break
                else:  # fall back to a random action
                    return Action(self._action_space.sample())
            act = cur_option.policy(state)
            cur_option_ind += 1
            return act

        return policy
