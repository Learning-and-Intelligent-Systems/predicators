"""Definitions of ground truth options for all environments."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State


def get_gt_options(env_name) -> Set[ParameterizedOption]:
    """Create ground truth options for an env."""
    # This is a work in progress. Gradually moving options out of environments
    # until we can remove them from the environment API entirely.
    if env_name == "cover":
        options = _create_cover_options()
    else:
        # In the final version of this function, we will instead raise an
        # error in this case.
        env = get_or_create_env(env_name)
        options = env.options
    # Seed the options for reproducibility.
    for option in options:
        option.params_space.seed(CFG.seed)
    return options


def _create_cover_options() -> Set[ParameterizedOption]:

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del state, memory, objects  # unused
        return Action(params)  # action is simply the parameter

    PickPlace = utils.SingletonParameterizedOption("PickPlace",
                                                   _policy,
                                                   params_space=Box(
                                                       0, 1, (1, )))

    return {PickPlace}
