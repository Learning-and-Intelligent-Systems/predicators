"""Handle creation of environments."""

import logging

from predicators import utils
from predicators.envs.base_env import BaseEnv

__all__ = ["BaseEnv"]
_MOST_RECENT_ENV_INSTANCE = {}

# Find the subclasses.
utils.import_submodules(__path__, __name__)


def create_new_env(name: str,
                   do_cache: bool = True,
                   use_gui: bool = True) -> BaseEnv:
    """Create a new instance of an environment from its name.

    If do_cache is True, then cache this env instance so that it can
    later be loaded using get_or_create_env().
    """
    for cls in utils.get_all_subclasses(BaseEnv):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            env = cls(use_gui)
            break
    else:
        raise NotImplementedError(f"Unknown env: {name}")
    if do_cache:
        _MOST_RECENT_ENV_INSTANCE[name] = env
    return env


def get_or_create_env(name: str) -> BaseEnv:
    """Get the most recent cached env instance. If one does not exist in the
    cache, create it using create_new_env().

    If you use this function, you should NOT be doing anything that
    relies on the environment's internal state (i.e., you should not
    call reset() or step()).

    Also note that the GUI is always turned off for environments that are
    newly created by this function. If you want to use the GUI, you should
    create the environment explicitly through create_new_env().
    """
    if name not in _MOST_RECENT_ENV_INSTANCE:
        logging.warning(
            "WARNING: you called get_or_create_env, but I couldn't "
            f"find {name} in the cache. Making a new instance.")
        create_new_env(name, do_cache=True, use_gui=False)
    return _MOST_RECENT_ENV_INSTANCE[name]
