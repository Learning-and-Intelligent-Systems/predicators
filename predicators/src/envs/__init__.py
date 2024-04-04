"""Handle creation of environments."""

import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING

from predicators.src import utils
from predicators.src.envs.base_env import BaseEnv

__all__ = ["BaseEnv"]
_MOST_RECENT_ENV_INSTANCE = {}

if not TYPE_CHECKING:
    # Load all modules so that utils.get_all_subclasses() works.
    for _, module_name, _ in pkgutil.walk_packages(__path__):
        if "__init__" not in module_name:
            # Important! We use an absolute import here to avoid issues
            # with isinstance checking when using relative imports.
            importlib.import_module(f"{__name__}.{module_name}")


def create_new_env(name: str, do_cache: bool = True) -> BaseEnv:
    """Create a new instance of an environment from its name.

    If do_cache is True, then cache this env instance so that it can
    later be loaded using get_or_create_env().
    """
    for cls in utils.get_all_subclasses(BaseEnv):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            env = cls()
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
    """
    if name not in _MOST_RECENT_ENV_INSTANCE:
        logging.warning(
            "WARNING: you called get_or_create_env, but I couldn't "
            f"find {name} in the cache. Making a new instance.")
        create_new_env(name, do_cache=True)
    return _MOST_RECENT_ENV_INSTANCE[name]
