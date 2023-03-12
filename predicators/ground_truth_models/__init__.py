"""Implements ground-truth NSRTs and options."""
import abc
from typing import Dict, List, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv, get_or_create_env
from predicators.settings import CFG
from predicators.structs import NSRT, ParameterizedOption, Predicate, Type


class GroundTruthOptionFactory(abc.ABC):
    """Parent class for ground-truth option definitions."""

    @classmethod
    @abc.abstractmethod
    def get_env_names(cls) -> Set[str]:
        """Get the env names that this factory builds options for."""
        raise NotImplementedError("Override me!")

    @classmethod
    @abc.abstractmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        """Create options for the given env name."""
        raise NotImplementedError("Override me!")


class GroundTruthNSRTFactory(abc.ABC):
    """Parent class for ground-truth NSRT definitions."""

    @classmethod
    @abc.abstractmethod
    def get_env_names(cls) -> Set[str]:
        """Get the env names that this factory builds NSRTs for."""
        raise NotImplementedError("Override me!")

    @staticmethod
    @abc.abstractmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        """Create NSRTs for the given env name."""
        raise NotImplementedError("Override me!")


def get_gt_options(env_name: str) -> Set[ParameterizedOption]:
    """Create ground truth options for an env."""
    # This is a work in progress. Gradually moving options out of environments
    # until we can remove them from the environment API entirely.
    env = get_or_create_env(env_name)
    for cls in utils.get_all_subclasses(GroundTruthOptionFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            factory = cls()
            types = {t.name: t for t in env.types}
            predicates = {p.name: p for p in env.predicates}
            options = factory.get_options(env_name, types, predicates,
                                          env.action_space)
            break
    else:  # pragma: no cover
        raise NotImplementedError("Ground-truth options not implemented for "
                                  f"env: {env_name}")
    # Seed the options for reproducibility.
    for option in options:
        option.params_space.seed(CFG.seed)
    return options


def get_gt_nsrts(env_name: str, predicates_to_keep: Set[Predicate],
                 options_to_keep: Set[ParameterizedOption]) -> Set[NSRT]:
    """Create ground truth options for an env."""
    # This is a work in progress. Gradually moving NSRTs into env-specific
    # files; ground_truth_nsrts.py will be deleted.
    env = get_or_create_env(env_name)
    env_options = get_gt_options(env_name)
    assert predicates_to_keep.issubset(env.predicates)
    assert options_to_keep.issubset(env_options)
    for cls in utils.get_all_subclasses(GroundTruthNSRTFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            factory = cls()
            # Give all predicates and options, then filter based on kept ones
            # at the end of this function. This is easier than filtering within
            # the factory itself.
            types = {t.name: t for t in env.types}
            predicates = {p.name: p for p in env.predicates}
            options = {o.name: o for o in env_options}
            nsrts = factory.get_nsrts(env_name, types, predicates, options)
            break
    else:  # pragma: no cover
        raise NotImplementedError("Ground-truth NSRTs not implemented for "
                                  f"env: {env_name}")
    # Filter out excluded predicates from NSRTs, and filter out NSRTs whose
    # options are excluded.
    final_nsrts = set()
    for nsrt in nsrts:
        if nsrt.option not in options_to_keep:
            continue
        nsrt = nsrt.filter_predicates(predicates_to_keep)
        final_nsrts.add(nsrt)
    return final_nsrts


def parse_config_included_options(env: BaseEnv) -> Set[ParameterizedOption]:
    """Parse the CFG.included_options string, given an environment.

    Return the set of included oracle options.

    Note that "all" is not implemented because setting the option_learner flag
    to "no_learning" is the preferred way to include all options.
    """
    if not CFG.included_options:
        return set()
    env_options = get_gt_options(env.get_name())
    included_names = set(CFG.included_options.split(","))
    assert included_names.issubset({option.name for option in env_options}), \
        "Unrecognized option in included_options!"
    included_options = {o for o in env_options if o.name in included_names}
    return included_options


# Convenience functions used by external scripts, tests, and other oracles.


def _get_from_env_by_names(env_name: str, names: Sequence[str],
                           env_attr: str) -> List:
    """Helper for loading types, predicates, and options by name."""
    env = get_or_create_env(env_name)
    name_to_env_obj = {}
    for o in getattr(env, env_attr):
        name_to_env_obj[o.name] = o
    assert set(name_to_env_obj).issuperset(set(names))
    return [name_to_env_obj[name] for name in names]


def _get_types_by_names(env_name: str, names: Sequence[str]) -> List[Type]:
    """Load types from an env given their names."""
    return _get_from_env_by_names(env_name, names, "types")


def _get_predicates_by_names(env_name: str,
                             names: Sequence[str]) -> List[Predicate]:
    """Load predicates from an env given their names."""
    return _get_from_env_by_names(env_name, names, "predicates")


def _get_options_by_names(env_name: str,
                          names: Sequence[str]) -> List[ParameterizedOption]:
    """Load parameterized options from an env given their names."""
    options = get_gt_options(env_name)
    name_to_option = {o.name: o for o in options}
    return [name_to_option[name] for name in names]


# Find the factories.
utils.import_submodules(__path__, __name__)
