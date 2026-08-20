"""Implements ground-truth NSRTs and options."""
import abc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv, get_or_create_env
from predicators.settings import CFG
from predicators.structs import NSRT, CausalProcess, EndogenousProcess, \
    LiftedDecisionList, ParameterizedOption, ParameterizedSampler, Predicate, \
    State, Task, Type


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

    @classmethod
    @abc.abstractmethod
    def get_nsrts(cls, env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        """Create NSRTs for the given env name."""
        raise NotImplementedError("Override me!")


class GroundTruthProcessFactory(abc.ABC):
    """Parent class for ground-truth process definitions."""

    @classmethod
    @abc.abstractmethod
    def get_env_names(cls) -> Set[str]:
        """Get the env names that this factory builds processes for."""
        raise NotImplementedError("Override me!")

    @classmethod
    @abc.abstractmethod
    def get_processes(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        """Create processes for the given env name."""
        raise NotImplementedError("Override me!")


class GroundTruthSimulatorFactory(abc.ABC):
    """Parent class for ground-truth residual-dynamics simulator programs.

    The factory itself only pins an env-name binding. The actual
    simulator components (``RESIDUAL_RULES``, ``PARAM_SPECS``,
    ``RESIDUAL_FEATURES``) live as module-level globals on the same file
    as the subclass, matching the contract used by agent-synthesized
    simulators. ``get_gt_simulator`` reads them via
    ``read_simulator_components``.
    """

    @classmethod
    @abc.abstractmethod
    def get_env_names(cls) -> Set[str]:
        """Get the env names that this factory builds simulators for."""
        raise NotImplementedError("Override me!")


class GroundTruthSamplerFactory(abc.ABC):
    """Parent class for ground-truth per-skill samplers.

    Provides a mapping ``option name -> ParameterizedSampler`` consulted
    by bilevel-sketch refinement (the grid-free counterpart of the NSRT
    samplers in ``processes.py``). Lets an env supply hand-written
    samplers instead of having the agent synthesize them.
    """

    @classmethod
    @abc.abstractmethod
    def get_env_names(cls) -> Set[str]:
        """Get the env names that this factory builds samplers for."""
        raise NotImplementedError("Override me!")

    @classmethod
    @abc.abstractmethod
    def get_samplers(cls, env_name: str) -> Dict[str, ParameterizedSampler]:
        """Return ``option name -> ParameterizedSampler`` for the given env."""
        raise NotImplementedError("Override me!")


class GroundTruthLDLBridgePolicyFactory(abc.ABC):
    """Ground-truth policies implemented with LDLs saved in text files."""

    @classmethod
    @abc.abstractmethod
    def get_env_names(cls) -> Set[str]:
        """Get the env names that this factory builds bridge policies for."""
        raise NotImplementedError("Override me!")

    @classmethod
    @abc.abstractmethod
    def _get_ldl_file(cls) -> Path:
        """Get the path to the LDL file."""
        raise NotImplementedError("Override me!")

    @classmethod
    def get_ldl_bridge_policy(cls, env_name: str, types: Set[Type],
                              predicates: Set[Predicate],
                              options: Set[ParameterizedOption],
                              nsrts: Set[NSRT]) -> LiftedDecisionList:
        """Create LDL bridge policy for the given env name."""
        del env_name, options  # not used
        ldl_file = cls._get_ldl_file()
        with open(ldl_file, "r", encoding="utf-8") as f:
            ldl_str = f.read()
        return utils.parse_ldl_from_str(ldl_str, types, predicates, nsrts)


class GroundTruthTypeFactory(abc.ABC):
    """Parent class for environment-specific helper types."""

    @classmethod
    @abc.abstractmethod
    def get_env_names(cls) -> Set[str]:
        """Get the env names that this factory provides helper types for."""
        raise NotImplementedError("Override me!")

    @classmethod
    @abc.abstractmethod
    def get_helper_types(cls, env_name: str) -> Set[Type]:
        """Get helper types for the given env name."""
        raise NotImplementedError("Override me!")

    @classmethod
    def augment_task_with_helper_objects(cls, task: Task) -> Task:
        """Augment task's initial state with helper objects and features.

        By default, returns the task unchanged. Override this method to
        add environment-specific helper objects to the initial state.
        """
        return task

    @classmethod
    def augment_state_with_helper_objects(cls, state: State) -> State:
        """Augment a single state with helper objects and features.

        By default, returns the state unchanged. Override to re-derive
        helper objects on execution states (e.g. so closed-loop oracle
        policies can keep evaluating helper predicates when the executed
        state is otherwise helper-free).
        """
        return state


class GroundTruthPredicateFactory(abc.ABC):
    """Parent class for environment-specific helper predicates."""

    @classmethod
    @abc.abstractmethod
    def get_env_names(cls) -> Set[str]:
        """Get the env names that this factory provides helper predicates
        for."""
        raise NotImplementedError("Override me!")

    @classmethod
    @abc.abstractmethod
    def get_helper_predicates(cls, env_name: str,
                              types: Dict[str, Type]) -> Set[Predicate]:
        """Get helper predicates for the given env name."""
        raise NotImplementedError("Override me!")


def get_gt_options(env_name: str) -> Set[ParameterizedOption]:
    """Create ground truth options for an env."""
    env = get_or_create_env(env_name)
    for cls in utils.get_all_subclasses(GroundTruthOptionFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            factory = cls()
            # Get environment types and helper types
            helper_types = get_gt_helper_types(env_name)
            all_types = env.types | helper_types
            types = {t.name: t for t in all_types}
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


def get_gt_processes(env_name: str,
                     predicates_to_keep: Set[Predicate],
                     options_to_keep: Set[ParameterizedOption],
                     only_endogenous: bool = False) -> Set[CausalProcess]:
    """Create ground truth processes for an env."""
    env = get_or_create_env(env_name)
    env_options = get_gt_options(env_name)
    helper_predicates = get_gt_helper_predicates(env_name)
    # Helper predicates take precedence over env predicates on name collisions
    # (e.g. the grid's derived InFront replaces the position-based InFront).
    all_predicates = helper_predicates | env.predicates
    helper_types = get_gt_helper_types(env_name)
    all_types = env.types | helper_types
    assert predicates_to_keep.issubset(all_predicates)
    assert options_to_keep.issubset(env_options)
    for cls in utils.get_all_subclasses(GroundTruthProcessFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            factory = cls()
            # Give all predicates and options, then filter based on kept ones
            # at the end of this function. This is easier than filtering within
            # the factory itself.
            types = {t.name: t for t in all_types}
            predicates = {p.name: p for p in all_predicates}
            options = {o.name: o for o in env_options}
            processes = factory.get_processes(env_name, types, predicates,
                                              options)
            break
    else:  # pragma: no cover
        raise NotImplementedError("Ground-truth processes not implemented for "
                                  f"env: {env_name}")
    # Filter out excluded predicates from processes, and filter out processes
    # options are excluded.
    final_processes = set()
    for process in processes:
        if (isinstance(process, EndogenousProcess)
                and process.option not in options_to_keep):
            continue
        process = process.filter_predicates(predicates_to_keep)
        final_processes.add(process)

    # Filter out exogenous processes if only_endogenous is True.
    if only_endogenous:
        final_processes = {
            p
            for p in final_processes if isinstance(p, EndogenousProcess)
        }
    return final_processes


def get_gt_simulator(env_name: str) -> tuple:
    """Load ground-truth residual rules and param specs for an env.

    Returns ``(rules, param_specs, residual_features)``: *rules* is the
    list of residual rule functions, *param_specs* is the list of
    ``ParamSpec`` objects whose ``init_value`` is the GT value, and
    *residual_features* is the ``{type_name: [feat_names]}`` mapping that
    scopes which features the rules predict.

    Locates the right module via the ``GroundTruthSimulatorFactory``
    registry (env-name binding) and reads the three components from
    that module's globals via ``read_simulator_components``. This
    mirrors the loader used for agent-synthesized simulators.
    """
    # Local import to avoid pulling code_sim_learning into ground_truth_models
    # at import time.
    # pylint: disable=import-outside-toplevel
    from predicators.code_sim_learning.utils import read_simulator_components

    for cls in utils.get_all_subclasses(GroundTruthSimulatorFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            module = sys.modules[cls.__module__]
            rules, specs, features = read_simulator_components(vars(module))
            if rules is None or specs is None or features is None:
                raise RuntimeError(
                    f"GT simulator module {cls.__module__} is missing one "
                    "of RESIDUAL_RULES / PARAM_SPECS / RESIDUAL_FEATURES.")
            return rules, specs, features
    raise NotImplementedError("Ground-truth simulator not implemented for "
                              f"env: {env_name}")


def get_gt_samplers(
        env_name: str) -> Optional[Dict[str, ParameterizedSampler]]:
    """Return ``option name -> ground-truth ParameterizedSampler`` for an env.

    Merges the samplers from every ``GroundTruthSamplerFactory`` bound
    to ``env_name``. Returns ``None`` when no factory provides samplers
    for the env, so callers can fall back to learning/uniform sampling.
    """
    out: Dict[str, ParameterizedSampler] = {}
    for cls in utils.get_all_subclasses(GroundTruthSamplerFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            out.update(cls.get_samplers(env_name))
    return out or None


def get_gt_ldl_bridge_policy(env_name: str, types: Set[Type],
                             predicates: Set[Predicate],
                             options: Set[ParameterizedOption],
                             nsrts: Set[NSRT]) -> LiftedDecisionList:
    """Create a lifted decision list for an oracle bridge policy."""
    for cls in utils.get_all_subclasses(GroundTruthLDLBridgePolicyFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            factory = cls()
            return factory.get_ldl_bridge_policy(env_name, types, predicates,
                                                 options, nsrts)
    raise NotImplementedError("Ground-truth bridge policy not implemented for "
                              f"env: {env_name}")


def get_gt_helper_types(env_name: str) -> Set[Type]:
    """Get environment-specific helper types if defined.

    Returns an empty set if no helper types are defined for this
    environment.
    """
    for cls in utils.get_all_subclasses(GroundTruthTypeFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            factory = cls()
            return factory.get_helper_types(env_name)
    return set()


def augment_task_with_helper_objects(task: Task, env_name: str) -> Task:
    """Augment task with environment-specific helper objects if defined.

    Returns the task unchanged if no helper object augmentation is
    defined for this environment.
    """
    for cls in utils.get_all_subclasses(GroundTruthTypeFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            factory = cls()
            return factory.augment_task_with_helper_objects(task)
    return task


def augment_state_with_helper_objects(state: State, env_name: str) -> State:
    """Augment a state with environment-specific helper objects if defined.

    Returns the state unchanged if no helper augmentation is defined for
    this environment. Used to re-derive helper objects on execution
    states so closed-loop oracle policies can keep evaluating helper
    predicates.
    """
    for cls in utils.get_all_subclasses(GroundTruthTypeFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            factory = cls()
            return factory.augment_state_with_helper_objects(state)
    return state


def get_gt_helper_predicates(env_name: str) -> Set[Predicate]:
    """Get environment-specific helper predicates if defined.

    Returns an empty set if no helper predicates are defined for this
    environment.
    """
    for cls in utils.get_all_subclasses(GroundTruthPredicateFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            factory = cls()
            # Get environment types and helper types
            env = get_or_create_env(env_name)
            helper_types = get_gt_helper_types(env_name)
            all_types = env.types | helper_types
            types_dict = {t.name: t for t in all_types}
            return factory.get_helper_predicates(env_name, types_dict)
    return set()


def merge_gt_helper_types(base_types: Set[Type], env_name: str) -> Set[Type]:
    """Union ``base_types`` with the env's GT helper types.

    No-op for envs without a helper factory. Used by both the oracle /
    process-planning approaches and the (opt-in) agent-planning
    approaches so the two paths share one definition of the helper
    vocabulary.
    """
    return base_types | get_gt_helper_types(env_name)


def merge_gt_helper_predicates(base_preds: Set[Predicate],
                               env_name: str) -> Set[Predicate]:
    """Union ``base_preds`` with the env's GT helper predicates.

    Helper predicates take precedence on name collisions (e.g. the
    domino grid's derived ``InFront`` replaces the position-based
    ``InFront``). A plain set union does NOT enforce this: two same-
    named predicates are ``==``-equal but hash differently
    (DerivedPredicate vs Predicate), so both survive the union and
    ``abstract`` then evaluates BOTH -- the looser base predicate can
    inject spurious atoms. Drop any base predicate whose name a helper
    predicate already provides, then union. No-op for envs without a
    helper factory.
    """
    helper_preds = get_gt_helper_predicates(env_name)
    helper_names = {p.name for p in helper_preds}
    return helper_preds | {p for p in base_preds if p.name not in helper_names}


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
