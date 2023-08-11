"""Handle creation of explorers."""

from typing import Callable, Dict, List, Optional, Set

from gym.spaces import Box

from predicators import utils
from predicators.explorers.base_explorer import BaseExplorer
from predicators.explorers.bilevel_planning_explorer import \
    BilevelPlanningExplorer
from predicators.option_model import _OptionModelBase
from predicators.rl.policies import TorchStochasticPolicy
from predicators.settings import CFG
from predicators.structs import NSRT, GroundAtom, NSRTSampler, \
    ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT, \
    _GroundSTRIPSOperator

__all__ = ["BaseExplorer"]

# Find the subclasses.
utils.import_submodules(__path__, __name__)


def create_explorer(
    name: str,
    initial_predicates: Set[Predicate],
    initial_options: Set[ParameterizedOption],
    types: Set[Type],
    action_space: Box,
    train_tasks: List[Task],
    nsrts: Optional[Set[NSRT]] = None,
    option_model: Optional[_OptionModelBase] = None,
    babble_predicates: Optional[Set[Predicate]] = None,
    atom_score_fn: Optional[Callable[[Set[GroundAtom]], float]] = None,
    state_score_fn: Optional[Callable[[Set[GroundAtom], State], float]] = None,
    max_steps_before_termination: Optional[int] = None,
    ground_op_hist: Optional[Dict[_GroundSTRIPSOperator, List[bool]]] = None,
    nsrt_to_explorer_sampler: Optional[Dict[NSRT, NSRTSampler]] = None,
    seen_train_task_idxs: Optional[Set[int]] = None,
    ground_nsrts: Optional[List[_GroundNSRT]] = None,
    exploration_policy: Optional[TorchStochasticPolicy] = None,
    observations_size: Optional[int] = None,
    discrete_actions_size: Optional[int] = None,
    continuous_actions_size: Optional[int] = None
) -> BaseExplorer:
    """Create an explorer given its name."""
    if max_steps_before_termination is None:
        max_steps_before_termination = CFG.max_num_steps_interaction_request
    for cls in utils.get_all_subclasses(BaseExplorer):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            # Special case GLIB because it uses babble predicates and an atom
            # score function.
            if name == "glib":
                assert nsrts is not None
                assert option_model is not None
                assert babble_predicates is not None
                assert atom_score_fn is not None
                explorer = cls(initial_predicates, initial_options, types,
                               action_space, train_tasks,
                               max_steps_before_termination, nsrts,
                               option_model, babble_predicates, atom_score_fn)
            # Special case greedy lookahead because it uses a state score
            # function.
            elif name == "greedy_lookahead":
                assert nsrts is not None
                assert option_model is not None
                assert state_score_fn is not None
                explorer = cls(initial_predicates, initial_options, types,
                               action_space, train_tasks,
                               max_steps_before_termination, nsrts,
                               option_model, state_score_fn)
            # Bilevel planning approaches use NSRTs and an option model.
            elif issubclass(cls, BilevelPlanningExplorer):
                assert nsrts is not None
                assert option_model is not None
                explorer = cls(initial_predicates, initial_options, types,
                               action_space, train_tasks,
                               max_steps_before_termination, nsrts,
                               option_model)
            # Random NSRTs explorer uses NSRTs, but not an option model
            elif name == "random_nsrts":
                assert nsrts is not None
                explorer = cls(initial_predicates, initial_options, types,
                               action_space, train_tasks,
                               max_steps_before_termination, nsrts)
            # Active sampler explorer uses ground_op_hist and no option model.
            elif name == "active_sampler":
                assert ground_op_hist is not None
                assert nsrt_to_explorer_sampler is not None
                assert seen_train_task_idxs is not None
                explorer = cls(initial_predicates, initial_options, types,
                               action_space, train_tasks,
                               max_steps_before_termination, nsrts,
                               ground_op_hist, nsrt_to_explorer_sampler,
                               seen_train_task_idxs)
            # MAPLE explorer uses ground_nsrts, exploration policy, and observations,
            # discrete actions and continuous actions sizes
            elif name == "maple_explorer":
                assert ground_nsrts is not None
                assert observations_size is not None
                assert discrete_actions_size is not None
                assert continuous_actions_size is not None
                explorer = cls(initial_predicates, initial_options, types,
                               action_space, train_tasks,
                               max_steps_before_termination, ground_nsrts, exploration_policy, observations_size, discrete_actions_size, continuous_actions_size)
            else:
                explorer = cls(initial_predicates, initial_options, types,
                               action_space, train_tasks,
                               max_steps_before_termination)
            break
    else:
        raise NotImplementedError(f"Unrecognized explorer: {name}")
    return explorer
