"""Handle creation of explorers."""

from typing import Callable, Dict, List, Optional, Set

from gym.spaces import Box

from predicators import utils
from predicators.competence_models import SkillCompetenceModel
from predicators.explorers.base_explorer import BaseExplorer
from predicators.explorers.bilevel_planning_explorer import \
    BilevelPlanningExplorer
from predicators.ml_models import MapleQFunction
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import NSRT, GroundAtom, \
    NSRTSamplerWithEpsilonIndicator, ParameterizedOption, Predicate, State, \
    Task, Type, _GroundSTRIPSOperator

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
    competence_models: Optional[Dict[_GroundSTRIPSOperator,
                                     SkillCompetenceModel]] = None,
    nsrt_to_explorer_sampler: Optional[Dict[
        NSRT, NSRTSamplerWithEpsilonIndicator]] = None,
    seen_train_task_idxs: Optional[Set[int]] = None,
    pursue_task_goal_first: Optional[bool] = None,
    maple_q_function: Optional[MapleQFunction] = None,
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
                assert competence_models is not None
                assert nsrt_to_explorer_sampler is not None
                assert seen_train_task_idxs is not None
                assert pursue_task_goal_first is not None
                explorer = cls(initial_predicates, initial_options, types,
                               action_space, train_tasks,
                               max_steps_before_termination, nsrts,
                               ground_op_hist, competence_models,
                               nsrt_to_explorer_sampler, seen_train_task_idxs,
                               pursue_task_goal_first)
            elif name == "maple_q":
                assert maple_q_function is not None
                explorer = cls(initial_predicates, initial_options, types,
                               action_space, train_tasks,
                               max_steps_before_termination, nsrts,
                               maple_q_function)
            else:
                explorer = cls(initial_predicates, initial_options, types,
                               action_space, train_tasks,
                               max_steps_before_termination)
            break
    else:
        raise NotImplementedError(f"Unrecognized explorer: {name}")
    return explorer
