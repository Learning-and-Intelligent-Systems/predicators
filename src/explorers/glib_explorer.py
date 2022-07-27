"""An explorer that uses goal-literal babbling (GLIB)."""

import logging
from typing import Callable, List, Set

from gym.spaces import Box

from predicators.src import utils
from predicators.src.explorers.bilevel_planning_explorer import \
    BilevelPlanningExplorer
from predicators.src.explorers.random_options_explorer import \
    RandomOptionsExplorer
from predicators.src.option_model import _OptionModelBase
from predicators.src.planning import PlanningFailure, PlanningTimeout
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, ExplorationStrategy, GroundAtom, \
    ParameterizedOption, Predicate, Task, Type


class GLIBExplorer(BilevelPlanningExplorer):
    """GLIBExplorer implementation.

    Sample goals, score each of them, and then try planning starting
    from the highest-scoring goal, terminating at the first goal for
    which we find a plan.
    """

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task], nsrts: Set[NSRT],
                 option_model: _OptionModelBase,
                 babble_predicates: Set[Predicate],
                 atom_score_fn: Callable[[Set[GroundAtom]], float]) -> None:
        super().__init__(predicates, options, types, action_space, train_tasks,
                         nsrts, option_model)
        self._babble_predicates = babble_predicates
        self._atom_score_fn = atom_score_fn  # higher is better
        # GLIB falls back to random options.
        self._fallback_explorer = RandomOptionsExplorer(
            predicates, options, types, action_space, train_tasks)

    @classmethod
    def get_name(cls) -> str:
        return "glib"

    def get_exploration_strategy(self, train_task_idx: int,
                                 timeout: int) -> ExplorationStrategy:
        # The goal of the task is ignored.
        task = self._train_tasks[train_task_idx]
        init = task.init
        # Detect and filter out static predicates.
        static_preds = utils.get_static_preds(self._nsrts,
                                              self._babble_predicates)
        preds = self._babble_predicates - static_preds
        # Sample possible goals to plan toward.
        ground_atom_universe = utils.all_possible_ground_atoms(init, preds)
        # If there are no possible goals, fall back to random immediately.
        if not ground_atom_universe:
            logging.info("No possible goals, falling back to random.")
            return self._fallback_explorer.get_exploration_strategy(
                train_task_idx, timeout)
        possible_goals = utils.sample_subsets(
            ground_atom_universe,
            num_samples=CFG.glib_num_babbles,
            min_set_size=CFG.glib_min_goal_size,
            max_set_size=CFG.glib_max_goal_size,
            rng=self._rng)
        # Exclude goals that hold in the initial state to prevent trivial
        # interaction requests.
        possible_goal_lst = [
            g for g in possible_goals if not all(a.holds(init) for a in g)
        ]
        goal_list = sorted(possible_goal_lst,
                           key=self._atom_score_fn,
                           reverse=True)  # largest to smallest
        task_list = [Task(init, goal) for goal in goal_list]
        # Find the first solvable task.
        for glib_task in task_list:
            try:
                logging.info("Solving for policy...")
                strategy = self._solve(glib_task, timeout=timeout)
                logging.info(f"GLIB found a plan with goal {glib_task.goal}.")
                return strategy
            except (PlanningFailure, PlanningTimeout):
                continue
        # Fall back to a random exploration strategy if no solvable task
        # can be found.
        logging.info("No solvable task found, falling back to random.")
        return self._fallback_explorer.get_exploration_strategy(
            train_task_idx, timeout)
