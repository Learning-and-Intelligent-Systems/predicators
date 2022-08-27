"""An explorer that performs a lookahead to maximize a state score function."""

from typing import Callable, List, Optional, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.explorers.base_explorer import BaseExplorer
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import NSRT, ExplorationStrategy, GroundAtom, \
    ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT


class GreedyLookaheadExplorer(BaseExplorer):
    """GreedyLookaheadExplorer implementation.

    Sample a certain number of max-length trajectories and pick the one that
    has the highest cumulative score.

    The score function takes the atoms and state as input and returns a
    score, with higher better.
    """
    def __init__(
            self, predicates: Set[Predicate],
            options: Set[ParameterizedOption], types: Set[Type],
            action_space: Box, train_tasks: List[Task], nsrts: Set[NSRT],
            option_model: _OptionModelBase,
            state_score_fn: Callable[[Set[GroundAtom], State], float]) -> None:
        super().__init__(predicates, options, types, action_space, train_tasks)
        self._nsrts = nsrts
        self._option_model = option_model
        self._state_score_fn = state_score_fn

    @classmethod
    def get_name(cls) -> str:
        return "greedy_lookahead"

    def get_exploration_strategy(self, train_task_idx: int,
                                 timeout: int) -> ExplorationStrategy:
        # The goal of the task is ignored.
        task = self._train_tasks[train_task_idx]
        init = task.init
        # Create all applicable ground NSRTs.
        ground_nsrts: List[_GroundNSRT] = []
        for nsrt in sorted(self._nsrts):
            ground_nsrts.extend(utils.all_ground_nsrts(nsrt, list(init)))
        # Sample trajectories by sampling random sequences of NSRTs.
        best_score = -np.inf
        best_options = []
        for _ in range(CFG.greedy_lookahead_max_num_trajectories):
            state = init.copy()
            options = []
            trajectory_length = 0
            total_score = 0.0
            while trajectory_length < CFG.greedy_lookahead_max_traj_length:
                # Sample an NSRT that has preconditions satisfied in the
                # current state.
                ground_nsrt = self._sample_applicable_ground_nsrt(
                    state, ground_nsrts)
                if ground_nsrt is None:  # No applicable NSRTs
                    break
                assert all(a.holds for a in ground_nsrt.preconditions)
                # Sample an option. Note that goal is assumed not used.
                assert not CFG.sampler_learning_use_goals
                option = ground_nsrt.sample_option(state,
                                                   goal=set(),
                                                   rng=self._rng)
                # Assume for now that options will be initiable when the
                # preconditions of the NSRT are satisfied.
                assert option.initiable(state)
                state, num_actions = \
                    self._option_model.get_next_state_and_num_actions(state,
                                                                      option)
                # Special case: if the num actions is 0, something went wrong,
                # and we don't want to use this option after all. To prevent
                # possible infinite loops, just break immediately in this case.
                if num_actions == 0:
                    break
                options.append(option)
                trajectory_length += num_actions
                # Update the total score.
                atoms = utils.abstract(state, self._predicates)
                total_score += self._state_score_fn(atoms, state)
            if total_score > best_score:
                best_score = total_score
                best_options = options
        act_policy = utils.option_plan_to_policy(best_options)
        # When the act policy finishes, an OptionExecutionFailure is raised
        # and caught, terminating the episode.
        termination_function = lambda s: False

        return act_policy, termination_function

    def _sample_applicable_ground_nsrt(
            self, state: State,
            ground_nsrts: Sequence[_GroundNSRT]) -> Optional[_GroundNSRT]:
        """Choose uniformly among the ground NSRTs that are applicable in the
        state."""
        atoms = utils.abstract(state, self._predicates)
        applicable_nsrts = sorted(
            utils.get_applicable_operators(ground_nsrts, atoms))
        if len(applicable_nsrts) == 0:
            return None
        idx = self._rng.choice(len(applicable_nsrts))
        return applicable_nsrts[idx]
