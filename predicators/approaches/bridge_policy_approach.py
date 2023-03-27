"""A simulator-free planning approach that uses a bridge policy to compensate
for lack of downward refinability.

Like bilevel-planning-without-sim, the approach makes an abstract plan and then
greedily samples and executes each skill in the abstract plan. If at any point
the skill fails to achieve its operator effects, control switches to a bridge
policy, which takes the current state (low-level and abstract) and the last
failed skill identity and returns a ground option or "done" (DummyOption). If
"done", control is given back to the planner, which replans. If the bridge
policy outputs a ground option, that ground option is executed until
termination and then the bridge policy is called again.

The bridge policy is so named because it's meant to serve as a "bridge back to
plannability" in states where the planner has gotten stuck.
"""

import logging
from pathlib import Path
from typing import Any, Callable, List, Set, Tuple

from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.oracle_approach import OracleApproach
from predicators.bridge_policies import create_bridge_policy
from predicators.settings import CFG
from predicators.structs import NSRT, Action, DummyOption, Metrics, \
    ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT, _Option
from predicators.utils import OptionExecutionFailure


class BridgePolicyApproach(OracleApproach):
    """A simulator-free bilevel planning approach that uses a bridge policy."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks, task_planning_heuristic,
                         max_skeletons_optimized)
        self._bridge_policy = create_bridge_policy(CFG.bridge_policy)

    @classmethod
    def get_name(cls) -> str:
        return "bridge_policy"

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        goal = task.goal

        # Start by planning. Note that we cannot start with the bridge policy
        # because the bridge policy takes as input the last failed NSRT.
        current_control = "planner"
        current_option, current_nsrt, remaining_nsrt_plan = \
            self._get_next_option_by_planning(task, timeout)

        # Used to detect if an NSRT has failed / is stuck.
        state_history: List[State] = []
        option_history: List[Tuple[_Option, int]] = []  # (option, init time)

        def _policy(s: State) -> Action:
            nonlocal current_control, current_option, current_nsrt, remaining_nsrt_plan
            current_task = Task(s, goal)
            state_history.append(s)

            # Try to execute the current option.
            if not self._current_option_is_stuck(current_option, current_nsrt,
                                                 state_history,
                                                 option_history):
                try:
                    return current_option.policy(s)
                except OptionExecutionFailure:
                    # Something went wrong, so we need a new option.
                    pass

            found_new_option = False

            # Case 1: planner is in control and there is another NSRT in
            # the queue.
            if current_control == "planner" and remaining_nsrt_plan:
                try:
                    current_option, current_nsrt, remaining_nsrt_plan = \
                        self._pop_option_from_nsrt_plan(remaining_nsrt_plan, current_task)
                    found_new_option = True
                except OptionExecutionFailure:
                    # The next option in the queue failed to initiate. Go
                    # on to Case 2.
                    pass

            # Case 2: planner is in control, but failed, so need to switch
            # to bridge policy.
            if current_control == "planner" and not found_new_option:
                current_control = "bridge"
                atoms = utils.abstract(s, self._get_current_predicates())
                current_option = self._bridge_policy(s, atoms, current_nsrt)
                # If the bridge policy is done immediately, then we should
                # attempt to replan.
                if current_option is not DummyOption:
                    found_new_option = True

            # Case 3: bridge policy is in control, but just finished, so
            # we need to switch back to the planner.
            if current_control == "bridge" and not found_new_option:
                assert current_option is DummyOption
                current_control = "planner"
                try:
                    current_option, current_nsrt, remaining_nsrt_plan = \
                        self._get_next_option_by_planning(current_task, timeout)
                    found_new_option = True
                except OptionExecutionFailure:
                    raise ApproachFailure("Bridge policy terminated and "
                                          "then planning failed.")

            assert found_new_option
            t = len(state_history) - 1
            option_history.append((current_option, t))

            try:
                return current_option.policy(s)
            except OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _current_option_is_stuck(
            self, current_option: _Option, current_nsrt: _GroundNSRT,
            state_history: List[State],
            option_history: List[Tuple[_Option, int]]) -> bool:
        # TODO!!!!
        return False

    def _get_next_option_by_planning(
            self, task: Task,
            timeout: float) -> Tuple[_Option, _GroundNSRT, List[_GroundNSRT]]:
        """Returns an initiated option and the remainder of an option plan."""

        # Ensure random over successive calls.
        self._num_calls += 1
        seed = self._seed + self._num_calls
        nsrts = self._get_current_nsrts()
        preds = self._get_current_predicates()

        nsrt_plan, _ = self._run_task_plan(task, nsrts, preds, timeout, seed)
        return self._pop_option_from_nsrt_plan(nsrt_plan, task)

    def _pop_option_from_nsrt_plan(
            self, nsrt_plan: List[_GroundNSRT],
            task: Task) -> Tuple[_Option, _GroundNSRT, List[_GroundNSRT]]:
        if not nsrt_plan:
            raise ApproachFailure("Failed to find an initial abstract plan.")
        nsrt = nsrt_plan.pop(0)
        option = nsrt.sample_option(task.init, task.goal, self._rng)
        if not option.initiable(task.init):
            raise OptionExecutionFailure("Greedy option not initiable.")
        return option, nsrt, nsrt_plan
