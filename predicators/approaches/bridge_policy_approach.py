"""A simulator-free planning approach that uses a bridge policy to compensate
for lack of downward refinability.

Like bilevel-planning-without-sim, the approach makes an abstract plan and then
greedily samples and executes each skill in the abstract plan. If at any point
the skill fails to achieve its operator effects, control switches to a bridge
policy, which takes the current state (low-level and abstract) and the last
failed skill identity and returns a ground option or raises BridgePolicyDone.
If done, control is given back to the planner, which replans. If the bridge
policy outputs a ground option, that ground option is executed until
termination and then the bridge policy is called again.

The bridge policy is so named because it's meant to serve as a "bridge back to
plannability" in states where the planner has gotten stuck.


Oracle bridge policy in painting:
    python predicators/main.py --env painting --approach bridge_policy \
        --seed 0 --painting_lid_open_prob 0.0 \
        --painting_raise_environment_failure False \
        --bridge_policy oracle --debug

Oracle bridge policy in stick button:
    python predicators/main.py --env stick_button --approach bridge_policy \
        --seed 0 --bridge_policy oracle --horizon 10000

Oracle bridge policy in cluttered table:
    python predicators/main.py --env cluttered_table --approach bridge_policy \
        --seed 0 --bridge_policy oracle

Oracle bridge policy in exit garage:
    python predicators/main.py --env exit_garage --approach bridge_policy \
        --seed 0 --bridge_policy oracle \
        --exit_garage_motion_planning_ignore_obstacles True \
        --exit_garage_raise_environment_failure True
"""

import logging
import time
from typing import Callable, List, Set

from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.oracle_approach import OracleApproach
from predicators.bridge_policies import BridgePolicyDone, create_bridge_policy
from predicators.settings import CFG
from predicators.structs import Action, BridgePolicyDoneNSRT, \
    ParameterizedOption, Predicate, State, Task, Type, _Option
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
        predicates = self._get_current_predicates()
        options = initial_options
        nsrts = self._get_current_nsrts() | {BridgePolicyDoneNSRT}
        self._bridge_policy = create_bridge_policy(CFG.bridge_policy, types,
                                                   predicates, options, nsrts)

    @classmethod
    def get_name(cls) -> str:
        return "bridge_policy"

    @property
    def is_learning_based(self) -> bool:
        return self._bridge_policy.is_learning_based

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        start_time = time.perf_counter()
        self._bridge_policy.reset()
        # The bridge policy's internal history is updated every time a new
        # option is selected or a failure is encountered.
        new_option_callback = self._create_new_option_callback()
        # Start by planning. Note that we cannot start with the bridge policy
        # because the bridge policy takes as input the last failed NSRT.
        current_control = "planner"
        option_policy = self._get_option_policy_by_planning(task, timeout)
        current_policy = utils.option_policy_to_policy(
            option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout,
            raise_error_on_repeated_state=True,
            new_option_callback=new_option_callback,
        )

        def _policy(s: State) -> Action:
            nonlocal current_control, current_policy

            if time.perf_counter() - start_time > timeout:
                raise ApproachTimeout("Bridge policy timed out.")

            # Normal execution. Either keep executing the current option, or
            # switch to the next option if it has terminated.
            try:
                return current_policy(s)
            except BridgePolicyDone:
                # Bridge policy declared itself done so switch back to planner.
                assert current_control == "bridge"
                current_control = "planner"
                logging.debug("Switching control from bridge to planner.")
                failed_option = None
                offending_objects = None
            except OptionExecutionFailure as e:
                # An error was encountered, so we need the bridge policy.
                current_control = "bridge"
                failed_option = e.info["last_failed_option"]
                offending_objects = e.info.get("offending_objects", None)
                bridge_policy_failure = (failed_option, offending_objects)
                self._bridge_policy.record_failure(bridge_policy_failure)
                logging.debug("Giving control to bridge.")

            if current_control == "bridge":
                # Planning failed on the first time step.
                if failed_option is None:
                    assert s.allclose(task.init)
                    raise ApproachFailure("Planning failed on init state.")
                option_policy = self._bridge_policy.get_option_policy()
                current_policy = utils.option_policy_to_policy(
                    option_policy,
                    max_option_steps=CFG.max_num_steps_option_rollout,
                    raise_error_on_repeated_state=True,
                    new_option_callback=new_option_callback,
                )
            else:
                assert current_control == "planner"
                current_task = Task(s, task.goal)
                current_control = "planner"
                duration = time.perf_counter() - start_time
                remaining_time = timeout - duration
                option_policy = self._get_option_policy_by_planning(
                    current_task, remaining_time)
                current_policy = utils.option_policy_to_policy(
                    option_policy,
                    max_option_steps=CFG.max_num_steps_option_rollout,
                    raise_error_on_repeated_state=True,
                    new_option_callback=new_option_callback,
                )

            return _policy(s)

        return _policy

    def _get_option_policy_by_planning(
            self, task: Task, timeout: float) -> Callable[[State], _Option]:
        """Raises an OptionExecutionFailure with the last_failed_option in its
        info dict in the case where execution fails."""

        # Ensure random over successive calls.
        self._num_calls += 1
        seed = self._seed + self._num_calls
        nsrts = self._get_current_nsrts()
        preds = self._get_current_predicates()

        nsrt_plan, atoms_seq, _ = self._run_task_plan(task, nsrts, preds,
                                                      timeout, seed)
        return utils.nsrt_plan_to_greedy_option_policy(
            nsrt_plan,
            goal=task.goal,
            rng=self._rng,
            necessary_atoms_seq=atoms_seq)

    def _create_new_option_callback(self) -> Callable[[State, _Option], None]:

        def _new_option_callback(state: State, option: _Option):
            self._bridge_policy.record_state_option(state, option)
            try:
                # Use the option model ONLY to predict environment failures.
                # Will raise an EnvironmentFailure if one is predicted.
                self._option_model.get_next_state_and_num_actions(state, option)
            except EnvironmentFailure as e:
                raise OptionExecutionFailure(
                    f"Environment failure predicted: {repr(e)}.",
                    info={
                        "last_failed_option": cur_option,
                        **e.info
                    })
        return _new_option_callback
