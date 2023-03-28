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
"""

import logging
from typing import Callable, List, Optional, Set, Sequence

from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.oracle_approach import OracleApproach
from predicators.bridge_policies import BridgePolicyDone, create_bridge_policy
from predicators.settings import CFG
from predicators.structs import Action, DummyOption, ParameterizedOption, \
    Predicate, State, Task, Type, _GroundNSRT, InteractionRequest, InteractionResult, Query, HumanNSRTDemoQuery
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
        nsrts = self._get_current_nsrts()
        self._bridge_policy = create_bridge_policy(CFG.bridge_policy,
                                                   predicates, nsrts)

    @classmethod
    def get_name(cls) -> str:
        return "bridge_policy"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        # Start by planning. Note that we cannot start with the bridge policy
        # because the bridge policy takes as input the last failed NSRT.
        current_control = "planner"
        current_policy = self._get_policy_by_planning(task, timeout)

        def _policy(s: State) -> Action:
            nonlocal current_control, current_policy

            # Normal execution. Either keep executing the current option, or
            # switch to the next option if it has terminated.
            try:
                return current_policy(s)
            except BridgePolicyDone:
                assert current_control == "bridge"
                failed_nsrt = None  # not used, but satisfy linting
            except OptionExecutionFailure as e:
                failed_nsrt = e.info.get("last_failed_nsrt", None)

            # Switch control from planner to bridge.
            if current_control == "planner":
                # Planning failed on the first time step.
                if failed_nsrt is None:
                    assert s.allclose(task.init)
                    raise ApproachFailure("Planning failed on init state.")
                logging.debug(f"Failed NSRT: {failed_nsrt.name}"
                              f"{failed_nsrt.objects}.")
                logging.debug("Switching control from planner to bridge.")
                current_control = "bridge"
                current_policy = self._bridge_policy.get_policy(failed_nsrt)
                # Special case: bridge policy passes control immediately back
                # to the planner. For example, if this happened on every time
                # step, then this approach would be performing MPC.
                try:
                    return current_policy(s)
                except BridgePolicyDone:
                    pass
                except OptionExecutionFailure as e:
                    raise ApproachFailure(e.args[0], e.info)

            # Switch control from bridge to planner.
            logging.debug("Switching control from bridge to planner.")
            assert current_control == "bridge"
            current_task = Task(s, task.goal)
            current_control = "planner"
            current_policy = self._get_policy_by_planning(
                current_task, timeout)
            try:
                return current_policy(s)
            except OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _get_policy_by_planning(self, task: Task,
                                timeout: float) -> Callable[[State], Action]:
        """Raises an OptionExecutionFailure with the last_failed_nsrt in its
        info dict in the case where execution fails."""

        # Ensure random over successive calls.
        self._num_calls += 1
        seed = self._seed + self._num_calls
        nsrts = self._get_current_nsrts()
        preds = self._get_current_predicates()

        nsrt_queue, atoms_seq, _ = self._run_task_plan(task, nsrts, preds,
                                                       timeout, seed)
        atoms_queue = utils.compute_necessary_atoms_seq(
            nsrt_queue, atoms_seq, task.goal)[1:]
        cur_nsrt: Optional[_GroundNSRT] = None
        last_nsrt: Optional[_GroundNSRT] = None
        cur_option = DummyOption

        def _policy(state: State) -> Action:
            nonlocal cur_nsrt, last_nsrt, cur_option
            if cur_option is DummyOption or cur_option.terminal(state):
                last_nsrt = cur_nsrt
                if not nsrt_queue:
                    raise OptionExecutionFailure(
                        "Greedy option plan exhausted.",
                        info={"last_failed_nsrt": last_nsrt})
                if last_nsrt is not None:
                    expected_atoms = atoms_queue.pop(0)
                    if not all(a.holds(state) for a in expected_atoms):
                        raise OptionExecutionFailure(
                            "Executing the option "
                            "failed to achieve the NSRT effects.",
                            info={"last_failed_nsrt": last_nsrt})
                cur_nsrt = nsrt_queue.pop(0)
                logging.debug(f"Using NSRT {cur_nsrt.name}{cur_nsrt.objects} "
                              "from planner.")
                cur_option = cur_nsrt.sample_option(state, task.goal,
                                                    self._rng)
                if not cur_option.initiable(state):
                    raise OptionExecutionFailure(
                        "Greedy option not initiable.",
                        info={"last_failed_nsrt": last_nsrt})
            act = cur_option.policy(state)
            return act

        return _policy


    ########################### Active learning ###############################

    def get_interaction_requests(self) -> List[InteractionRequest]:
        requests = []

        # TODO make less ugly
        def create_interaction_request(train_task_idx):
            task = self._train_tasks[train_task_idx]
            planning_policy = self._get_policy_by_planning(task, timeout=CFG.timeout)

            reached_stuck_state = False
            failed_nsrt = None

            def act_policy(s: State) -> Action:
                nonlocal reached_stuck_state, failed_nsrt
                try:
                    return planning_policy(s)
                except OptionExecutionFailure as e:
                    reached_stuck_state = True
                    failed_nsrt = e.info["last_failed_nsrt"]
                    raise e

            def termination_fn(s: State) -> bool:
                return reached_stuck_state

            def query_policy(s: State) -> Optional[Query]:
                if not reached_stuck_state:
                    return None
                assert failed_nsrt is not None
                return HumanNSRTDemoQuery(train_task_idx, failed_nsrt)

            request = InteractionRequest(train_task_idx, act_policy,
                                         query_policy, termination_fn)
            
            return request

        for train_task_idx in self._select_interaction_train_task_idxs():
            request = create_interaction_request(train_task_idx)
            requests.append(request)
        assert len(requests) == CFG.interactive_num_requests_per_cycle
        return requests

    def _select_interaction_train_task_idxs(self) -> List[int]:
        # At the moment, we select train task indices uniformly at
        # random, with replacement. In the future, we may want to
        # try other strategies.
        return self._rng.choice(len(self._train_tasks),
                                size=CFG.interactive_num_requests_per_cycle)
    
    def learn_from_interaction_results(
        self, results: Sequence[InteractionResult]) -> None:

        nsrts = self._get_current_nsrts()
        preds = self._get_current_predicates()

        bridge_demos = []

        for result in results:
            response = result.responses[-1]
            query = response.query
            goal = self._train_tasks[response.query.train_task_idx].goal
            failed_nsrt = response.query.failed_nsrt
            seq_len = len(response.atoms)
            # Find the end of the bridge.
            # The bridge is done if we can generate the remainder of the plan
            # by ourselves, up to random tiebreaking, assuming optimal planning.
            # Equivalently, the remainder of the plan looks rational -- the
            # action selected has optimal cost-to-go.

            # Start by computing the optimal costs to go for each state.
            optimal_ctgs = []
            for state in response.states:
                task = Task(state, goal)
                # Assuming optimal task planning here
                nsrt_plan, _, _ = self._run_task_plan(task, nsrts, preds,
                                                      CFG.timeout, self._seed)
                ctg = len(nsrt_plan)
                optimal_ctgs.append(ctg)

            # Look for first time where the plan suffix decreases by 1.
            bridge_end = seq_len - 1
            for t in range(seq_len):
                suffix = optimal_ctgs[t:]
                decreasing_smoothly = True
                for i, j in zip(suffix[:-1], suffix[1:]):
                    if i != j + 1:
                        decreasing_smoothly = False
                        break
                if decreasing_smoothly:
                    bridge_end = t
                    break

            bridge_demos.append((
                failed_nsrt,
                response.ground_nsrts[:bridge_end],
                response.atoms[:bridge_end+1],
                response.states[:bridge_end+1],
            ))

        return self._bridge_policy.learn_from_demos(bridge_demos)
