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

Learned bridge policy in painting:
    python predicators/main.py --env painting --approach bridge_policy \
        --seed 0 --painting_lid_open_prob 0.0 \
        --painting_raise_environment_failure False --max_initial_demos 0 \
        --interactive_num_requests_per_cycle 1 --num_online_learning_cycles 1 \
        --debug --num_test_tasks 3 --segmenter every_step --demonstrator human

Oracle bridge policy in stick button:
    python predicators/main.py --env stick_button --approach bridge_policy \
        --seed 0 --bridge_policy oracle --debug --horizon 10000
"""

import logging
from typing import Callable, List, Optional, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.oracle_approach import OracleApproach
from predicators.bridge_policies import BridgePolicyDone, create_bridge_policy
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.settings import CFG
from predicators.structs import Action, BridgeDataset, DemonstrationQuery, \
    DemonstrationResponse, InteractionRequest, InteractionResult, \
    ParameterizedOption, Predicate, Query, State, Task, Type, _Option
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
        nsrts = self._get_current_nsrts()
        self._bridge_policy = create_bridge_policy(CFG.bridge_policy,
                                                   predicates, options, nsrts)

    @classmethod
    def get_name(cls) -> str:
        return "bridge_policy"

    @property
    def is_learning_based(self) -> bool:
        return self._bridge_policy.is_learning_based

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._bridge_policy.reset()
        # Start by planning. Note that we cannot start with the bridge policy
        # because the bridge policy takes as input the last failed NSRT.
        current_control = "planner"
        option_policy = self._get_option_policy_by_planning(task, timeout)
        current_policy = utils.option_policy_to_policy(
            option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout,
            raise_error_on_repeated_state=True,
        )

        def _policy(s: State) -> Action:
            nonlocal current_control, current_policy

            # Normal execution. Either keep executing the current option, or
            # switch to the next option if it has terminated.
            try:
                return current_policy(s)
            except BridgePolicyDone:
                assert current_control == "bridge"
                failed_option = None  # not used, but satisfy linting
            except OptionExecutionFailure as e:
                failed_option = e.info["last_failed_option"]
                self._bridge_policy.record_failed_option(failed_option)

            # Switch control from planner to bridge.
            if current_control == "planner":
                # Planning failed on the first time step.
                if failed_option is None:
                    assert s.allclose(task.init)
                    raise ApproachFailure("Planning failed on init state.")
                logging.debug(f"Failed option: {failed_option.name}"
                              f"{failed_option.objects}.")
                logging.debug("Switching control from planner to bridge.")
                current_control = "bridge"
                option_policy = self._bridge_policy.get_option_policy()
                current_policy = utils.option_policy_to_policy(
                    option_policy,
                    max_option_steps=CFG.max_num_steps_option_rollout,
                    raise_error_on_repeated_state=True,
                )
                # Special case: bridge policy passes control immediately back
                # to the planner. For example, if this happened on every time
                # step, then this approach would be performing MPC.
                try:
                    return current_policy(s)
                except BridgePolicyDone:
                    pass

            # Switch control from bridge to planner.
            logging.debug("Switching control from bridge to planner.")
            assert current_control == "bridge"
            current_task = Task(s, task.goal)
            current_control = "planner"
            option_policy = self._get_option_policy_by_planning(
                current_task, timeout)
            current_policy = utils.option_policy_to_policy(
                option_policy,
                max_option_steps=CFG.max_num_steps_option_rollout,
                raise_error_on_repeated_state=True,
            )
            try:
                return current_policy(s)
            except OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

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

    ########################### Active learning ###############################

    def get_interaction_requests(self) -> List[InteractionRequest]:
        requests = []
        for train_task_idx in self._select_interaction_train_task_idxs():
            request = self._create_interaction_request(train_task_idx)
            requests.append(request)
        assert len(requests) == CFG.interactive_num_requests_per_cycle
        return requests

    def _select_interaction_train_task_idxs(self) -> List[int]:
        # At the moment, we select train task indices uniformly at
        # random, with replacement. In the future, we may want to
        # try other strategies.
        return self._rng.choice(len(self._train_tasks),
                                size=CFG.interactive_num_requests_per_cycle)

    def _create_interaction_request(self,
                                    train_task_idx: int) -> InteractionRequest:
        task = self._train_tasks[train_task_idx]
        option_policy = self._get_option_policy_by_planning(task, CFG.timeout)
        planning_policy = utils.option_policy_to_policy(
            option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout,
            raise_error_on_repeated_state=True,
        )

        reached_stuck_state = False
        failed_option = None

        def _act_policy(s: State) -> Action:
            nonlocal reached_stuck_state, failed_option
            try:
                return planning_policy(s)
            except OptionExecutionFailure as e:
                reached_stuck_state = True
                failed_option = e.info["last_failed_option"]
                raise e

        def _termination_fn(s: State) -> bool:
            del s  # unused
            return reached_stuck_state

        def _query_policy(s: State) -> Optional[Query]:
            del s  # unused
            if not reached_stuck_state:
                return None
            assert failed_option is not None
            return DemonstrationQuery(train_task_idx,
                                      {"failed_option": failed_option})

        request = InteractionRequest(train_task_idx, _act_policy,
                                     _query_policy, _termination_fn)

        return request

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:

        nsrts = self._get_current_nsrts()
        preds = self._get_current_predicates()

        bridge_dataset: BridgeDataset = []

        for result in results:
            response = result.responses[-1]
            assert isinstance(response, DemonstrationResponse)
            query = response.query
            assert isinstance(query, DemonstrationQuery)
            goal = self._train_tasks[query.train_task_idx].goal
            failed_option = query.get_info("failed_option")

            # Abstract and segment the trajectory.
            traj = response.teacher_traj
            assert traj is not None
            atom_traj = [utils.abstract(s, preds) for s in traj.states]
            segmented_traj = segment_trajectory((traj, atom_traj))
            states = utils.segment_trajectory_to_state_sequence(segmented_traj)
            atoms = utils.segment_trajectory_to_atoms_sequence(segmented_traj)
            assert len(states) == len(atoms)
            seq_len = len(atoms)

            # Find the end of the bridge.
            # The bridge is done if we can generate the remainder of the plan
            # by ourselves, up to random tiebreaking, assuming optimal planning.
            # Equivalently, the remainder of the plan looks rational -- the
            # action selected has optimal cost-to-go.

            # Start by computing the optimal costs to go for each state.
            optimal_ctgs: List[float] = []
            for state in states:
                task = Task(state, goal)
                # Assuming optimal task planning here.
                try:
                    nsrt_plan, _, _ = self._run_task_plan(
                        task, nsrts, preds, CFG.timeout, self._seed)
                    ctg: float = len(nsrt_plan)
                except ApproachFailure:
                    # Planning failed, put in infinite cost to go.
                    ctg = float("inf")
                optimal_ctgs.append(ctg)

            # Look for first time where the plan suffix decreases by 1.
            bridge_end = seq_len - 1
            for t in range(seq_len):
                suffix = optimal_ctgs[t:]
                decreasing_smoothly = True
                for i, j in zip(suffix[:-1], suffix[1:]):
                    if int(i) != int(j + 1):
                        decreasing_smoothly = False
                        break
                if decreasing_smoothly:
                    bridge_end = t
                    break

            # Convert atom bridge into ground NSRT bridge.
            ground_nsrt_bridge = []
            objects = set(states[0])
            effects_to_ground_nsrt = {}
            for nsrt in nsrts:
                for ground_nsrt in utils.all_ground_nsrts(nsrt, objects):
                    add_atoms = frozenset(ground_nsrt.add_effects)
                    effects_to_ground_nsrt[add_atoms] = ground_nsrt

            # Assume all atom changes were necessary; we don't know otherwise.
            for t in range(bridge_end):
                add_atoms = frozenset(atoms[t + 1] - atoms[t])
                # If no ground NSRT matches, terminate the bridge early because
                # there's nothing we can do.
                try:
                    ground_nsrt = effects_to_ground_nsrt[add_atoms]
                except KeyError:
                    bridge_end = t
                    break
                ground_nsrt_bridge.append(ground_nsrt)

            atoms_bridge = atoms[:bridge_end + 1]
            states_bridge = states[:bridge_end + 1]

            bridge_dataset.append((
                failed_option,
                ground_nsrt_bridge,
                atoms_bridge,
                states_bridge,
            ))

        return self._bridge_policy.learn_from_demos(bridge_dataset)
