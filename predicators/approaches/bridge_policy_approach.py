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
        --debug --num_test_tasks 1 --segmenter oracle --demonstrator human

Oracle bridge policy in stick button:
    python predicators/main.py --env stick_button --approach bridge_policy \
        --seed 0 --bridge_policy oracle --horizon 10000

Learned bridge policy in stick button:
    python predicators/main.py --env stick_button --approach bridge_policy \
        --seed 0 --horizon 10000 --max_initial_demos 0 \
        --interactive_num_requests_per_cycle 1 \
        --num_online_learning_cycles 100 \
        --num_test_tasks 10 --segmenter contacts --demonstrator human \
        --stick_button_num_buttons_train '[3,4]'

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
from typing import Callable, List, Optional, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.oracle_approach import OracleApproach
from predicators.bridge_policies import BridgePolicyDone, create_bridge_policy
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.settings import CFG
from predicators.structs import Action, BridgeDataset, DefaultState, \
    DemonstrationQuery, DemonstrationResponse, InteractionRequest, \
    InteractionResult, ParameterizedOption, Predicate, Query, State, Task, \
    Type, _Option, BridgePolicyDoneNSRT
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
        self._bridge_dataset: BridgeDataset = []

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
                offending_objects = set()
            except OptionExecutionFailure as e:
                # An error was encountered, so we need the bridge policy.
                current_control = "bridge"
                failed_option = e.info["last_failed_option"]
                offending_objects = e.info.get("offending_objects", set())
                bridge_policy_failure = (failed_option, offending_objects)
                self._bridge_policy.record_failure(bridge_policy_failure)
                logging.debug("Giving control to bridge.")
            except ApproachFailure as e:
                # The bridge policy got stuck.
                assert "LDL bridge policy not applicable" in str(e)
                policy_input = self._bridge_policy.get_policy_input_atoms(s)
                raise ApproachFailure("Bridge policy stuck.",
                    info={"policy_input": policy_input},
                )

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

        def _new_option_callback(state: State, option: _Option) -> None:
            self._bridge_policy.record_state_option(state, option)
            try:
                # Use the option model ONLY to predict environment failures.
                # Will raise an EnvironmentFailure if one is predicted.
                self._option_model.get_next_state_and_num_actions(
                    state, option)
            except utils.EnvironmentFailure as e:
                logging.debug(f"Environment failure: {repr(e)}")
                raise OptionExecutionFailure(
                    f"Environment failure predicted: {repr(e)}.",
                    info={
                        "last_failed_option": option,
                        **e.info
                    })

        return _new_option_callback

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
        policy = self._solve(task, timeout=CFG.timeout)

        reached_stuck_state = False
        policy_input = None

        def _act_policy(s: State) -> Action:
            nonlocal reached_stuck_state, policy_input
            try:
                return policy(s)
            except ApproachFailure as e:
                reached_stuck_state = True
                policy_input = e.info["policy_input"]
                # Approach failures not caught in interaction loop.
                raise OptionExecutionFailure(e.args[0], e.info)

        def _termination_fn(s: State) -> bool:
            return reached_stuck_state or task.goal_holds(s)

        def _query_policy(s: State) -> Optional[Query]:
            if not reached_stuck_state or task.goal_holds(s):
                return None
            assert policy_input is not None
            return DemonstrationQuery(
                train_task_idx, {"policy_input": policy_input})

        request = InteractionRequest(train_task_idx, _act_policy,
                                     _query_policy, _termination_fn)

        return request

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:

        nsrts = self._get_current_nsrts()
        preds = self._get_current_predicates()

        # If we haven't collected any new results on this cycle, skip learning
        # for efficiency.
        if not results:
            return None

        for result in results:
            response = result.responses[-1]
            # Interaction didn't involve any queries.
            if response is None:
                continue
            assert isinstance(response, DemonstrationResponse)
            query = response.query
            assert isinstance(query, DemonstrationQuery)
            goal = self._train_tasks[query.train_task_idx].goal
            policy_input = query.get_info("policy_input")

            # Abstract and segment the trajectory.
            traj = response.teacher_traj
            assert traj is not None
            atom_traj = [utils.abstract(s, preds) for s in traj.states]
            segmented_traj = segment_trajectory((traj, atom_traj))
            if not segmented_traj:
                assert len(atom_traj) == 1
                states = [traj.states[0]]
                atoms = atom_traj
            else:
                states = utils.segment_trajectory_to_state_sequence(
                    segmented_traj)
                atoms = utils.segment_trajectory_to_atoms_sequence(
                    segmented_traj)
            assert len(states) == len(atoms)
            seq_len = len(atoms)

            # Prepare to map the rational transitions to done.
            optimal_ctgs: List[float] = []
            for state in states:
                task = Task(state, goal)
                # Assuming optimal task planning here.
                assert (CFG.sesame_task_planner == "astar" and \
                        CFG.sesame_task_planning_heuristic == "lmcut") or \
                        CFG.sesame_task_planner == "fdopt"
                try:
                    nsrt_plan, _, _ = self._run_task_plan(
                        task, nsrts, preds, CFG.timeout, self._seed)
                    ctg: float = len(nsrt_plan)
                except ApproachFailure:  # pragma: no cover
                    # Planning failed, put in infinite cost to go.
                    ctg = float("inf")
                optimal_ctgs.append(ctg)

            # For later converting atoms into ground NSRTs.
            objects = set(states[0])
            effects_to_ground_nsrt = {}
            for nsrt in nsrts:
                for ground_nsrt in utils.all_ground_nsrts(nsrt, objects):
                    add_atoms = frozenset(ground_nsrt.add_effects)
                    effects_to_ground_nsrt[add_atoms] = ground_nsrt

            # Collect the irrational transitions and turn atom changes into
            # ground NSRTs.
            last_transition_was_rational = True
            for t in range(seq_len - 1):
                # Step was rational.
                if optimal_ctgs[t] == optimal_ctgs[t + 1] + 1:
                    if not last_transition_was_rational:
                        # Bridge policy should be done now.
                        ground_nsrt = BridgePolicyDoneNSRT.ground([])
                    last_transition_was_rational = True
                # Step was irrational.
                else:
                    last_transition_was_rational = False
                    # Assume all changes were necessary; we don't know otherwise.
                    add_atoms = frozenset(atoms[t + 1] - atoms[t])
                    try:
                        ground_nsrt = effects_to_ground_nsrt[add_atoms]
                    except KeyError:  # pragma: no cover
                        logging.warning("WARNING: no NSRT found for add atoms "
                                        f"{add_atoms}. Skipping transition.")
                        continue
                # Add to data.
                self._bridge_dataset.append((
                    policy_input,
                    ground_nsrt,
                ))

        return self._bridge_policy.learn_from_demos(self._bridge_dataset)
