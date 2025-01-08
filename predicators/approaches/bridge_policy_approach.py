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
"""

import logging
import time
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.maple_q_approach import MapleQApproach, MPDQNApproach
from predicators.ml_models import MapleQFunction, MPDQNFunction
from predicators.approaches.oracle_approach import OracleApproach
from predicators.bridge_policies import BridgePolicyDone, create_bridge_policy
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import NSRT, Action, Array, BridgeDataset, \
    DefaultState, DemonstrationQuery, DemonstrationResponse, GroundAtom, \
    InteractionRequest, InteractionResult, LiftedAtom, LowLevelTrajectory, \
    Object, ParameterizedOption, Predicate, Query, State, Task, Type, \
    Variable, _GroundNSRT, _Option


class BridgePolicyApproach(OracleApproach):
    """A simulator-free bilevel planning approach that uses a bridge policy."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1,
                 nsrts: Optional[Set[NSRT]] = None,
                 option_model: Optional[_OptionModelBase] = None) -> None:
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         task_planning_heuristic,
                         max_skeletons_optimized,
                         nsrts=nsrts,
                         option_model=option_model)
        predicates = self._get_current_predicates()
        options = initial_options
        nsrts = self._get_current_nsrts()
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
        # Start by planning. Note that we cannot start with the bridge policy
        # because the bridge policy takes as input the last failed NSRT.
        current_control = "planner"
        option_policy = self._get_option_policy_by_planning(task, timeout)
        current_policy = utils.option_policy_to_policy(
            option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout,
            raise_error_on_repeated_state=True,
        )
        all_failed_options: List[_Option] = []

        # Prevent infinite loops by detecting if the bridge policy is called
        # twice with the same state.
        last_bridge_policy_state = DefaultState

        def _policy(s: State) -> Action:
            nonlocal current_control, current_policy, last_bridge_policy_state
            if time.perf_counter() - start_time > timeout:
                raise ApproachTimeout("Bridge policy timed out.")

            # Normal execution. Either keep executing the current option, or
            # switch to the next option if it has terminated.
            try:
                return current_policy(s)
            except BridgePolicyDone:
                assert current_control == "bridge"
                failed_option = None  # not used, but satisfy linting
            except utils.OptionExecutionFailure as e:
                failed_option = e.info["last_failed_option"]
                if failed_option is not None:
                    all_failed_options.append(failed_option)
                    logging.debug(f"Failed option: {failed_option.name}"
                                  f"{failed_option.objects}.")
                    logging.debug(f"Error: {e.args[0]}")
                    self._bridge_policy.record_failed_option(failed_option)

            # Switch control from planner to bridge.
            if current_control == "planner":
                # Planning failed on the first time step.
                if failed_option is None:
                    assert s.allclose(task.init)
                    raise ApproachFailure("Planning failed on init state.")
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
                    if last_bridge_policy_state.allclose(s):
                        raise ApproachFailure(
                            "Loop detected, giving up.",
                            info={"all_failed_options": all_failed_options})
                last_bridge_policy_state = s

            # Switch control from bridge to planner.
            logging.debug("Switching control from bridge to planner.")
            assert current_control == "bridge"
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
            )
            try:
                return current_policy(s)
            except utils.OptionExecutionFailure as e:
                all_failed_options.append(e.info["last_failed_option"])
                raise ApproachFailure(
                    e.args[0], info={"all_failed_options": all_failed_options})

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
        policy = self._solve(task, timeout=CFG.timeout)

        reached_stuck_state = False
        all_failed_options = None

        def _act_policy(s: State) -> Action:
            nonlocal reached_stuck_state, all_failed_options
            try:
                return policy(s)
            except ApproachFailure as e:
                reached_stuck_state = True
                all_failed_options = e.info["all_failed_options"]
                # Approach failures not caught in interaction loop.
                raise utils.OptionExecutionFailure(e.args[0], e.info)

        def _termination_fn(s: State) -> bool:
            return reached_stuck_state or task.goal_holds(s)

        def _query_policy(s: State) -> Optional[Query]:
            if not reached_stuck_state or task.goal_holds(s):
                return None
            assert all_failed_options is not None
            return DemonstrationQuery(
                train_task_idx, {"all_failed_options": all_failed_options})

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
            all_failed_options = query.get_info("all_failed_options")

            # Abstract and segment the trajectory.
            traj = response.teacher_traj
            assert traj is not None
            atom_traj = [utils.abstract(s, preds) for s in traj.states]
            segmented_traj = segment_trajectory(traj, preds, atom_traj)
            if not segmented_traj:
                assert len(atom_traj) == 1
                states = [traj.states[0]]
                atoms = atom_traj
            else:
                states = utils.segment_trajectory_to_start_end_state_sequence(
                    segmented_traj)
                atoms = utils.segment_trajectory_to_atoms_sequence(
                    segmented_traj)
            assert len(states) == len(atoms)
            seq_len = len(atoms)

            # Prepare to excise the rational transitions.
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
                    del_atoms = frozenset(ground_nsrt.delete_effects)
                    ground_effects = (add_atoms, del_atoms)
                    effects_to_ground_nsrt[ground_effects] = ground_nsrt

            # Collect the irrational transitions and turn atom changes into
            # ground NSRTs.
            for t in range(seq_len - 1):
                # Step was rational, so skip it.
                if optimal_ctgs[t] == optimal_ctgs[t + 1] + 1:
                    continue
                # Step was irrational, so include it.
                # Assume all changes were necessary; we don't know otherwise.
                add_atoms = frozenset(atoms[t + 1] - atoms[t])
                del_atoms = frozenset(atoms[t] - atoms[t + 1])
                ground_effects = (add_atoms, del_atoms)
                try:
                    ground_nsrt = effects_to_ground_nsrt[ground_effects]
                except KeyError:  # pragma: no cover
                    logging.warning("WARNING: no NSRT found for add atoms "
                                    f"{add_atoms}. Skipping transition.")
                    continue
                self._bridge_dataset.append((
                    all_failed_options,
                    ground_nsrt,
                    atoms[t],
                    states[t],
                ))

        return self._bridge_policy.learn_from_demos(self._bridge_dataset)


class RLBridgePolicyApproach(BridgePolicyApproach):
    """A simulator-free bilevel planning approach that uses a Deep RL (Maple Q)
    bridge policy."""

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
        self._maple_initialized = False
        if task_planning_heuristic == "default":
            task_planning_heuristic = CFG.sesame_task_planning_heuristic
        self._task_planning_heuristic = task_planning_heuristic
        self._trajs: List[LowLevelTrajectory] = []
        self.CanPlan = Predicate("CanPlan", [], self._Can_plan)
        if CFG.use_callplanner:
            self.CallPlanner = utils.SingletonParameterizedOption(
            "CallPlanner",
            types=None,
            policy=self.call_planner_policy,
            params_space=Box(low=np.array([]), high=np.array([]), shape=(0, )),
            )
            initial_options.add(self.CallPlanner)
        else:
            self.CallPlanner = None
        self._initial_options = initial_options
        self.mapleq=MPDQNApproach(self._get_current_predicates(), \
                                   self._initial_options, self._types, \
                                    self._action_space, self._train_tasks, self.CallPlanner)
        self._current_control = ""
        self._current_plan = None
        try:
            _, option_policy = self._get_option_policy_by_planning(
            self._train_tasks[0], CFG.timeout)
        except:
            import ipdb;ipdb.set_trace()
        self._current_policy = utils.option_policy_to_policy(
            option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout,
            raise_error_on_repeated_state=True,
        )
        self._bridge_called_state = State(data={})
        self._policy_logs: List[str] = []
        self._plan_logs = []
        self._current_task: Optional[Task] = None
        self.num_bridge_steps = 0
        self.bridge_done = False
        self.planning_policy = None
        self.num_planner_steps = 0

    def _Can_plan(self, state: State, _: Sequence[Object]) -> bool:
        if (MPDQNFunction._old_vectorize_state(self.mapleq._q_function, state) !=  # pylint: disable=protected-access
                MPDQNFunction._old_vectorize_state(self.mapleq._q_function,  # pylint: disable=protected-access
                    self._bridge_called_state)).any():  # pylint: disable=protected-access
            return True
        return False

    def call_planner_policy(self, state: State, _: Dict, __: Sequence[Object],
                            params: Array) -> Action:
        """policy for CallPlanner option."""
        self._current_control = "planner"
        # create a new task where the init state is our current state
        current_task = Task(state, self._current_task.goal)
        task_list, option_policy = self._get_option_policy_by_planning(
            current_task, CFG.timeout)
        self._current_plan = task_list
        self._current_policy = utils.option_policy_to_policy(
            option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout,
            raise_error_on_repeated_state=True,
        )
        if CFG.env == "grid_row_door":
            return Action(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        elif CFG.env == "doorknobs":
            return Action(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        elif CFG.env in {"coffee", "coffeelids"}:
            return Action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    def call_planner_nsrt(self) -> NSRT:
        """CallPlanner NSRT."""
        parameters: Sequence[Variable] = []
        option_vars = parameters
        option = self.CallPlanner
        preconditions = {LiftedAtom(self.CanPlan, [])}
        add_effects: Set[LiftedAtom] = set()
        delete_effects: Set[LiftedAtom] = set()

        ignore_effects: Set[Predicate] = set()
        call_planner_nsrt = NSRT("CallPlanner", parameters, preconditions,
                                 add_effects, delete_effects, ignore_effects,
                                 option, option_vars, utils.null_sampler)
        return call_planner_nsrt

    @classmethod
    def get_name(cls) -> str:
        return "rl_bridge_policy"

    @property
    def is_learning_based(self) -> bool:
        return True
    
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
        return atoms_seq, utils.nsrt_plan_to_greedy_option_policy(
            nsrt_plan,
            goal=task.goal,
            rng=self._rng,
            necessary_atoms_seq=atoms_seq)
    
    def _init_nsrts(self) -> None:
        """Initializing nsrts for MAPLE Q."""
        options = self._initial_options
        nsrts = self._get_current_nsrts()
        callplanner_nsrt = self.call_planner_nsrt()
        if CFG.use_callplanner:
            nsrts.add(callplanner_nsrt)
        predicates = self._get_current_predicates()
        all_ground_nsrts: Set[_GroundNSRT] = set()
        if CFG.sesame_grounder == "naive":
            for nsrt in nsrts:
                all_objects = {o for o in self._current_task.init}
                all_ground_nsrts.update(
                    utils.all_ground_nsrts(nsrt, all_objects))
        elif CFG.sesame_grounder == "fd_translator":  # pragma: no cover
            all_objects = set()
            t = self._current_task
            curr_task_objects = set(t.init)
            curr_task_types = {o.type for o in t.init}
            curr_init_atoms = utils.abstract(t.init, predicates)
            all_ground_nsrts.update(
                utils.all_ground_nsrts_fd_translator(
                    nsrts, curr_task_objects, predicates, curr_task_types,
                    curr_init_atoms, t.goal))
            all_objects.update(curr_task_objects)
        else:  # pragma: no cover
            raise ValueError(
                f"Unrecognized sesame_grounder: {CFG.sesame_grounder}")
        goals = [self._current_task.goal]  # pylint: disable=protected-access
        self.mapleq._q_function.set_grounding(  # pylint: disable=protected-access
            all_objects, goals, all_ground_nsrts, options)
        # self.mapleq._q_function.nsrts = nsrts
        # import ipdb;ipdb.set_trace()

    def _solve(self,
               task: Task,
               timeout: int,
               train_or_test: str = "test") -> Callable[[State], Action]:
        # Start by planning. Note that we cannot start with the bridge policy
        # because the bridge policy takes as input the last failed NSRT.
        self._current_task = task
        # import ipdb;ipdb.set_trace()
        self._current_control = "planner"
        task_list, option_policy = self._get_option_policy_by_planning(task, timeout)
        self._current_policy = utils.option_policy_to_policy(
            option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout,
            raise_error_on_repeated_state=True,
        )
        self._current_plan = task_list
        if not self._maple_initialized:
            self.mapleq = MPDQNApproach(self._get_current_predicates(),
                                         self._initial_options, self._types,
                                         self._action_space, self._train_tasks, self.CallPlanner)
            self._maple_initialized = True
        self._init_nsrts()

        # self.mapleq._q_function.set_grounding(  # pylint: disable=protected-access
        #     all_objects, goals, all_ground_nsrts)

        # Prevent infinite loops by detecting if the bridge policy is called
        # twice with the same state.
        last_bridge_policy_state = DefaultState

        def _policy(s: State) -> Action:
            nonlocal last_bridge_policy_state

            # Normal execution. Either keep executing the current option, or
            # switch to the next option if it has terminated.
            try:
                action = self._current_policy(s)
                if train_or_test == "train":
                    self._policy_logs.append(self._current_control)
                    self._plan_logs.append(self._current_plan)

                return action
            except utils.OptionExecutionFailure:
                failed = True
                if self._current_control == "bridge":
                    import ipdb;ipdb.set_trace()
                # logging.debug("failed" + self._current_control)

            # Switch control from planner to bridge.
            # assert self._current_control == "planner"
            self._current_control = "bridge"
            self.mapleq._q_function._last_planner_state = s
            if train_or_test == "train":
                self._policy_logs.append(self._current_control)
                self._plan_logs.append(self._current_plan)
            self._bridge_called_state = s
            self._current_policy = self.mapleq._solve(  # pylint: disable=protected-access
                task, timeout, train_or_test)
            action = self._current_policy(s)
            return action

        return _policy

    def _create_interaction_request(self,
                                    train_task_idx: int) -> InteractionRequest:
        task = self._train_tasks[train_task_idx]
        policy = self._solve(task, timeout=CFG.timeout, train_or_test="train")
        just_starting = True

        def _act_policy(s: State) -> Action:
            nonlocal just_starting
            if just_starting:
                task = self._train_tasks[train_task_idx]
                self._current_control = "planner"
                self.num_bridge_steps = 0
                self.bridge_done = False
                self.num_planner_steps = 0
                _, option_policy = self._get_option_policy_by_planning(
                    task, CFG.timeout)
                self._current_policy = utils.option_policy_to_policy(
                    option_policy,
                    max_option_steps=CFG.max_num_steps_option_rollout,
                    raise_error_on_repeated_state=True,
                )
                self.planning_policy = self._current_policy
                just_starting = False
            return policy(s)

        def _termination_fn(s: State) -> bool:
            task = self._train_tasks[train_task_idx]
            return task.goal_holds(s)

        # The request's acting policy is from mapleq
        # The resulting trajectory is from maple q's sampling
        request = InteractionRequest(train_task_idx, _act_policy,
                                     lambda s: None, _termination_fn)
        return request

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        # Turn state action pairs from results into trajectories
        # If we haven't collected any new results on this cycle, skip learning
        # for efficiency.
        if not results:
            return None
        all_states = []
        all_actions = []
        policy_logs = self._policy_logs
        plan_logs = self._plan_logs
        reward_bonuses = []
        for i in range(len(results)):
            reward_bonus = []
            result = results[i]
            policy_log = policy_logs[:len(result.states[:-1])]
            plan_log = plan_logs[:len(result.states[:-1])]
            mapleq_states = [
                state for j, state in enumerate(result.states[:-1])
                if policy_log[j] == "bridge"
                or policy_log[max(j - 1, 0)] == "bridge"
            ]
            mapleq_actions = [
                action for j, action in enumerate(result.actions)
                if policy_log[j] == "bridge"
                or policy_log[max(j - 1, 0)] == "bridge"
            ]
            
        # if CFG.rl_rwd_shape:
        #     for j, _ in enumerate(result.actions):
        #         reward_bonus = []
        #         sussy = plan_log[j]  # This is the planned sequence of states at timestep j
        #         rwd = 0
        #         # Check if we're currently in bridge mode or were in bridge mode in the previous step
        #         if policy_log[j] == "bridge" or policy_log[max(j - 1, 0)] == "bridge":
        #             # Get the abstract state (set of atoms) for the next state
        #             current_atoms = utils.abstract(result.states[j+1], self._get_current_predicates())
                    
        #             # Loop through each state in the planned sequence
        #             for g in range(len(sussy)):
        #                 # If our current state contains all atoms from a planned state
        #                 if sussy[g].issubset(current_atoms):
        #                     # Give reward based on how far along the plan we are
        #                     # e.g., if we match state 4 out of 10 states, reward would be 4/(10*2) = 0.2
        #                     rwd = (g/(len(sussy)*2))
        #                     break
        #             reward_bonus.append(rwd)

        #     reward_bonuses.extend(reward_bonus)


            num_rewards = 0
            if CFG.rl_rwd_shape:
                for j, _ in enumerate(result.actions):
                    rwd = -1  # Default reward is -1
                    
                    # Only apply reward shaping during bridge mode or right after
                    if policy_log[j] == "bridge" or policy_log[max(j - 1, 0)] == "bridge":
                        current_atoms = utils.abstract(result.states[j+1], self._get_current_predicates())
                        
                        # Find the last planner state before this point
                        last_planner_idx = -1
                        for i in range(j, -1, -1):
                            if policy_log[i] == "planner":
                                last_planner_idx = i
                                break
                        
                        if last_planner_idx != -1:
                            # Get all states from last planner state onwards in the plan
                            safe_states = plan_log[j][last_planner_idx+1:]
                            
                            # Check if current state satisfies postconditions of all remaining operators
                            # by checking if it contains atoms from any future state in the plan
                            for future_state in safe_states:
                                if future_state.issubset(current_atoms):
                                    print("FOUND MATCH")
                                    print(future_state)
                                    print(current_atoms)
                                    print(safe_states)
                                    # import ipdb;ipdb.set_trace()
                                    rwd = 1000  # We found a match, this is a safe state
                                    num_rewards += 1
                                    break


                        reward_bonus.append(rwd)
            print("num rewards", num_rewards)
            # if num_rewards > 1:
            #     import ipdb;ipdb.set_trace()
            reward_bonuses.append(reward_bonus)
            mapleq_states.append(result.states[-1])
            new_traj = LowLevelTrajectory(mapleq_states, mapleq_actions)
            self._trajs.append(new_traj)
            all_states.extend(mapleq_states)
            all_actions.extend(mapleq_actions)
            policy_logs = policy_logs[len(result.states) - 1:]
            plan_logs = plan_logs[len(result.states) - 1:]
        self.mapleq.get_interaction_requests()
        self.mapleq._learn_nsrts(self._trajs, 0, [] * len(self._trajs), reward_bonuses)  # pylint: disable=protected-access
        self._policy_logs = []
        self._plan_logs = []
        return None


class RLFirstBridgeApproach(RLBridgePolicyApproach):
    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1) -> None:
        super().__init__(initial_predicates, 
                 initial_options, 
                 types,
                 action_space,
                 train_tasks,
                 task_planning_heuristic,
                 max_skeletons_optimized)


    @classmethod
    def get_name(cls) -> str:
        return "rl_first_bridge"
    
    def _solve(self,
               task: Task,
               timeout: int,
               train_or_test: str = "test") -> Callable[[State], Action]:
        # Start by planning. Note that we cannot start with the bridge policy
        # because the bridge policy takes as input the last failed NSRT.
        self._current_task = task
        # import ipdb;ipdb.set_trace()
        self._current_control = "bridge"

        if not self._maple_initialized:
            self.mapleq = MPDQNApproach(self._get_current_predicates(),
                                         self._initial_options, self._types,
                                         self._action_space, self._train_tasks, self.CallPlanner)
            self._maple_initialized = True
        self._init_nsrts()
        self._current_policy = self.mapleq._solve(  # pylint: disable=protected-access
                task, timeout, train_or_test)
        # self.mapleq._q_function.set_grounding(  # pylint: disable=protected-access
        #     all_objects, goals, all_ground_nsrts)

        # Prevent infinite loops by detecting if the bridge policy is called
        # twice with the same state.
        last_bridge_policy_state = DefaultState

        def _policy(s: State) -> Action:
            nonlocal last_bridge_policy_state

            # Normal execution. Either keep executing the current option, or
            # switch to the next option if it has terminated.
            try:
                action = self._current_policy(s)
                if train_or_test == "train":
                    self._policy_logs.append(self._current_control)
                return action
            except utils.OptionExecutionFailure:
                failed = True
                if self._current_control == "bridge":
                    import ipdb;ipdb.set_trace()
                # logging.debug("failed" + self._current_control)

            # Switch control from planner to bridge.
            # assert self._current_control == "planner"
            self._current_control = "bridge"
            self.mapleq._q_function._last_planner_state = s
            if train_or_test == "train":
                self._policy_logs.append(self._current_control)
            self._bridge_called_state = s
            self._current_policy = self.mapleq._solve(  # pylint: disable=protected-access
                task, timeout, train_or_test)
            action = self._current_policy(s)
            return action

        return _policy
    

class RwdShapeBridgeApproach(RLBridgePolicyApproach):
    #basically same approach but stop when u hit a state thats further than the state we broke at and give a positive reward
    # so u need to modify solve function to switch back to planner after u finished (bridge_done = True)
    # so like check if the state we get inside of solve function has the atoms of a later state in the plan, 
    # if so then self.bridge_done = True
    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1) -> None:
        super().__init__(initial_predicates, 
                 initial_options, 
                 types,
                 action_space,
                 train_tasks,
                 task_planning_heuristic,
                 max_skeletons_optimized)
        self.bridge_done = False
 

    def _solve(self,
               task: Task,
               timeout: int,
               train_or_test: str = "test") -> Callable[[State], Action]:
        # Start by planning. Note that we cannot start with the bridge policy
        # because the bridge policy takes as input the last failed NSRT.
        self._current_task = task
        # import ipdb;ipdb.set_trace()
        self._current_control = "planner"
        task_list, option_policy = self._get_option_policy_by_planning(task, timeout)
        self._current_policy = utils.option_policy_to_policy(
            option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout,
            raise_error_on_repeated_state=True,
        )
        self._current_plan = task_list
        if not self._maple_initialized:
            self.mapleq = MPDQNApproach(self._get_current_predicates(),
                                         self._initial_options, self._types,
                                         self._action_space, self._train_tasks, self.CallPlanner)
            self._maple_initialized = True
        self._init_nsrts()

        # self.mapleq._q_function.set_grounding(  # pylint: disable=protected-access
        #     all_objects, goals, all_ground_nsrts)

        # Prevent infinite loops by detecting if the bridge policy is called
        # twice with the same state.
        last_bridge_policy_state = DefaultState

        def _policy(s: State) -> Action:
            nonlocal last_bridge_policy_state

            # Normal execution. Either keep executing the current option, or
            # switch to the next option if it has terminated.
            try:
                action = self._current_policy(s)
                if train_or_test == "train":
                    self._policy_logs.append(self._current_control)
                    self._plan_logs.append(self._current_plan)

                return action
            except utils.OptionExecutionFailure:
                failed = True
                if self._current_control == "bridge":
                    import ipdb;ipdb.set_trace()
                # logging.debug("failed" + self._current_control)

            # Switch control from planner to bridge.
            # assert self._current_control == "planner"
            self._current_control = "bridge"
            self.mapleq._q_function._last_planner_state = s
            if train_or_test == "train":
                self._policy_logs.append(self._current_control)
                self._plan_logs.append(self._current_plan)
            self._bridge_called_state = s
            self._current_policy = self.mapleq._solve(  # pylint: disable=protected-access
                task, timeout, train_or_test)
            action = self._current_policy(s)
            return action

        return _policy























class RapidLearnApproach(RLBridgePolicyApproach):
    #basically same approach but stop when u hit a state thats further than the state we broke at and give a positive reward
    # so u need to modify solve function to switch back to planner after u finished (bridge_done = True)
    # so like check if the state we get inside of solve function has the atoms of a later state in the plan, 
    # if so then self.bridge_done = True

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1) -> None:
        super().__init__(initial_predicates, 
                 initial_options, 
                 types,
                 action_space,
                 train_tasks,
                 task_planning_heuristic,
                 max_skeletons_optimized)
        self.bridge_done = False
        self.last_action = None
        self.num_bridge_steps = 0
        self.planning_policy = None
 
    @classmethod
    def get_name(cls) -> str:
        return "rapid_learn"

    def _solve(self,
               task: Task,
               timeout: int,
               train_or_test: str = "test") -> Callable[[State], Action]:
        # Start by planning. Note that we cannot start with the bridge policy
        # because the bridge policy takes as input the last failed NSRT.

        # print("\n=== NEW TASK ===")
        # print("bridge_done before reset:", self.bridge_done)
        self.bridge_done = False
        # print("bridge_done after reset:", self.bridge_done)

        self.num_planner_steps = 0
        self._current_task = task
        # import ipdb;ipdb.set_trace()
        self._current_control = "planner"
        task_list, option_policy = self._get_option_policy_by_planning(task, timeout)
        self._current_policy = utils.option_policy_to_policy(
            option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout,
            raise_error_on_repeated_state=True,
        )
        self.planning_policy = self._current_policy
        self._current_plan = task_list
        if not self._maple_initialized:
            self.mapleq = MPDQNApproach(self._get_current_predicates(),
                                         self._initial_options, self._types,
                                         self._action_space, self._train_tasks, self.CallPlanner)
            self._maple_initialized = True
        self._init_nsrts()

        # self.mapleq._q_function.set_grounding(  # pylint: disable=protected-access
        #     all_objects, goals, all_ground_nsrts)

        # Prevent infinite loops by detecting if the bridge policy is called
        # twice with the same state.
        last_bridge_policy_state = DefaultState
        self.num_bridge_steps=0
        self.num_planner_steps = 0


        def _policy(s: State) -> Action:
            nonlocal last_bridge_policy_state

            # if we are in bridge policy mode, check if our current state has the atoms of a later state in the plan
            # if so then self.bridge_done = True
        
            
            # print("num bridge steps", self.num_bridge_steps)
            if self._current_control == "bridge":
                print(self._current_control)
                if self.num_bridge_steps > CFG.max_num_bridge_steps:
                    self.bridge_done = True
                    self._current_control = "planner"
                    self._current_policy = self.planning_policy
                    self.num_bridge_steps = 0
                    print("bridge done")
                else:
                    current_atoms = utils.abstract(s, self._get_current_predicates())
                    print("NUM PLANNER STEPS", self.num_planner_steps)
                    for state in self._current_plan[self.num_planner_steps:]:
                        if state.issubset(current_atoms):
                            self.bridge_done = True
                            self._current_control = "planner"
                            self._current_policy = self.planning_policy
                            # import ipdb;ipdb.set_trace()
                            print("bridge done")
                            print(self._current_plan)
                            break
                
            # # Normal execution. Either keep executing the current option, or
            # # switch to the next option if it has terminated.
            # try:
            #     action = self._current_policy(s)
            #     if train_or_test == "train":
            #         self._policy_logs.append(self._current_control)
            #         self._plan_logs.append(self._current_plan)
                
            #     if self._current_control == "bridge":
            #         num_bridge_steps += 1

            #     return action
            # except utils.OptionExecutionFailure as e:
            #     # failed = True
            #     if self._current_control == "bridge":
            #         import ipdb;ipdb.set_trace()
            #     logging.debug("failed" + self._current_control)



            try:
                action = self._current_policy(s)
                print(f"Current control: {self._current_control}, Action: {action}")
                if self._current_control == "planner":
                    self.num_planner_steps += 1
                if train_or_test == "train":
                    self._policy_logs.append(self._current_control)
                    self._plan_logs.append(self._current_plan)
                if self._current_control == "bridge":
                    self.num_bridge_steps += 1
                self.last_action = action
                return action
            except utils.OptionExecutionFailure as e:
                print("current state", s)
                print(f"Option execution failed: {e}")
                print(f"Current control: {self._current_control}")
                print(f"Bridge done: {self.bridge_done}")
                if self.bridge_done or self._current_control == "planner":
                    # Get the next option from the option policy
                    try:
                        action = self._current_policy(s)
                        # current_option = option_policy(s)
                    except:
                        # If option_policy fails, use the one from the error info
                        current_option = e.info["last_failed_option"]
                    
                        if current_option is not None:
                            # Force the option's policy to execute
                            memory = e.info.get("memory", {})
                            action = current_option.policy(s)
                            
                        else:
                            action = self.last_action
                            print("planner execution error", action)



            # Switch control from planner to bridge.
            # assert self._current_control == "planner"
            if not self.bridge_done:
                print("switching to bridge", self.bridge_done)
                self._current_control = "bridge"
                self.num_bridge_steps=0
                self.mapleq._q_function._last_planner_state = s

                self._bridge_called_state = s
                self._current_policy = self.mapleq._solve(  # pylint: disable=protected-access
                    task, timeout, train_or_test)
                action = self._current_policy(s)
                if self._current_control == "bridge":
                    self.num_bridge_steps += 1


            if train_or_test == "train":
                self._policy_logs.append(self._current_control)
                self._plan_logs.append(self._current_plan)
            return action

        return _policy

