"""An abstract approach that does planning to solve tasks.

Uses the SeSamE bilevel planning strategy: SEarch-and-SAMple planning,
then Execution.
"""

import abc
import os
import sys
import time
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple

from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    BaseApproach
from predicators.option_model import _OptionModelBase, create_option_model
from predicators.planning import PlanningFailure, PlanningTimeout, \
    fd_plan_from_sas_file, generate_sas_file_for_fd, sesame_plan, task_plan, \
    task_plan_grounding
from predicators.settings import CFG
from predicators.structs import NSRT, Action, DummyOption, GroundAtom, \
    Metrics, ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT, \
    _Option


class BilevelPlanningApproach(BaseApproach):
    """Bilevel planning approach."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        if task_planning_heuristic == "default":
            task_planning_heuristic = CFG.sesame_task_planning_heuristic
        if max_skeletons_optimized == -1:
            max_skeletons_optimized = CFG.sesame_max_skeletons_optimized
        self._task_planning_heuristic = task_planning_heuristic
        self._max_skeletons_optimized = max_skeletons_optimized
        self._plan_without_sim = CFG.bilevel_plan_without_sim
        self._option_model = create_option_model(CFG.option_model_name)
        self._num_calls = 0
        self._last_plan: List[_Option] = []

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._num_calls += 1
        # ensure random over successive calls
        seed = self._seed + self._num_calls
        nsrts = self._get_current_nsrts()
        preds = self._get_current_predicates()

        # Run task planning only and then greedily sample and execute in the
        # policy.
        if self._plan_without_sim:
            nsrt_plan, metrics = self._run_task_plan(task, nsrts, preds,
                                                     timeout, seed)
            self._save_metrics(metrics, nsrts, preds)
            return self._task_plan_to_greedy_policy(nsrt_plan, task.goal)

        # Run full bilevel planning.
        plan, metrics = self._run_sesame_plan(task, nsrts, preds, timeout,
                                              seed)
        self._save_metrics(metrics, nsrts, preds)
        self._last_plan = plan
        option_policy = utils.option_plan_to_policy(plan)

        def _policy(s: State) -> Action:
            try:
                return option_policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _run_sesame_plan(self, task: Task, nsrts: Set[NSRT],
                         preds: Set[Predicate], timeout: float, seed: int,
                         **kwargs: Any) -> Tuple[List[_Option], Metrics]:
        """Subclasses may override.

        For example, PG4 inserts an abstract policy into kwargs.
        """
        try:
            plan, metrics = sesame_plan(
                task,
                self._option_model,
                nsrts,
                preds,
                self._types,
                timeout,
                seed,
                self._task_planning_heuristic,
                self._max_skeletons_optimized,
                max_horizon=CFG.horizon,
                allow_noops=CFG.sesame_allow_noops,
                use_visited_state_set=CFG.sesame_use_visited_state_set,
                **kwargs)
        except PlanningFailure as e:
            raise ApproachFailure(e.args[0], e.info)
        except PlanningTimeout as e:
            raise ApproachTimeout(e.args[0], e.info)

        return plan, metrics

    def _run_task_plan(self, task: Task, nsrts: Set[NSRT],
                       preds: Set[Predicate], timeout: float, seed: int,
                       **kwargs: Any) -> Tuple[List[_GroundNSRT], Metrics]:
        init_atoms = utils.abstract(task.init, preds)
        goal = task.goal
        objects = set(task.init)
        try:
            start_time = time.perf_counter()
            ground_nsrts, reachable_atoms = task_plan_grounding(
                init_atoms, objects, nsrts)
            heuristic = utils.create_task_planning_heuristic(
                self._task_planning_heuristic, init_atoms, goal, ground_nsrts,
                preds, objects)
            duration = time.perf_counter() - start_time
            timeout -= duration
            if CFG.sesame_task_planner == "astar":
                plan, _, metrics = next(
                    task_plan(init_atoms,
                              goal,
                              ground_nsrts,
                              reachable_atoms,
                              heuristic,
                              seed,
                              timeout,
                              max_skeletons_optimized=1,
                              use_visited_state_set=True,
                              **kwargs))
            elif "fd" in CFG.sesame_task_planner:
                fd_exec_path = os.environ["FD_EXEC_PATH"]
                exec_str = os.path.join(fd_exec_path, "fast-downward.py")
                timeout_cmd = "gtimeout" if sys.platform == "darwin" \
                    else "timeout"
                # Run Fast Downward followed by cleanup. Capture the output.
                assert "FD_EXEC_PATH" in os.environ, \
                    "Please follow instructions in the docstring of the" +\
                    "_sesame_plan_with_fast_downward method in planning.py"
                
                import ipdb; ipdb.set_trace()

                if CFG.sesame_task_planner == "fdopt":
                    alias_flag = "--alias seq-opt-lmcut"
                elif CFG.sesame_task_planner == "fdsat":
                    alias_flag = "--alias lama-first"
                else:
                    raise ValueError("Unrecognized sesame_task_planner: "
                                     f"{CFG.sesame_task_planner}")

                sas_file = generate_sas_file_for_fd(task, nsrts, preds,
                                                    self._types, timeout,
                                                    timeout_cmd, alias_flag,
                                                    exec_str, list(objects),
                                                    init_atoms)
                plan, _, metrics = fd_plan_from_sas_file(
                    sas_file, timeout_cmd, timeout, exec_str, alias_flag,
                    start_time, list(objects), init_atoms, nsrts, CFG.horizon)
            else:
                raise ValueError("Unrecognized sesame_task_planner: "
                                 f"{CFG.sesame_task_planner}")

        except PlanningFailure as e:
            raise ApproachFailure(e.args[0], e.info)
        except PlanningTimeout as e:
            raise ApproachTimeout(e.args[0], e.info)

        return plan, metrics

    def reset_metrics(self) -> None:
        super().reset_metrics()
        # Initialize min to inf (max gets initialized to 0 by default).
        self._metrics["min_num_samples"] = float("inf")
        self._metrics["min_num_skeletons_optimized"] = float("inf")

    def _save_metrics(self, metrics: Metrics, nsrts: Set[NSRT],
                      predicates: Set[Predicate]) -> None:
        for metric in [
                "num_samples", "num_skeletons_optimized",
                "num_failures_discovered", "num_nodes_expanded",
                "num_nodes_created", "plan_length"
        ]:
            self._metrics[f"total_{metric}"] += metrics[metric]
        self._metrics["total_num_nsrts"] += len(nsrts)
        self._metrics["total_num_preds"] += len(predicates)
        for metric in [
                "num_samples",
                "num_skeletons_optimized",
        ]:
            self._metrics[f"min_{metric}"] = min(
                metrics[metric], self._metrics[f"min_{metric}"])
            self._metrics[f"max_{metric}"] = max(
                metrics[metric], self._metrics[f"max_{metric}"])

    @abc.abstractmethod
    def _get_current_nsrts(self) -> Set[NSRT]:
        """Get the current set of NSRTs."""
        raise NotImplementedError("Override me!")

    def _get_current_predicates(self) -> Set[Predicate]:
        """Get the current set of predicates.

        Defaults to initial predicates.
        """
        return self._initial_predicates

    def get_option_model(self) -> _OptionModelBase:
        """For ONLY an oracle approach, we allow the user to get the current
        option model."""
        assert self.get_name() == "oracle"
        return self._option_model

    def get_last_plan(self) -> List[_Option]:
        """Note that this doesn't fit into the standard API for an Approach,
        since solve() returns a policy, which abstracts away the details of
        whether that policy is actually a plan under the hood."""
        assert self.get_name() == "oracle"
        return self._last_plan

    def _task_plan_to_greedy_policy(
            self, nsrt_plan: Sequence[_GroundNSRT],
            goal: Set[GroundAtom]) -> Callable[[State], Action]:

        cur_nsrt: Optional[_GroundNSRT] = None
        cur_option = DummyOption
        nsrt_queue = list(nsrt_plan)

        def _policy(state: State) -> Action:
            nonlocal cur_nsrt, cur_option
            if cur_option is DummyOption or cur_option.terminal(state):
                if not nsrt_queue:
                    raise ApproachFailure("Greedy option plan exhausted.")
                cur_nsrt = nsrt_queue.pop(0)
                cur_option = cur_nsrt.sample_option(state, goal, self._rng)
                if not cur_option.initiable(state):
                    raise ApproachFailure("Greedy option not initiable.")
            act = cur_option.policy(state)
            return act

        return _policy
