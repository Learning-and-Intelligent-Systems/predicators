"""An abstract approach that does planning to solve tasks.

Uses the SeSamE bilevel planning strategy: SEarch-and-SAMple planning,
then Execution.
"""
import abc
import logging
from typing import Any, Callable, List, Optional, Set, Tuple

from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    BaseApproach
from predicators.option_model import _OptionModelBase, create_option_model
from predicators.planning import PlanningFailure, PlanningTimeout, \
    run_task_plan_once, sesame_plan
from predicators.settings import CFG
from predicators.structs import NSRT, Action, GroundAtom, Metrics, \
    ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT, _Option


class BilevelPlanningApproach(BaseApproach):
    """Bilevel planning approach."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1,
                 bilevel_plan_without_sim: Optional[bool] = None,
                 option_model: Optional[_OptionModelBase] = None) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        if task_planning_heuristic == "default":
            task_planning_heuristic = CFG.sesame_task_planning_heuristic
        if max_skeletons_optimized == -1:
            max_skeletons_optimized = CFG.sesame_max_skeletons_optimized
        if bilevel_plan_without_sim is None:
            bilevel_plan_without_sim = CFG.bilevel_plan_without_sim
        self._task_planning_heuristic = task_planning_heuristic
        self._max_skeletons_optimized = max_skeletons_optimized
        self._plan_without_sim = bilevel_plan_without_sim
        if option_model is None:
            option_model = create_option_model(CFG.option_model_name)
        self._option_model = option_model
        self._num_calls = 0
        self._last_plan: List[_Option] = []  # used if plan WITH sim
        self._last_nsrt_plan: List[_GroundNSRT] = []  # plan WITHOUT sim
        self._last_atoms_seq: List[Set[GroundAtom]] = []  # plan WITHOUT sim

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._num_calls += 1
        # ensure random over successive calls
        seed = self._seed + self._num_calls
        nsrts = self._get_current_nsrts()
        preds = self._get_current_predicates()

        # import pdb; pdb.set_trace()
        # nsrts = list(nsrts)
        # MoveWhenFacingStart = [n for n in nsrts if n.name ==  "MoveWhenFacingStart"][0]
        # MoveWhenAlreadyAdjacent = [n for n in nsrts if n.name ==  "MoveWhenAlreadyAdjacent"][0]
        # MoveFromNothing = [n for n in nsrts if n.name ==  "MoveFromNothing"][0]
        # PickSingleAdjacent = [n for n in nsrts if n.name == "PickSingleAdjacent"][0]
        # Place = [n for n in nsrts if n.name == "Place"][0]
        # Cook = [n for n in nsrts if n.name == "Cook"][0]
        # Slice = [n for n in nsrts if n.name == "Slice"][0]
        # PickFromStack = [n for n in nsrts if n.name == "PickFromStack"][0]
        #
        # grill = [obj for obj in task.init if obj.name == "grill"][0]
        # patty = [obj for obj in task.init if obj.name == "patty"][0]
        # robot = [obj for obj in task.init if obj.name == "robby"][0]
        # tomato = [obj for obj in task.init if obj.name == "tomato"][0]
        # cutting_board = [obj for obj in task.init if obj.name == "cutting_board"][0]
        # cheese = [obj for obj in task.init if obj.name == "cheese"][0]
        # top_bun = [obj for obj in task.init if obj.name == "top_bun"][0]
        # bottom_bun = [obj for obj in task.init if obj.name == "bottom_bun"][0]
        #
        # init = utils.abstract(task.init, preds)

        # plan = [
        #     MoveWhenFacingStart.ground([robot, patty, grill]),
        #     PickSingleAdjacent.ground([robot, patty]),
        #     MoveFromNothing.ground([robot, grill]),
        #     Place.ground([robot, patty, grill]),
        #     Cook.ground([robot, patty, grill]),
        #     PickFromStack.ground([robot, patty, grill]),
        #     MoveWhenFacingStart.ground([robot, bottom_bun, grill]),
        #     Place.ground([robot, patty, bottom_bun]),
        #
        #     MoveWhenFacingStart.ground([robot, cheese, patty]),
        #     PickSingleAdjacent.ground([robot, cheese]),
        #     MoveFromNothing.ground([robot, patty]),
        #     Place.ground([robot, cheese, patty]),
        #
        #     MoveWhenFacingStart.ground([robot, tomato, cheese]),
        #     PickSingleAdjacent.ground([robot, tomato]),
        #     MoveFromNothing.ground([robot, cutting_board]),
        #     Place.ground([robot, tomato, cutting_board]),
        #     Slice.ground([robot, tomato, cutting_board]),
        #     PickFromStack.ground([robot, tomato, cutting_board]),
        #     MoveWhenFacingStart.ground([robot, cheese, cutting_board]),
        #     Place.ground([robot, tomato, cheese]),
        #
        #     MoveWhenFacingStart.ground([robot, top_bun, cheese]),
        #     PickSingleAdjacent.ground([robot, top_bun]),
        #     MoveFromNothing.ground([robot, cheese]),
        #     Place.ground([robot, top_bun, tomato])
        # ]
        #
        # atoms_seq = [init]
        # curr_atoms = init
        # curr_nsrt = None
        # for i, n in enumerate(plan):
        #     curr_nsrt = n
        #     try:
        #         assert n.preconditions.issubset(curr_atoms)
        #     except:
        #         import pdb; pdb.set_trace()
        #     curr_atoms = (curr_atoms | n.add_effects) - n.delete_effects
        #     atoms_seq.append(curr_atoms)
        # option_plan = [n.option.ground(n.option_objs, []) for n in plan]

        # curr_atoms = init
        # count = 0
        # import pdb; pdb.set_trace()
        # curr_nsrt = plan[count]
        # curr_nsrt.preconditions.issubset(curr_atoms)
        # curr_atoms = (curr_atoms | curr_nsrt.add_effects) - curr_nsrt.delete_effects
        # count += 1

        # import pdb; pdb.set_trace()
        # nsrt_plan, atoms_seq, metrics = self._run_task_plan(task, nsrts, preds, timeout, seed)
        # import pdb; pdb.set_trace()

        # Run task planning only and then greedily sample and execute in the
        # policy.
        if self._plan_without_sim:
            nsrt_plan, atoms_seq, metrics = self._run_task_plan(
                task, nsrts, preds, timeout, seed)
            self._last_nsrt_plan = nsrt_plan
            self._last_atoms_seq = atoms_seq
            policy = utils.nsrt_plan_to_greedy_policy(nsrt_plan, task.goal,
                                                      self._rng)
            # self._last_plan = plan
            # self._last_nsrt_plan = plan
            # self._last_atoms_seq = atoms_seq
            # policy = utils.nsrt_plan_to_greedy_policy(plan, task.goal,
            #                                           self._rng)
            # logging.debug("Current Task Plan:")
            # for act in nsrt_plan:
            #     logging.debug(act)

        # Run full bilevel planning.
        else:
            option_plan, nsrt_plan, metrics = self._run_sesame_plan(
                task, nsrts, preds, timeout, seed)
            self._last_plan = option_plan
            self._last_nsrt_plan = nsrt_plan
            # self._last_plan = option_plan
            # self._last_nsrt_plan = plan
            policy = utils.option_plan_to_policy(option_plan)


        import pdb; pdb.set_trace()

        # self._save_metrics(metrics, nsrts, preds)

        def _policy(s: State) -> Action:
            try:
                return policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _run_sesame_plan(
            self, task: Task, nsrts: Set[NSRT], preds: Set[Predicate],
            timeout: float, seed: int,
            **kwargs: Any) -> Tuple[List[_Option], List[_GroundNSRT], Metrics]:
        """Subclasses may override.

        For example, PG4 inserts an abstract policy into kwargs.
        """
        try:
            option_plan, nsrt_plan, metrics = sesame_plan(
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

        return option_plan, nsrt_plan, metrics

    def _run_task_plan(
        self, task: Task, nsrts: Set[NSRT], preds: Set[Predicate],
        timeout: float, seed: int, **kwargs: Any
    ) -> Tuple[List[_GroundNSRT], List[Set[GroundAtom]], Metrics]:

        try:
            plan, atoms_seq, metrics = run_task_plan_once(
                task,
                nsrts,
                preds,
                self._types,
                timeout,
                seed,
                task_planning_heuristic=self._task_planning_heuristic,
                max_horizon=float(CFG.horizon),
                **kwargs)
        except PlanningFailure as e:
            raise ApproachFailure(e.args[0], e.info)
        except PlanningTimeout as e:
            raise ApproachTimeout(e.args[0], e.info)

        return plan, atoms_seq, metrics

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
                "num_nodes_created", "plan_length", "refinement_time"
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
        assert not self._plan_without_sim
        return self._last_plan

    def get_last_nsrt_plan(self) -> List[_GroundNSRT]:
        """Similar to get_last_plan() in that only oracle should use this.

        And this will only be used when bilevel_plan_without_sim is
        True.
        """
        assert self.get_name() == "oracle"
        return self._last_nsrt_plan

    def get_execution_monitoring_info(self) -> List[Set[GroundAtom]]:
        if self._plan_without_sim:
            remaining_atoms_seq = list(self._last_atoms_seq)
            if remaining_atoms_seq:
                self._last_atoms_seq.pop(0)
            return remaining_atoms_seq
        return []
