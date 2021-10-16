"""Algorithms for task and motion planning.

Mainly, "SeSamE": SEarch-and-SAMple planning, then Execution.
"""

from __future__ import annotations
from collections import defaultdict
import heapq as hq
import time
from typing import Collection, Callable, List, Set, Optional, Tuple, Dict
from dataclasses import dataclass, field
import numpy as np
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.structs import State, Task, Operator, Predicate, \
    GroundAtom, _GroundOperator, DefaultOption, DefaultState, _Option, Action, \
    PyperplanFacts, Metrics
from predicators.src import utils
from predicators.src.envs import EnvironmentFailure
from predicators.src.settings import CFG

_NOT_CAUSES_FAILURE = "NotCausesFailure"


@dataclass(repr=False, eq=False)
class _Node:
    """A node for the search over skeletons.
    """
    atoms: Collection[GroundAtom]
    skeleton: List[_GroundOperator]
    atoms_sequence: List[Collection[GroundAtom]]  # expected state sequence
    parent: Optional[_Node]
    pyperplan_facts: PyperplanFacts = field(
        init=False, default_factory=frozenset)

    def __post_init__(self) -> None:
        self.pyperplan_facts = utils.atoms_to_tuples(self.atoms)


def sesame_plan(task: Task,
                simulator: Callable[[State, Action], State],
                current_operators: Set[Operator],
                initial_predicates: Set[Predicate],
                timeout: float, seed: int,
                check_dr_reachable: bool = True
                ) -> Tuple[List[Action], Metrics]:
    """Run TAMP. Return a sequence of low-level actions, and a dictionary
    of metrics for this run of the planner. Uses the SeSamE strategy:
    SEarch-and-SAMple planning, then Execution.
    """
    op_preds, _ = utils.extract_preds_and_types(current_operators)
    # Ensure that initial predicates are always included.
    predicates = initial_predicates | set(op_preds.values())
    atoms = utils.abstract(task.init, predicates)
    objects = list(task.init)
    ground_operators = []
    for op in current_operators:
        for ground_op in utils.all_ground_operators(op, objects):
            ground_operators.append(ground_op)
    ground_operators = utils.filter_static_operators(
        ground_operators, atoms)
    # Keep restarting the A* search while we get new discovered failures.
    start_time = time.time()
    metrics: Metrics = defaultdict(float)
    while True:
        if check_dr_reachable and \
           not utils.is_dr_reachable(ground_operators, atoms, task.goal):
            raise ApproachFailure(f"Goal {task.goal} not dr-reachable")
        try:
            new_seed = seed+int(metrics["num_failures_discovered"])
            plan = _run_search(
                task, simulator, ground_operators, atoms, predicates,
                timeout-(time.time()-start_time), new_seed, metrics)
            break  # planning succeeded, break out of loop
        except _DiscoveredFailureException as e:
            metrics["num_failures_discovered"] += 1
            _update_operators_with_failure(
                e.discovered_failure, ground_operators)
    print(f"Planning succeeded! Found plan of length {len(plan)} after trying "
          f"{int(metrics['num_skeletons_optimized'])} skeletons, discovering "
          f"{int(metrics['num_failures_discovered'])} failures")
    metrics["plan_length"] = len(plan)
    return plan, metrics


def _run_search(task: Task,
                simulator: Callable[[State, Action], State],
                ground_operators: List[_GroundOperator],
                init_atoms: Collection[GroundAtom],
                predicates: Set[Predicate],
                timeout: float, seed: int,
                metrics: Metrics) -> List[Action]:
    """A* search over skeletons (sequences of ground operators).
    """
    start_time = time.time()
    queue: List[Tuple[float, float, _Node]] = []
    root_node = _Node(atoms=init_atoms, skeleton=[],
                     atoms_sequence=[init_atoms], parent=None)
    rng_prio = np.random.default_rng(seed)
    rng_sampler = np.random.default_rng(seed)
    # Set up stuff for pyperplan heuristic.
    relaxed_operators = frozenset({utils.RelaxedOperator(
        op.name, utils.atoms_to_tuples(op.preconditions),
        utils.atoms_to_tuples(op.add_effects)) for op in ground_operators})
    heuristic_cache: Dict[PyperplanFacts, float] = {}
    heuristic: Callable[[PyperplanFacts], float] = utils.HAddHeuristic(
        utils.atoms_to_tuples(init_atoms),
        utils.atoms_to_tuples(task.goal), relaxed_operators)
    heuristic_cache[root_node.pyperplan_facts] = heuristic(
        root_node.pyperplan_facts)
    hq.heappush(queue, (heuristic_cache[root_node.pyperplan_facts],
                        rng_prio.uniform(),
                        root_node))
    # Start search.
    while queue and (time.time()-start_time < timeout):
        if (int(metrics["num_skeletons_optimized"]) ==
            CFG.max_skeletons_optimized):
            raise ApproachFailure("Planning reached max_skeletons_optimized!")
        _, _, node = hq.heappop(queue)
        # Good debug point #1: print node.skeleton here to see what
        # the high-level search is doing.
        if task.goal.issubset(node.atoms):
            # If this skeleton satisfies the goal, run low-level search.
            metrics["num_skeletons_optimized"] += 1
            plan = _run_low_level_search(
                task, simulator, node.skeleton, node.atoms_sequence,
                rng_sampler, predicates, start_time, timeout)
            if plan is not None:
                return plan
        else:
            # Generate successors.
            for operator in utils.get_applicable_operators(
                    ground_operators, node.atoms):
                child_atoms = utils.apply_operator(
                    operator, set(node.atoms))
                child_node = _Node(
                    atoms=child_atoms,
                    skeleton=node.skeleton+[operator],
                    atoms_sequence=node.atoms_sequence+[child_atoms],
                    parent=node)
                if child_node.pyperplan_facts not in heuristic_cache:
                    heuristic_cache[child_node.pyperplan_facts] = heuristic(
                        child_node.pyperplan_facts)
                # priority is g [plan length] plus h [heuristic]
                priority = (len(child_node.skeleton)+
                            heuristic_cache[child_node.pyperplan_facts])
                hq.heappush(queue, (priority,
                                    rng_prio.uniform(),
                                    child_node))
    if not queue:
        raise ApproachFailure("Planning ran out of skeletons!")
    assert time.time()-start_time > timeout
    raise ApproachTimeout("Planning timed out in skeleton search!")


def _run_low_level_search(
        task: Task,
        simulator: Callable[[State, Action], State],
        skeleton: List[_GroundOperator],
        atoms_sequence: List[Collection[GroundAtom]],
        rng_sampler: np.random.Generator,
        predicates: Set[Predicate],
        start_time: float,
        timeout: float) -> Optional[List[Action]]:
    """Backtracking search over continuous values.
    """
    cur_idx = 0
    num_tries = [0 for _ in skeleton]
    options: List[_Option] = [DefaultOption for _ in skeleton]
    plan: List[List[Action]] = [[] for _ in skeleton]  # unflattened
    traj: List[State] = [task.init]+[DefaultState for _ in skeleton]
    # We'll use a maximum of one discovered failure per step, since
    # resampling can render old discovered failures obsolete.
    discovered_failures: List[
        Optional[_DiscoveredFailure]] = [None for _ in skeleton]
    while cur_idx < len(skeleton):
        if time.time()-start_time > timeout:
            raise ApproachTimeout("Planning timed out in backtracking!")
        assert num_tries[cur_idx] < CFG.max_samples_per_step
        # Good debug point #2: if you have a skeleton that you think is
        # reasonable, but sampling isn't working, print num_tries here to
        # see at what step the backtracking search is getting stuck.
        num_tries[cur_idx] += 1
        state = traj[cur_idx]
        operator = skeleton[cur_idx]
        # Ground the operator's ParameterizedOption into an _Option.
        # This invokes the operator's sampler.
        option = operator.sample_option(state, rng_sampler)
        options[cur_idx] = option
        try:
            option_traj_states, option_traj_acts = utils.option_to_trajectory(
                state, simulator, option,
                max_num_steps=CFG.max_num_steps_option_rollout)
            discovered_failures[cur_idx] = None  # no failure occurred
        except EnvironmentFailure as e:
            can_continue_on = False
            discovered_failures[cur_idx] = _DiscoveredFailure(
                e, operator)  # remember only the most recent failure
        if not discovered_failures[cur_idx]:
            traj[cur_idx+1] = option_traj_states[-1]  # ignore previous states
            plan[cur_idx] = option_traj_acts
            cur_idx += 1
            # Check atoms again expected atoms_sequence constraint.
            assert len(traj) == len(atoms_sequence)
            atoms = utils.abstract(traj[cur_idx], predicates)
            if atoms == {atom for atom in atoms_sequence[cur_idx]
                         if atom.predicate.name != _NOT_CAUSES_FAILURE}:
                can_continue_on = True
                if cur_idx == len(skeleton):  # success!
                    result = [act for acts in plan for act in acts]  # flatten
                    return result
            else:
                can_continue_on = False
        else:
            cur_idx += 1  # it's about to be decremented again
        if not can_continue_on:
            # Go back to re-do the step we just did. If necessary, backtrack.
            cur_idx -= 1
            while num_tries[cur_idx] == CFG.max_samples_per_step:
                num_tries[cur_idx] = 0
                options[cur_idx] = DefaultOption
                plan[cur_idx] = []
                traj[cur_idx+1] = DefaultState
                cur_idx -= 1
                if cur_idx < 0:
                    # Backtracking exhausted. If there were any failures,
                    # propagate up the EARLIEST one so that search restarts.
                    # Otherwise, return None so that search continues.
                    for failure in discovered_failures:
                        if CFG.propagate_failures and failure is not None:
                            raise _DiscoveredFailureException(
                                "Discovered a failure", failure)
                    return None
    # Should only get here if the skeleton was empty
    assert not skeleton
    return []


def _update_operators_with_failure(
        discovered_failure: _DiscoveredFailure,
        ground_operators: Collection[_GroundOperator]) -> None:
    """Update the given set of ground_operators based on the given
    DiscoveredFailure.
    """
    for obj in discovered_failure.env_failure.offending_objects:
        atom = GroundAtom(Predicate(_NOT_CAUSES_FAILURE, [obj.type],
                                    _classifier=lambda s, o: False), [obj])
        # Update the preconditions of the failing operator.
        discovered_failure.failing_operator.preconditions.add(atom)
        # Update the effects of all operators that use this object.
        for op in ground_operators:
            if obj in op.objects:
                op.add_effects.add(atom)


@dataclass(frozen=True, eq=False)
class _DiscoveredFailure:
    """Container class for holding information related to a low-level
    discovery of a failure which must be propagated up to the main
    search function, in order to restart A* search with new operators.
    """
    env_failure: EnvironmentFailure
    failing_operator: _GroundOperator


class _DiscoveredFailureException(Exception):
    """Exception class for DiscoveredFailure propagation.
    """
    def __init__(self, message: str, discovered_failure: _DiscoveredFailure):
        super().__init__(message)
        self.discovered_failure = discovered_failure
