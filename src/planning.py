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
from predicators.src.structs import State, Task, NSRT, Predicate, \
    GroundAtom, _GroundNSRT, DefaultOption, DefaultState, _Option, \
    PyperplanFacts, Metrics
from predicators.src import utils
from predicators.src.envs import EnvironmentFailure
from predicators.src.option_model import _OptionModel
from predicators.src.settings import CFG

_NOT_CAUSES_FAILURE = "NotCausesFailure"


@dataclass(repr=False, eq=False)
class _Node:
    """A node for the search over skeletons.
    """
    atoms: Collection[GroundAtom]
    skeleton: List[_GroundNSRT]
    atoms_sequence: List[Collection[GroundAtom]]  # expected state sequence
    parent: Optional[_Node]
    pyperplan_facts: PyperplanFacts = field(
        init=False, default_factory=frozenset)

    def __post_init__(self) -> None:
        self.pyperplan_facts = utils.atoms_to_tuples(self.atoms)


def sesame_plan(task: Task,
                option_model: _OptionModel,
                nsrts: Set[NSRT],
                initial_predicates: Set[Predicate],
                timeout: float, seed: int,
                check_dr_reachable: bool = True
                ) -> Tuple[List[_Option], Metrics]:
    """Run TAMP. Return a sequence of options, and a dictionary
    of metrics for this run of the planner. Uses the SeSamE strategy:
    SEarch-and-SAMple planning, then Execution.
    """
    nsrt_preds, _ = utils.extract_preds_and_types(nsrts)
    # Ensure that initial predicates are always included.
    predicates = initial_predicates | set(nsrt_preds.values())
    atoms = utils.abstract(task.init, predicates)
    objects = list(task.init)
    ground_nsrts = []
    for nsrt in nsrts:
        for ground_nsrt in utils.all_ground_nsrts(nsrt, objects):
            ground_nsrts.append(ground_nsrt)
    ground_nsrts = utils.filter_static_nsrts(ground_nsrts, atoms)
    # Keep restarting the A* search while we get new discovered failures.
    start_time = time.time()
    metrics: Metrics = defaultdict(float)
    while True:
        # There is no point in using NSRTs with empty effects, and they can
        # slow down search significantly, so we exclude them. Note however
        # that we need to do this inside the while True here, because an NSRT
        # that initially has empty effects may later have a _NOT_CAUSES_FAILURE.
        nonempty_ground_nsrts = [nsrt for nsrt in ground_nsrts
                                 if nsrt.add_effects | nsrt.delete_effects]
        if check_dr_reachable and \
           not utils.is_dr_reachable(nonempty_ground_nsrts, atoms, task.goal):
            raise ApproachFailure(f"Goal {task.goal} not dr-reachable")
        try:
            new_seed = seed+int(metrics["num_failures_discovered"])
            plan = _run_search(
                task, option_model, nonempty_ground_nsrts, atoms, predicates,
                timeout-(time.time()-start_time), new_seed, metrics)
            break  # planning succeeded, break out of loop
        except _DiscoveredFailureException as e:
            metrics["num_failures_discovered"] += 1
            _update_nsrts_with_failure(e.discovered_failure, ground_nsrts)
    print(f"Planning succeeded! Found plan of length {len(plan)} after trying "
          f"{int(metrics['num_skeletons_optimized'])} skeletons, discovering "
          f"{int(metrics['num_failures_discovered'])} failures")
    metrics["plan_length"] = len(plan)
    return plan, metrics


def _run_search(task: Task,
                option_model: _OptionModel,
                ground_nsrts: List[_GroundNSRT],
                init_atoms: Collection[GroundAtom],
                predicates: Set[Predicate],
                timeout: float, seed: int,
                metrics: Metrics) -> List[_Option]:
    """A* search over skeletons (sequences of ground NSRTs).
    """
    start_time = time.time()
    queue: List[Tuple[float, float, _Node]] = []
    root_node = _Node(atoms=init_atoms, skeleton=[],
                     atoms_sequence=[init_atoms], parent=None)
    rng_prio = np.random.default_rng(seed)
    rng_sampler = np.random.default_rng(seed)
    # Set up stuff for pyperplan heuristic.
    relaxed_operators = frozenset({utils.RelaxedOperator(
        nsrt.name, utils.atoms_to_tuples(nsrt.preconditions),
        utils.atoms_to_tuples(nsrt.add_effects)) for nsrt in ground_nsrts})
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
                task, option_model, node.skeleton, node.atoms_sequence,
                rng_sampler, predicates, start_time, timeout)
            if plan is not None:
                return plan
        else:
            # Generate successors.
            for nsrt in utils.get_applicable_nsrts(ground_nsrts, node.atoms):
                child_atoms = utils.apply_nsrt(nsrt, set(node.atoms))
                child_node = _Node(
                    atoms=child_atoms,
                    skeleton=node.skeleton+[nsrt],
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
        option_model: _OptionModel,
        skeleton: List[_GroundNSRT],
        atoms_sequence: List[Collection[GroundAtom]],
        rng_sampler: np.random.Generator,
        predicates: Set[Predicate],
        start_time: float,
        timeout: float) -> Optional[List[_Option]]:
    """Backtracking search over continuous values.
    """
    assert CFG.sesame_propagate_failures in \
        {"after_exhaust", "immediately", "never"}
    cur_idx = 0
    num_tries = [0 for _ in skeleton]
    plan: List[_Option] = [DefaultOption for _ in skeleton]
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
        nsrt = skeleton[cur_idx]
        # Ground the NSRT's ParameterizedOption into an _Option.
        # This invokes the NSRT's sampler.
        option = nsrt.sample_option(state, rng_sampler)
        plan[cur_idx] = option
        if option.initiable(state):
            try:
                next_state = option_model.get_next_state(state, option)
                discovered_failures[cur_idx] = None  # no failure occurred
            except EnvironmentFailure as e:
                can_continue_on = False
                failure = _DiscoveredFailure(e, nsrt)
                # Remember only the most recent failure.
                discovered_failures[cur_idx] = failure
                # If we're immediately propagating failures, raise it now.
                if CFG.sesame_propagate_failures == "immediately":
                    raise _DiscoveredFailureException(
                        "Discovered a failure", failure)
            if not discovered_failures[cur_idx]:
                traj[cur_idx+1] = next_state
                cur_idx += 1
                # Check atoms against expected atoms_sequence constraint.
                assert len(traj) == len(atoms_sequence)
                atoms = utils.abstract(traj[cur_idx], predicates)
                if atoms == {atom for atom in atoms_sequence[cur_idx]
                             if atom.predicate.name != _NOT_CAUSES_FAILURE}:
                    can_continue_on = True
                    if cur_idx == len(skeleton):  # success!
                        result = plan
                        return result
                else:
                    can_continue_on = False
            else:
                cur_idx += 1  # it's about to be decremented again
        else:
            # If the option is not initiable, need to resample / backtrack.
            can_continue_on = False
            cur_idx += 1  # it's about to be decremented again
        if not can_continue_on:
            # Go back to re-do the step we just did. If necessary, backtrack.
            cur_idx -= 1
            assert cur_idx >= 0
            while num_tries[cur_idx] == CFG.max_samples_per_step:
                num_tries[cur_idx] = 0
                plan[cur_idx] = DefaultOption
                traj[cur_idx+1] = DefaultState
                cur_idx -= 1
                if cur_idx < 0:
                    # Backtracking exhausted. If we're only propagating failures
                    # after exhaustion, and if there are any failures,
                    # propagate up the EARLIEST one so that search restarts.
                    # Otherwise, return None so that search continues.
                    for earliest_failure in discovered_failures:
                        if (CFG.sesame_propagate_failures == "after_exhaust"
                            and earliest_failure is not None):
                            raise _DiscoveredFailureException(
                                "Discovered a failure", earliest_failure)
                    return None
    # Should only get here if the skeleton was empty
    assert not skeleton
    return []


def _update_nsrts_with_failure(
        discovered_failure: _DiscoveredFailure,
        ground_nsrts: Collection[_GroundNSRT]) -> None:
    """Update the given set of ground_nsrts based on the given
    DiscoveredFailure.
    """
    for obj in discovered_failure.env_failure.offending_objects:
        atom = GroundAtom(Predicate(_NOT_CAUSES_FAILURE, [obj.type],
                                    _classifier=lambda s, o: False), [obj])
        # Update the preconditions of the failing NSRT.
        discovered_failure.failing_nsrt.preconditions.add(atom)
        # Update the effects of all nsrts that use this object.
        for nsrt in ground_nsrts:
            if obj in nsrt.objects:
                nsrt.add_effects.add(atom)


@dataclass(frozen=True, eq=False)
class _DiscoveredFailure:
    """Container class for holding information related to a low-level
    discovery of a failure which must be propagated up to the main
    search function, in order to restart A* search with new NSRTs.
    """
    env_failure: EnvironmentFailure
    failing_nsrt: _GroundNSRT


class _DiscoveredFailureException(Exception):
    """Exception class for DiscoveredFailure propagation.
    """
    def __init__(self, message: str, discovered_failure: _DiscoveredFailure):
        super().__init__(message)
        self.discovered_failure = discovered_failure
