"""Algorithms for bilevel planning.

Mainly, "SeSamE": SEarch-and-SAMple planning, then Execution.
"""

from __future__ import annotations

import heapq as hq
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators.src import utils
from predicators.src.option_model import _OptionModelBase
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, DefaultState, DummyOption, \
    GroundAtom, Metrics, Object, OptionSpec, Predicate, State, \
    STRIPSOperator, Task, Type, _GroundNSRT, _Option
from predicators.src.utils import EnvironmentFailure, ExceptionWithInfo, \
    _TaskPlanningHeuristic

_NOT_CAUSES_FAILURE = "NotCausesFailure"


@dataclass(repr=False, eq=False)
class _Node:
    """A node for the search over skeletons."""
    atoms: Set[GroundAtom]
    skeleton: List[_GroundNSRT]
    atoms_sequence: List[Set[GroundAtom]]  # expected state sequence
    parent: Optional[_Node]


def sesame_plan(
    task: Task,
    option_model: _OptionModelBase,
    nsrts: Set[NSRT],
    initial_predicates: Set[Predicate],
    types: Set[Type],
    timeout: float,
    seed: int,
    task_planning_heuristic: str,
    max_skeletons_optimized: int,
    max_horizon: int,
    check_dr_reachable: bool = True,
    allow_noops: bool = False,
) -> Tuple[List[_Option], Metrics]:
    """Run bilevel planning.

    Return a sequence of options, and a dictionary of metrics for this
    run of the planner. Uses the SeSamE strategy: SEarch-and-SAMple
    planning, then Execution.
    """
    # Note: the types that would be extracted from the NSRTs here may not
    # include all the environment's types, so it's better to use the
    # types that are passed in as an argument instead.
    nsrt_preds, _ = utils.extract_preds_and_types(nsrts)
    # Ensure that initial predicates are always included.
    predicates = initial_predicates | set(nsrt_preds.values())
    init_atoms = utils.abstract(task.init, predicates)
    objects = list(task.init)
    start_time = time.time()
    if CFG.sesame_grounder == "naive":
        ground_nsrts = []
        for nsrt in sorted(nsrts):
            for ground_nsrt in utils.all_ground_nsrts(nsrt, objects):
                ground_nsrts.append(ground_nsrt)
                if time.time() - start_time > timeout:
                    raise PlanningTimeout("Planning timed out in grounding!")
    elif CFG.sesame_grounder == "fd_translator":
        # WARNING: there is no easy way to check the timeout within this call,
        # since Fast Downward's translator is a third-party function. We'll
        # just check the timeout afterward.
        ground_nsrts = list(
            utils.all_ground_nsrts_fd_translator(nsrts, objects, predicates,
                                                 types, init_atoms, task.goal))
        if time.time() - start_time > timeout:
            raise PlanningTimeout("Planning timed out in grounding!")
    else:
        raise ValueError(
            f"Unrecognized sesame_grounder: {CFG.sesame_grounder}")
    # Keep restarting the A* search while we get new discovered failures.
    metrics: Metrics = defaultdict(float)
    # Keep track of partial refinements: skeletons and partial plans. This is
    # for making videos of failed planning attempts.
    partial_refinements = []
    while True:
        # Optionally exclude NSRTs with empty effects, because they can slow
        # the search significantly, so we may want to exclude them. Note however
        # that we need to do this inside the while True here, because an NSRT
        # that initially has empty effects may later have a _NOT_CAUSES_FAILURE.
        nonempty_ground_nsrts = [
            nsrt for nsrt in ground_nsrts
            if allow_noops or (nsrt.add_effects | nsrt.delete_effects)
        ]
        all_reachable_atoms = utils.get_reachable_atoms(
            nonempty_ground_nsrts, init_atoms)
        if check_dr_reachable and not task.goal.issubset(all_reachable_atoms):
            raise PlanningFailure(f"Goal {task.goal} not dr-reachable")
        reachable_nsrts = [
            nsrt for nsrt in nonempty_ground_nsrts
            if nsrt.preconditions.issubset(all_reachable_atoms)
        ]
        heuristic = utils.create_task_planning_heuristic(
            task_planning_heuristic, init_atoms, task.goal, reachable_nsrts,
            predicates, objects)
        try:
            new_seed = seed + int(metrics["num_failures_discovered"])
            for skeleton, atoms_sequence in _skeleton_generator(
                    task, reachable_nsrts, init_atoms, heuristic, new_seed,
                    timeout - (time.time() - start_time), metrics,
                    max_skeletons_optimized):
                plan, suc = _run_low_level_search(
                    task, option_model, skeleton, atoms_sequence, new_seed,
                    timeout - (time.time() - start_time), max_horizon)
                if suc:
                    # Success! It's a complete plan.
                    logging.info(
                        f"Planning succeeded! Found plan of length "
                        f"{len(plan)} after "
                        f"{int(metrics['num_skeletons_optimized'])} "
                        f"skeletons, discovering "
                        f"{int(metrics['num_failures_discovered'])} failures")
                    metrics["plan_length"] = len(plan)
                    return plan, metrics
                partial_refinements.append((skeleton, plan))
                if time.time() - start_time > timeout:
                    raise PlanningTimeout(
                        "Planning timed out in backtracking!",
                        info={"partial_refinements": partial_refinements})
        except _DiscoveredFailureException as e:
            metrics["num_failures_discovered"] += 1
            new_predicates, ground_nsrts = _update_nsrts_with_failure(
                e.discovered_failure, ground_nsrts)
            predicates |= new_predicates
            partial_refinements.append(
                (skeleton, e.info["longest_failed_refinement"]))
        except (_MaxSkeletonsFailure, _SkeletonSearchTimeout) as e:
            e.info["partial_refinements"] = partial_refinements
            raise e


def task_plan_grounding(
    init_atoms: Set[GroundAtom],
    objects: Set[Object],
    strips_ops: Sequence[STRIPSOperator],
    option_specs: Sequence[OptionSpec],
    allow_noops: bool = False,
) -> Tuple[List[_GroundNSRT], Set[GroundAtom]]:
    """Ground all operators for task planning into dummy _GroundNSRTs,
    filtering out ones that are unreachable or have empty effects.

    Also return the set of reachable atoms, which is used by task
    planning to quickly determine if a goal is unreachable.

    See the task_plan docstring for usage instructions.
    """
    nsrts = utils.ops_and_specs_to_dummy_nsrts(strips_ops, option_specs)
    ground_nsrts = []
    for nsrt in sorted(nsrts):
        for ground_nsrt in utils.all_ground_nsrts(nsrt, objects):
            if allow_noops or (ground_nsrt.add_effects
                               | ground_nsrt.delete_effects):
                ground_nsrts.append(ground_nsrt)
    reachable_atoms = utils.get_reachable_atoms(ground_nsrts, init_atoms)
    reachable_nsrts = [
        nsrt for nsrt in ground_nsrts
        if nsrt.preconditions.issubset(reachable_atoms)
    ]
    return reachable_nsrts, reachable_atoms


def task_plan(
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_nsrts: List[_GroundNSRT],
    reachable_atoms: Set[GroundAtom],
    heuristic: _TaskPlanningHeuristic,
    seed: int,
    timeout: float,
    max_skeletons_optimized: int,
) -> Iterator[Tuple[List[_GroundNSRT], List[Set[GroundAtom]], Metrics]]:
    """Run only the task planning portion of SeSamE. A* search is run, and
    skeletons that achieve the goal symbolically are yielded. Specifically,
    yields a tuple of (skeleton, atoms sequence, metrics dictionary).

    This method is NOT used by SeSamE, but is instead provided as a
    convenient wrapper around _skeleton_generator below (which IS used
    by SeSamE) that takes in only the minimal necessary arguments.

    This method is tightly coupled with task_plan_grounding -- the reason they
    are separate methods is that it is sometimes possible to ground only once
    and then plan multiple times (e.g. from different initial states, or to
    different goals). To run task planning once, call task_plan_grounding to
    get ground_nsrts and reachable_atoms; then create a heuristic using
    utils.create_task_planning_heuristic; then call this method. See the tests
    in tests/test_planning for usage examples.
    """
    if not goal.issubset(reachable_atoms):
        raise PlanningFailure(f"Goal {goal} not dr-reachable")
    dummy_task = Task(DefaultState, goal)
    metrics: Metrics = defaultdict(float)
    generator = _skeleton_generator(dummy_task, ground_nsrts, init_atoms,
                                    heuristic, seed, timeout, metrics,
                                    max_skeletons_optimized)
    # Note that we use this pattern to avoid having to catch an exception
    # when _skeleton_generator runs out of skeletons to optimize.
    for skeleton, atoms_sequence in islice(generator, max_skeletons_optimized):
        yield skeleton, atoms_sequence, metrics.copy()


def _skeleton_generator(
    task: Task, ground_nsrts: List[_GroundNSRT], init_atoms: Set[GroundAtom],
    heuristic: _TaskPlanningHeuristic, seed: int, timeout: float,
    metrics: Metrics, max_skeletons_optimized: int
) -> Iterator[Tuple[List[_GroundNSRT], List[Set[GroundAtom]]]]:
    """A* search over skeletons (sequences of ground NSRTs).
    Iterates over pairs of (skeleton, atoms sequence).
    """
    start_time = time.time()
    queue: List[Tuple[float, float, _Node]] = []
    root_node = _Node(atoms=init_atoms,
                      skeleton=[],
                      atoms_sequence=[init_atoms],
                      parent=None)
    metrics["num_nodes_created"] += 1
    rng_prio = np.random.default_rng(seed)
    hq.heappush(queue,
                (heuristic(root_node.atoms), rng_prio.uniform(), root_node))
    # Start search.
    while queue and (time.time() - start_time < timeout):
        if int(metrics["num_skeletons_optimized"]) == max_skeletons_optimized:
            raise _MaxSkeletonsFailure(
                "Planning reached max_skeletons_optimized!")
        _, _, node = hq.heappop(queue)
        # Good debug point #1: print out the skeleton here to see what
        # the high-level search is doing. You can accomplish this via:
        # for act in node.skeleton:
        #     logging.info(f"{act.name} {act.objects}")
        # logging.info("")
        if task.goal.issubset(node.atoms):
            # If this skeleton satisfies the goal, yield it.
            metrics["num_skeletons_optimized"] += 1
            yield node.skeleton, node.atoms_sequence
        else:
            # Generate successors.
            metrics["num_nodes_expanded"] += 1
            for nsrt in utils.get_applicable_operators(ground_nsrts,
                                                       node.atoms):
                child_atoms = utils.apply_operator(nsrt, set(node.atoms))
                child_node = _Node(atoms=child_atoms,
                                   skeleton=node.skeleton + [nsrt],
                                   atoms_sequence=node.atoms_sequence +
                                   [child_atoms],
                                   parent=node)
                metrics["num_nodes_created"] += 1
                # priority is g [plan length] plus h [heuristic]
                priority = (len(child_node.skeleton) +
                            heuristic(child_node.atoms))
                hq.heappush(queue, (priority, rng_prio.uniform(), child_node))
                if time.time() - start_time >= timeout:
                    break
    if not queue:
        raise _MaxSkeletonsFailure("Planning ran out of skeletons!")
    assert time.time() - start_time >= timeout
    raise _SkeletonSearchTimeout


def _run_low_level_search(task: Task, option_model: _OptionModelBase,
                          skeleton: List[_GroundNSRT],
                          atoms_sequence: List[Set[GroundAtom]], seed: int,
                          timeout: float,
                          max_horizon: int) -> Tuple[List[_Option], bool]:
    """Backtracking search over continuous values.

    Returns a sequence of options and a boolean. If the boolean is True,
    the option sequence is a complete low-level plan refining the given
    skeleton. Otherwise, the option sequence is the longest partial
    failed refinement, where the last step did not satisfy the skeleton,
    but all previous steps did. Note that there are multiple low-level
    plans in general; we return the first one found (arbitrarily).
    """
    start_time = time.time()
    rng_sampler = np.random.default_rng(seed)
    assert CFG.sesame_propagate_failures in \
        {"after_exhaust", "immediately", "never"}
    cur_idx = 0
    num_tries = [0 for _ in skeleton]
    # Optimization: if the params_space for the NSRT option is empty, only
    # sample it once, because all samples are just empty (so equivalent).
    max_tries = [
        CFG.sesame_max_samples_per_step
        if nsrt.option.params_space.shape[0] > 0 else 1 for nsrt in skeleton
    ]
    plan: List[_Option] = [DummyOption for _ in skeleton]
    # The number of actions taken by each option in the plan. This is to
    # make sure that we do not exceed the task horizon.
    num_actions_per_option = [0 for _ in plan]
    traj: List[State] = [task.init] + [DefaultState for _ in skeleton]
    longest_failed_refinement: List[_Option] = []
    # We'll use a maximum of one discovered failure per step, since
    # resampling can render old discovered failures obsolete.
    discovered_failures: List[Optional[_DiscoveredFailure]] = [
        None for _ in skeleton
    ]
    while cur_idx < len(skeleton):
        if time.time() - start_time > timeout:
            return longest_failed_refinement, False
        assert num_tries[cur_idx] < max_tries[cur_idx]
        # Good debug point #2: if you have a skeleton that you think is
        # reasonable, but sampling isn't working, print num_tries here to
        # see at what step the backtracking search is getting stuck.
        num_tries[cur_idx] += 1
        state = traj[cur_idx]
        nsrt = skeleton[cur_idx]
        # Ground the NSRT's ParameterizedOption into an _Option.
        # This invokes the NSRT's sampler.
        option = nsrt.sample_option(state, task.goal, rng_sampler)
        plan[cur_idx] = option
        # Increment cur_idx. It will be decremented later on if we get stuck.
        cur_idx += 1
        if option.initiable(state):
            try:
                next_state, num_actions = \
                    option_model.get_next_state_and_num_actions(state, option)
            except EnvironmentFailure as e:
                can_continue_on = False
                # Remember only the most recent failure.
                discovered_failures[cur_idx - 1] = _DiscoveredFailure(e, nsrt)
            else:  # an EnvironmentFailure was not raised
                discovered_failures[cur_idx - 1] = None
                num_actions_per_option[cur_idx - 1] = num_actions
                traj[cur_idx] = next_state
                # Check if we have exceeded the horizon.
                if np.sum(num_actions_per_option[:cur_idx]) > max_horizon:
                    can_continue_on = False
                # Check if the option was effectively a no-op.
                elif num_actions == 0:
                    can_continue_on = False
                elif CFG.sesame_check_expected_atoms:
                    # Check atoms against expected atoms_sequence constraint.
                    assert len(traj) == len(atoms_sequence)
                    # The expected atoms are ones that we definitely expect to
                    # be true at this point in the plan. They are not *all* the
                    # atoms that could be true.
                    expected_atoms = {
                        atom
                        for atom in atoms_sequence[cur_idx]
                        if atom.predicate.name != _NOT_CAUSES_FAILURE
                    }
                    # This "if all" statement is equivalent to, but faster
                    # than, checking whether expected_atoms is a subset of
                    # utils.abstract(traj[cur_idx], predicates).
                    if all(a.holds(traj[cur_idx]) for a in expected_atoms):
                        can_continue_on = True
                        if cur_idx == len(skeleton):
                            return plan, True  # success!
                    else:
                        can_continue_on = False
                else:
                    # If we're not checking expected_atoms, we need to
                    # explicitly check the goal on the final timestep.
                    can_continue_on = True
                    if cur_idx == len(skeleton):
                        if task.goal_holds(traj[cur_idx]):
                            return plan, True  # success!
                        can_continue_on = False
        else:
            # The option is not initiable.
            can_continue_on = False
        if not can_continue_on:  # we got stuck, time to resample / backtrack!
            # Update the longest_failed_refinement found so far.
            if cur_idx > len(longest_failed_refinement):
                longest_failed_refinement = list(plan[:cur_idx])
            # If we're immediately propagating failures, and we got a failure,
            # raise it now. We don't do this right after catching the
            # EnvironmentFailure because we want to make sure to update
            # the longest_failed_refinement first.
            possible_failure = discovered_failures[cur_idx - 1]
            if possible_failure is not None and \
               CFG.sesame_propagate_failures == "immediately":
                raise _DiscoveredFailureException(
                    "Discovered a failure", possible_failure,
                    {"longest_failed_refinement": longest_failed_refinement})
            # Decrement cur_idx to re-do the step we just did. If num_tries
            # is exhausted, backtrack.
            cur_idx -= 1
            assert cur_idx >= 0
            while num_tries[cur_idx] == max_tries[cur_idx]:
                num_tries[cur_idx] = 0
                plan[cur_idx] = DummyOption
                num_actions_per_option[cur_idx] = 0
                traj[cur_idx + 1] = DefaultState
                cur_idx -= 1
                if cur_idx < 0:
                    # Backtracking exhausted. If we're only propagating failures
                    # after exhaustion, and if there are any failures,
                    # propagate up the EARLIEST one so that high-level search
                    # restarts. Otherwise, return a partial refinement so that
                    # high-level search continues.
                    for possible_failure in discovered_failures:
                        if possible_failure is not None and \
                           CFG.sesame_propagate_failures == "after_exhaust":
                            raise _DiscoveredFailureException(
                                "Discovered a failure", possible_failure, {
                                    "longest_failed_refinement":
                                    longest_failed_refinement
                                })
                    return longest_failed_refinement, False
    # Should only get here if the skeleton was empty.
    assert not skeleton
    return [], True


def _update_nsrts_with_failure(
    discovered_failure: _DiscoveredFailure, ground_nsrts: List[_GroundNSRT]
) -> Tuple[Set[Predicate], List[_GroundNSRT]]:
    """Update the given set of ground_nsrts based on the given
    DiscoveredFailure.

    Returns a new list of ground NSRTs to replace the input one, where
    all ground NSRTs that need modification are replaced with new ones
    (because _GroundNSRTs are frozen).
    """
    new_predicates = set()
    new_ground_nsrts = []
    for obj in discovered_failure.env_failure.info["offending_objects"]:
        pred = Predicate(_NOT_CAUSES_FAILURE, [obj.type],
                         _classifier=lambda s, o: False)
        new_predicates.add(pred)
        atom = GroundAtom(pred, [obj])
        for ground_nsrt in ground_nsrts:
            # Update the preconditions of the failing NSRT.
            if ground_nsrt == discovered_failure.failing_nsrt:
                new_ground_nsrt = ground_nsrt.copy_with(
                    preconditions=ground_nsrt.preconditions | {atom})
            # Update the effects of all NSRTs that use this object.
            # Note that this is an elif rather than an if, because it would
            # never be possible to use the failing NSRT's effects to set
            # the _NOT_CAUSES_FAILURE precondition.
            elif obj in ground_nsrt.objects:
                new_ground_nsrt = ground_nsrt.copy_with(
                    add_effects=ground_nsrt.add_effects | {atom})
            else:
                new_ground_nsrt = ground_nsrt
            new_ground_nsrts.append(new_ground_nsrt)
    return new_predicates, new_ground_nsrts


@dataclass(frozen=True, eq=False)
class _DiscoveredFailure:
    """Container class for holding information related to a low-level discovery
    of a failure which must be propagated up to the main search function, in
    order to restart A* search with new NSRTs."""
    env_failure: EnvironmentFailure
    failing_nsrt: _GroundNSRT


class _DiscoveredFailureException(ExceptionWithInfo):
    """Exception class for DiscoveredFailure propagation."""

    def __init__(self,
                 message: str,
                 discovered_failure: _DiscoveredFailure,
                 info: Optional[Dict] = None):
        super().__init__(message, info)
        self.discovered_failure = discovered_failure


class PlanningFailure(utils.ExceptionWithInfo):
    """Raised when the planner fails."""


class PlanningTimeout(utils.ExceptionWithInfo):
    """Raised when the planner times out."""


class _MaxSkeletonsFailure(PlanningFailure):
    """Raised when the maximum number of skeletons has been reached."""


class _SkeletonSearchTimeout(PlanningTimeout):
    """Raised when timeout occurs in _run_low_level_search()."""

    def __init__(self) -> None:
        super().__init__("Planning timed out in skeleton search!")
