"""Algorithms for bilevel planning.

Mainly, "SeSamE": SEarch-and-SAMple planning, then Execution.
"""

from __future__ import annotations

import heapq as hq
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from typing import Any, Collection, Dict, FrozenSet, Iterator, List, \
    Optional, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.option_model import _OptionModelBase
from predicators.refinement_estimators import BaseRefinementEstimator
from predicators.settings import CFG
from predicators.structs import NSRT, AbstractPolicy, DefaultState, \
    DummyOption, GroundAtom, Metrics, Object, OptionSpec, \
    ParameterizedOption, Predicate, State, STRIPSOperator, Task, Type, \
    _GroundNSRT, _GroundSTRIPSOperator, _Option
from predicators.utils import EnvironmentFailure, _TaskPlanningHeuristic

_NOT_CAUSES_FAILURE = "NotCausesFailure"


@dataclass(repr=False, eq=False)
class _Node:
    """A node for the search over skeletons."""
    atoms: Set[GroundAtom]
    skeleton: List[_GroundNSRT]
    atoms_sequence: List[Set[GroundAtom]]  # expected state sequence
    parent: Optional[_Node]
    cumulative_cost: float


def sesame_plan(
    task: Task,
    option_model: _OptionModelBase,
    nsrts: Set[NSRT],
    predicates: Set[Predicate],
    types: Set[Type],
    timeout: float,
    seed: int,
    task_planning_heuristic: str,
    max_skeletons_optimized: int,
    max_horizon: int,
    abstract_policy: Optional[AbstractPolicy] = None,
    max_policy_guided_rollout: int = 0,
    refinement_estimator: Optional[BaseRefinementEstimator] = None,
    check_dr_reachable: bool = True,
    allow_noops: bool = False,
    use_visited_state_set: bool = False
) -> Tuple[List[_Option], List[_GroundNSRT], Metrics]:
    """Run bilevel planning.

    Return a sequence of options, and a dictionary of metrics for this
    run of the planner. Uses the SeSamE strategy: SEarch-and-SAMple
    planning, then Execution. The high-level planner can be either A* or
    Fast Downward (FD). In the latter case, we allow either optimal mode
    ("fdopt") or satisficing mode ("fdsat"). With Fast Downward, we can
    only consider at most one skeleton, and DiscoveredFailures cannot be
    handled.
    """
    if CFG.sesame_task_planner == "astar":
        return _sesame_plan_with_astar(
            task, option_model, nsrts, predicates, types, timeout, seed,
            task_planning_heuristic, max_skeletons_optimized, max_horizon,
            abstract_policy, max_policy_guided_rollout, refinement_estimator,
            check_dr_reachable, allow_noops, use_visited_state_set)
    if CFG.sesame_task_planner == "fdopt":
        assert abstract_policy is None
        return _sesame_plan_with_fast_downward(task,
                                               option_model,
                                               nsrts,
                                               predicates,
                                               types,
                                               timeout,
                                               seed,
                                               max_horizon,
                                               optimal=True)
    if CFG.sesame_task_planner == "fdsat":
        assert abstract_policy is None
        return _sesame_plan_with_fast_downward(task,
                                               option_model,
                                               nsrts,
                                               predicates,
                                               types,
                                               timeout,
                                               seed,
                                               max_horizon,
                                               optimal=False)
    raise ValueError("Unrecognized sesame_task_planner: "
                     f"{CFG.sesame_task_planner}")


def _sesame_plan_with_astar(
    task: Task,
    option_model: _OptionModelBase,
    nsrts: Set[NSRT],
    predicates: Set[Predicate],
    types: Set[Type],
    timeout: float,
    seed: int,
    task_planning_heuristic: str,
    max_skeletons_optimized: int,
    max_horizon: int,
    abstract_policy: Optional[AbstractPolicy] = None,
    max_policy_guided_rollout: int = 0,
    refinement_estimator: Optional[BaseRefinementEstimator] = None,
    check_dr_reachable: bool = True,
    allow_noops: bool = False,
    use_visited_state_set: bool = False
) -> Tuple[List[_Option], List[_GroundNSRT], Metrics]:
    """The default version of SeSamE, which runs A* to produce skeletons."""
    init_atoms = utils.abstract(task.init, predicates)
    objects = list(task.init)
    start_time = time.perf_counter()
    ground_nsrts = sesame_ground_nsrts(task, init_atoms, nsrts, objects,
                                       predicates, types, start_time, timeout)
    # Keep restarting the A* search while we get new discovered failures.
    metrics: Metrics = defaultdict(float)
    # Make a copy of the predicates set to avoid modifying the input set,
    # since we may be adding NotCausesFailure predicates to the set.
    predicates = predicates.copy()
    # Keep track of partial refinements: skeletons and partial plans. This is
    # for making videos of failed planning attempts.
    partial_refinements = []
    while True:
        # Optionally exclude NSRTs with empty effects, because they can slow
        # the search significantly, so we may want to exclude them. Note however
        # that we need to do this inside the while True here, because an NSRT
        # that initially has empty effects may later have a _NOT_CAUSES_FAILURE.
        reachable_nsrts = filter_nsrts(task, init_atoms, ground_nsrts,
                                       check_dr_reachable, allow_noops)
        heuristic = utils.create_task_planning_heuristic(
            task_planning_heuristic, init_atoms, task.goal, reachable_nsrts,
            predicates, objects)
        try:
            new_seed = seed + int(metrics["num_failures_discovered"])
            gen = _skeleton_generator(
                task, reachable_nsrts, init_atoms, heuristic, new_seed,
                timeout - (time.perf_counter() - start_time), metrics,
                max_skeletons_optimized, abstract_policy,
                max_policy_guided_rollout, use_visited_state_set)
            # If a refinement cost estimator is provided, generate a number of
            # skeletons first, then predict the refinement cost of each skeleton
            # and attempt to refine them in this order.
            if refinement_estimator is not None:
                estimator: BaseRefinementEstimator = refinement_estimator
                proposed_skeletons = []
                for _ in range(
                        CFG.refinement_estimation_num_skeletons_generated):
                    try:
                        proposed_skeletons.append(next(gen))
                    except _MaxSkeletonsFailure:
                        break
                gen = iter(
                    sorted(proposed_skeletons,
                           key=lambda s: estimator.get_cost(task, *s)))
            refinement_start_time = time.perf_counter()
            for skeleton, atoms_sequence in gen:
                logging.debug(f"Found skeleton: {[n.name for n in skeleton]}")
                if CFG.sesame_use_necessary_atoms:
                    atoms_seq = utils.compute_necessary_atoms_seq(
                        skeleton, atoms_sequence, task.goal)
                else:
                    atoms_seq = atoms_sequence
                plan, suc = run_low_level_search(
                    task, option_model, skeleton, atoms_seq, new_seed,
                    timeout - (time.perf_counter() - start_time), metrics,
                    max_horizon)
                if suc:
                    # Success! It's a complete plan.
                    logging.info(
                        f"Planning succeeded! Found plan of length "
                        f"{len(plan)} after "
                        f"{int(metrics['num_skeletons_optimized'])} "
                        f"skeletons with {int(metrics['num_samples'])}"
                        f" samples, discovering "
                        f"{int(metrics['num_failures_discovered'])} failures")
                    metrics["plan_length"] = len(plan)
                    metrics["refinement_time"] = (time.perf_counter() -
                                                  refinement_start_time)
                    return plan, skeleton, metrics
                partial_refinements.append((skeleton, plan))
                if time.perf_counter() - start_time > timeout:
                    logging.debug("Exiting search due to timeout.")
                    logging.debug(f"Partial refinements: {partial_refinements}")
                    raise PlanningTimeout(
                        "Planning timed out in refinement!",
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


def sesame_ground_nsrts(
    task: Task,
    init_atoms: Set[GroundAtom],
    nsrts: Set[NSRT],
    objects: List[Object],
    predicates: Set[Predicate],
    types: Set[Type],
    start_time: float,
    timeout: float,
) -> List[_GroundNSRT]:
    """Helper function for _sesame_plan_with_astar(); generate ground NSRTs."""
    if CFG.sesame_grounder == "naive":
        ground_nsrts = []
        for nsrt in sorted(nsrts):
            for ground_nsrt in utils.all_ground_nsrts(nsrt, objects):
                ground_nsrts.append(ground_nsrt)
                if time.perf_counter() - start_time > timeout:
                    raise PlanningTimeout("Planning timed out in grounding!")
    elif CFG.sesame_grounder == "fd_translator":
        # WARNING: there is no easy way to check the timeout within this call,
        # since Fast Downward's translator is a third-party function. We'll
        # just check the timeout afterward.
        ground_nsrts = list(
            utils.all_ground_nsrts_fd_translator(nsrts, objects, predicates,
                                                 types, init_atoms, task.goal))
        if time.perf_counter() - start_time > timeout:
            raise PlanningTimeout("Planning timed out in grounding!")
    else:
        raise ValueError(
            f"Unrecognized sesame_grounder: {CFG.sesame_grounder}")
    return ground_nsrts


def filter_nsrts(
    task: Task,
    init_atoms: Set[GroundAtom],
    ground_nsrts: List[_GroundNSRT],
    check_dr_reachable: bool = True,
    allow_noops: bool = False,
) -> List[_GroundNSRT]:
    """Helper function for _sesame_plan_with_astar(); optionally filter out
    NSRTs with empty effects and/or those that are unreachable."""
    nonempty_ground_nsrts = [
        nsrt for nsrt in ground_nsrts
        if allow_noops or (nsrt.add_effects | nsrt.delete_effects)
    ]
    all_reachable_atoms = utils.get_reachable_atoms(nonempty_ground_nsrts,
                                                    init_atoms)
    if check_dr_reachable and not task.goal.issubset(all_reachable_atoms):
        raise PlanningFailure(f"Goal {task.goal} not dr-reachable")
    reachable_nsrts = [
        nsrt for nsrt in nonempty_ground_nsrts
        if nsrt.preconditions.issubset(all_reachable_atoms)
    ]
    return reachable_nsrts


def task_plan_grounding(
    init_atoms: Set[GroundAtom],
    objects: Set[Object],
    nsrts: Collection[NSRT],
    allow_noops: bool = False,
) -> Tuple[List[_GroundNSRT], Set[GroundAtom]]:
    """Ground all operators for task planning into dummy _GroundNSRTs,
    filtering out ones that are unreachable or have empty effects.

    Also return the set of reachable atoms, which is used by task
    planning to quickly determine if a goal is unreachable.

    See the task_plan docstring for usage instructions.
    """
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
    use_visited_state_set: bool = False,
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
        logging.info(f"Detected goal unreachable. Goal: {goal}")
        logging.info(f"Initial atoms: {init_atoms}")
        raise PlanningFailure(f"Goal {goal} not dr-reachable")
    dummy_task = Task(DefaultState, goal)
    metrics: Metrics = defaultdict(float)
    generator = _skeleton_generator(
        dummy_task,
        ground_nsrts,
        init_atoms,
        heuristic,
        seed,
        timeout,
        metrics,
        max_skeletons_optimized,
        use_visited_state_set=use_visited_state_set)
    # Note that we use this pattern to avoid having to catch an exception
    # when _skeleton_generator runs out of skeletons to optimize.
    for skeleton, atoms_sequence in islice(generator, max_skeletons_optimized):
        yield skeleton, atoms_sequence, metrics.copy()


def _skeleton_generator(
    task: Task,
    ground_nsrts: List[_GroundNSRT],
    init_atoms: Set[GroundAtom],
    heuristic: _TaskPlanningHeuristic,
    seed: int,
    timeout: float,
    metrics: Metrics,
    max_skeletons_optimized: int,
    abstract_policy: Optional[AbstractPolicy] = None,
    sesame_max_policy_guided_rollout: int = 0,
    use_visited_state_set: bool = False
) -> Iterator[Tuple[List[_GroundNSRT], List[Set[GroundAtom]]]]:
    """A* search over skeletons (sequences of ground NSRTs).
    Iterates over pairs of (skeleton, atoms sequence).

    Note that we can't use utils.run_astar() here because we want to
    yield multiple skeletons, whereas that utility method returns only
    a single solution. Furthermore, it's easier to track and update our
    metrics dictionary if we re-implement the search here. If
    use_visited_state_set is False (which is the default), then we may revisit
    the same abstract states multiple times, unlike in typical A*. See
    Issue #1117 for a discussion on why this is False by default.
    """

    start_time = time.perf_counter()
    current_objects = set(task.init)
    queue: List[Tuple[float, float, _Node]] = []
    root_node = _Node(atoms=init_atoms,
                      skeleton=[],
                      atoms_sequence=[init_atoms],
                      parent=None,
                      cumulative_cost=0)
    metrics["num_nodes_created"] += 1
    rng_prio = np.random.default_rng(seed)
    hq.heappush(queue,
                (heuristic(root_node.atoms), rng_prio.uniform(), root_node))
    # Initialize with empty skeleton for root.
    # We want to keep track of the visited skeletons so that we avoid
    # repeatedly outputting the same faulty skeletons.
    visited_skeletons: Set[Tuple[_GroundNSRT, ...]] = set()
    visited_skeletons.add(tuple(root_node.skeleton))
    if use_visited_state_set:
        # This set will maintain (frozen) atom sets that have been fully
        # expanded already, and ensure that we never expand redundantly.
        visited_atom_sets = set()
    # Start search.
    while queue and (time.perf_counter() - start_time < timeout):
        if int(metrics["num_skeletons_optimized"]) == max_skeletons_optimized:
            raise _MaxSkeletonsFailure(
                "Planning reached max_skeletons_optimized!")
        _, _, node = hq.heappop(queue)
        if use_visited_state_set:
            frozen_atoms = frozenset(node.atoms)
            visited_atom_sets.add(frozen_atoms)
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
            # If an abstract policy is provided, generate policy-based
            # successors first.
            if abstract_policy is not None:
                current_node = node
                for _ in range(sesame_max_policy_guided_rollout):
                    if task.goal.issubset(current_node.atoms):
                        yield current_node.skeleton, current_node.atoms_sequence
                        break
                    ground_nsrt = abstract_policy(current_node.atoms,
                                                  current_objects, task.goal)
                    if ground_nsrt is None:
                        break
                    # Make sure ground_nsrt is applicable.
                    if not ground_nsrt.preconditions.issubset(
                            current_node.atoms):
                        break
                    child_atoms = utils.apply_operator(ground_nsrt,
                                                       set(current_node.atoms))
                    child_skeleton = current_node.skeleton + [ground_nsrt]
                    child_skeleton_tup = tuple(child_skeleton)
                    if child_skeleton_tup in visited_skeletons:
                        continue
                    visited_skeletons.add(child_skeleton_tup)
                    # Note: the cost of taking a policy-generated action is 1,
                    # but the policy-generated skeleton is immediately yielded
                    # once it reaches a goal. This allows the planner to always
                    # trust the policy first, but it also allows us to yield a
                    # policy-generated plan without waiting to exhaustively
                    # rule out the possibility that some other primitive plans
                    # are actually lower cost.
                    child_cost = 1 + current_node.cumulative_cost
                    child_node = _Node(
                        atoms=child_atoms,
                        skeleton=child_skeleton,
                        atoms_sequence=current_node.atoms_sequence +
                        [child_atoms],
                        parent=current_node,
                        cumulative_cost=child_cost)
                    metrics["num_nodes_created"] += 1
                    # priority is g [cost] plus h [heuristic]
                    priority = (child_node.cumulative_cost +
                                heuristic(child_node.atoms))
                    hq.heappush(queue,
                                (priority, rng_prio.uniform(), child_node))
                    current_node = child_node
                    if time.perf_counter() - start_time >= timeout:
                        break
            # Generate primitive successors.
            for nsrt in utils.get_applicable_operators(ground_nsrts,
                                                       node.atoms):
                child_atoms = utils.apply_operator(nsrt, set(node.atoms))
                if use_visited_state_set:
                    frozen_atoms = frozenset(child_atoms)
                    if frozen_atoms in visited_atom_sets:
                        continue
                child_skeleton = node.skeleton + [nsrt]
                child_skeleton_tup = tuple(child_skeleton)
                if child_skeleton_tup in visited_skeletons:  # pragma: no cover
                    continue
                visited_skeletons.add(child_skeleton_tup)
                # Action costs are unitary.
                child_cost = node.cumulative_cost + 1.0
                child_node = _Node(atoms=child_atoms,
                                   skeleton=child_skeleton,
                                   atoms_sequence=node.atoms_sequence +
                                   [child_atoms],
                                   parent=node,
                                   cumulative_cost=child_cost)
                metrics["num_nodes_created"] += 1
                # priority is g [cost] plus h [heuristic]
                priority = (child_node.cumulative_cost +
                            heuristic(child_node.atoms))
                hq.heappush(queue, (priority, rng_prio.uniform(), child_node))
                if time.perf_counter() - start_time >= timeout:
                    break
    if not queue:
        raise _MaxSkeletonsFailure("Planning ran out of skeletons!")
    assert time.perf_counter() - start_time >= timeout
    raise _SkeletonSearchTimeout


def run_low_level_search(
    task: Task,
    option_model: _OptionModelBase,
    skeleton: List[_GroundNSRT],
    atoms_sequence: List[Set[GroundAtom]],
    seed: int,
    timeout: float,
    metrics: Metrics,
    max_horizon: int,
    refinement_time: Optional[List[float]] = None
) -> Tuple[List[_Option], bool]:
    """Backtracking search over continuous values.

    Returns a sequence of options and a boolean. If the boolean is True,
    the option sequence is a complete low-level plan refining the given
    skeleton. Otherwise, the option sequence is the longest partial
    failed refinement, where the last step did not satisfy the skeleton,
    but all previous steps did. Note that there are multiple low-level
    plans in general; we return the first one found (arbitrarily).
    """
    start_time = time.perf_counter()
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
    # If refinement_time list is passed, record the refinement time
    # distributed across each step of the skeleton
    if refinement_time is not None:
        assert len(refinement_time) == 0
        for _ in skeleton:
            refinement_time.append(0)
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
    plan_found = False
    while cur_idx < len(skeleton):
        if time.perf_counter() - start_time > timeout:
            logging.debug("Exiting low-level search due to timeout.")
            return longest_failed_refinement, False
        assert num_tries[cur_idx] < max_tries[cur_idx]
        try_start_time = time.perf_counter()
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
        # Increment num_samples metric by 1
        metrics["num_samples"] += 1
        # Increment cur_idx. It will be decremented later on if we get stuck.
        cur_idx += 1
        if option.initiable(state):
            try:
                logging.debug(f"Running option {option}")
                next_state, num_actions = \
                    option_model.get_next_state_and_num_actions(state, option)
            except EnvironmentFailure as e:
                logging.debug(f"Discovered a failure: {e}")
                can_continue_on = False
                # Remember only the most recent failure.
                discovered_failures[cur_idx - 1] = _DiscoveredFailure(e, nsrt)
            else:  # an EnvironmentFailure was not raised
                discovered_failures[cur_idx - 1] = None
                num_actions_per_option[cur_idx - 1] = num_actions
                traj[cur_idx] = next_state
                # Check if objects that were outside the scope had a change
                # in state.
                static_obj_changed = False
                if CFG.sesame_check_static_object_changes:
                    static_objs = set(state) - set(nsrt.objects)
                    for obj in sorted(static_objs):
                        if not np.allclose(
                                traj[cur_idx][obj],
                                traj[cur_idx - 1][obj],
                                atol=CFG.sesame_static_object_change_tol):
                            static_obj_changed = True
                            break
                if static_obj_changed:
                    can_continue_on = False
                # Check if we have exceeded the horizon in total.
                elif np.sum(num_actions_per_option[:cur_idx]) > max_horizon:
                    logging.debug("Cannot continue: exceeded total horizon.")
                    can_continue_on = False
                # Check if we have exceeded the horizon individually.
                elif num_actions >= CFG.max_num_steps_option_rollout:
                    logging.debug("Cannot continue: exceeded individual "
                                  "horizon.")
                    can_continue_on = False
                # Check if the option was effectively a noop.
                elif num_actions == 0:
                    logging.debug("Cannot continue: an noop")
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
                            plan_found = True
                    else:
                        can_continue_on = False
                else:
                    # If we're not checking expected_atoms, we need to
                    # explicitly check the goal on the final timestep.
                    can_continue_on = True
                    if cur_idx == len(skeleton):
                        if task.goal_holds(traj[cur_idx]):
                            plan_found = True
                        else:
                            can_continue_on = False
        else:
            # The option is not initiable.
            can_continue_on = False
        if refinement_time is not None:
            try_end_time = time.perf_counter()
            refinement_time[cur_idx - 1] += try_end_time - try_start_time
        if plan_found:
            return plan, True  # success!
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


def _update_sas_file_with_failure(discovered_failure: _DiscoveredFailure,
                                  sas_file: str) -> None:  # pragma: no cover
    """Update the given sas_file of ground NSRTs for FD based on the given
    DiscoveredFailure.

    We directly update the sas_file with the new ground NSRTs.
    """
    # Get string representation of the ground NSRTs with the Discovered Failure.
    ground_op_str = discovered_failure.failing_nsrt.name.lower(
    ) + " " + " ".join(o.name for o in discovered_failure.failing_nsrt.objects)
    # Add Discovered Failure for each offending object.
    for obj in discovered_failure.env_failure.info["offending_objects"]:
        with open(sas_file, 'r', encoding="utf-8") as f:
            sas_lines = f.readlines()
        # For every line in our sas_file we are going to copy it to our
        # new_sas_file_lines and make edits as needed.
        new_sas_file_lines = []
        sas_file_i = 0

        # We use the Fast Downward documentation to parse the sas_file
        # For more info: https://www.fast-downward.org/TranslatorOutputFormat
        # First we fix sas_file Variables:
        # The first line is "begin_variable".
        # The second line contains the name of the variable (which is
        # usually a nondescriptive name like "var7").
        # The third line specifies the axiom layer of the variable.
        # Single variables are always -1.
        # The fourth line specifies the variable's range, i.e., the
        # number of different values it can take it on. The value of
        # a variable is always from the set {0, 1, 2, ..., range - 1}.
        # The following range lines specify the symbolic names for
        # each of the range values the variable can take on, one at a
        # time. These typically correspond to grounded PDDL facts,
        # except for values that represent that none out a set of PDDL
        # facts is true.
        # The final line is "end_variable".
        count_variables = 0
        for i, line in enumerate(sas_lines):
            # We copy lines until we've reached end_metric. Then we
            # increment the number of variables by 1 and add our new
            # not-causes-failure variable in the new_sas_file_lines.
            if i > 0 and "end_metric" in sas_lines[i - 1]:
                line = line.strip()
                assert line.isdigit()
                num_variables = int(line)
                # Change num variables
                new_sas_file_lines.append(f"{num_variables + 1}\n")
            elif "end_variable" in line:
                count_variables += 1
                new_sas_file_lines.append(line)
                if count_variables == num_variables:
                    # Add new variables here
                    new_sas_file_lines.append("begin_variable\n")
                    new_sas_file_lines.append(f"var{count_variables}\n")
                    new_sas_file_lines.append("-1\n")
                    new_sas_file_lines.append("2\n")
                    new_sas_file_lines.append(
                        f"Atom not-causes-failure({obj.name.lower()})\n")
                    new_sas_file_lines.append(
                        f"NegatedAtom not-causes-failure({obj.name.lower()})\n"
                    )
                    new_sas_file_lines.append("end_variable\n")
                    sas_file_i = i + 1
                    break
            else:
                new_sas_file_lines.append(line)

        # Add sas_file init_state, goal, and mutex.
        num_operators = None
        for i, line in enumerate(sas_lines[sas_file_i:]):
            if i > 0 and "end_goal" in sas_lines[sas_file_i + i - 1]:
                # Save num_operators for use later.
                line = line.strip()
                assert line.isdigit()
                num_operators = int(line)
                sas_file_i = sas_file_i + i + 1
                new_sas_file_lines.append(f"{num_operators}\n")
                break
            if "end_state" in line:
                new_sas_file_lines.append("1\n")
                new_sas_file_lines.append(line)
            else:
                new_sas_file_lines.append(line)
        assert num_operators is not None

        # We use the Fast Downward documentation to parse the sas_file
        # For more info: https://www.fast-downward.org/TranslatorOutputFormat
        # Second we fix sas_file Operators:
        # The first line is "begin_operator".
        # The second line contains the name of the operator.
        # The third line contains a single number, denoting the number of
        # precondition conditions.
        # The following lines describe the precondition conditions, one line
        # for each condition. A precondition condition is given by two numbers
        # separated by spaces, denoting a variable/value pairing in the
        # same notation for goals described above.
        # The first line after the precondition conditions contains a single
        # number, denoting the number of effects.
        # The following lines describe the effects, one line for each effect
        # (read on).
        # The line before last gives the operator cost. This line only
        # matters if metric is 1 (otherwise, any number here will be treated
        # as 1).
        # The final line is "end_operator".
        count_operators = 0
        for i, line in enumerate(sas_lines[sas_file_i:]):
            # We copy each operator from the sas_file and add our new
            # not-causes-failure variable to the necessary operators in
            # the new_sas_file_lines.
            if "begin_operator" in line:
                # Parse Operator from sas_lines.
                count_operators += 1
                begin_operator_str = sas_lines[sas_file_i + i]
                operator_str = sas_lines[sas_file_i + i + 1]
                line = sas_lines[sas_file_i + i + 2].strip()
                assert line.isdigit()
                num_precondition_conditons = int(line)
                line = sas_lines[sas_file_i + i + 3 +
                                 num_precondition_conditons].strip()
                assert line.isdigit()
                num_effects = int(line)
                line = sas_lines[sas_file_i + i + 4 +
                                 num_precondition_conditons +
                                 num_effects].strip()
                assert line.isdigit()
                cost = int(line)
                end_operator_str = sas_lines[sas_file_i + i + 5 +
                                             num_precondition_conditons +
                                             num_effects]
                # Begin Operator
                new_sas_file_lines.append(begin_operator_str)
                new_sas_file_lines.append(operator_str)
                # Append preconditions
                if operator_str.replace("\n", "") == ground_op_str:
                    new_sas_file_lines.append(
                        f"{num_precondition_conditons + 1}\n")
                    new_sas_file_lines.append(
                        f"{num_variables} 0\n")  # additional precondition
                else:
                    new_sas_file_lines.append(
                        f"{num_precondition_conditons}\n")
                for j in range(num_precondition_conditons):
                    new_sas_file_lines.append(sas_lines[sas_file_i + i + 3 +
                                                        j])
                # Append effects
                if obj.name.lower() in operator_str:
                    new_sas_file_lines.append(f"{num_effects + 1}\n")
                    new_sas_file_lines.append(
                        f"0 {num_variables} -1 0\n")  # additional effect
                else:
                    new_sas_file_lines.append(f"{num_effects}\n")
                for j in range(num_effects):
                    new_sas_file_lines.append(
                        sas_lines[sas_file_i + i + 4 +
                                  num_precondition_conditons + j])
                # End Operator
                new_sas_file_lines.append(f"{cost}\n")
                new_sas_file_lines.append(end_operator_str)
                if count_operators == num_operators:
                    sas_file_i = sas_file_i + i + 1
                    break
        # Copy the rest of the file.
        for i, line in enumerate(sas_lines[sas_file_i:]):
            new_sas_file_lines.append(line)
        # Overwrite sas_file with new_sas_file_lines.
        with open(sas_file, 'w', encoding="utf-8") as f:
            f.writelines(new_sas_file_lines)


def task_plan_with_option_plan_constraint(
    objects: Set[Object],
    predicates: Set[Predicate],
    strips_ops: List[STRIPSOperator],
    option_specs: List[OptionSpec],
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    option_plan: List[Tuple[ParameterizedOption, Sequence[Object]]],
    atoms_seq: Optional[List[Set[GroundAtom]]] = None,
) -> Optional[List[_GroundNSRT]]:
    """Turn an option plan into a plan of ground NSRTs that achieves the goal
    from the initial atoms.

    If atoms_seq is not None, the ground NSRT plan must also match up with
    the given sequence of atoms. Otherwise, atoms are not checked.

    If no goal-achieving sequence of ground NSRTs corresponds to
    the option plan, return None.
    """
    dummy_nsrts = utils.ops_and_specs_to_dummy_nsrts(strips_ops, option_specs)
    ground_nsrts, _ = task_plan_grounding(init_atoms,
                                          objects,
                                          dummy_nsrts,
                                          allow_noops=True)
    heuristic = utils.create_task_planning_heuristic(
        CFG.sesame_task_planning_heuristic, init_atoms, goal, ground_nsrts,
        predicates, objects)

    def _check_goal(
            searchnode_state: Tuple[FrozenSet[GroundAtom], int]) -> bool:
        return goal.issubset(searchnode_state[0])

    def _get_successor_with_correct_option(
        searchnode_state: Tuple[FrozenSet[GroundAtom], int]
    ) -> Iterator[Tuple[_GroundNSRT, Tuple[FrozenSet[GroundAtom], int],
                        float]]:
        atoms = searchnode_state[0]
        idx_into_traj = searchnode_state[1]

        if idx_into_traj > len(option_plan) - 1:
            return

        gt_param_option = option_plan[idx_into_traj][0]
        gt_objects = option_plan[idx_into_traj][1]
        for applicable_nsrt in utils.get_applicable_operators(
                ground_nsrts, atoms):
            # NOTE: we check that the ParameterizedOptions are equal before
            # attempting to ground because otherwise, we might
            # get a parameter mismatch and trigger an AssertionError
            # during grounding.
            if applicable_nsrt.option != gt_param_option:
                continue
            if applicable_nsrt.option_objs != gt_objects:
                continue
            if atoms_seq is not None and not \
                applicable_nsrt.preconditions.issubset(
                    atoms_seq[idx_into_traj]):
                continue
            next_atoms = utils.apply_operator(applicable_nsrt, set(atoms))
            # The returned cost is uniform because we don't
            # actually care about finding the shortest path;
            # just one that matches!
            yield (applicable_nsrt, (frozenset(next_atoms), idx_into_traj + 1),
                   1.0)

    init_atoms_frozen = frozenset(init_atoms)
    init_searchnode_state = (init_atoms_frozen, 0)
    # NOTE: each state in the below GBFS is a tuple of
    # (current_atoms, idx_into_traj). The idx_into_traj is necessary because
    # we need to check whether the atoms that are true at this particular
    # index into the trajectory is what we would expect given the demo
    # trajectory.
    state_seq, action_seq = utils.run_gbfs(
        init_searchnode_state, _check_goal, _get_successor_with_correct_option,
        lambda searchnode_state: heuristic(searchnode_state[0]))

    if not _check_goal(state_seq[-1]):
        return None

    return action_seq


def generate_sas_file_for_fd(
        task: Task, nsrts: Set[NSRT], predicates: Set[Predicate],
        types: Set[Type], timeout: float, timeout_cmd: str, alias_flag: str,
        exec_str: str, objects: List[Object],
        init_atoms: Set[GroundAtom]) -> str:  # pragma: no cover
    """Generates a SAS file for a particular PDDL planning problem so that FD
    can be used for search."""
    # Create the domain and problem strings, then write them to tempfiles.
    dom_str = utils.create_pddl_domain(nsrts, predicates, types, "mydomain")
    prob_str = utils.create_pddl_problem(objects, init_atoms, task.goal,
                                         "mydomain", "myproblem")
    dom_file = tempfile.NamedTemporaryFile(delete=False).name
    with open(dom_file, "w", encoding="utf-8") as f:
        f.write(dom_str)
    prob_file = tempfile.NamedTemporaryFile(delete=False).name
    with open(prob_file, "w", encoding="utf-8") as f:
        f.write(prob_str)
    # The SAS file is used when augmenting the grounded operators,
    # during dicovered failures, and it's important that we give
    # it a name, because otherwise Fast Downward uses a fixed
    # default name, which will cause issues if you run multiple
    # processes simultaneously.
    sas_file = tempfile.NamedTemporaryFile(delete=False).name
    # Run to generate sas
    cmd_str = (f"{timeout_cmd} {timeout} {exec_str} {alias_flag} "
               f"--sas-file {sas_file} {dom_file} {prob_file}")
    fd_translation_cmd_output = subprocess.getoutput(cmd_str)
    if "Driver aborting" in fd_translation_cmd_output:
        logging.debug(fd_translation_cmd_output)
        logging.debug(prob_str)
        raise PlanningFailure("FD failed to translate PDDL "
                              "to sas, there is likely a "
                              "dr-reachability issue! Run "
                              "with '--debug' flag to see the "
                              "output from FD.")
    return sas_file


def _ground_op_to_sas_op(
        ground_op: _GroundSTRIPSOperator) -> str:  # pragma: no cover
    name = ground_op.parent.name.lower()
    objs = [o.name.lower() for o in ground_op.objects]
    objs_str = " ".join(objs)
    return f"{name} {objs_str}".strip()


def _update_sas_file_with_costs(
        sas_file: str, ground_op_costs: Dict[_GroundSTRIPSOperator, float],
        default_ground_op_cost: float,
        cost_precision: int) -> None:  # pragma: no cover
    """Modifies the SAS file in place.

    See https://www.fast-downward.org/TranslatorOutputFormat for info on SAS.
    """
    with open(sas_file, "r", encoding="utf-8") as f:
        sas_str = f.read()
    # Make sure that 'metric' is turned on.
    metric_off_str = "begin_metric\n0\nend_metric"
    metric_on_str = "begin_metric\n1\nend_metric"
    sas_str = sas_str.replace(metric_off_str, metric_on_str)
    # Convert ground op names to SAS format.
    remaining_sas_ground_op_costs = {
        _ground_op_to_sas_op(op): c
        for op, c in ground_op_costs.items()
    }
    # Replace costs for all operators.
    sas_lines = sas_str.split("\n")
    num_lines = len(sas_lines)
    for idx in range(num_lines):
        if sas_lines[idx] == "begin_operator":
            name_idx = idx + 1
            end_idx = next(i for i in range(idx + 1, num_lines)
                           if sas_lines[i] == "end_operator")
            cost_idx = end_idx - 1
            assert sas_lines[end_idx] == "end_operator"
            assert sas_lines[cost_idx] == "1"  # original cost
            sas_op_name = sas_lines[name_idx]
            if sas_op_name in remaining_sas_ground_op_costs:
                cost = remaining_sas_ground_op_costs.pop(sas_op_name)
            else:
                cost = default_ground_op_cost
            int_cost = int((10**cost_precision) * cost)
            sas_lines[cost_idx] = str(int_cost)
    if remaining_sas_ground_op_costs:
        # Operators can get filtered out if they are not needed for the goal.
        unmatched_ops = sorted(remaining_sas_ground_op_costs)
        logging.warning(f"No SAS file matches found for ops: {unmatched_ops}")
    new_sas_str = "\n".join(sas_lines)
    with open(sas_file, "w", encoding="utf-8") as f:
        f.write(new_sas_str)


def fd_plan_from_sas_file(
    sas_file: str, timeout_cmd: str, timeout: float, exec_str: str,
    alias_flag: str, start_time: float, objects: List[Object],
    init_atoms: Set[GroundAtom], nsrts: Set[NSRT], max_horizon: float
) -> Tuple[List[_GroundNSRT], List[Set[GroundAtom]],
           Metrics]:  # pragma: no cover
    """Given a SAS file, runs search on it to generate a plan."""
    cmd_str = (f"{timeout_cmd} {timeout} {exec_str} {alias_flag} {sas_file}")
    output = subprocess.getoutput(cmd_str)
    cleanup_cmd_str = f"{exec_str} --cleanup"
    subprocess.getoutput(cleanup_cmd_str)
    if time.perf_counter() - start_time > timeout:
        raise PlanningTimeout("Planning timed out in call to FD!")
    # Parse and log metrics.
    metrics: Metrics = defaultdict(float)
    num_nodes_expanded = re.findall(r"Expanded (\d+) state", output)
    num_nodes_created = re.findall(r"Evaluated (\d+) state", output)
    if len(num_nodes_expanded) != 1:
        raise PlanningFailure(f"Plan not found with FD! Error: {output}")
    assert len(num_nodes_created) == 1
    metrics["num_nodes_expanded"] = float(num_nodes_expanded[0])
    metrics["num_nodes_created"] = float(num_nodes_created[0])
    # Extract the skeleton from the output and compute the atoms_sequence.
    if "Solution found!" not in output:
        raise PlanningFailure(f"Plan not found with FD! Error: {output}")
    if "Plan length: 0 step" in output:
        # Handle the special case where the plan is found to be trivial.
        skeleton_str = []
    else:
        skeleton_str = re.findall(r"(.+) \(\d+?\)", output)
        if not skeleton_str:
            raise PlanningFailure(f"Plan not found with FD! Error: {output}")
    skeleton: List[_GroundNSRT] = []
    atoms_sequence = [init_atoms]
    nsrt_name_to_nsrt = {nsrt.name.lower(): nsrt for nsrt in nsrts}
    obj_name_to_obj = {obj.name.lower(): obj for obj in objects}
    for nsrt_str in skeleton_str:
        str_split = nsrt_str.split()
        nsrt = nsrt_name_to_nsrt[str_split[0]]
        objs = [obj_name_to_obj[obj_name] for obj_name in str_split[1:]]
        ground_nsrt = nsrt.ground(objs)
        skeleton.append(ground_nsrt)
        atoms_sequence.append(
            utils.apply_operator(ground_nsrt, atoms_sequence[-1]))
    if len(skeleton) > max_horizon:
        raise PlanningFailure("Skeleton produced by FD exceeds horizon!")
    metrics["num_skeletons_optimized"] = 1
    metrics["num_failures_discovered"] = 0
    metrics["plan_length"] = len(skeleton_str)
    return (skeleton, atoms_sequence, metrics)


def _sesame_plan_with_fast_downward(
    task: Task, option_model: _OptionModelBase, nsrts: Set[NSRT],
    predicates: Set[Predicate], types: Set[Type], timeout: float, seed: int,
    max_horizon: int, optimal: bool
) -> Tuple[List[_Option], List[_GroundNSRT], Metrics]:  # pragma: no cover
    """A version of SeSamE that runs the Fast Downward planner to produce a
    single skeleton, then calls run_low_level_search() to turn it into a plan.

    Usage: Build and compile the Fast Downward planner, then set the environment
    variable FD_EXEC_PATH to point to the `downward` directory. For example:
    1) git clone https://github.com/aibasel/downward.git
    2) cd downward && ./build.py
    3) export FD_EXEC_PATH="<your path here>/downward"

    On MacOS, to use gtimeout:
    4) brew install coreutils

    Important Note: Fast Downward will potentially not work with null operators
    (i.e. operators that have an empty effect set). This happens when
    Fast Downward grounds the operators, null operators get pruned because they
    cannot help satisfy the goal. In A* search Discovered Failures could
    potentially add effects to null operators, but this ability is not
    implemented here.
    """
    init_atoms = utils.abstract(task.init, predicates)
    objects = list(task.init)
    timeout_cmd = "gtimeout" if sys.platform == "darwin" else "timeout"
    if optimal:
        alias_flag = "--alias seq-opt-lmcut"
    else:  # satisficing
        alias_flag = "--alias lama-first"
    # Run Fast Downward followed by cleanup. Capture the output.
    assert "FD_EXEC_PATH" in os.environ, \
        "Please follow the instructions in the docstring of this method!"
    fd_exec_path = os.environ["FD_EXEC_PATH"]
    exec_str = os.path.join(fd_exec_path, "fast-downward.py")
    start_time = time.perf_counter()
    sas_file = generate_sas_file_for_fd(task, nsrts, predicates, types,
                                        timeout, timeout_cmd, alias_flag,
                                        exec_str, objects, init_atoms)

    while True:
        skeleton, atoms_sequence, metrics = fd_plan_from_sas_file(
            sas_file, timeout_cmd, timeout, exec_str, alias_flag, start_time,
            objects, init_atoms, nsrts, float(max_horizon))
        # Run low-level search on this skeleton.
        low_level_timeout = timeout - (time.perf_counter() - start_time)
        try:
            necessary_atoms_seq = utils.compute_necessary_atoms_seq(
                skeleton, atoms_sequence, task.goal)
            refinement_start_time = time.perf_counter()
            plan, suc = run_low_level_search(task, option_model, skeleton,
                                             necessary_atoms_seq, seed,
                                             low_level_timeout, metrics,
                                             max_horizon)
            if not suc:
                if time.perf_counter() - start_time > timeout:
                    raise PlanningTimeout("Planning timed out in refinement!")
                raise PlanningFailure("Skeleton produced by FD not refinable!")
            metrics["plan_length"] = len(plan)
            metrics["refinement_time"] = (time.perf_counter() -
                                          refinement_start_time)
            return plan, skeleton, metrics
        except _DiscoveredFailureException as e:
            metrics["num_failures_discovered"] += 1
            _update_sas_file_with_failure(e.discovered_failure, sas_file)
        except (_MaxSkeletonsFailure, _SkeletonSearchTimeout) as e:
            raise e


def run_task_plan_once(
        task: Task,
        nsrts: Set[NSRT],
        preds: Set[Predicate],
        types: Set[Type],
        timeout: float,
        seed: int,
        task_planning_heuristic: Optional[str] = None,
        ground_op_costs: Optional[Dict[_GroundSTRIPSOperator, float]] = None,
        default_cost: float = 1.0,
        cost_precision: int = 3,
        max_horizon: float = np.inf,
        **kwargs: Any
) -> Tuple[List[_GroundNSRT], List[Set[GroundAtom]], Metrics]:
    """Get a single abstract plan for a task.

    The sequence of ground atom sets returned represent NECESSARY atoms.
    """

    init_atoms = utils.abstract(task.init, preds)
    goal = task.goal
    objects = set(task.init)

    start_time = time.perf_counter()

    if CFG.sesame_task_planner == "astar":
        ground_nsrts, reachable_atoms = task_plan_grounding(
            init_atoms, objects, nsrts)
        assert task_planning_heuristic is not None
        heuristic = utils.create_task_planning_heuristic(
            task_planning_heuristic, init_atoms, goal, ground_nsrts, preds,
            objects)
        duration = time.perf_counter() - start_time
        timeout -= duration
        plan, atoms_seq, metrics = next(
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
        if len(plan) > max_horizon:
            raise PlanningFailure(
                "Skeleton produced by A-star exceeds horizon!")
    elif "fd" in CFG.sesame_task_planner:  # pragma: no cover
        fd_exec_path = os.environ["FD_EXEC_PATH"]
        exec_str = os.path.join(fd_exec_path, "fast-downward.py")
        timeout_cmd = "gtimeout" if sys.platform == "darwin" \
            else "timeout"
        # Run Fast Downward followed by cleanup. Capture the output.
        assert "FD_EXEC_PATH" in os.environ, \
            "Please follow instructions in the docstring of the" +\
            "_sesame_plan_with_fast_downward method in planning.py"

        sesame_task_planner = CFG.sesame_task_planner
        if sesame_task_planner.endswith("-costs"):
            use_costs = True
            sesame_task_planner = sesame_task_planner[:-len("-costs")]
        else:
            use_costs = False

        if sesame_task_planner == "fdopt":
            alias_flag = "--alias seq-opt-lmcut"
        elif sesame_task_planner == "fdsat":
            alias_flag = "--alias lama-first"
        else:
            raise ValueError("Unrecognized sesame_task_planner: "
                             f"{CFG.sesame_task_planner}")

        sas_file = generate_sas_file_for_fd(task, nsrts, preds, types, timeout,
                                            timeout_cmd, alias_flag, exec_str,
                                            list(objects), init_atoms)

        if use_costs:
            assert ground_op_costs is not None
            assert all(c >= 0 for c in ground_op_costs.values())
            _update_sas_file_with_costs(sas_file,
                                        ground_op_costs,
                                        default_ground_op_cost=default_cost,
                                        cost_precision=cost_precision)

        plan, atoms_seq, metrics = fd_plan_from_sas_file(
            sas_file, timeout_cmd, timeout, exec_str, alias_flag, start_time,
            list(objects), init_atoms, nsrts, float(max_horizon))
    else:
        raise ValueError("Unrecognized sesame_task_planner: "
                         f"{CFG.sesame_task_planner}")

    necessary_atoms_seq = utils.compute_necessary_atoms_seq(
        plan, atoms_seq, goal)

    return plan, necessary_atoms_seq, metrics


class PlanningFailure(utils.ExceptionWithInfo):
    """Raised when the planner fails."""


class PlanningTimeout(utils.ExceptionWithInfo):
    """Raised when the planner times out."""


@dataclass(frozen=True, eq=False)
class _DiscoveredFailure:
    """Container class for holding information related to a low-level discovery
    of a failure which must be propagated up to the main search function, in
    order to restart A* search with new NSRTs."""
    env_failure: EnvironmentFailure
    failing_nsrt: _GroundNSRT


class _DiscoveredFailureException(PlanningFailure):
    """Exception class for DiscoveredFailure propagation."""

    def __init__(self,
                 message: str,
                 discovered_failure: _DiscoveredFailure,
                 info: Optional[Dict] = None):
        super().__init__(message, info)
        self.discovered_failure = discovered_failure


class _MaxSkeletonsFailure(PlanningFailure):
    """Raised when the maximum number of skeletons has been reached."""


class _SkeletonSearchTimeout(PlanningTimeout):
    """Raised when timeout occurs in run_low_level_search()."""

    def __init__(self) -> None:
        super().__init__("Planning timed out in skeleton search!")
