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
from typing import Dict, FrozenSet, Iterator, List, Optional, Sequence, Set, \
    Tuple

import numpy as np

from predicators import utils
from predicators.option_model import _BehaviorOptionModel, _OptionModelBase
from predicators.settings import CFG
from predicators.structs import NSRT, AbstractPolicy, Action, DefaultState, \
    DummyOption, GroundAtom, LowLevelTrajectory, Metrics, Object, OptionSpec, \
    ParameterizedOption, Predicate, State, STRIPSOperator, Task, Type, \
    _GroundNSRT, _Option
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
    check_dr_reachable: bool = True,
    allow_noops: bool = False,
    use_visited_state_set: bool = False
) -> Tuple[List[_Option], Metrics, List[State]]:
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
            abstract_policy, max_policy_guided_rollout, check_dr_reachable,
            allow_noops, use_visited_state_set)
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
    check_dr_reachable: bool = True,
    allow_noops: bool = False,
    use_visited_state_set: bool = False
) -> Tuple[List[_Option], Metrics, List[State]]:
    """The default version of SeSamE, which runs A* to produce skeletons."""
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
            gen = _skeleton_generator(
                task, reachable_nsrts, init_atoms, heuristic, new_seed,
                timeout - (time.time() - start_time), metrics,
                max_skeletons_optimized, abstract_policy,
                max_policy_guided_rollout, use_visited_state_set)
            for skeleton, atoms_sequence in gen:
                plan, suc, traj = run_low_level_search(
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
                    return plan, metrics, traj
                partial_refinements.append((skeleton, plan))
                if time.time() - start_time > timeout:
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

    start_time = time.time()
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
    while queue and (time.time() - start_time < timeout):
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
                    # Note: the cost of taking a policy-generated action is 0.
                    # This encourages the planner to trust the policy, and
                    # also allows us to yield a policy-generated plan without
                    # waiting to exhaustively rule out the possibility that
                    # some other primitive plans are actually lower cost.
                    child_cost = current_node.cumulative_cost
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
                    if time.time() - start_time >= timeout:
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
                if child_skeleton_tup in visited_skeletons:
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
                if time.time() - start_time >= timeout:
                    break
    if not queue:
        raise _MaxSkeletonsFailure("Planning ran out of skeletons!")
    assert time.time() - start_time >= timeout
    raise _SkeletonSearchTimeout


def run_low_level_search(
        task: Task, option_model: _OptionModelBase,
        skeleton: List[_GroundNSRT], atoms_sequence: List[Set[GroundAtom]],
        seed: int, timeout: float,
        max_horizon: int) -> Tuple[List[_Option], bool, List[State]]:
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
            return longest_failed_refinement, False, traj
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
                # Check if the option was effectively a noop.
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
                        logging.info("Success: Expected Atoms Check Passed!")
                        if cur_idx == len(skeleton):
                            return plan, True, traj  # success!
                    else:
                        logging.info("Failure: Expected Atoms Check Failed.")
                        for a in expected_atoms:
                            if not a.holds(traj[cur_idx]):
                                logging.info(a)
                        can_continue_on = False
                else:
                    # If we're not checking expected_atoms, we need to
                    # explicitly check the goal on the final timestep.
                    can_continue_on = True
                    logging.info("Success: Goal Atoms Check Passed!")
                    if cur_idx == len(skeleton):
                        if task.goal_holds(traj[cur_idx]):
                            return plan, True, traj  # success!
                        can_continue_on = False
                        logging.info("Failure: Goal Atoms Check Failed.")
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
                    return longest_failed_refinement, False, traj
    # Should only get here if the skeleton was empty.
    assert not skeleton
    return [], True, traj


def _run_plan_with_option_model(
        task: Task, task_idx: int, option_model: _OptionModelBase,
        plan: List[_Option],
        last_traj: List[State]) -> Tuple[LowLevelTrajectory, bool]:
    """Runs a plan on an option model to generate a low-level trajectory.

    Returns a LowLevelTrajectory and a boolean. If the boolean is True,
    the option sequence successfully executed to achieve the goal and
    generated a LowLevelTrajectory. Otherwise, it returns an empty list
    and False. Since option models return only states, we will add dummy
    actions to the states to create our low level trajectories.
    """
    traj: List[State] = [task.init] + [DefaultState for _ in plan]
    actions: List[Action] = [Action(np.array([0.0])) for _ in plan]
    for idx in range(len(plan)):
        state = traj[idx]
        option = plan[idx]
        if not option.initiable(state):
            # The option is not initiable.
            return LowLevelTrajectory(_states=[task.init],
                                      _actions=[],
                                      _is_demo=False,
                                      _train_task_idx=task_idx), False
        if CFG.plan_only_eval:  # pragma: no cover
            assert isinstance(option_model, _BehaviorOptionModel)
            next_state = option_model.load_state(last_traj[idx + 1])
        else:
            next_state, _ = option_model.get_next_state_and_num_actions(
                state, option)
        traj[idx + 1] = next_state
        # Need to make a new option without policy, initiable, and
        # terminal in order to make it a picklable trajectory for
        # BEHAVIOR environment trajectories.
        action_option = ParameterizedOption(
            option.name, option.parent.types, option.parent.params_space,
            lambda s, m, o, p: Action(np.array([0.0])),
            lambda s, m, o, p: True,
            lambda s, m, o, p: True).ground(option.objects, option.params)
        action_option.memory = option.memory
        actions[idx].set_option(action_option)
    # Since we're not checking the expected_atoms, we need to
    # explicitly check if the goal is achieved.
    if task.goal_holds(traj[-1]):
        return LowLevelTrajectory(_states=traj,
                                  _actions=actions,
                                  _is_demo=True,
                                  _train_task_idx=task_idx), True  # success!
    return LowLevelTrajectory(_states=[task.init],
                              _actions=[],
                              _is_demo=False,
                              _train_task_idx=task_idx), False


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
    ground_nsrts, _ = task_plan_grounding(init_atoms,
                                          objects,
                                          strips_ops,
                                          option_specs,
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
        if atoms_seq is not None:
            expected_next_atoms = atoms_seq[idx_into_traj + 1]

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
            next_atoms = utils.apply_operator(applicable_nsrt, set(atoms))
            if atoms_seq is not None and \
                not next_atoms.issubset(expected_next_atoms):
                continue
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


def _sesame_plan_with_fast_downward(
    task: Task, option_model: _OptionModelBase, nsrts: Set[NSRT],
    predicates: Set[Predicate], types: Set[Type], timeout: float, seed: int,
    max_horizon: int, optimal: bool
) -> Tuple[List[_Option], Metrics, List[State]]:  # pragma: no cover
    """A version of SeSamE that runs the Fast Downward planner to produce a
    single skeleton, then calls run_low_level_search() to turn it into a plan.

    Usage: Build and compile the Fast Downward planner, then set the environment
    variable FD_EXEC_PATH to point to the `downward` directory. For example:
    1) git clone https://github.com/ronuchit/downward.git
    2) cd downward && ./build.py
    3) export FD_EXEC_PATH="<your path here>/downward"
    """
    init_atoms = utils.abstract(task.init, predicates)
    objects = list(task.init)
    start_time = time.time()
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
    # The SAS file isn't actually used, but it's important that we give it a
    # name, because otherwise Fast Downward uses a fixed default name, which
    # will cause issues if you run multiple processes simultaneously.
    sas_file = tempfile.NamedTemporaryFile(delete=False).name
    # Run Fast Downward followed by cleanup. Capture the output.
    timeout_cmd = "gtimeout" if sys.platform == "darwin" else "timeout"
    if optimal:
        alias_flag = "--alias seq-opt-lmcut"
    else:  # satisficing
        alias_flag = "--alias lama-first"
    assert "FD_EXEC_PATH" in os.environ, \
        "Please follow the instructions in the docstring of this method!"
    fd_exec_path = os.environ["FD_EXEC_PATH"]
    exec_str = os.path.join(fd_exec_path, "fast-downward.py")
    cmd_str = (f"{timeout_cmd} {timeout} {exec_str} {alias_flag} "
               f"--sas-file {sas_file} {dom_file} {prob_file}")
    output = subprocess.getoutput(cmd_str)
    cleanup_cmd_str = f"{exec_str} --cleanup"
    subprocess.getoutput(cleanup_cmd_str)
    if time.time() - start_time > timeout:
        raise PlanningTimeout("Planning timed out in call to FD!")
    # Parse and log metrics.
    metrics: Metrics = defaultdict(float)
    num_nodes_expanded = re.findall(r"Evaluated (\d+) state", output)
    num_nodes_created = re.findall(r"Generated (\d+) state", output)
    assert len(num_nodes_expanded) == 1
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
    skeleton = []
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
    # Run low-level search on this skeleton.
    low_level_timeout = timeout - (time.time() - start_time)
    metrics["num_skeletons_optimized"] = 1
    metrics["num_failures_discovered"] = 0
    try:
        plan, suc, traj = run_low_level_search(task, option_model, skeleton,
                                               atoms_sequence, seed,
                                               low_level_timeout, max_horizon)
    except _DiscoveredFailureException:
        # If we get a DiscoveredFailure, give up. Note that we cannot
        # modify the NSRTs as we do in SeSamE with A*, because we don't ever
        # compute all the ground NSRTs ourselves when using Fast Downward.
        raise PlanningFailure("Got a DiscoveredFailure when using FD!")
    if not suc:
        if time.time() - start_time > timeout:
            raise PlanningTimeout("Planning timed out in refinement!")
        raise PlanningFailure("Skeleton produced by FD not refinable!")
    metrics["plan_length"] = len(plan)
    return plan, metrics, traj


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
