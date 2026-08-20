"""Planning with processes module."""
# pylint: disable=redefined-outer-name
from __future__ import annotations

import heapq as hq
import logging
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import islice
from pprint import pformat
from typing import Any, Callable, Collection, Dict, Iterator, List, Optional, \
    Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.option_model import _OptionModelBase
from predicators.planning import PlanningFailure, PlanningTimeout, \
    _DiscoveredFailureException, _MaxSkeletonsFailure, \
    _SkeletonSearchTimeout, run_low_level_search
from predicators.settings import CFG
from predicators.structs import AbstractProcessPolicy, CausalProcess, \
    DefaultState, DerivedPredicate, EndogenousProcess, GroundAtom, Metrics, \
    Object, Predicate, Task, Type, _GroundCausalProcess, \
    _GroundEndogenousProcess, _GroundExogenousProcess, _Option
from predicators.utils import _TaskPlanningHeuristic


def _build_exogenous_process_index(
    ground_processes: List[_GroundCausalProcess],
) -> Dict[Predicate, List[_GroundExogenousProcess]]:
    """Build index mapping predicates to exogenous processes that have those
    predicates in their condition_at_start.

    This helps efficiently find which exogenous processes might be
    triggered when new facts become true.
    """
    precondition_to_exogenous_processes: Dict[
        Predicate, List[_GroundExogenousProcess]] = defaultdict(list)
    for p in ground_processes:
        if isinstance(p, _GroundExogenousProcess):
            for atom in p.condition_at_start:
                precondition_to_exogenous_processes[atom.predicate].append(p)
    return precondition_to_exogenous_processes


def get_reachable_atoms_from_processes(
    ground_processes: List[_GroundCausalProcess],
    atoms: Set[GroundAtom],
    derived_predicates: Optional[Set[DerivedPredicate]] = None,
    objects: Optional[Set[Object]] = None,
) -> Set[GroundAtom]:
    """Get all atoms that are reachable from the init atoms using ground
    processes.

    This function builds a relaxed planning graph by applying
    exogenous processes and derived predicates similar to when
    building the relaxed planning graph
    in the ff_heuristic.

    Args:
        ground_processes: List of grounded causal processes
        atoms: Initial set of atoms
        derived_predicates: Set of derived predicates to consider
        objects: Set of objects for derived predicate evaluation

    Returns:
        Set of all reachable atoms
    """
    if derived_predicates is None:
        derived_predicates = set()
    if objects is None:
        objects = set()
    # Pre-compute dependencies for incremental
    # derived predicates
    dep_to_derived_preds: Dict[Predicate,
                               List[DerivedPredicate]] = defaultdict(list)
    if derived_predicates:
        for der_pred in derived_predicates:
            if der_pred.auxiliary_predicates is not None:
                for aux_pred in der_pred.auxiliary_predicates:
                    dep_to_derived_preds[aux_pred].append(der_pred)

    # Initialize with input atoms and any initial derived facts
    reachable_atoms = atoms.copy()
    if derived_predicates:
        reachable_atoms.update(
            utils.abstract_with_derived_predicates(reachable_atoms,
                                                   derived_predicates,
                                                   objects))

    # Build relaxed planning graph until fixed point
    while True:
        fixed_point_reached = True
        previous_atoms = reachable_atoms.copy()

        # Apply all applicable ground processes
        newly_added_primitive_facts = set()
        for process in ground_processes:
            if process.condition_at_start.issubset(reachable_atoms):
                # Add effects that aren't already reachable
                new_effects = process.add_effects - reachable_atoms
                if new_effects:
                    fixed_point_reached = False
                    newly_added_primitive_facts.update(new_effects)
                    reachable_atoms.update(new_effects)

        # Handle derived predicates incrementally
        # if we added new primitive facts
        if newly_added_primitive_facts and derived_predicates:
            newly_derived_facts = _run_incremental_derived_predicate_logic(
                newly_added_primitive_facts,
                previous_atoms,
                objects,
                dep_to_derived_preds,
            )
            if newly_derived_facts:
                fixed_point_reached = False
                reachable_atoms.update(newly_derived_facts)

        if fixed_point_reached:
            break

    return reachable_atoms


def process_task_plan_grounding(
    init_atoms: Set[GroundAtom],
    objects: Set[Object],
    cps: Collection[CausalProcess],
    allow_waits: bool = True,
    compute_reachable_atoms: bool = False,
    derived_predicates: Optional[Set[DerivedPredicate]] = None,
) -> Tuple[List[_GroundCausalProcess], Set[GroundAtom]]:
    """Ground all operators for task planning.

    Filter out unreachable or empty-effect operators. Also return the
    set of reachable atoms.
    """
    if derived_predicates is None:
        derived_predicates = set()
    ground_cps = []
    for cp in sorted(cps):
        for ground_cp in utils.all_ground_nsrts(cp, objects):
            if allow_waits or (ground_cp.add_effects
                               | ground_cp.delete_effects):
                ground_cps.append(ground_cp)
    if compute_reachable_atoms:
        reachable_atoms = get_reachable_atoms_from_processes(
            ground_cps,  # type: ignore[arg-type]
            init_atoms,
            derived_predicates,
            objects)
    else:
        reachable_atoms = set()

    reachable_nsrts = ground_cps
    return reachable_nsrts, reachable_atoms  # type: ignore[return-value]


@dataclass(repr=False, eq=False)
class _ProcessPlanningNode():
    """
    Args:
        state_history: a finegrained, per-step history of the state trajectory
            compared to atoms_sequence which is segmented by action.
        action_history: a finegrained, per-step history of the action trajectory
            compared to skeleton which is segmented by action.
    """
    atoms: Set[GroundAtom]  # per big step state
    skeleton: List[_GroundEndogenousProcess]  # per big step action
    atoms_sequence: List[Set[GroundAtom]]  # expected state sequence
    parent: Optional[_ProcessPlanningNode]
    cumulative_cost: float
    state_history: List[Set[GroundAtom]]  # per small step state
    action_history: List[
        Optional[_GroundEndogenousProcess]]  # per small step action
    scheduled_events: Dict[int, List[Tuple[_GroundCausalProcess, int]]]


class ProcessWorldModel:
    """Simulates process execution for planning."""

    def __init__(
        self,
        ground_processes: List[_GroundCausalProcess],
        state: Set[GroundAtom],
        state_history: Optional[List[Set[GroundAtom]]] = None,
        action_history: Optional[List[
            Optional[_GroundEndogenousProcess]]] = None,
        scheduled_events: Optional[Dict[int, List[Tuple[_GroundCausalProcess,
                                                        int]]]] = None,
        t: int = 0,
        derived_predicates: Optional[Set[DerivedPredicate]] = None,
        objects: Optional[Set[Object]] = None,
        precondition_to_exogenous_processes: Optional[Dict[
            Predicate, List[_GroundExogenousProcess]]] = None,
        dep_to_derived_preds: Optional[Dict[Predicate,
                                            List[DerivedPredicate]]] = None
    ) -> None:

        self.ground_processes = ground_processes
        self.state = state
        if state_history is None:
            state_history = []
        self.state_history = state_history
        if action_history is None:
            action_history = []
        if scheduled_events is None:
            scheduled_events = {}
        if derived_predicates is None:
            derived_predicates = set()
        if objects is None:
            objects = set()
        self.current_action: Optional[_GroundEndogenousProcess] = None
        self.action_history = action_history
        self.scheduled_events: Dict[int, List[Tuple[_GroundCausalProcess,
                                                    int]]] = scheduled_events
        self.t = t
        self.derived_predicates = derived_predicates
        self.objects = objects

        # --- Use provided indexes or build them if not provided ---
        self._precondition_to_exogenous_processes: Dict[
            Predicate, List[_GroundExogenousProcess]]
        if precondition_to_exogenous_processes is not None:
            self._precondition_to_exogenous_processes = (
                precondition_to_exogenous_processes)
        elif CFG.build_exogenous_process_index_for_planning:
            # Fallback: build the index if not provided
            # and CFG allows it
            self._precondition_to_exogenous_processes = (
                _build_exogenous_process_index(self.ground_processes))
        else:
            # Don't build the index
            self._precondition_to_exogenous_processes = defaultdict(list)

        self._dep_to_derived_preds: Dict[Predicate, List[DerivedPredicate]]
        if dep_to_derived_preds is not None:
            self._dep_to_derived_preds = dep_to_derived_preds
        else:
            # Fallback: build the index if not provided
            self._dep_to_derived_preds = defaultdict(list)
            for der_pred in self.derived_predicates:
                assert der_pred.auxiliary_predicates is not None
                for aux_pred in der_pred.auxiliary_predicates:
                    self._dep_to_derived_preds[aux_pred].append(der_pred)

    def small_step(
            self,
            small_step_action: Optional[_GroundEndogenousProcess] = None
    ) -> None:
        """Will keep the current action as a class variable for now, as opposed
        to a part of the state variable as in the demo code."""
        # 1. self.current_action is set to an action when this small_step is
        # first called. And is set back to None when `duration` timesteps
        # sampled from its distribution passes.
        # `small_step_action` is not None in the first call but becomes None in
        # subsequent calls.
        if small_step_action is not None:
            self.current_action = small_step_action.copy()
        self.action_history.append(self.current_action.copy() if self.
                                   current_action is not None else None)

        # 2. Process effects scheduled for this timestep.
        if self.t in self.scheduled_events:
            primitive_facts_before = {
                a
                for a in self.state
                if not isinstance(a.predicate, DerivedPredicate)
            }

            for g_process, start_time in self.scheduled_events[self.t]:
                if (all(
                        g_process.condition_overall.issubset(s)
                        for s in self.state_history[start_time + 1:])
                        and g_process.condition_at_end.issubset(self.state)):
                    for atom in g_process.delete_effects:
                        self.state.discard(atom)
                    for atom in g_process.add_effects:
                        self.state.add(atom)
                    if isinstance(g_process, _GroundEndogenousProcess) and\
                        small_step_action is None:
                        self.current_action = None
            del self.scheduled_events[self.t]

            if len(self.derived_predicates) > 0:
                primitive_facts_after = {
                    a
                    for a in self.state
                    if not isinstance(a.predicate, DerivedPredicate)
                }

                # Only update if the primitive facts have changed.
                if primitive_facts_before != primitive_facts_after:
                    deleted_facts = (primitive_facts_before -
                                     primitive_facts_after)

                    # If any primitive fact was deleted, a full re-computation
                    # is the safest way to ensure correctness.
                    if deleted_facts:
                        # Remove all old derived facts.
                        self.state = {
                            atom
                            for atom in self.state
                            if not isinstance(atom.predicate, DerivedPredicate)
                        }
                        # Re-compute all derived facts from the new state.
                        self.state |= utils.abstract_with_derived_predicates(
                            self.state, self.derived_predicates, self.objects)

                    # Otherwise, only additions occurred; we can be incremental.
                    else:
                        added_facts = (primitive_facts_after -
                                       primitive_facts_before)
                        # existing_facts includes primitive
                        # and derived facts before additions.
                        existing_facts_before_increment = (self.state -
                                                           added_facts)
                        fn = _run_incremental_derived_predicate_logic
                        newly_derived_facts = fn(
                            added_facts, existing_facts_before_increment,
                            self.objects, self._dep_to_derived_preds)
                        self.state.update(newly_derived_facts)

        # 3. Schedule new events whose conditions are met.
        # 3a. Handle the endogenous process (action) passed to this step.
        # This is for starting a new action.
        if (small_step_action is not None
                and isinstance(small_step_action.parent, EndogenousProcess)
                and small_step_action.parent.option.name != 'Wait'
                and small_step_action.condition_at_start.issubset(self.state)):
            delay = small_step_action.delay_distribution.sample()
            delay = max(1, delay)
            scheduled_time = self.t + delay
            if scheduled_time not in self.scheduled_events:
                self.scheduled_events[scheduled_time] = []
            self.scheduled_events[scheduled_time].append(
                (small_step_action, self.t))

        # 3b. Handle exogenous processes.
        if CFG.build_exogenous_process_index_for_planning:
            # Use the index for efficiency.
            # Find newly true primitive facts by comparing current vs. previous.
            previous_facts = self.state_history[-1] if self.state_history \
                                else set()
            newly_added_facts = self.state - previous_facts

            # Gather all candidate processes touched by these new facts.
            candidate_processes_to_check: Set[_GroundExogenousProcess] = set()
            for fact in newly_added_facts:
                candidate_processes_to_check.update(
                    self._precondition_to_exogenous_processes[fact.predicate])

            # Check the full preconditions for only the candidate processes.
            for g_process in candidate_processes_to_check:
                if g_process.condition_at_start.issubset(self.state):
                    delay = g_process.delay_distribution.sample()
                    delay = max(1, delay)
                    scheduled_time = self.t + delay
                    if scheduled_time not in self.scheduled_events:
                        self.scheduled_events[scheduled_time] = []
                    self.scheduled_events[scheduled_time].append(
                        (g_process, self.t))
        else:
            # Fallback: check all exogenous processes (less efficient)
            for g_process in self.ground_processes:
                if isinstance(g_process, _GroundExogenousProcess):
                    first_state_or_prev_state_doesnt_satisfy = (
                        len(self.state_history) == 0
                        or not g_process.condition_at_start.issubset(
                            self.state_history[-1]))
                    if g_process.condition_at_start.issubset(self.state) and\
                        first_state_or_prev_state_doesnt_satisfy:
                        delay = g_process.delay_distribution.sample()
                        delay = max(1, delay)
                        scheduled_time = self.t + delay
                        if scheduled_time not in self.scheduled_events:
                            self.scheduled_events[scheduled_time] = []
                        self.scheduled_events[scheduled_time].append(
                            (g_process, self.t))

        # --- END MODIFIED ---

        self.state_history.append(self.state.copy())

        # if the action has finished and set to None.
        if self.current_action is None:
            return
        self.t += 1

    def big_step(self,
                 action_process: _GroundEndogenousProcess,
                 max_num_steps: int = 50) -> Set[GroundAtom]:
        """current_action is set to an action in the first call to small_step
        and is set to None when 1) the action reaches the end of its duration
        2) some aspects of the state changes; removing this because this can
        cause action to stop before the end of its duration 3) reaches
        max_num_steps."""
        initial_state = self.state.copy()
        num_steps = 0
        action_not_finished = True

        while action_not_finished and num_steps < max_num_steps:
            self.small_step(action_process)
            num_steps += 1

            if action_process is not None:
                action_process = None  # type: ignore[assignment]

            action_not_finished = self.current_action is not None

            # if currently executing Wait and state has changed, then break
            if (self.current_action is not None and isinstance(
                    self.current_action.parent, EndogenousProcess)
                    and self.current_action.parent.option.name == 'Wait'
                    and self.state != initial_state):
                break
        return self.state


def _skeleton_generator_with_processes(
    task: Task,
    ground_processes: List[_GroundCausalProcess],
    init_atoms: Set[GroundAtom],
    heuristic: _TaskPlanningHeuristic,
    seed: int,
    timeout: float,
    metrics: Metrics,
    max_skeletons_optimized: int,
    abstract_policy: Optional[AbstractProcessPolicy] = None,
    sesame_max_policy_guided_rollout: int = 0,
    use_visited_state_set: bool = False,
    log_sucessful_small_steps: bool = False,
    log_heuristic: bool = False,
    time_heuristic: bool = True,
    heuristic_weight: float = 10,
    derived_predicates: Optional[Set[DerivedPredicate]] = None,
    objects: Optional[Set[Object]] = None,
) -> Iterator[Tuple[List[_GroundEndogenousProcess], List[Set[GroundAtom]]]]:

    if derived_predicates is None:
        derived_predicates = set()
    if objects is None:
        objects = set()
    # Filter out all the action from processes
    # zero heuristic
    objects = objects.copy()

    # --- Build indexes once for all ProcessWorldModel instances ---
    # Index for efficient scheduling of exogenous processes
    precondition_to_exogenous_processes: Optional[Dict[
        Predicate, List[_GroundExogenousProcess]]] = None
    if CFG.build_exogenous_process_index_for_planning:
        precondition_to_exogenous_processes = _build_exogenous_process_index(
            ground_processes)

    # Pre-compute dependencies for incremental derived predicates
    dep_to_derived_preds: Dict[Predicate,
                               List[DerivedPredicate]] = defaultdict(list)
    for der_pred in derived_predicates:
        assert der_pred.auxiliary_predicates is not None
        for aux_pred in der_pred.auxiliary_predicates:
            dep_to_derived_preds[aux_pred].append(der_pred)
    # --- End index building ---
    ground_action_processes = [
        p for p in ground_processes if isinstance(p, _GroundEndogenousProcess)
    ]
    start_time = time.perf_counter()
    queue: List[Tuple[float, float, _ProcessPlanningNode]] = []
    root_node = _ProcessPlanningNode(
        atoms=init_atoms,
        skeleton=[],
        atoms_sequence=[init_atoms],
        parent=None,
        cumulative_cost=0,
        state_history=[],
        action_history=[],
        scheduled_events={},
    )
    metrics["num_nodes_created"] += 1
    rng_prio = np.random.default_rng(seed)
    if time_heuristic:
        heuristic_call_count = 0
        total_heuristic_time = 0.0
        heuristic_start_time = time.perf_counter()
        h = heuristic(root_node.atoms) * heuristic_weight
        heuristic_end_time = time.perf_counter()
        heuristic_call_count += 1
        total_heuristic_time += (heuristic_end_time - heuristic_start_time)
    else:
        h = heuristic(root_node.atoms) * heuristic_weight
    if log_heuristic:
        logging.debug(f"Root heuristic: {h}")
    hq.heappush(queue, (h, rng_prio.uniform(), root_node))
    # Initialize with empty skeleton for root.
    # We want to keep track of the visited skeletons so that we avoid
    # repeatedly outputting the same faulty skeletons.
    visited_skeletons: Set[Tuple[_GroundCausalProcess, ...]] = set()
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
            time_taken = time.perf_counter() - start_time
            logging.info(f"\n[Task Planner] Found Plan of length "
                         f"{len(node.skeleton)} in {time_taken:.2f}s:")
            for process in node.skeleton:
                logging.debug(process.name_and_objects_str())
            logging.debug("")

            if log_sucessful_small_steps:
                prev_state: Optional[Set[GroundAtom]] = None
                for i, (state, action) in enumerate(
                        zip(node.state_history, node.action_history)):
                    if i == 0:
                        logging.debug(f"State {i}: {sorted(state)}")
                    else:
                        assert prev_state is not None
                        logging.debug(
                            f"State {i}: "
                            f"Add atoms: {sorted(state - prev_state)} "
                            f"Del atoms: {sorted(prev_state - state)}")
                    action_str = action.name_and_objects_str() \
                                    if action is not None else None
                    logging.info(f"Action {i}: {action_str}\n")
                    prev_state = state
                if prev_state is not None:
                    logging.debug(
                        f"State {len(node.state_history)}: "
                        f"Add atoms: "
                        f"{sorted(node.state_history[-1] - prev_state)} "
                        f"Del atoms: "
                        f"{sorted(prev_state - node.state_history[-1])}")

            # Log heuristic timing stats when a solution is found
            if time_heuristic:
                average_heuristic_time = total_heuristic_time / \
                    heuristic_call_count if heuristic_call_count > 0 else 0.0
                logging.debug(f"Heuristic timing stats - "
                              f"Calls: {heuristic_call_count}, "
                              f"Total: {total_heuristic_time:.4f}s, "
                              f"Avg: {average_heuristic_time:.4f}s")

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
                    ground_process = abstract_policy(current_node.atoms,
                                                     objects, task.goal)
                    if ground_process is None:
                        break
                    if not ground_process.condition_at_start.issubset(
                            current_node.atoms):
                        break

                    # Run the process through the world model
                    # to get the resulting state
                    world_model = ProcessWorldModel(
                        ground_processes=ground_processes.copy(),
                        state=current_node.atoms.copy(),
                        state_history=current_node.state_history.copy(),
                        action_history=current_node.action_history.copy(),
                        scheduled_events=deepcopy(
                            current_node.scheduled_events),
                        t=len(current_node.state_history),
                        derived_predicates=derived_predicates,
                        objects=objects,
                        precondition_to_exogenous_processes=
                        precondition_to_exogenous_processes,
                        dep_to_derived_preds=dep_to_derived_preds)

                    world_model.big_step(ground_process)
                    child_atoms = world_model.state.copy()

                    child_skeleton = current_node.skeleton + [ground_process]
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
                    child_node = _ProcessPlanningNode(
                        atoms=child_atoms,
                        skeleton=child_skeleton,
                        atoms_sequence=current_node.atoms_sequence +
                        [child_atoms],
                        parent=current_node,
                        cumulative_cost=child_cost,
                        state_history=world_model.state_history.copy(),
                        action_history=world_model.action_history.copy(),
                        scheduled_events=deepcopy(
                            world_model.scheduled_events))
                    metrics["num_nodes_created"] += 1
                    # priority is g [cost] plus h [heuristic]
                    if time_heuristic:
                        heuristic_start_time = time.perf_counter()
                        h = heuristic(child_node.atoms) * heuristic_weight
                        heuristic_end_time = time.perf_counter()
                        heuristic_call_count += 1
                        total_heuristic_time += (heuristic_end_time -
                                                 heuristic_start_time)
                    else:
                        h = heuristic(child_node.atoms) * heuristic_weight
                    priority = (child_node.cumulative_cost + h)
                    hq.heappush(queue,
                                (priority, rng_prio.uniform(), child_node))
                    current_node = child_node
                    if time.perf_counter() - start_time >= timeout:
                        break
            applicable_actions: List[Any] = list(
                utils.get_applicable_operators(ground_action_processes,
                                               node.atoms))

            # Domain-specific pruning for domino environment
            if CFG.env == "pybullet_domino_grid" and CFG.domino_prune_actions:
                # Filter out backwards placements and redundant picks
                filtered_actions: List[Any] = []
                placed_dominos = set()  # Track which dominos have been placed

                # First pass: identify already placed dominos
                for prev_action in node.skeleton:
                    if prev_action.parent.name == "PlaceDomino":
                        # The domino being placed is the second argument
                        if len(prev_action.objects) > 1:
                            placed_dominos.add(prev_action.objects[1])

                for action in applicable_actions:
                    assert action is not None
                    # Always keep Wait and Push actions
                    if action.parent.name in ["Wait", "PushStartBlock"]:
                        filtered_actions.append(action)
                    # For Pick, only pick dominos that haven't been placed yet
                    elif action.parent.name == "PickDomino":
                        domino_to_pick = action.objects[1] if len(
                            action.objects) > 1 else None
                        if domino_to_pick and \
                                domino_to_pick not in placed_dominos:
                            filtered_actions.append(action)
                    # For Place, apply heuristics
                    elif action.parent.name == "PlaceDomino":
                        # Keep all place actions for now,
                        # but could add more pruning.
                        # E.g., only place in forward
                        # direction, avoid cycles, etc.
                        filtered_actions.append(action)
                    else:
                        filtered_actions.append(action)

                # If pruning removed all actions, fall back to unpruned
                if filtered_actions:
                    applicable_actions = filtered_actions

            for action_process in applicable_actions:

                # --- Run the action process on the world model
                world_model = ProcessWorldModel(
                    ground_processes=ground_processes.copy(),
                    state=node.atoms.copy(),
                    state_history=node.state_history.copy(),
                    action_history=node.action_history.copy(),
                    scheduled_events=deepcopy(node.scheduled_events),
                    t=len(node.state_history),
                    derived_predicates=derived_predicates,
                    objects=objects,
                    precondition_to_exogenous_processes=
                    precondition_to_exogenous_processes,
                    dep_to_derived_preds=dep_to_derived_preds)

                assert isinstance(action_process, _GroundEndogenousProcess)
                # (debug logging removed)
                # # action_names = [p.name for p in node.skeleton]
                # # target_action_names = ['PickJugFromOutsideFaucetAndBurner',
                # #                        'PlaceUnderFaucet',
                # #                        'SwitchFaucetOn',
                # #                        'SwitchBurnerOn',
                # #                        'SwitchFaucetOff',
                # #                        'PickJugFromFaucet',
                # #                        'PlaceOnBurner',
                # #                        'PickJugFromOutsideFaucetAndBurner',
                # #                        'PlaceUnderFaucet',
                # #                        'SwitchFaucetOn',
                # #                        'SwitchBurnerOn',
                # #                        ]
                # target_action_names = [
                #     # (debug action names removed)
                #     ]
                # # if action_names == target:
                #     # (debug condition removed)
                # if False:  # Update with actual action string when debugging
                # # if action_names == target_action_names:
                #     breakpoint()
                world_model.big_step(action_process)
                child_atoms = world_model.state.copy()
                # --- End

                # Same as standard skeleton generator
                if use_visited_state_set:
                    frozen_atoms = frozenset(child_atoms)
                    if frozen_atoms in visited_atom_sets:
                        continue
                child_skeleton = node.skeleton + [action_process]
                child_skeleton_tup = tuple(child_skeleton)
                if child_skeleton_tup in visited_skeletons:  # pragma: no cover
                    continue
                visited_skeletons.add(child_skeleton_tup)
                # Action costs are unitary.
                if action_process.option.name == 'Wait':
                    action_cost = 0.5
                else:
                    action_cost = 1.0
                child_cost = node.cumulative_cost + action_cost
                child_node = _ProcessPlanningNode(
                    atoms=child_atoms,
                    skeleton=child_skeleton.copy(),
                    atoms_sequence=node.atoms_sequence + [child_atoms],
                    parent=node,
                    cumulative_cost=child_cost,
                    state_history=world_model.state_history.copy(),
                    action_history=world_model.action_history.copy(),
                    scheduled_events=deepcopy(world_model.scheduled_events))
                metrics["num_nodes_created"] += 1
                # priority is g [cost] plus h [heuristic]
                if time_heuristic:
                    heuristic_start_time = time.perf_counter()
                    h = heuristic(child_node.atoms) * heuristic_weight
                    heuristic_end_time = time.perf_counter()
                    heuristic_call_count += 1
                    total_heuristic_time += (heuristic_end_time -
                                             heuristic_start_time)
                else:
                    h = heuristic(child_node.atoms) * heuristic_weight
                priority = (child_node.cumulative_cost + h)
                if log_heuristic:
                    logging.debug(
                        f"Heuristic: {h}, g: {child_node.cumulative_cost}")
                hq.heappush(queue, (priority, rng_prio.uniform(), child_node))
                if time.perf_counter() - start_time >= timeout:
                    logging.debug(f"Planning timeout of {timeout} reached.")
                    break
    if time_heuristic:
        average_heuristic_time = total_heuristic_time / \
            heuristic_call_count if heuristic_call_count > 0 else 0.0
        logging.debug(
            f"Heuristic timing stats - Calls: {heuristic_call_count}, "
            f"Total time: {total_heuristic_time:.4f}s, "
            f"Average time: {average_heuristic_time:.4f}s, "
            f"Num_nodes_created: {metrics['num_nodes_created']}, "
            f"Num_nodes_expanded: {metrics['num_nodes_expanded']}")

    if not queue:
        raise _MaxSkeletonsFailure("Planning ran out of skeletons!")
    assert time.perf_counter() - start_time >= timeout
    raise _SkeletonSearchTimeout


def task_plan_from_task(
    task: Task,
    predicates: Collection[Predicate],
    processes: Set[CausalProcess],
    seed: int,
    timeout: float,
    max_skeletons_optimized: int,
    use_visited_state_set: bool = True,
    abstract_policy: Optional[AbstractProcessPolicy] = None,
    max_policy_guided_rollout: int = 0,
) -> Iterator[Tuple[List[_GroundEndogenousProcess], List[Set[GroundAtom]],
                    Metrics]]:
    """Task plan from task."""
    predicates_set = set(predicates)
    all_predicates = utils.add_in_auxiliary_predicates(predicates_set)
    derived_predicates = utils.get_derived_predicates(all_predicates)

    init_atoms = utils.abstract(task.init, all_predicates)
    logging.debug("[Task Planner] Task goal atoms: "
                  f"{pformat(sorted(task.goal))}")
    logging.debug("[Task Planner] Task init atoms: "
                  f"{pformat(sorted(init_atoms))}")
    goal = task.goal
    objects = set(task.init)
    ground_processes, reachable_atoms = process_task_plan_grounding(
        init_atoms,
        objects,
        processes,
        allow_waits=True,
        compute_reachable_atoms=True,
        derived_predicates=derived_predicates)

    if CFG.process_task_planning_heuristic == "goal_count":
        heuristic = utils.create_task_planning_heuristic(
            CFG.process_task_planning_heuristic,
            init_atoms,  # type: ignore[type-var]
            goal,
            ground_processes,
            all_predicates,
            objects)
    elif CFG.process_task_planning_heuristic == "lm_cut":
        heuristic = create_lm_cut_heuristic(  # type: ignore[assignment]
            goal,
            ground_processes,
            derived_predicates,
            objects,
            use_derived_predicates=CFG.use_derived_predicate_in_heuristic)
    elif CFG.process_task_planning_heuristic == "h_max":
        heuristic = create_h_max_heuristic(  # type: ignore[assignment]
            goal,
            ground_processes,
            derived_predicates,
            objects,
            use_derived_predicates=CFG.use_derived_predicate_in_heuristic)

    elif CFG.process_task_planning_heuristic == "h_ff":
        heuristic = create_ff_heuristic(  # type: ignore[assignment]
            goal,
            ground_processes,
            derived_predicates,
            objects,
            use_derived_predicates=CFG.use_derived_predicate_in_heuristic)
    else:
        raise ValueError("Unrecognized "
                         "process_task_planning_heuristic: "
                         f"{CFG.process_task_planning_heuristic}")

    return task_plan(
        init_atoms,
        goal,
        ground_processes,
        reachable_atoms,
        heuristic,
        seed,
        timeout,
        max_skeletons_optimized,
        use_visited_state_set=use_visited_state_set,
        derived_predicates=derived_predicates,
        objects=objects,
        abstract_policy=abstract_policy,
        max_policy_guided_rollout=max_policy_guided_rollout,
    )


def task_plan(
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_processes: List[_GroundCausalProcess],
    reachable_atoms: Set[GroundAtom],
    heuristic: _TaskPlanningHeuristic,
    seed: int,
    timeout: float,
    max_skeletons_optimized: int,
    use_visited_state_set: bool = True,
    derived_predicates: Optional[Set[DerivedPredicate]] = None,
    objects: Optional[Set[Object]] = None,
    abstract_policy: Optional[AbstractProcessPolicy] = None,
    max_policy_guided_rollout: int = 0,
) -> Iterator[Tuple[List[_GroundEndogenousProcess], List[Set[GroundAtom]],
                    Metrics]]:
    """Run task planning portion of SeSamE.

    A* search is run, and skeletons that achieve the
    goal symbolically are yielded. Specifically, yields
    a tuple of (skeleton, atoms sequence, metrics dict).

    This method is NOT used by SeSamE, but is instead
    provided as a convenient wrapper around
    _skeleton_generator below (which IS used by SeSamE)
    that takes in only the minimal necessary arguments.

    This method is tightly coupled with
    task_plan_grounding -- the reason they are separate
    methods is that it is sometimes possible to ground
    only once and then plan multiple times (e.g. from
    different initial states, or to
    different goals). To run task planning once, call task_plan_grounding to
    get ground_nsrts and reachable_atoms; then create a heuristic using
    utils.create_task_planning_heuristic; then call this method. See the tests
    in tests/test_planning for usage examples.
    """
    if derived_predicates is None:
        derived_predicates = set()
    if objects is None:
        objects = set()
    if CFG.planning_check_dr_reachable and \
            not goal.issubset(reachable_atoms):
        logging.info(f"Detected goal unreachable. Goal: {goal}")
        logging.info(f"Initial atoms: {init_atoms}")
        raise PlanningFailure(f"Goal {goal} not dr-reachable")
    dummy_task = Task(DefaultState, goal)
    metrics: Metrics = defaultdict(float)
    # logging.debug(f"init_atoms: {init_atoms}")
    generator = _skeleton_generator_with_processes(
        dummy_task,
        ground_processes,
        init_atoms,
        heuristic,
        seed,
        timeout,
        metrics,
        max_skeletons_optimized,
        abstract_policy=abstract_policy,
        sesame_max_policy_guided_rollout=max_policy_guided_rollout,
        use_visited_state_set=use_visited_state_set,
        derived_predicates=derived_predicates,
        objects=objects,
        heuristic_weight=CFG.process_planning_heuristic_weight,
    )

    # Note that we use this pattern to avoid having to catch an exception
    # when _skeleton_generator runs out of skeletons to optimize.
    for skeleton, atoms_sequence in islice(generator, max_skeletons_optimized):
        yield skeleton, atoms_sequence, metrics.copy()


def run_task_plan_with_processes_once(
    task: Task,
    processes: Set[CausalProcess],
    preds: Set[Predicate],
    _types: Set[Type],
    timeout: float,
    seed: int,
    _task_planning_heuristic: str,
    max_horizon: float = np.inf,
    _compute_reachable_atoms: bool = False,
    abstract_policy: Optional[AbstractProcessPolicy] = None,
    max_policy_guided_rollout: int = 0,
) -> Tuple[List[_GroundEndogenousProcess], List[Set[GroundAtom]], Metrics]:
    """Get a single abstract plan for a task.

    The sequence of ground atom sets returned represent NECESSARY atoms.
    """

    start_time = time.perf_counter()

    if CFG.sesame_task_planner == "astar":
        duration = time.perf_counter() - start_time
        timeout -= duration
        plan, _atoms_seq, metrics = next(
            task_plan_from_task(
                task,
                preds,
                processes,
                seed,
                timeout,
                max_skeletons_optimized=1,
                abstract_policy=abstract_policy,
                max_policy_guided_rollout=max_policy_guided_rollout,
            ))
        if len(plan) > max_horizon:
            raise PlanningFailure(
                "Skeleton produced by A-star exceeds horizon!")
    else:
        raise ValueError("Unrecognized sesame_task_planner: "
                         f"{CFG.sesame_task_planner}")

    # comment out for now
    # necessary_atoms_seq = utils.compute_necessary_atoms_seq(
    #     plan, atoms_seq, goal)
    necessary_atoms_seq: List[Set[GroundAtom]] = []

    return plan, necessary_atoms_seq, metrics


def sesame_plan_with_processes(
    task: Task,
    option_model: _OptionModelBase,
    processes: Set[CausalProcess],
    predicates: Set[Predicate],
    timeout: float,
    seed: int,
    max_skeletons_optimized: int,
    max_horizon: int,
    abstract_policy: Optional[AbstractProcessPolicy] = None,
    max_policy_guided_rollout: int = 0,
) -> Tuple[List[_Option], List[_GroundEndogenousProcess], Metrics]:
    """Run bilevel planning with processes (SeSamE-style).

    Generates process skeletons via A* search and refines each with low-
    level search (backtracking over continuous parameter samples).
    Returns a sequence of options, the process skeleton, and metrics.
    """
    start_time = time.perf_counter()

    gen = task_plan_from_task(
        task,
        predicates,
        processes,
        seed,
        timeout - (time.perf_counter() - start_time),
        max_skeletons_optimized,
        abstract_policy=abstract_policy,
        max_policy_guided_rollout=max_policy_guided_rollout,
    )

    partial_refinements: list = []
    metrics: Metrics = defaultdict(float)
    refinement_start_time = time.perf_counter()

    for skeleton, atoms_sequence, skel_metrics in gen:
        # Update metrics from skeleton generation.
        for k, v in skel_metrics.items():
            metrics[k] = v

        logging.debug(f"Found process skeleton: "
                      f"{[p.name_and_objects_str() for p in skeleton]}")

        try:
            plan, suc = run_low_level_search(
                task,
                option_model,
                skeleton,  # type: ignore[arg-type]
                atoms_sequence,
                seed,
                timeout - (time.perf_counter() - start_time),
                metrics,
                max_horizon)
        except _DiscoveredFailureException:
            # Process planning doesn't support failure discovery;
            # treat as a failed skeleton.
            suc = False
            plan = []

        if suc:
            logging.info(
                f"Process planning succeeded! Found plan of length "
                f"{len(plan)} after "
                f"{int(metrics['num_skeletons_optimized'])} "
                f"skeletons with {int(metrics['num_samples'])} samples")
            metrics["plan_length"] = len(plan)
            metrics["refinement_time"] = (time.perf_counter() -
                                          refinement_start_time)
            # Inject Wait target atoms from atoms_sequence so
            # execution terminates on specific atoms, not noise.
            _inject_wait_targets(plan, skeleton, atoms_sequence)
            return plan, skeleton, metrics

        partial_refinements.append((skeleton, plan))
        if time.perf_counter() - start_time > timeout:
            raise PlanningTimeout(
                "Process planning timed out in refinement!",
                info={"partial_refinements": partial_refinements})

    raise PlanningFailure("Process planning exhausted all skeletons!",
                          info={"partial_refinements": partial_refinements})


def _inject_wait_targets(
    plan: List[_Option],
    _skeleton: List[_GroundEndogenousProcess],
    atoms_sequence: Sequence[Set[GroundAtom]],
) -> None:
    """Inject Wait target atoms into all Wait options in a plan."""
    for i, option in enumerate(plan):
        utils.inject_wait_targets_for_option(option, i, atoms_sequence)


def create_ff_heuristic(
    goal: Set[GroundAtom],
    ground_processes: List[_GroundCausalProcess],
    derived_predicates: Optional[Set[DerivedPredicate]] = None,
    objects: Optional[Set[Object]] = None,
    use_derived_predicates: bool = True,
    debug_log: bool = False,
) -> Callable[[Set[GroundAtom]], float]:
    """Creates a callable FF heuristic."""
    if derived_predicates is None:
        derived_predicates = set()
    if objects is None:
        objects = set()

    adds_map: Dict[GroundAtom, List[_GroundCausalProcess]] = defaultdict(list)
    for process in ground_processes:
        for atom in process.add_effects:
            adds_map[atom].append(process)

    # --- CHANGE START: Use pre-computation for the shared function ---
    dep_to_derived_preds: Dict[Predicate,
                               List[DerivedPredicate]] = defaultdict(list)
    if use_derived_predicates:
        for der_pred in derived_predicates:
            assert der_pred.auxiliary_predicates is not None, \
                "Can't find auxiliary predicates for derived predicate " +\
                f"{der_pred.name}"
            for aux_pred in der_pred.auxiliary_predicates:
                dep_to_derived_preds[aux_pred].append(der_pred)
    # --- CHANGE END ---

    def _ff_heuristic(atoms: Set[GroundAtom]) -> float:
        """The FF heuristic using incremental RPG generation."""
        if goal.issubset(atoms):
            return 0.0

        # --- 1. Build the Relaxed Planning Graph (RPG) ---
        initial_facts = atoms.copy()
        if use_derived_predicates:
            # The first layer must be a full, non-incremental computation.
            initial_facts.update(
                utils.abstract_with_derived_predicates(initial_facts,
                                                       derived_predicates,
                                                       objects))

        fact_layers: List[Set[GroundAtom]] = [initial_facts]
        process_layers: List[Set[_GroundCausalProcess]] = []

        if debug_log:
            count = 1
            logging.debug(f"Initial facts: {sorted(initial_facts)}")
        while not goal.issubset(fact_layers[-1]):
            if debug_log:
                logging.debug(f"Applying actions {count}...")
                count += 1
            current_facts = fact_layers[-1]

            # Find all processes whose preconditions
            # are met in the current layer.
            applicable_processes: Set[_GroundCausalProcess] = set()
            for process in ground_processes:
                if process.condition_at_start.issubset(current_facts):
                    applicable_processes.add(process)

            process_layers.append(applicable_processes)

            # --- Incremental Fact Generation ---
            # a) Collect all new primitive facts from applicable processes.
            primitive_add_effects = set()
            for process in applicable_processes:
                primitive_add_effects.update(process.add_effects)

            newly_added_primitive_facts = primitive_add_effects - current_facts
            if debug_log:
                logging.debug("Newly added primitive facts: "
                              f"{sorted(newly_added_primitive_facts)}")

            # b) Incrementally compute new derived facts.
            newly_derived_facts = set()
            if use_derived_predicates:
                # --- CHANGE START: Call the shared function ---
                newly_derived_facts = _run_incremental_derived_predicate_logic(
                    newly_added_primitive_facts,
                    current_facts,
                    objects,
                    dep_to_derived_preds,
                )
                # --- CHANGE END ---
                if debug_log:
                    logging.debug(
                        f"Newly derived facts: {sorted(newly_derived_facts)}\n"
                    )

            next_facts = (current_facts
                          | newly_added_primitive_facts
                          | newly_derived_facts)

            # If the new layer is identical to the old one, we've stagnated.
            if next_facts == current_facts:
                return float('inf')

            fact_layers.append(next_facts)

        # --- 2. Extract a Relaxed Plan (Backward Search through the RPG) ---
        relaxed_plan_actions: Set[_GroundEndogenousProcess] = set()
        subgoals_to_achieve = goal.copy()

        for i in range(len(fact_layers) - 1, 0, -1):

            if use_derived_predicates:
                for subgoal in subgoals_to_achieve.copy():
                    # Case 1: The subgoal is a DERIVED predicate.
                    # It is achieved 'for free' by its
                    # supporting auxiliary predicates.
                    if isinstance(subgoal.predicate, DerivedPredicate):
                        # The new subgoals are the auxiliary
                        # predicates that support it.
                        # In a relaxed plan, we conservatively
                        # add all atoms from the
                        # previous layer that could be supporters.
                        try:
                            supporter_predicates =\
                                utils.get_base_supporter_predicates(
                                    subgoal.predicate)
                        except Exception as e:
                            logging.error("Error getting base supporter "
                                          f"predicates for "
                                          f"{subgoal.predicate}: {e}")
                            raise
                        new_subgoals = {
                            atom
                            for atom in fact_layers[i - 1]
                            if atom.predicate in supporter_predicates
                        }

                        subgoals_to_achieve.update(new_subgoals)
                        subgoals_to_achieve.discard(subgoal)
            if debug_log:
                logging.debug(f"\nLayer {i} Subgoals to achieve: "
                              f"{sorted(subgoals_to_achieve)}")

            unachieved_subgoals = subgoals_to_achieve.copy()
            for subgoal in unachieved_subgoals:
                # If the subgoal appeared for the first time in this layer...
                if subgoal in fact_layers[i] and subgoal not in fact_layers[i -
                                                                            1]:

                    if debug_log:
                        logging.debug(f"Considering subgoal: {subgoal}")

                    # Case 2: The subgoal is a PRIMITIVE
                    # predicate (original logic).
                    best_supporter = None
                    # Find a process from the previous layer that achieves it.
                    for process in adds_map.get(subgoal, []):
                        if process in process_layers[i - 1]:
                            if debug_log:
                                logging.debug(
                                    f"Found supporter for {subgoal}: "
                                    f"{process.name_and_objects_str()}")
                            best_supporter = process
                            break

                    if best_supporter:
                        # Only agent actions (endogenous)
                        # contribute to the plan cost.
                        if isinstance(best_supporter,
                                      _GroundEndogenousProcess):
                            relaxed_plan_actions.add(best_supporter)

                        # Add the supporter's preconditions
                        # to our set of subgoals.
                        subgoals_to_achieve.update(
                            best_supporter.condition_at_start)
                        subgoals_to_achieve.discard(subgoal)

        return float(len(relaxed_plan_actions))

    return _ff_heuristic


def create_lm_cut_heuristic(
    goal: Set[GroundAtom],
    ground_processes: List[_GroundCausalProcess],
    derived_predicates: Optional[Set[DerivedPredicate]] = None,
    objects: Optional[Set[Object]] = None,
    use_derived_predicates: bool = True,
) -> Callable[[Set[GroundAtom]], float]:
    """Creates a callable LM-cut heuristic function.

    This heuristic iteratively finds landmarks by computing a relaxed
    plan, calculating its cost, and then assuming its effects have been
    achieved before solving for the next landmark.
    """
    if derived_predicates is None:
        derived_predicates = set()
    if objects is None:
        objects = set()

    # --- Pre-computation ---
    adds_map: Dict[GroundAtom, List[_GroundCausalProcess]] = defaultdict(list)
    for process in ground_processes:
        for atom in process.add_effects:
            adds_map[atom].append(process)

    # --- CHANGE START: Use pre-computation for the shared function ---
    dep_to_derived_preds: Dict[Predicate,
                               List[DerivedPredicate]] = defaultdict(list)
    if use_derived_predicates:
        for der_pred in derived_predicates:
            assert der_pred.auxiliary_predicates is not None
            for aux_pred in der_pred.auxiliary_predicates:
                dep_to_derived_preds[aux_pred].append(der_pred)
    # --- CHANGE END ---

    def _calculate_relaxed_plan(
        current_atoms: Set[GroundAtom], current_goal: Set[GroundAtom]
    ) -> Tuple[float, Set[_GroundCausalProcess]]:
        """Helper that computes one relaxed plan (our landmark) from a given
        state."""
        initial_facts = current_atoms.copy()
        if use_derived_predicates:
            initial_facts.update(
                utils.abstract_with_derived_predicates(initial_facts,
                                                       derived_predicates,
                                                       objects))

        if current_goal.issubset(initial_facts):
            return 0.0, set()

        fact_layers: List[Set[GroundAtom]] = [initial_facts]
        process_layers: List[Set[_GroundCausalProcess]] = []

        while not current_goal.issubset(fact_layers[-1]):
            current_facts = fact_layers[-1]

            applicable_processes: Set[_GroundCausalProcess] = set()
            for process in ground_processes:
                if process.condition_at_start.issubset(current_facts):
                    applicable_processes.add(process)

            process_layers.append(applicable_processes)

            primitive_add_effects = set()
            for process in applicable_processes:
                primitive_add_effects.update(process.add_effects)
            newly_added_primitive_facts = primitive_add_effects - current_facts

            newly_derived_facts = set()
            if use_derived_predicates:
                # --- CHANGE START: Call the shared function ---
                newly_derived_facts = _run_incremental_derived_predicate_logic(
                    newly_added_primitive_facts,
                    current_facts,
                    objects,
                    dep_to_derived_preds,
                )
                # --- CHANGE END ---

            next_facts = (current_facts
                          | newly_added_primitive_facts
                          | newly_derived_facts)

            if next_facts == current_facts:
                return float('inf'), set()

            fact_layers.append(next_facts)

        # 2. Extract one relaxed plan via backward search.
        relaxed_plan: Set[_GroundCausalProcess] = set()
        subgoals_to_achieve = current_goal.copy()

        for i in range(len(fact_layers) - 1, 0, -1):
            for subgoal in subgoals_to_achieve.copy():
                if subgoal in fact_layers[i] and subgoal not in fact_layers[i -
                                                                            1]:
                    best_supporter = None
                    for process in adds_map.get(subgoal, []):
                        if process in process_layers[i - 1]:
                            best_supporter = process
                            break

                    if best_supporter:
                        relaxed_plan.add(best_supporter)
                        subgoals_to_achieve.update(
                            best_supporter.condition_at_start)
                        subgoals_to_achieve.discard(subgoal)

        # 3. Calculate the cost of the relaxed plan.
        cost = 0.0
        for process in relaxed_plan:
            # Endogenous processes (agent actions) have a cost.
            if isinstance(process, _GroundEndogenousProcess):
                # Use axiom_cost if it's a derived
                # predicate axiom, else default to 1.
                cost += getattr(process, 'axiom_cost', 1.0)

        return cost, relaxed_plan

    def _lm_cut_heuristic(atoms: Set[GroundAtom]) -> float:
        """The main heuristic function.

        It iteratively calls the relaxed plan solver to find and sum the
        costs of landmarks.
        """
        total_cost = 0.0
        current_atoms = atoms.copy()

        # Loop until the goal is satisfied in our simulated state.
        while not goal.issubset(current_atoms):
            # Find the cost and plan for the next landmark.
            landmark_cost, landmark_plan = _calculate_relaxed_plan(
                current_atoms, goal)

            # If a landmark is infinitely costly, the goal is unreachable.
            if landmark_cost == float('inf'):
                return float('inf')

            # If we found a plan with no cost (e.g., only free events),
            # but haven't reached the goal, we must force progress by adding
            # at least one real action. A cost of 1 is the minimum.
            if landmark_cost == 0.0:
                total_cost += 1.0

            total_cost += landmark_cost

            # "Apply" the landmark by adding the effects
            # of its plan to our state.
            if not landmark_plan:
                # Should not be reachable if cost is
                # not inf, but as a safeguard...
                return float('inf')

            for process in landmark_plan:
                current_atoms.update(process.add_effects)

        return total_cost

    return _lm_cut_heuristic


def create_h_max_heuristic(
    goal: Set[GroundAtom],
    ground_processes: List[_GroundCausalProcess],
    derived_predicates: Optional[Set[DerivedPredicate]] = None,
    objects: Optional[Set[Object]] = None,
    use_derived_predicates: bool = True,
) -> Callable[[Set[GroundAtom]], float]:
    """Creates a callable h_max heuristic function.

    Compatible with exogenous processes (zero-cost) and derived
    predicates (zero-cost).
    """
    if derived_predicates is None:
        derived_predicates = set()
    if objects is None:
        objects = set()

    # Pre-computation for derived predicate deps.
    dep_to_derived_preds: Dict[Predicate,
                               List[DerivedPredicate]] = defaultdict(list)
    if use_derived_predicates:
        for der_pred in derived_predicates:
            assert der_pred.auxiliary_predicates is not None
            for aux_pred in der_pred.auxiliary_predicates:
                dep_to_derived_preds[aux_pred].append(der_pred)

    def _h_max_heuristic(atoms: Set[GroundAtom]) -> float:
        """The h_max heuristic function."""
        if goal.issubset(atoms):
            return 0.0

        # Initialize costs: 0 for initial atoms, infinity otherwise.
        atom_costs = defaultdict(lambda: float('inf'))
        for atom in atoms:
            atom_costs[atom] = 0.0

        # Iteratively relax costs until a fixed point is reached.
        while True:
            costs_changed = False

            # --- 1. Propagate costs through primitive processes ---
            for process in ground_processes:
                # Cost of preconditions is the max cost of any single precond.
                precond_cost = max(
                    [atom_costs[p] for p in process.condition_at_start]
                    or [0.0])

                if precond_cost == float('inf'):
                    continue

                # Actions (endogenous) have cost 1,
                # others (exogenous) have cost 0.
                process_cost = 1.0 if isinstance(
                    process, _GroundEndogenousProcess) else 0.0
                total_cost = precond_cost + process_cost

                # Update costs of effects if we found a
                # cheaper way to achieve them.
                for effect in process.add_effects:
                    if total_cost < atom_costs[effect]:
                        atom_costs[effect] = total_cost
                        costs_changed = True

            # --- 2. Propagate costs through derived predicates (zero-cost) ---
            if use_derived_predicates:
                # We need to loop here to handle chains of derived predicates.
                while True:
                    derived_costs_changed = False
                    # This logic is a simplified version of
                    # the incremental approach, adapted for
                    # h_max's cost propagation.
                    current_facts_for_eval = {
                        a
                        for a, c in atom_costs.items() if c != float('inf')
                    }

                    # Check all derived predicates whose
                    # inputs might have changed.
                    # pylint: disable=protected-access
                    _fn = utils \
                        ._abstract_with_derived_predicates
                    # pylint: enable=protected-access
                    derived_atoms = _fn(current_facts_for_eval,
                                        derived_predicates, objects)

                    for derived_atom in derived_atoms:
                        # To determine the cost, we need to find the specific
                        # atoms that make this derived predicate true. This is
                        # complex, so we approximate by taking the max cost
                        # of any atom in the current state. This is a safe
                        # over-approximation for the preconditions. A more
                        # precise implementation would require inspecting the
                        # logic inside the 'holds' function. For now, we
                        # find the cost of the supporter atoms.
                        # NOTE: This is a simplification. A fully correct h_max
                        # would need to know the specific atoms that satisfy
                        # the 'holds' condition. We find the supporters by
                        # checking the auxiliary predicates.
                        supporter_atoms: Set[GroundAtom] = set()
                        assert isinstance(derived_atom.predicate,
                                          DerivedPredicate)
                        assert derived_atom.predicate.auxiliary_predicates \
                            is not None
                        for p in derived_atom.predicate.auxiliary_predicates:
                            supporter_atoms.update(
                                a for a in current_facts_for_eval
                                if a.predicate == p)

                        if not supporter_atoms:
                            continue

                        derived_cost = max(
                            [atom_costs[a] for a in supporter_atoms] or [0.0])

                        if derived_cost < atom_costs[derived_atom]:
                            atom_costs[derived_atom] = derived_cost
                            derived_costs_changed = True
                            costs_changed = True

                    if not derived_costs_changed:
                        break

            # If no costs were updated in a full pass,
            # we've reached a fixed point.
            if not costs_changed:
                break

        # The heuristic value is the max cost of any goal atom.
        goal_costs = [atom_costs[g] for g in goal]

        # If any goal atom is infinitely costly, the goal is unreachable.
        if not goal_costs or max(goal_costs) == float('inf'):
            return float('inf')

        return max(goal_costs)

    return _h_max_heuristic


def _run_incremental_derived_predicate_logic(
    newly_added_facts: Set[GroundAtom],
    existing_facts: Set[GroundAtom],
    objects: Set[Object],
    dep_to_derived_preds: Dict[Predicate, List[DerivedPredicate]],
) -> Set[GroundAtom]:
    """Incrementally compute the fixed point of derived predicate atoms."""
    all_newly_derived_facts: Set[GroundAtom] = set()
    facts_for_next_iter = newly_added_facts.copy()

    while facts_for_next_iter:
        derived_preds_to_check: Set[DerivedPredicate] = set()
        for fact in facts_for_next_iter:
            if fact.predicate in dep_to_derived_preds:
                derived_preds_to_check.update(
                    dep_to_derived_preds[fact.predicate])

        if not derived_preds_to_check:
            break

        current_state_for_eval = existing_facts | all_newly_derived_facts |\
                                    newly_added_facts
        # pylint: disable=protected-access
        _fn = utils \
            ._abstract_with_derived_predicates
        # pylint: enable=protected-access
        potential_new_atoms = _fn(current_state_for_eval,
                                  derived_preds_to_check, objects)

        truly_new_atoms = potential_new_atoms - (existing_facts
                                                 | all_newly_derived_facts)

        if not truly_new_atoms:
            break

        all_newly_derived_facts.update(truly_new_atoms)
        facts_for_next_iter = truly_new_atoms

    return all_newly_derived_facts


if __name__ == "__main__":
    from predicators.envs.pybullet_boil import PyBulletBoilEnv
    from predicators.ground_truth_models import get_gt_options, \
        get_gt_processes
    args = utils.parse_args()
    utils.update_config(args)
    str_args = " ".join(sys.argv)
    utils.configure_logging()
    CFG.seed = 0
    CFG.env = "pybullet_boil"
    CFG.planning_filter_unreachable_nsrt = False
    CFG.planning_check_dr_reachable = False

    env = PyBulletBoilEnv()
    # objects
    robot = env._robot  # pylint: disable=protected-access
    faucet = env._faucet  # pylint: disable=protected-access
    jug1 = env._jugs[0]  # pylint: disable=protected-access
    burner1 = env._burners[0]  # pylint: disable=protected-access

    # Processes
    options = get_gt_options(env.get_name())
    processes = get_gt_processes(env.get_name(), env.predicates, options)
    action_processes = [
        p for p in processes if isinstance(p, EndogenousProcess)
    ]
    pick = [p for p in action_processes if p.name == 'PickJugFromFaucet'][0]
    # place = [p for p in action_processes if p.name == 'PlaceUnderFaucet'][0]
    switch_on = [p for p in action_processes if p.name == 'SwitchFaucetOn'][0]
    switch_off = [p for p in action_processes
                  if p.name == 'SwitchFaucetOff'][0]
    wait_proc = [p for p in action_processes if p.name == 'Wait'][0]

    plan: List[_GroundEndogenousProcess] = [
        switch_on.ground([robot, faucet]),
        switch_off.ground([robot, faucet]),
        wait_proc.ground([robot]),
        wait_proc.ground([robot])
    ]

    # Predicates
    predicates = env.predicates

    def policy() -> Optional[_GroundEndogenousProcess]:
        """Policy."""
        if len(plan) > 0:
            return plan.pop(0)
        return None

    # Task
    rng = np.random.default_rng(CFG.seed)
    task = env._make_tasks(  # pylint: disable=protected-access
        1,
        [1],
        [1],  # type: ignore[call-arg, arg-type]
        rng)[0]
    ground_processes, _reachable_atoms = process_task_plan_grounding(
        init_atoms=task.init,  # type: ignore[arg-type]
        objects=set(task.init),
        cps=processes,
        allow_waits=True,
        compute_reachable_atoms=False)

    world_model = ProcessWorldModel(ground_processes=ground_processes,
                                    state=utils.abstract(
                                        task.init, predicates),
                                    state_history=[],
                                    action_history=[],
                                    scheduled_events={},
                                    t=0)
    for _ in range(100):
        action = policy()
        if action is not None:
            world_model.big_step(action)
        else:
            break
