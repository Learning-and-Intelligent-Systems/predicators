"""Learn operators by searching over sets of PNADs."""

from __future__ import annotations

import abc
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

from predicators import utils
from predicators.nsrt_learning.strips_learning.gen_to_spec_learner import \
    GeneralToSpecificSTRIPSLearner
from predicators.structs import GroundAtom, LowLevelTrajectory, \
    ParameterizedOption, PartialNSRTAndDatastore, Predicate, Segment, Task, \
    _GroundSTRIPSOperator


class _PNADSearchOperator(abc.ABC):
    """An operator that proposes successor sets of PNAD sets."""

    def __init__(self, trajectories: List[LowLevelTrajectory],
                 train_tasks: List[Task], predicates: Set[Predicate],
                 segmented_trajs: List[List[Segment]],
                 learner: PNADSearchSTRIPSLearner) -> None:
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs
        self._learner = learner

    @abc.abstractmethod
    def get_successors(
        self, pnads: FrozenSet[PartialNSRTAndDatastore]
    ) -> Iterator[FrozenSet[PartialNSRTAndDatastore]]:
        """Generate zero or more successor PNAD sets."""
        raise NotImplementedError("Override me!")


class _BackChainingPNADSearchOperator(_PNADSearchOperator):
    """An operator that uses backchaining to propose a new PNAD set."""

    def get_successors(
        self, pnads: FrozenSet[PartialNSRTAndDatastore]
    ) -> Iterator[FrozenSet[PartialNSRTAndDatastore]]:
        pnads_list = list(pnads)
        pnads_list = sorted(pnads_list)
        uncovered_segment = self._get_first_uncovered_segment(pnads_list)
        if uncovered_segment is not None:
            # We will need to induce an operator to cover this
            # segment, and thus it must have some necessary add effects.
            new_pnad = self._learner.spawn_new_pnad(uncovered_segment)
            ret_pnads_list = self._append_new_pnad_and_keep_effects(
                new_pnad, pnads_list)
            ret_pnads = frozenset(ret_pnads_list)
            yield ret_pnads

    def _append_new_pnad_and_keep_effects(
        self, new_pnad: PartialNSRTAndDatastore,
        current_pnads: List[PartialNSRTAndDatastore]
    ) -> List[PartialNSRTAndDatastore]:
        """Given some newly-created PNAD and a set of existing PNADs, correctly
        repartition data amongst all these PNADs and induce keep effects for
        the newly-created PNAD.

        Return the final set of all PNADs (current, newly-created, and
        ones with keep effects).
        """
        new_pnads = current_pnads + [new_pnad]
        # We first repartition data and ensure delete and ignore
        # effects for the newly-created PNAD are correct.
        new_pnads = self._learner.recompute_pnads_from_effects(new_pnads)
        # Ensure that the unnecessary keep effs sub and poss
        # keep effects are both cleared before backchaining. This is
        # important because we will be inducing keep effects after this
        # backchaining.
        for pnad in new_pnads:
            pnad.poss_keep_effects.clear()
            self._learner.clear_unnecessary_keep_effs(pnad)
        # We rerun backchaining to make sure the seg_to_keep_effects_sub
        # is up-to-date.
        self._get_backchaining_results(new_pnads)
        # Now we can induce keep effects for new_pnad.
        # NOTE: we cannot use new_pnad directly here, since that's the old
        # object before running backchaining (thus, its poss_keep_effects are
        # incorrect). Instead, we need to find which of the new_pnads
        # corresponds to the new_pnad we just induced.
        newly_added_pnad = None
        for new_p in new_pnads:
            if new_p.op.add_effects == new_pnad.op.add_effects and \
                new_p.op.parameters == new_pnad.op.parameters and \
                new_p.option_spec == new_pnad.option_spec:
                newly_added_pnad = new_p
        assert newly_added_pnad is not None
        new_pnads_with_keep_effs = self._learner.get_pnads_with_keep_effects(
            newly_added_pnad)
        new_pnads += sorted(new_pnads_with_keep_effs)
        # We recompute pnads again here to delete keep effect operators
        # that are unnecessary.
        new_pnads = self._learner.recompute_pnads_from_effects(new_pnads)
        return new_pnads

    def _get_backchaining_results(
        self, pnads: List[PartialNSRTAndDatastore]
    ) -> Tuple[int, List[Tuple[List[Segment], List[_GroundSTRIPSOperator]]]]:
        backchaining_results = []
        max_chain_len = 0
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if ll_traj.is_demo:
                traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
                chain = self._learner.backchain(seg_traj, pnads, traj_goal)
                max_chain_len = max(max_chain_len, len(chain))
                backchaining_results.append((seg_traj, chain))
        return max_chain_len, backchaining_results

    def _get_first_uncovered_segment(
        self,
        pnads: List[PartialNSRTAndDatastore],
    ) -> Optional[Segment]:
        # Find the first uncovered segment. Do this in a kind of breadth-first
        # backward search over trajectories.
        # Compute all the chains once up front.
        max_chain_len, backchaining_results = self._get_backchaining_results(
            pnads)
        # Now look for an uncovered segment. If one cannot be found, this
        # method will automatically return None.
        for depth in range(max_chain_len + 1):
            for seg_traj, op_chain in backchaining_results:
                if not (len(op_chain) > depth
                        or len(op_chain) == len(seg_traj)):
                    # We found an uncovered transition: we now need to return
                    # the information necessary to induce a new operator to
                    # cover it.
                    # The timestep of the uncovered transition is the number of
                    # segments - 1 - (numer of actions in our backchained plan)
                    t = (len(seg_traj) - 1) - len(op_chain)
                    assert t >= 0
                    segment = seg_traj[t]
                    return segment
        return None


class _PruningPNADSearchOperator(_PNADSearchOperator):
    """An operator that prunes PNAD sets."""

    def get_successors(
        self, pnads: FrozenSet[PartialNSRTAndDatastore]
    ) -> Iterator[FrozenSet[PartialNSRTAndDatastore]]:
        # NOTE: Sorting done here to maintain determinism.
        for pnad_to_remove in sorted(pnads,
                                     key=lambda p: len(p.op.add_effects)):
            yield frozenset([pnad for pnad in pnads if pnad != pnad_to_remove])


class _PNADSearchHeuristic(abc.ABC):
    """Given a set of PNAD sets, produce a score, with lower better."""

    def __init__(self, trajectories: List[LowLevelTrajectory],
                 train_tasks: List[Task], predicates: Set[Predicate],
                 segmented_trajs: List[List[Segment]],
                 learner: PNADSearchSTRIPSLearner) -> None:
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs
        self._learner = learner
        # We compute the total number of segments, which is also the
        # maximum number of operators that we will induce (since, in
        # the worst case, we induce a different operator for every
        # segment).
        self._total_num_segments = 0
        for seg_traj in self._segmented_trajs:
            self._total_num_segments += len(seg_traj)

    @abc.abstractmethod
    def __call__(self,
                 curr_pnads: FrozenSet[PartialNSRTAndDatastore]) -> float:
        """Compute the heuristic value for the given PNAD sets."""
        raise NotImplementedError("Override me!")


class _BackChainingHeuristic(_PNADSearchHeuristic):
    """Counts the number of transitions that are not yet covered by some
    operator in the backchaining sense."""

    def __call__(self,
                 curr_pnads: FrozenSet[PartialNSRTAndDatastore]) -> float:
        # Start by recomputing all PNADs from their effects.
        recomp_curr_pnads = self._learner.recompute_pnads_from_effects(
            sorted(curr_pnads))
        # Next, run backchaining using these PNADs.
        uncovered_transitions = 0
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if ll_traj.is_demo:
                traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
                chain = self._learner.backchain(seg_traj, recomp_curr_pnads,
                                                traj_goal)
                assert len(chain) <= len(seg_traj)
                uncovered_transitions += len(seg_traj) - len(chain)
        # Our objective is such that covering more data is *always*
        # more important than creating a less complex set of operators.
        # Thus, we multiply the coverage term by the maximum number of
        # possible operators, so it will always be beneficial to
        # cover more data over deleting operators to make a less complex
        # hypothesis.
        coverage_term = uncovered_transitions * self._total_num_segments
        # NOTE: for now, we measure complexity by simply counting the number
        # of learned PNADs. We could come up with more intricate and
        # accurate measures that also take into account the add effects,
        # arity, etc. (though this might involve changing the weighting
        # of the coverage term).
        complexity_term = len(recomp_curr_pnads)
        return coverage_term + complexity_term


class PNADSearchSTRIPSLearner(GeneralToSpecificSTRIPSLearner):
    """Base class for a effect search STRIPS learner."""

    @classmethod
    def get_name(cls) -> str:
        return "pnad_search"

    def recompute_pnads_from_effects(
            self, pnads: List[PartialNSRTAndDatastore]
    ) -> List[PartialNSRTAndDatastore]:
        """Given some input PNADs, strips away everything except the add and
        keep effects, then re-partitions data amongst these and uses this to
        recompute these components."""
        # First, reset all PNADs to only maintain their add and
        # keep effects. Ensure they ignore all predicates in the domain.
        for pnad in pnads:
            keep_effects = pnad.op.preconditions & pnad.op.add_effects
            new_pnad_op = pnad.op.copy_with(
                preconditions=keep_effects,
                add_effects=pnad.op.add_effects,
                ignore_effects=self._predicates.copy())
            pnad.op = new_pnad_op
        # Repartition all data amongst these new PNADs.
        self._recompute_datastores_from_segments(pnads)
        # Prune any PNADs with empty datastores.
        pnads = [p for p in pnads if p.datastore]
        # Add new preconditions.
        for pnad in pnads:
            preconditions = self._induce_preconditions_via_intersection(pnad)
            pnad.op = pnad.op.copy_with(preconditions=preconditions)
        # Add delete and ignore effects.
        for pnad in pnads:
            self._compute_pnad_delete_effects(pnad)
            self._compute_pnad_ignore_effects(pnad)
        # Fix naming.
        pnad_map: Dict[ParameterizedOption, List[PartialNSRTAndDatastore]] = {
            p.option_spec[0]: []
            for p in pnads
        }
        for p in pnads:
            p.op = p.op.copy_with(name=p.option_spec[0].name)
            pnad_map[p.option_spec[0]].append(p)
        pnads = self._get_uniquely_named_nec_pnads(pnad_map)
        return pnads

    def _learn(self) -> List[PartialNSRTAndDatastore]:
        # Set up hill-climbing search over PNAD sets.
        # Create the search operators.
        search_operators = self._create_search_operators()
        # Create the heuristic.
        heuristic = self._create_heuristic()
        # Initialize the search.
        initial_state: FrozenSet[PartialNSRTAndDatastore] = frozenset()

        def get_successors(
            pnads: FrozenSet[PartialNSRTAndDatastore]
        ) -> Iterator[Tuple[Tuple[_PNADSearchOperator, int],
                            FrozenSet[PartialNSRTAndDatastore], float]]:
            for op in search_operators:
                for i, child in enumerate(op.get_successors(pnads)):
                    yield (op, i), child, 1.0  # cost always 1

        # Run hill-climbing search.
        path, _, _ = utils.run_hill_climbing(initial_state=initial_state,
                                             check_goal=lambda _: False,
                                             get_successors=get_successors,
                                             heuristic=heuristic)

        # Extract the best PNADs set.
        final_pnads = path[-1]
        # Recompute these PNADs so that they exactly match the PNADs used
        # to compute the final heuristic.
        recomp_final_pnads = self.recompute_pnads_from_effects(
            sorted(final_pnads))
        # Fix naming.
        pnad_map: Dict[ParameterizedOption, List[PartialNSRTAndDatastore]] = {
            p.option_spec[0]: []
            for p in recomp_final_pnads
        }
        for p in recomp_final_pnads:
            p.op = p.op.copy_with(name=p.option_spec[0].name)
            pnad_map[p.option_spec[0]].append(p)
        ret_pnads = self._get_uniquely_named_nec_pnads(pnad_map)
        return ret_pnads

    def _create_search_operators(self) -> List[_PNADSearchOperator]:
        op_classes = [
            _BackChainingPNADSearchOperator, _PruningPNADSearchOperator
        ]
        ops = [
            cls(self._trajectories, self._train_tasks, self._predicates,
                self._segmented_trajs, self) for cls in op_classes
        ]
        return ops

    def _create_heuristic(self) -> _PNADSearchHeuristic:
        backchaining_heur = _BackChainingHeuristic(self._trajectories,
                                                   self._train_tasks,
                                                   self._predicates,
                                                   self._segmented_trajs, self)
        return backchaining_heur

    def backchain(self, segmented_traj: List[Segment],
                  pnads: List[PartialNSRTAndDatastore],
                  traj_goal: Set[GroundAtom]) -> List[_GroundSTRIPSOperator]:
        """Returns chain of ground operators in REVERSE order."""
        operator_chain: List[_GroundSTRIPSOperator] = []
        atoms_seq = utils.segment_trajectory_to_atoms_sequence(segmented_traj)
        objects = set(segmented_traj[0].states[0])
        assert traj_goal.issubset(atoms_seq[-1])
        necessary_image = set(traj_goal)
        for t in range(len(atoms_seq) - 2, -1, -1):
            segment = segmented_traj[t]
            segment.necessary_image = necessary_image
            segment.necessary_add_effects = necessary_image - atoms_seq[t]
            pnad, var_to_obj = self._find_best_matching_pnad_and_sub(
                segment, objects, pnads)
            # If no match found, terminate.
            if pnad is None:
                break
            assert var_to_obj is not None
            obj_to_var = {v: k for k, v in var_to_obj.items()}
            assert len(var_to_obj) == len(obj_to_var)
            ground_op = pnad.op.ground(
                tuple(var_to_obj[var] for var in pnad.op.parameters))
            next_atoms = utils.apply_operator(ground_op, segment.init_atoms)
            # Update the PNAD's seg_to_keep_effs_sub dict.
            self._update_pnad_seg_to_keep_effs(pnad, necessary_image,
                                               ground_op, obj_to_var, segment)
            # If we're missing something in the necessary image, terminate.
            if not necessary_image.issubset(next_atoms):
                break
            # Otherwise, extend the chain.
            operator_chain.append(ground_op)
            # Update necessary_image for this timestep. It no longer
            # needs to include the ground add effects of this PNAD, but
            # must now include its ground preconditions.
            necessary_image = necessary_image.copy()
            necessary_image -= {
                a.ground(var_to_obj)
                for a in pnad.op.add_effects
            }
            necessary_image |= {
                a.ground(var_to_obj)
                for a in pnad.op.preconditions
            }
        return operator_chain
