"""Learn operators by searching over sets of add effect sets."""

from __future__ import annotations

import abc
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, \
    Tuple

from predicators import utils
from predicators.nsrt_learning.strips_learning.gen_to_spec_learner import \
    GeneralToSpecificSTRIPSLearner
from predicators.structs import GroundAtom, LiftedAtom, LowLevelTrajectory, \
    Object, OptionSpec, ParameterizedOption, PartialNSRTAndDatastore, \
    Predicate, Segment, Task, Variable, _GroundSTRIPSOperator

# Necessary images and ground operators, in reverse order. Also, if the chain
# is not full-length, then the "best" operator that backchaining tried to use
# but couldn't.
_Chain = Tuple[List[Set[GroundAtom]], List[_GroundSTRIPSOperator],
               Optional[_GroundSTRIPSOperator]]


def _update_pnad_seg_to_keep_effs(pnad: PartialNSRTAndDatastore,
                                  necessary_image: Set[GroundAtom],
                                  ground_op: _GroundSTRIPSOperator,
                                  obj_to_var: Dict[Object, Variable],
                                  segment: Segment) -> None:
    """Updates the pnad's seg_to_keep_effs_sub dictionary, which is necesssary
    for correctly grounding keep effects to data."""
    # Every atom in the necessary_image that wasn't in the
    # ground_op's add effects is a possible keep effect. This
    # may add new variables, whose mappings for this segment
    # we keep track of in the seg_to_keep_effects_sub dict.
    for atom in necessary_image - ground_op.add_effects:
        keep_eff_sub = {}
        for obj in atom.objects:
            if obj in obj_to_var:
                continue
            new_var = utils.create_new_variables([obj.type],
                                                 obj_to_var.values())[0]
            obj_to_var[obj] = new_var
            keep_eff_sub[new_var] = obj
        pnad.poss_keep_effects.add(atom.lift(obj_to_var))
        if segment not in pnad.seg_to_keep_effects_sub:
            pnad.seg_to_keep_effects_sub[segment] = {}
        pnad.seg_to_keep_effects_sub[segment].update(keep_eff_sub)


class _EffectSearchOperator(abc.ABC):
    """An operator that proposes successor sets of effect sets."""

    def __init__(
        self, trajectories: List[LowLevelTrajectory], train_tasks: List[Task],
        predicates: Set[Predicate], segmented_trajs: List[List[Segment]],
        backchain: Callable[
            [List[Segment], List[PartialNSRTAndDatastore], Set[GroundAtom]],
            _Chain], associated_heuristic: _EffectSearchHeuristic,
        spawn_new_pnad: Callable[[Segment], PartialNSRTAndDatastore],
        get_pnads_with_keep_effects: Callable[[PartialNSRTAndDatastore],
                                              Set[PartialNSRTAndDatastore]],
        recompute_pnads_from_effects: Callable[
            [Sequence[PartialNSRTAndDatastore]],
            Sequence[PartialNSRTAndDatastore]]
    ) -> None:
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs
        self._backchain = backchain
        self._associated_heuristic = associated_heuristic
        self._spawn_new_pnad = spawn_new_pnad
        self._get_pnads_with_keep_effects = get_pnads_with_keep_effects
        self._recompute_pnads_from_effects = recompute_pnads_from_effects

    @abc.abstractmethod
    def get_successors(
        self, pnads: Sequence[PartialNSRTAndDatastore]
    ) -> Iterator[PartialNSRTAndDatastore]:
        """Generate zero or more successor effect sets."""
        raise NotImplementedError("Override me!")


class _BackChainingEffectSearchOperator(_EffectSearchOperator):
    """An operator that uses backchaining to propose a new effect set."""

    # TODO: Add method that removes all components of all PNADs except add and
    # keep effects. Call this before we get the first uncovered transition?

    def get_successors(
        self, pnads: Sequence[PartialNSRTAndDatastore]
    ) -> Iterator[PartialNSRTAndDatastore]:
        uncovered_segment = self._get_first_uncovered_segment(pnads)
        new_pnads = pnads[:]
        if uncovered_segment is not None:
            # We will need to induce an operator to cover this
            # segment, and thus it must have some necessary add effects.
            assert uncovered_segment.necessary_add_effects is not None
            new_pnad = self._spawn_new_pnad(uncovered_segment)
            new_pnads_with_keep_effs = self._get_pnads_with_keep_effects(
                new_pnad)
            new_pnads_set = {new_pnad} | new_pnads_with_keep_effs
            new_pnads += list(new_pnads_set)
            new_pnads = self._recompute_pnads_from_effects(new_pnads)
            yield new_pnads

    def _get_first_uncovered_segment(
        self,
        pnads: List[PartialNSRTAndDatastore],
    ) -> Optional[Segment]:
        # Find the first uncovered segment. Do this in a kind of breadth-first
        # backward search over trajectories.
        # Compute all the chains once up front.
        backchaining_results = []
        max_chain_len = 0
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if ll_traj.is_demo:
                atoms_seq = utils.segment_trajectory_to_atoms_sequence(
                    seg_traj)
                traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
                chain = self._backchain(seg_traj, pnads, traj_goal)
                max_chain_len = max(max_chain_len, len(chain[1]))
                backchaining_results.append(
                    (seg_traj, atoms_seq, traj_goal, chain))
        # Now look for an uncovered segment. If one cannot be found, this
        # method will automatically return None.
        for depth in range(max_chain_len + 1):
            for seg_traj, atoms_seq, traj_goal, chain in backchaining_results:
                image_chain, op_chain, last_used_op = chain
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


class _PruningEffectSearchOperator(_EffectSearchOperator):
    """An operator that prunes effect sets."""

    def get_successors(
        self, pnads: Sequence[PartialNSRTAndDatastore]
    ) -> Iterator[PartialNSRTAndDatastore]:
        for pnad_to_remove in pnads:
            yield [pnad for pnad in pnads if pnad != pnad_to_remove]


class _EffectSearchHeuristic(abc.ABC):
    """Given a set of effect sets, produce a score, with lower better."""

    def __init__(
        self,
        trajectories: List[LowLevelTrajectory],
        train_tasks: List[Task],
        predicates: Set[Predicate],
        segmented_trajs: List[List[Segment]],
        backchain: Callable[
            [List[Segment], List[PartialNSRTAndDatastore], Set[GroundAtom]],
            _Chain],
    ) -> None:
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs
        self._backchain = backchain
        self._recompute_pnads_from_effects = Callable[
            [Sequence[PartialNSRTAndDatastore]],
            Sequence[PartialNSRTAndDatastore]]
        self._clear_unnecessary_keep_effs_subs = Callable[
            [PartialNSRTAndDatastore], None]

    @abc.abstractmethod
    def __call__(self, curr_pnads: Sequence[PartialNSRTAndDatastore]) -> float:
        """Compute the heuristic value for the given effect sets."""
        raise NotImplementedError("Override me!")


class _BackChainingHeuristic(_EffectSearchHeuristic):
    """Counts the number of transitions that are not yet covered by some
    operator in the backchaining sense."""

    def __call__(self, curr_pnads: Sequence[PartialNSRTAndDatastore]) -> float:
        # Start by recomputing all PNADs from their effects.
        curr_pnads = self._recompute_pnads_from_effects(curr_pnads)
        # Ensure that the unnecessary keep effs sub and poss
        # keep effects are both cleared before backchaining.
        for pnad in curr_pnads:
            pnad.poss_keep_effects.clear()
            self._clear_unnecessary_keep_effs_sub(pnad)

        uncovered_transitions = 0
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if ll_traj.is_demo:
                traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
                _, chain, _ = self._backchain(seg_traj, curr_pnads, traj_goal)
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
        complexity_term = len(curr_pnads)
        return coverage_term + complexity_term


class EffectSearchSTRIPSLearner(GeneralToSpecificSTRIPSLearner):
    """Base class for a effect search STRIPS learner."""

    @classmethod
    def get_name(cls) -> str:
        return "effect_search"

    def _recompute_pnads_from_effects(
        self, pnads: Sequence[PartialNSRTAndDatastore]
    ) -> Sequence[PartialNSRTAndDatastore]:
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
        # TODO: Should we add PNADs with keep effects here? Seems wrong because then
        # say we delete a PNAD from our set and then want to compute the heuristic
        # value of this. We would be potentially adding a bunch of new PNADs just
        # at heuristic computation time (but this is deterministic, so maybe it's
        # correct)?
        # Fix naming.
        pnad_map: Dict[ParameterizedOption, List[PartialNSRTAndDatastore]] = {
            p.option_spec[0]: []
            for p in pnads
        }
        for p in pnads:
            pnad_map[p.option_spec[0]].append(p)
        pnads = self._get_uniquely_named_nec_pnads(pnad_map)

    def _learn(self) -> List[PartialNSRTAndDatastore]:
        # Set up hill-climbing search over effect sets.
        # Create the search operators.
        search_operators = self._create_search_operators()

        # Create the heuristic.
        heuristic = self._create_heuristic()

        # Initialize the search.
        initial_state = []

        def get_successors(
            pnads: Sequence[PartialNSRTAndDatastore]
        ) -> Iterator[Tuple[Tuple[_EffectSearchOperator, int],
                            Sequence[PartialNSRTAndDatastore], float]]:
            for op in search_operators:
                for i, child in enumerate(op.get_successors(pnads)):
                    yield (op, i), child, 1.0  # cost always 1

        # Run hill-climbing search.
        path, _, _ = utils.run_hill_climbing(initial_state=initial_state,
                                             check_goal=lambda _: False,
                                             get_successors=get_successors,
                                             heuristic=heuristic)

        # Extract the best effect set.
        final_pnads = path[-1]
        return final_pnads

    def _create_search_operators(self) -> List[_EffectSearchOperator]:
        op_classes = [
            _BackChainingEffectSearchOperator, _PruningEffectSearchOperator
        ]
        ops = [
            cls(self._trajectories, self._train_tasks,
                self._predicates, self._segmented_trajs, self._backchain,
                self._create_heuristic()) for cls in op_classes
        ]
        return ops

    def _create_heuristic(self) -> _EffectSearchHeuristic:
        backchaining_heur = _BackChainingHeuristic(
            self._trajectories, self._train_tasks, self._predicates,
            self._segmented_trajs, self._backchain,
            self._recompute_pnads_from_effects,
            self._clear_unnecessary_keep_effs_sub)
        return backchaining_heur

    def _backchain(self, segmented_traj: List[Segment],
                   pnads: List[PartialNSRTAndDatastore],
                   traj_goal: Set[GroundAtom]) -> _Chain:
        """Returns chain of ground operators in REVERSE order."""
        image_chain: List[Set[GroundAtom]] = []
        operator_chain: List[_GroundSTRIPSOperator] = []
        final_failed_op: Optional[_GroundSTRIPSOperator] = None
        atoms_seq = utils.segment_trajectory_to_atoms_sequence(segmented_traj)
        objects = set(segmented_traj[0].states[0])
        assert traj_goal.issubset(atoms_seq[-1])
        necessary_image = set(traj_goal)
        image_chain = [necessary_image]
        for t in range(len(atoms_seq) - 2, -1, -1):
            segment = segmented_traj[t]
            segment.necessary_image = necessary_image
            segment.necessary_add_effects = necessary_image - atoms_seq[t]
            pnad, var_to_obj = self._find_best_matching_pnad_and_sub(
                segment, objects, pnads)

            # If no match found, terminate.
            if pnad is None:
                break

            # TODO: Understand why TF this assertion fails so much!
            # # Assert that this segment is in the PNAD's datastore.
            # segs_in_pnad = {
            #     datapoint[0]
            #     for datapoint in pnad.datastore
            # }
            # assert segment in segs_in_pnad

            assert var_to_obj is not None
            obj_to_var = {v: k for k, v in var_to_obj.items()}
            assert len(var_to_obj) == len(obj_to_var)
            ground_op = pnad.op.ground(
                tuple(var_to_obj[var] for var in pnad.op.parameters))
            next_atoms = utils.apply_operator(ground_op, segment.init_atoms)
            # Update the PNAD's seg_to_keep_effs_sub dict.
            _update_pnad_seg_to_keep_effs(pnad, necessary_image, ground_op,
                                          obj_to_var, segment)
            # If we're missing something in the necessary image, terminate.
            if not necessary_image.issubset(next_atoms):
                final_failed_op = ground_op
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
            image_chain.append(necessary_image)
        return (image_chain, operator_chain, final_failed_op)
