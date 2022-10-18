"""Base class for a STRIPS operator learning algorithm."""

import abc
import logging
from typing import Dict, List, Optional, Set, Tuple

from predicators import utils
from predicators.planning import task_plan_with_option_plan_constraint
from predicators.settings import CFG
from predicators.structs import DummyOption, GroundAtom, LiftedAtom, \
    LowLevelTrajectory, Object, OptionSpec, PartialNSRTAndDatastore, \
    Predicate, Segment, State, STRIPSOperator, Task, Variable, \
    _GroundSTRIPSOperator


class BaseSTRIPSLearner(abc.ABC):
    """Base class definition."""

    def __init__(
        self,
        trajectories: List[LowLevelTrajectory],
        train_tasks: List[Task],
        predicates: Set[Predicate],
        segmented_trajs: List[List[Segment]],
        verify_harmlessness: bool,
        verbose: bool = True,
    ) -> None:
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs
        self._verify_harmlessness = verify_harmlessness
        self._verbose = verbose
        self._num_segments = sum(len(t) for t in segmented_trajs)
        assert len(self._trajectories) == len(self._segmented_trajs)

    def learn(self) -> List[PartialNSRTAndDatastore]:
        """The public method for a STRIPS operator learning strategy.

        A wrapper around self._learn() to sanity check that harmlessness
        holds on the training data, and then filter out operators
        without enough data. We check harmlessness first because
        filtering may break it.
        """
        learned_pnads = self._learn()
        if self._verify_harmlessness and not CFG.disable_harmlessness_check:
            logging.info("\nRunning harmlessness check...")
            assert self._check_harmlessness(learned_pnads)
        # Remove pnads by increasing min_data_perc until harmlessness breaks.
        if CFG.enable_harmless_op_pruning:
            assert self._verify_harmlessness
            assert not CFG.disable_harmlessness_check
            # Keeps track of latest set of harmless pnads.
            min_harmless_pnads = learned_pnads
            # Find the percentage of data in each PNAD uses from lowest
            # to highest.
            pnad_perc_data_low_to_high = [
                len(pnad.datastore) / float(self._num_segments)
                for pnad in learned_pnads
            ]
            pnad_perc_data_low_to_high.sort()
        else:
            # If we are not doing harmless operator pruning, return
            # PNADs at current min_perc_data_for_nsrts.
            pnad_perc_data_low_to_high = [CFG.min_perc_data_for_nsrt / 100.0]
        # Iterates over each PNAD in the learned PNADs removing the
        # PNAD that uses the least amount of data.
        for min_perc_data_for_nsrt in pnad_perc_data_low_to_high:
            learned_pnads = self._learn()
            min_data = max(CFG.min_data_for_nsrt,
                           self._num_segments * min_perc_data_for_nsrt)
            learned_pnads = [
                pnad for pnad in learned_pnads
                if len(pnad.datastore) >= min_data
            ]
            if not CFG.enable_harmless_op_pruning:
                # If we are not doing harmless operator pruning, return
                # PNADs at current min_perc_data_for_nsrts.
                return learned_pnads
            # Runs harmlessness check after we have pruned operators.
            logging.info("\nRunning harmlessness check...")
            if not self._check_harmlessness(learned_pnads):
                break
            # We successfully verified harmlessness than we save this set of
            # pnads and continue reducing min_perc_data_for_nsrts.
            min_harmless_pnads = learned_pnads
        learned_pnads = min_harmless_pnads
        return learned_pnads

    @abc.abstractmethod
    def _learn(self) -> List[PartialNSRTAndDatastore]:
        """The key method that a STRIPS operator learning strategy must
        implement.

        Returns a new list of PNADs learned from the data, with op
        (STRIPSOperator), datastore, and option_spec fields filled in
        (but not sampler).
        """
        raise NotImplementedError("Override me!")

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this STRIPS learner, used as the
        strips_learner setting in settings.py."""
        raise NotImplementedError("Override me!")

    def _check_harmlessness(self,
                            pnads: List[PartialNSRTAndDatastore]) -> bool:
        """Function to check whether the given PNADs holistically preserve
        harmlessness over demonstrations on the training tasks.

        Preserving harmlessness roughly means that the set of operators
        and predicates supports the agent's ability to plan to achieve
        all of the training tasks in the same way as was demonstrated
        (i.e., the predicates and operators don't render any
        demonstrated trajectory impossible).
        """
        strips_ops = [pnad.op for pnad in pnads]
        option_specs = [pnad.option_spec for pnad in pnads]
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            atoms_seq = utils.segment_trajectory_to_atoms_sequence(seg_traj)
            task = self._train_tasks[ll_traj.train_task_idx]
            traj_goal = task.goal
            if not traj_goal.issubset(atoms_seq[-1]):
                # In this case, the goal predicates are not correct (e.g.,
                # we are learning them), so we skip this demonstration.
                continue
            demo_preserved = self._check_single_demo_preservation(
                seg_traj, ll_traj.states[0], atoms_seq, traj_goal, strips_ops,
                option_specs)
            if not demo_preserved:
                logging.debug("Harmlessness not preserved for demo!")
                logging.debug(f"Initial atoms: {atoms_seq[0]}")
                for t in range(1, len(atoms_seq)):
                    logging.debug(f"Timestep {t} add effects: "
                                  f"{atoms_seq[t] - atoms_seq[t-1]}")
                    logging.debug(f"Timestep {t} del effects: "
                                  f"{atoms_seq[t-1] - atoms_seq[t]}")
                return False
        return True

    def _check_single_demo_preservation(
            self, seg_traj: List[Segment], init_state: State,
            atoms_seq: List[Set[GroundAtom]], traj_goal: Set[GroundAtom],
            strips_ops: List[STRIPSOperator],
            option_specs: List[OptionSpec]) -> bool:
        """Function to check whether a given set of operators preserves a
        single training trajectory."""
        init_atoms = utils.abstract(init_state, self._predicates)
        objects = set(init_state)
        option_plan = []
        for seg in seg_traj:
            if seg.has_option():
                option = seg.get_option()
            else:
                option = DummyOption
            option_plan.append((option.parent, option.objects))
        ground_nsrt_plan = task_plan_with_option_plan_constraint(
            objects, self._predicates, strips_ops, option_specs, init_atoms,
            traj_goal, option_plan, atoms_seq)
        return ground_nsrt_plan is not None

    def _recompute_datastores_from_segments(
            self, pnads: List[PartialNSRTAndDatastore]) -> None:
        """For the given PNADs, wipe and recompute the datastores.

        Uses a "rationality" heuristic, where for each segment, we
        select, among the ground PNADs covering it, the one whose add
        and delete effects match the segment's most closely (breaking
        ties arbitrarily). At the end of this procedure, each segment is
        guaranteed to be in at most one PNAD's datastore.
        """
        for pnad in pnads:
            pnad.datastore = []  # reset all PNAD datastores
        # Note: we want to loop over all segments, NOT just the ones
        # associated with demonstrations.
        for seg_traj in self._segmented_trajs:
            objects = set(seg_traj[0].states[0])
            for segment in seg_traj:
                best_pnad, best_sub = self._find_best_matching_pnad_and_sub(
                    segment, objects, pnads)
                if best_pnad is not None:
                    assert best_sub is not None
                    best_pnad.add_to_datastore((segment, best_sub),
                                               check_effect_equality=False)

    def _find_best_matching_pnad_and_sub(
        self,
        segment: Segment,
        objects: Set[Object],
        pnads: List[PartialNSRTAndDatastore],
        check_only_preconditions: bool = False
    ) -> Tuple[Optional[PartialNSRTAndDatastore], Optional[Dict[Variable,
                                                                Object]]]:
        """Find the best matching PNAD (if any) given our rationality-based
        score function, and return the PNAD and substitution necessary to
        ground it. If no PNAD from the input list matches the segment, then
        return Nones.

        If check_only_preconditions is True, we must be calling this function
        during spawning of a new PNAD during backchaining. In this case,
        we want to find a grounding whose preconditions are satisfied in
        the segment.init_atoms. Otherwise, we want to find a grounding that
        not only satisfies the above check, but also is such that calling
        utils.apply_operator() from the segment.init_atoms results in a subset
        of the segment's final atoms, and - if the
        segment.necessary_add_effects are not None - that these are satisfied
        by calling utils.apply_operator() from the segment.init_atoms. This
        effectively checks that the grounding can be applied to this segment
        in a harmless way.
        """
        if segment.has_option():
            segment_option = segment.get_option()
        else:
            segment_option = DummyOption
        segment_param_option = segment_option.parent
        segment_option_objs = tuple(segment_option.objects)
        # Loop over all ground operators, looking for the most
        # rational match for this segment.
        best_score = float("inf")
        best_pnad = None
        best_sub = None
        for pnad in pnads:
            param_opt, opt_vars = pnad.option_spec
            if param_opt != segment_param_option:
                continue
            isub = dict(zip(opt_vars, segment_option_objs))
            if segment in pnad.seg_to_keep_effects_sub:
                # If there are any variables only in the keep effects,
                # their mappings should be put into isub, since their
                # grounding is underconstrained by the segment itself.
                keep_eff_sub = pnad.seg_to_keep_effects_sub[segment]
                for var in pnad.op.parameters:
                    if var in keep_eff_sub:
                        assert var not in isub
                        isub[var] = keep_eff_sub[var]
            for ground_op in utils.all_ground_operators_given_partial(
                    pnad.op, objects, isub):
                if len(ground_op.objects) != len(set(ground_op.objects)):
                    continue
                # If the preconditions don't hold in the segment's
                # initial atoms, skip.
                if not ground_op.preconditions.issubset(segment.init_atoms):
                    continue
                next_atoms = utils.apply_operator(ground_op,
                                                  segment.init_atoms)
                if not check_only_preconditions:
                    # If the atoms resulting from apply_operator() don't
                    # all hold in the segment's final atoms, skip.
                    if not next_atoms.issubset(segment.final_atoms):
                        continue
                    # If the segment has a non-None necessary_add_effects,
                    # and the ground operator's add effects don't fit this,
                    # skip.
                    if segment.necessary_add_effects is not None and \
                       not segment.necessary_add_effects.issubset(
                           ground_op.add_effects):
                        continue
                else:
                    # If check_only_preconditions is True, we must be
                    # calling this from spawning during backchaining
                    # with a most-general PNAD that has no add effects
                    # and all other predicates sidelined, and thus this
                    # assertion must hold.
                    assert next_atoms.issubset(segment.final_atoms)
                # This ground PNAD covers this segment. Score it!
                score = self._score_segment_ground_op_match(segment, ground_op)
                if score < best_score:  # we want a closer match
                    best_score = score
                    best_pnad = pnad
                    best_sub = dict(zip(pnad.op.parameters, ground_op.objects))
        return best_pnad, best_sub

    @staticmethod
    def _score_segment_ground_op_match(
            segment: Segment, ground_op: _GroundSTRIPSOperator) -> float:
        """Return a score for how well the given segment matches the given
        ground operator, used in recompute_datastores_from_segments().

        A lower score is a CLOSER match. We use a heuristic to estimate
        the quality of the match, where we check how many ground atoms
        are different between the segment's add/delete effects and the
        operator's add/delete effects. However, we must be careful to
        treat keep effects specially, since they will not appear in
        segment.add_effects. In general, we favor more keep effects
        (hence we subtract len(keep_effects)), since we can only ever
        call this function on ground operators whose preconditions are
        satisfied in segment.init_atoms.
        """
        keep_effects = ground_op.preconditions & ground_op.add_effects
        nonkeep_add_effects = ground_op.add_effects - keep_effects
        return len(segment.add_effects - nonkeep_add_effects) + \
            len(nonkeep_add_effects - segment.add_effects) + \
            len(segment.delete_effects - ground_op.delete_effects) + \
            len(ground_op.delete_effects - segment.delete_effects) - \
            len(keep_effects)

    @staticmethod
    def _induce_preconditions_via_intersection(
            pnad: PartialNSRTAndDatastore) -> Set[LiftedAtom]:
        """Given a PNAD with a nonempty datastore, compute the preconditions
        for the PNAD's operator by intersecting all lifted preimages."""
        assert len(pnad.datastore) > 0
        for i, (segment, var_to_obj) in enumerate(pnad.datastore):
            objects = set(var_to_obj.values())
            obj_to_var = {o: v for v, o in var_to_obj.items()}
            atoms = {
                atom
                for atom in segment.init_atoms
                if all(o in objects for o in atom.objects)
            }
            lifted_atoms = {atom.lift(obj_to_var) for atom in atoms}
            if i == 0:
                preconditions = lifted_atoms
            else:
                preconditions &= lifted_atoms
        return preconditions
