"""Methods for learning to sideline predicates in NSRTs."""

import abc
from typing import FrozenSet, Iterator, List, Optional, Set, Tuple

from predicators.src import utils
from predicators.src.nsrt_learning.strips_learning import \
    induce_pnad_preconditions
from predicators.src.planning import task_plan_grounding
from predicators.src.predicate_search_score_functions import \
    _PredictionErrorScoreFunction
from predicators.src.settings import CFG
from predicators.src.structs import GroundAtom, LiftedAtom, \
    LowLevelTrajectory, OptionSpec, ParameterizedOption, \
    PartialNSRTAndDatastore, Predicate, Segment, STRIPSOperator, Task, \
    _GroundNSRT


class SidePredicateLearner(abc.ABC):
    """Base class for a side predicate learning strategy."""

    def __init__(self, initial_pnads: List[PartialNSRTAndDatastore],
                 trajectories: List[LowLevelTrajectory],
                 train_tasks: List[Task], predicates: Set[Predicate],
                 segmented_trajs: List[List[Segment]]) -> None:
        self._initial_pnads = initial_pnads
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs

    def sideline(self) -> List[PartialNSRTAndDatastore]:
        """The public method for a side predicate learning strategy.

        A simple wrapper around self._sideline() that remembers to call
        self._recompute_datastores_from_segments() afterward.
        """
        sidelined_pnads = self._sideline()
        # Recompute the datastores in the PNADs. We need to do this
        # because when we have side predicates, each transition may be
        # assigned to *multiple* datastores.
        self._recompute_datastores_from_segments(sidelined_pnads)
        return sidelined_pnads

    @abc.abstractmethod
    def _sideline(self) -> List[PartialNSRTAndDatastore]:
        """The key method that a side predicate learning strategy must
        implement.

        Returns a new list of PNADs with sidelining applied as desired.
        Note that self._initial_pnads is not modified.
        """
        raise NotImplementedError("Override me!")

    def _recompute_datastores_from_segments(
            self, sidelined_pnads: List[PartialNSRTAndDatastore]) -> None:
        for pnad in sidelined_pnads:
            pnad.datastore = []  # reset all PNAD datastores
        for seg_traj in self._segmented_trajs:
            objects = set(seg_traj[0].states[0])
            for segment in seg_traj:
                assert segment.has_option()
                segment_option = segment.get_option()
                segment_param_option = segment_option.parent
                segment_option_objs = tuple(segment_option.objects)
                # Get ground operators given these objects and option objs.
                for pnad in sidelined_pnads:
                    param_opt, opt_vars = pnad.option_spec
                    if param_opt != segment_param_option:
                        continue
                    isub = dict(zip(opt_vars, segment_option_objs))
                    # Consider adding this segment to each datastore.
                    for ground_op in utils.all_ground_operators_given_partial(
                            pnad.op, objects, isub):
                        # Check if preconditions hold.
                        if not ground_op.preconditions.issubset(
                                segment.init_atoms):
                            continue
                        # Check if effects match. Note that we're using the side
                        # predicates semantics here!
                        atoms = utils.apply_operator(ground_op,
                                                     segment.init_atoms)
                        if not atoms.issubset(segment.final_atoms):
                            continue
                        # Skip over segments that have multiple possible
                        # bindings.
                        if (len(set(ground_op.objects)) != len(
                                ground_op.objects)):
                            continue
                        # This segment belongs in this datastore, so add it.
                        sub = dict(zip(pnad.op.parameters, ground_op.objects))
                        pnad.add_to_datastore((segment, sub),
                                              check_effect_equality=False)


class HillClimbingSidePredicateLearner(SidePredicateLearner):
    """An abstract side predicate learning strategy that performs hill climbing
    over candidate sidelinings of add effects, one at a time.

    Leaves the evaluation function unspecified.
    """

    def _sideline(self) -> List[PartialNSRTAndDatastore]:
        # Run the search, starting from original PNADs.
        path, _, _ = utils.run_hill_climbing(tuple(self._initial_pnads),
                                             self._check_goal,
                                             self._get_successors,
                                             self._evaluate)
        # The last state in the search holds the final PNADs.
        return list(path[-1])

    @abc.abstractmethod
    def _evaluate(self, state: Tuple[PartialNSRTAndDatastore, ...]) -> float:
        """Abstract evaluation/score function for search.

        Lower is better.
        """
        raise NotImplementedError("Override me!")

    @staticmethod
    def _check_goal(s: Tuple[PartialNSRTAndDatastore, ...]) -> bool:
        del s  # unused
        # There are no goal states for this search; run until exhausted.
        return False

    @staticmethod
    def _get_successors(
        s: Tuple[PartialNSRTAndDatastore, ...],
    ) -> Iterator[Tuple[None, Tuple[PartialNSRTAndDatastore, ...], float]]:
        # For each PNAD/operator...
        for i in range(len(s)):
            pnad = s[i]
            _, option_vars = pnad.option_spec
            # ...consider changing each of its add effects to a side predicate.
            for effect in pnad.op.add_effects:
                if len(pnad.op.add_effects) > 1:
                    # We don't want sidelining to result in a no-op.
                    new_pnad = PartialNSRTAndDatastore(
                        pnad.op.effect_to_side_predicate(
                            effect, option_vars, "add"), pnad.datastore,
                        pnad.option_spec)
                    sprime = list(s)
                    sprime[i] = new_pnad
                    yield (None, tuple(sprime), 1.0)

            # ...consider removing it.
            sprime = list(s)
            del sprime[i]
            yield (None, tuple(sprime), 1.0)


class PredictionErrorHillClimbingSidePredicateLearner(
        HillClimbingSidePredicateLearner):
    """A side predicate learning strategy that does hill climbing with a
    prediction error score function."""

    def __init__(self, pnads: List[PartialNSRTAndDatastore],
                 trajectories: List[LowLevelTrajectory],
                 train_tasks: List[Task], predicates: Set[Predicate],
                 segmented_trajs: List[List[Segment]]) -> None:
        super().__init__(pnads, trajectories, train_tasks, predicates,
                         segmented_trajs)
        self._score_func = _PredictionErrorScoreFunction(
            self._predicates, [], {}, self._train_tasks)

    def _evaluate(self, state: Tuple[PartialNSRTAndDatastore, ...]) -> float:
        strips_ops = [pnad.op for pnad in state]
        option_specs = [pnad.option_spec for pnad in state]
        score = self._score_func.evaluate_with_operators(
            frozenset(), self._trajectories, self._segmented_trajs, strips_ops,
            option_specs)
        return score


class PreserveSkeletonsHillClimbingSidePredicateLearner(
        HillClimbingSidePredicateLearner):
    """A side predicate learning strategy that does hill climbing with a
    skeleton preservation (harmlessness) score function."""

    def _evaluate(self, state: Tuple[PartialNSRTAndDatastore, ...]) -> float:
        preserves_harmlessness = self._check_harmlessness(state)
        if preserves_harmlessness:
            # If harmlessness is preserved, the score is the number of
            # operators in the state, minus the number of side predicates.
            # This means we prefer fewer operators and more side predicates.
            score = 2 * len(state)
            for pnad in state:
                score -= len(pnad.op.side_predicates)
        else:
            # If harmlessness is not preserved, the score is an arbitrary
            # constant bigger than the total number of operators at the
            # start of the search. This is guaranteed to be worse (higher)
            # than any score that occurs if harmlessness is preserved.
            score = 10 * len(self._initial_pnads)
        return score

    def _check_harmlessness(
            self, state: Tuple[PartialNSRTAndDatastore, ...]) -> bool:
        """Function to check whether the given state in the search preserves
        harmlessness over demonstrations on the training tasks.

        Preserving harmlessness roughly means that the set of operators
        and predicates supports the agent's ability to plan to achieve
        all of the training tasks in the same way as was demonstrated
        (i.e., the predicates and operators don't render any
        demonstrated trajectory impossible).
        """
        strips_ops = [pnad.op for pnad in state]
        option_specs = [pnad.option_spec for pnad in state]
        assert len(self._trajectories) == len(self._segmented_trajs)
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            atoms_seq = utils.segment_trajectory_to_atoms_sequence(seg_traj)
            traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
            demo_preserved = self._check_single_demo_preservation(
                ll_traj, atoms_seq, traj_goal, strips_ops, option_specs)
            if not demo_preserved:
                return False
        return True

    def _check_single_demo_preservation(
            self, ll_traj: LowLevelTrajectory,
            atoms_seq: List[Set[GroundAtom]], traj_goal: Set[GroundAtom],
            strips_ops: List[STRIPSOperator],
            option_specs: List[OptionSpec]) -> bool:
        """Function to check whether a given set of operators and predicates
        preserves a single training trajectory."""
        init_atoms = utils.abstract(ll_traj.states[0], self._predicates)
        objects = set(ll_traj.states[0])
        ground_nsrts, _ = task_plan_grounding(init_atoms, objects, strips_ops,
                                              option_specs)
        heuristic = utils.create_task_planning_heuristic(
            CFG.sesame_task_planning_heuristic, init_atoms, traj_goal,
            ground_nsrts, self._predicates, objects)

        def _check_goal(state: Tuple[FrozenSet[GroundAtom], int]) -> bool:
            return traj_goal.issubset(state[0])

        def _get_successor_with_correct_option(
            searchnode_state: Tuple[FrozenSet[GroundAtom], int]
        ) -> Iterator[Tuple[_GroundNSRT, Tuple[FrozenSet[GroundAtom], int],
                            float]]:
            state = searchnode_state[0]
            idx_into_traj = searchnode_state[1]

            if idx_into_traj > len(ll_traj.actions) - 1:
                return

            gt_option = ll_traj.actions[idx_into_traj].get_option()
            expected_next_hl_state = atoms_seq[idx_into_traj + 1]

            for applicable_nsrt in utils.get_applicable_operators(
                    ground_nsrts, state):
                # NOTE: we check that the ParameterizedOptions are equal before
                # attempting to ground because otherwise, we might
                # get a parameter mismatch and trigger an AssertionError
                # during grounding.
                if applicable_nsrt.option != gt_option.parent:
                    continue
                if applicable_nsrt.option_objs != gt_option.objects:
                    continue
                next_hl_state = utils.apply_operator(applicable_nsrt,
                                                     set(state))
                if next_hl_state.issubset(expected_next_hl_state):
                    # The returned cost is uniform because we don't
                    # actually care about finding the shortest path;
                    # just one that matches!
                    yield (applicable_nsrt, (frozenset(next_hl_state),
                                             idx_into_traj + 1), 1.0)

        init_atoms_frozen = frozenset(init_atoms)
        init_searchnode_state = (init_atoms_frozen, 0)
        # NOTE: each state in the below GBFS is a tuple of
        # (current_atoms, idx_into_traj). The idx_into_traj is necessary because
        # we need to check whether the atoms that are true at this particular
        # index into the trajectory is what we would expect given the demo
        # trajectory.
        state_seq, _ = utils.run_gbfs(
            init_searchnode_state, _check_goal,
            _get_successor_with_correct_option,
            lambda searchnode_state: heuristic(searchnode_state[0]))

        return _check_goal(state_seq[-1])


class BackchainingSidePredicateLearner(SidePredicateLearner):
    """Sideline by backchaining, making PNADs increasingly specific."""

    def _sideline(self) -> List[PartialNSRTAndDatastore]:
        # Initialize the most general PNADs by merging self._initial_pnads.
        # As a result, we will have one very general PNAD per option.
        param_opt_to_pnad = {}
        parameterized_options = {p.option_spec[0] for p in self._initial_pnads}
        for param_opt in parameterized_options:
            pnad = self._initialize_general_pnad_for_option(param_opt)
            param_opt_to_pnad[param_opt] = pnad
        del self._initial_pnads  # no longer used
        # Go through each demonstration from the end back to the start,
        # making the PNADs more specific whenever needed.
        assert len(self._trajectories) == len(self._segmented_trajs)
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
            atoms_seq = utils.segment_trajectory_to_atoms_sequence(seg_traj)
            assert traj_goal.issubset(atoms_seq[-1])
            # This variable, necessary_image, gets updated as we
            # backchain. It always holds the set of ground atoms that
            # are necessary for the remainder of the plan to reach the
            # goal. At the start, necessary_image is simply the goal.
            necessary_image = set(traj_goal)
            for t in range(len(atoms_seq) - 2, -1, -1):
                segment = seg_traj[t]
                option = segment.get_option()
                # Find the PNAD associated with this option.
                pnad = param_opt_to_pnad[option.parent]
                var_to_obj = pnad.get_sub_for_member_segment(segment)
                # Compute the ground atoms that must be added on this timestep.
                necessary_add_effects = necessary_image - atoms_seq[t]
                # Compute the add effects that this PNAD currently predicts.
                predicted_add_effects = {
                    a.ground(var_to_obj)
                    for a in pnad.op.add_effects
                }
                # If the predicted add effects are missing necessary add
                # effects, then we need to make the add effects more specific.
                if not necessary_add_effects.issubset(predicted_add_effects):
                    obj_to_var = {o: v for v, o in var_to_obj.items()}
                    assert len(obj_to_var) == len(var_to_obj)
                    updated_add_effects = pnad.op.add_effects | {
                        a.lift(obj_to_var)
                        for a in necessary_add_effects
                    }
                    pnad = self._update_pnad_add_effects(
                        pnad, updated_add_effects)
                    param_opt_to_pnad[option.parent] = pnad
                    # Note that the parameters for the PNAD may have changed.
                    # So we need to re-obtain var_to_obj.
                    var_to_obj = pnad.get_sub_for_member_segment(segment)
                    predicted_add_effects = {
                        a.ground(var_to_obj)
                        for a in pnad.op.add_effects
                    }
                    # After we just made the add effects more specific, the
                    # predicted add effects should cover the necessary ones.
                    assert necessary_add_effects.issubset(
                        predicted_add_effects)
                # Update necessary_image for this timestep. It no longer
                # needs to include the add effects of this PNAD, but
                # must now include its preconditions.
                necessary_image -= predicted_add_effects
                necessary_image |= {
                    a.ground(var_to_obj)
                    for a in pnad.op.preconditions
                }

        return list(param_opt_to_pnad.values())

    def _initialize_general_pnad_for_option(
            self, parameterized_option: ParameterizedOption
    ) -> PartialNSRTAndDatastore:

        # There could be multiple PNADs in self._initial_pnads corresponding
        # to this option. The goal of this function will be to merge them into
        # a new, very general PNAD.
        initial_pnads_for_option = [
            p for p in self._initial_pnads
            if p.option_spec[0] == parameterized_option
        ]
        assert len(initial_pnads_for_option) > 0

        # The PNAD's side predicates are computed by unioning all add effects
        # or unpredicted delete effects in the initial PNADs for this option.
        # Example for unpredicted delete effects: NextToNothing sometimes, but
        # not always, appears as a delete effect of Move in RepeatedNextToEnv.
        # We need to include this as a side predicate because otherwise, if
        # the agent starts in a state where NextToNothing holds and then moves,
        # it would expect that NextToNothing still holds in the next state,
        # unless we include NextToNothing as a delete effect or side predicate.

        # The PNAD's delete effects are complicated. For now, we intersect,
        # representing all delete effects that always follow the execution of
        # the parameterized option. But we need to do this in a lifted way. That
        # in turn is complicated because to perform the lifted intersection,
        # we would need to consistently lift all atoms in the same way so that
        # they have the same variables. Unlike in normal NSRT learning, this
        # consistency would need to work across multiple PNADs. (Normally, we
        # just need to lift consistently within a single PNAD.) To make this
        # simple for now, we will only consider delete effects where all its
        # object are option objects, for the sake of computing this lifted
        # delete effect intersection. It should be possible to do better.

        # First we determine the delete effects, as described above, to figure
        # out which of the remaining delete effects are unpredicted (and thus
        # need to be sidelined, as described in the first paragraph above).
        # Arbitrarily choose an initial PNAD whose option spec will be used
        # as the option spec of the new PNAD.
        new_option_spec = initial_pnads_for_option[0].option_spec
        delete_effects: Optional[Set[LiftedAtom]] = None
        # For reuse in sidelining.
        all_segment_lifted_delete_effects = set()
        for initial_pnad in initial_pnads_for_option:
            assert initial_pnad.option_spec[0] == new_option_spec[0]
            # Map this PNAD's option spec variables to the variables
            # of the new PNAD's option spec. This ensures the
            # variables are consistent across all PNADs.
            var_to_remapped_var = dict(
                zip(initial_pnad.option_spec[1], new_option_spec[1]))
            for (seg, sub) in initial_pnad.datastore:
                # Remap the variables in this PNAD to be consistent
                # with the variables in the new PNAD.
                remapped_var_to_obj = {
                    var_to_remapped_var[v]: o
                    for v, o in sub.items() if v in var_to_remapped_var
                }
                # Determine the delete effects.
                obj_to_remapped_var = {
                    o: v
                    for v, o in remapped_var_to_obj.items()
                }
                assert len(obj_to_remapped_var) == len(remapped_var_to_obj)
                seg_lifted_del_effs = {
                    a.lift(obj_to_remapped_var)
                    for a in seg.delete_effects
                    # Ignore delete effects that have objects not
                    # contained in the option objects.
                    if set(a.objects).issubset(obj_to_remapped_var)
                }
                # Intersect the lifted delete effects over the PNADs.
                if delete_effects is None:
                    delete_effects = seg_lifted_del_effs
                else:
                    delete_effects &= seg_lifted_del_effs
                all_segment_lifted_delete_effects.add(
                    frozenset(seg_lifted_del_effs))
        assert delete_effects is not None

        # We're ready to compute the side predicates.
        # First, sideline all predicates of all add effects.
        side_predicates = set()
        for initial_pnad in initial_pnads_for_option:
            for (seg, _) in initial_pnad.datastore:
                for ground_atom in seg.add_effects:
                    side_predicates.add(ground_atom.predicate)
        # Now, sideline all predicates of all unpredicted delete effects.
        for lifted_delete_effects in all_segment_lifted_delete_effects:
            for lifted_atom in lifted_delete_effects - delete_effects:
                side_predicates.add(lifted_atom.predicate)

        # Construct and return the PNAD. This PNAD has trivial add effects.
        return self._construct_pnad_and_recompute(parameterized_option.name,
                                                  set(), delete_effects,
                                                  side_predicates,
                                                  new_option_spec)

    def _update_pnad_add_effects(
            self, pnad: PartialNSRTAndDatastore,
            add_effects: Set[LiftedAtom]) -> PartialNSRTAndDatastore:
        """Return a new PNAD resulting from changing the add effects in the
        given PNAD to the given add_effects."""
        # Recompute the side predicates by collecting all unpredicted effects
        # of the segments in the datastore.
        side_predicates = set()
        for seg, sub in pnad.datastore:
            predicted_add_effects = {a.ground(sub) for a in add_effects}
            for atom in seg.add_effects - predicted_add_effects:
                side_predicates.add(atom.predicate)
            predicted_delete_effects = {
                a.ground(sub)
                for a in pnad.op.delete_effects
            }
            for atom in seg.delete_effects - predicted_delete_effects:
                side_predicates.add(atom.predicate)
        # Note that in general, we may need to recompute the datastore and
        # preconditions, even though the segments don't change, because the
        # parameters may change, so the VarToObjSub may also change.
        return self._construct_pnad_and_recompute(pnad.op.name, add_effects,
                                                  pnad.op.delete_effects,
                                                  side_predicates,
                                                  pnad.option_spec)

    def _construct_pnad_and_recompute(
            self, name: str, add_effects: Set[LiftedAtom],
            delete_effects: Set[LiftedAtom], side_predicates: Set[Predicate],
            option_spec: OptionSpec) -> PartialNSRTAndDatastore:
        """Construct a PNAD with trivial datastore and preconditions, then
        recompute both."""
        parameter_set = {
            v
            for a in add_effects | delete_effects for v in a.variables
        }
        parameter_set |= set(option_spec[1])
        parameters = sorted(parameter_set)
        op = STRIPSOperator(name, parameters, set(), add_effects,
                            delete_effects, side_predicates)
        pnad = PartialNSRTAndDatastore(op, [], option_spec)
        # Recompute datastore.
        self._recompute_datastores_from_segments([pnad])
        # Determine the preconditions.
        preconditions = induce_pnad_preconditions(pnad)
        # Finalize and return PNAD.
        new_op = STRIPSOperator(name, parameters, preconditions, add_effects,
                                delete_effects, side_predicates)
        return PartialNSRTAndDatastore(new_op, pnad.datastore, option_spec)
