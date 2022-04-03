"""Methods for learning to sideline predicates in NSRTs."""

import abc
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

from predicators.src import utils
from predicators.src.nsrt_learning.strips_learning import \
    induce_pnad_preconditions
from predicators.src.planning import task_plan_grounding
from predicators.src.predicate_search_score_functions import \
    _PredictionErrorScoreFunction
from predicators.src.settings import CFG
from predicators.src.structs import GroundAtom, LiftedAtom, \
    LowLevelTrajectory, OptionSpec, ParameterizedOption, \
    PartialNSRTAndDatastore, Predicate, Segment, State, STRIPSOperator, Task, \
    Variable, _GroundNSRT, _GroundSTRIPSOperator


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
            self,
            pnads: List[PartialNSRTAndDatastore],
            semantics: str = "apply_operator") -> None:
        """For the given PNADs, wipe and recompute the datastores.

        If semantics is "apply_operator", then a segment is included in
        a datastore if, for some ground PNAD, the preconditions are
        satisfied and apply_operator() results in an abstract next state
        that is a subset of the segment's final_atoms. If semantics is
        "add_effects", then rather than using apply_operator(), we
        simply check that the add effects are a subset of the segment's
        add effects.
        """
        assert semantics in ("apply_operator", "add_effects")
        for pnad in pnads:
            pnad.datastore = []  # reset all PNAD datastores
        for seg_traj in self._segmented_trajs:
            objects = set(seg_traj[0].states[0])
            for segment in seg_traj:
                assert segment.has_option()
                segment_option = segment.get_option()
                segment_param_option = segment_option.parent
                segment_option_objs = tuple(segment_option.objects)
                # Get ground operators given these objects and option objs.
                for pnad in pnads:
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
                        # Check if effects match.
                        if semantics == "apply_operator":
                            atoms = utils.apply_operator(
                                ground_op, segment.init_atoms)
                            if not atoms.issubset(segment.final_atoms):
                                continue
                        elif semantics == "add_effects":
                            if not ground_op.add_effects.issubset(
                                    segment.add_effects):
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
        assert len(self._trajectories) == len(self._segmented_trajs)
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            atoms_seq = utils.segment_trajectory_to_atoms_sequence(seg_traj)
            traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
            demo_preserved = self._check_single_demo_preservation(
                seg_traj, ll_traj.states[0], atoms_seq, traj_goal, strips_ops,
                option_specs)
            if not demo_preserved:
                return False
        return True

    def _check_single_demo_preservation(
            self, seg_traj: List[Segment], init_state: State,
            atoms_seq: List[Set[GroundAtom]], traj_goal: Set[GroundAtom],
            strips_ops: List[STRIPSOperator],
            option_specs: List[OptionSpec]) -> bool:
        """Function to check whether a given set of operators and predicates
        preserves a single training trajectory."""
        init_atoms = utils.abstract(init_state, self._predicates)
        objects = set(init_state)
        options = [seg.get_option() for seg in seg_traj]
        ground_nsrts, _ = task_plan_grounding(init_atoms,
                                              objects,
                                              strips_ops,
                                              option_specs,
                                              allow_noops=True)
        heuristic = utils.create_task_planning_heuristic(
            CFG.sesame_task_planning_heuristic, init_atoms, traj_goal,
            ground_nsrts, self._predicates, objects)

        def _check_goal(
                searchnode_state: Tuple[FrozenSet[GroundAtom], int]) -> bool:
            return traj_goal.issubset(searchnode_state[0])

        def _get_successor_with_correct_option(
            searchnode_state: Tuple[FrozenSet[GroundAtom], int]
        ) -> Iterator[Tuple[_GroundNSRT, Tuple[FrozenSet[GroundAtom], int],
                            float]]:
            atoms = searchnode_state[0]
            idx_into_traj = searchnode_state[1]

            if idx_into_traj > len(options) - 1:
                return

            gt_option = options[idx_into_traj]
            expected_next_atoms = atoms_seq[idx_into_traj + 1]

            for applicable_nsrt in utils.get_applicable_operators(
                    ground_nsrts, atoms):
                # NOTE: we check that the ParameterizedOptions are equal before
                # attempting to ground because otherwise, we might
                # get a parameter mismatch and trigger an AssertionError
                # during grounding.
                if applicable_nsrt.option != gt_option.parent:
                    continue
                if applicable_nsrt.option_objs != gt_option.objects:
                    continue
                next_atoms = utils.apply_operator(applicable_nsrt, set(atoms))
                if next_atoms.issubset(expected_next_atoms):
                    # The returned cost is uniform because we don't
                    # actually care about finding the shortest path;
                    # just one that matches!
                    yield (applicable_nsrt, (frozenset(next_atoms),
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
    def _evaluate(self, s: Tuple[PartialNSRTAndDatastore, ...]) -> float:
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

    def _evaluate(self, s: Tuple[PartialNSRTAndDatastore, ...]) -> float:
        strips_ops = [pnad.op for pnad in s]
        option_specs = [pnad.option_spec for pnad in s]
        score = self._score_func.evaluate_with_operators(
            frozenset(), self._trajectories, self._segmented_trajs, strips_ops,
            option_specs)
        return score


class PreserveSkeletonsHillClimbingSidePredicateLearner(
        HillClimbingSidePredicateLearner):
    """A side predicate learning strategy that does hill climbing with a
    skeleton preservation (harmlessness) score function."""

    def _evaluate(self, s: Tuple[PartialNSRTAndDatastore, ...]) -> float:
        preserves_harmlessness = self._check_harmlessness(list(s))
        if preserves_harmlessness:
            # If harmlessness is preserved, the score is the number of
            # operators that we have, minus the number of side predicates.
            # This means we prefer fewer operators and more side predicates.
            score = 2 * len(s)
            for pnad in s:
                score -= len(pnad.op.side_predicates)
        else:
            # If harmlessness is not preserved, the score is an arbitrary
            # constant bigger than the total number of operators at the
            # start of the search. This is guaranteed to be worse (higher)
            # than any score that occurs if harmlessness is preserved.
            score = 10 * len(self._initial_pnads)
        return score


class GeneralToSpecificSidePredicateLearner(SidePredicateLearner):
    """Sideline by starting with the most general operators and then refining
    them by looking through the data."""

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

        # The side predicates are simply all predicates that appear in any
        # add or delete effects.
        side_predicates = set()
        for initial_pnad in initial_pnads_for_option:
            for effect in initial_pnad.op.add_effects:
                side_predicates.add(effect.predicate)
            for effect in initial_pnad.op.delete_effects:
                side_predicates.add(effect.predicate)

        # Now, we use an arbitrarily-chosen option to set the option spec
        # and parameters for the PNAD.
        option_spec = initial_pnads_for_option[0].option_spec
        parameters = sorted(option_spec[1])

        # There are no add effects or delete effects. The preconditions
        # are initialized to be trivial. They will be recomputed next.
        op = STRIPSOperator(parameterized_option.name, parameters, set(),
                            set(), set(), side_predicates)
        pnad = PartialNSRTAndDatastore(op, [], option_spec)

        # Recompute datastore. This simply clusters by option, since the
        # side predicates contain all predicates, and effects are trivial.
        self._recompute_datastores_from_segments([pnad])

        # Determine the initial preconditions. These are just a lifted
        # intersection of atoms that are true in every state from which
        # this PNAD's option was executed.
        preconditions = induce_pnad_preconditions(pnad)
        pnad.op = pnad.op.copy_with(preconditions=preconditions)

        # Finally, remove the side predicates from these operators.
        # These will be filled in later.
        pnad.op = pnad.op.copy_with(side_predicates=set())

        return pnad


class BackchainingSidePredicateLearner(GeneralToSpecificSidePredicateLearner):
    """Sideline by backchaining, making PNADs increasingly specific."""

    def _sideline(self) -> List[PartialNSRTAndDatastore]:
        # Initialize the most general PNADs by merging self._initial_pnads.
        # As a result, we will have one very general PNAD per option.
        param_opt_to_general_pnad = {}
        param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                     List[PartialNSRTAndDatastore]] = {}
        parameterized_options = {p.option_spec[0] for p in self._initial_pnads}
        total_datastore_len = 0
        for param_opt in parameterized_options:
            pnad = self._initialize_general_pnad_for_option(param_opt)
            param_opt_to_general_pnad[param_opt] = pnad
            param_opt_to_nec_pnads[param_opt] = []
            total_datastore_len += len(pnad.datastore)
        del self._initial_pnads  # no longer used
        # Assert that all data is in some PNAD's datastore.
        assert total_datastore_len == sum(
            len(seg_traj) for seg_traj in self._segmented_trajs)

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
                if len(param_opt_to_nec_pnads[option.parent]) == 0:
                    pnads_for_option = [
                        param_opt_to_general_pnad[option.parent]
                    ]
                else:
                    pnads_for_option = param_opt_to_nec_pnads[option.parent]

                # Compute the ground atoms that must be added on this timestep.
                # They must be a subset of the current PNAD's add effects.
                necessary_add_effects = necessary_image - atoms_seq[t]
                assert necessary_add_effects.issubset(segment.add_effects)

                # We start by checking if any of the PNADs associated with the
                # demonstrated option are able to match this transition.
                for pnad in pnads_for_option:
                    ground_op = self._find_unification(necessary_add_effects,
                                                       pnad, segment)
                    if ground_op is not None:
                        obj_to_var = dict(
                            zip(ground_op.objects, pnad.op.parameters))
                        if len(param_opt_to_nec_pnads[option.parent]) == 0:
                            pnad_op_name = pnad.op.name + "0"
                            pnad.op = pnad.op.copy_with(name=pnad_op_name)
                            param_opt_to_nec_pnads[option.parent].append(pnad)
                        break

                # If we weren't able to find a substitution (i.e, the above)
                # for loop did not break), we need to try
                # specialize each of our PNADs.
                else:
                    for pnad in pnads_for_option:
                        new_pnad = self._try_specializing_pnad(
                            necessary_add_effects, pnad, segment)
                        if new_pnad is not None:
                            assert new_pnad.option_spec == pnad.option_spec
                            if len(param_opt_to_nec_pnads[option.parent]) > 0:
                                param_opt_to_nec_pnads[option.parent].remove(
                                    pnad)
                            del pnad
                            break

                    # If we were unable to specialize any of the PNADs, we need
                    # to split from the most general PNAD and make a new PNAD
                    # to cover these necessary add effects.
                    else:
                        new_pnad = self._try_specializing_pnad(
                            necessary_add_effects,
                            param_opt_to_general_pnad[option.parent], segment)
                        assert new_pnad is not None

                    # Add the new PNAD to the dictionary mapping options to
                    # PNADs.
                    assert isinstance(new_pnad, PartialNSRTAndDatastore)
                    if not new_pnad.op.name[-1].isdigit():
                        op_num = len(param_opt_to_nec_pnads[option.parent])
                        new_pnad_op_name = new_pnad.op.name + str(op_num)
                        new_pnad.op = new_pnad.op.copy_with(
                            name=new_pnad_op_name)
                    pnad = new_pnad
                    param_opt_to_nec_pnads[option.parent].append(pnad)
                    # After all this, the unification call that failed earlier
                    # (leading us into the current if statement) should work.
                    ground_op = self._find_unification(necessary_add_effects,
                                                       pnad, segment)
                    assert ground_op is not None
                    obj_to_var = dict(
                        zip(ground_op.objects, pnad.op.parameters))

                # Update necessary_image for this timestep. It no longer
                # needs to include the ground add effects of this PNAD, but
                # must now include its ground preconditions.
                var_to_obj = {v: k for k, v in obj_to_var.items()}
                assert len(var_to_obj) == len(obj_to_var)

                necessary_image -= {
                    a.ground(var_to_obj)
                    for a in pnad.op.add_effects
                }
                necessary_image |= {
                    a.ground(var_to_obj)
                    for a in pnad.op.preconditions
                }

        # Now that the add effects and preconditions are correct,
        # make a list of all final PNADs. Note
        # that these final PNADs only come from the
        # param_opt_to_nec_pnads dict, since we can be assured
        # that our backchaining process ensured that the
        # PNADs in this dict cover all of the data!
        all_pnads = []
        for pnad_list in param_opt_to_nec_pnads.values():
            for pnad in pnad_list:
                all_pnads.append(pnad)

        # At this point, all PNADs have correct parameters, preconditions,
        # and add effects. We now finalize the delete effects and side
        # predicates. Note that we have to do delete effects first, and
        # then side predicates, because the latter rely on the former.
        print()
        for pnad in all_pnads:
            self._finalize_pnad_delete_effects(pnad)
            self._finalize_pnad_side_predicates(pnad)
            print(pnad)

        # Before returning, sanity check that harmlessness holds.
        assert self._check_harmlessness(all_pnads)
        return all_pnads

    @staticmethod
    def _find_unification(
        necessary_add_effects: Set[GroundAtom],
        pnad: PartialNSRTAndDatastore,
        segment: Segment,
        find_partial_grounding: bool = False
    ) -> Optional[_GroundSTRIPSOperator]:
        # Find a mapping from the variables in the PNAD add effects
        # and option to the objects in necessary_add_effects and the
        # segment's option. If one exists, we don't need to modify this
        # PNAD. Otherwise, we must make its add effects more specific.
        # Note that we are assuming all variables in the parameters
        # of the PNAD will appear in either the option arguments or
        # the add effects. This is in contrast to strips_learning.py,
        # where delete effect variables also contribute to parameters.
        objects = list(segment.states[0])
        option_objs = segment.get_option().objects
        isub = dict(zip(pnad.option_spec[1], option_objs))
        # Loop over all groundings.
        for ground_op in utils.all_ground_operators_given_partial(
                pnad.op, objects, isub):
            if not ground_op.preconditions.issubset(segment.init_atoms):
                continue
            # If find_partial_grounding is True, we want to find a grounding
            # that achieves some subset of the necessary add effects. Else,
            # we want to find a grounding that is some superset of the
            # necessary_add_effects
            if find_partial_grounding:
                if not ground_op.add_effects.issubset(necessary_add_effects):
                    continue
            else:
                if not ground_op.add_effects.issubset(segment.final_atoms):
                    continue
                if not necessary_add_effects.issubset(ground_op.add_effects):
                    continue
            return ground_op
        return None

    def _try_specializing_pnad(
        self,
        necessary_add_effects: Set[GroundAtom],
        pnad: PartialNSRTAndDatastore,
        segment: Segment,
    ) -> Optional[PartialNSRTAndDatastore]:
        """Given a PNAD and some necessary add effects that the PNAD must
        achieve, try to make the PNAD's add effects more specific
        ("specialize") so that they cover these necessary add effects.

        Returns the new constructed PNAD, without modifying the
        original. If the PNAD does not have a grounding that can even
        partially satisfy the necessary add effects, returns None.
        """

        # Get an arbitrary grounding of the PNAD's operator whose
        # preconditions hold in segment.init_atoms and whose add
        # effects are a subset of necessary_add_effects.
        ground_op = self._find_unification(necessary_add_effects,
                                           pnad,
                                           segment,
                                           find_partial_grounding=True)
        # If no such grounding exists, specializing is not possible.
        if ground_op is None:
            return None
        # To figure out the effects we need to add to this PNAD,
        # we first look at the ground effects that are missing
        # from this arbitrary ground operator.
        missing_effects = necessary_add_effects - ground_op.add_effects
        obj_to_var = dict(zip(ground_op.objects, pnad.op.parameters))
        # Before we can lift missing_effects, we need to add new
        # entries to obj_to_var to account for the situation where
        # missing_effects contains objects that were not in
        # the ground operator's parameters.
        all_objs = {o for eff in missing_effects for o in eff.objects}
        missing_objs = sorted(all_objs - set(obj_to_var))
        new_var_types = [o.type for o in missing_objs]
        new_vars = utils.create_new_variables(new_var_types,
                                              existing_vars=pnad.op.parameters)
        obj_to_var.update(dict(zip(missing_objs, new_vars)))
        # Finally, we can lift missing_effects.
        updated_params = sorted(obj_to_var.values())
        updated_add_effects = pnad.op.add_effects | {
            a.lift(obj_to_var)
            for a in missing_effects
        }
        # Create a new PNAD with the updated parameters and add effects.
        new_pnad = self._create_new_pnad_with_params_and_add_effects(
            pnad, updated_params, updated_add_effects)

        return new_pnad

    def _create_new_pnad_with_params_and_add_effects(
            self, pnad: PartialNSRTAndDatastore, parameters: List[Variable],
            add_effects: Set[LiftedAtom]) -> PartialNSRTAndDatastore:
        """Create a new PNAD that is the given PNAD with parameters and add
        effects changed to the given ones, and preconditions recomputed.

        Note that in general, changing the parameters means that we need
        to recompute all datastores, otherwise the precondition learning
        will not work correctly (since it relies on the substitution
        dictionaries in the datastores being correct).
        """
        # Create a new PNAD with the given parameters and add effects. Set
        # the preconditions to be trivial. They will be recomputed next.
        new_pnad_op = pnad.op.copy_with(parameters=parameters,
                                        preconditions=set(),
                                        add_effects=add_effects)
        new_pnad = PartialNSRTAndDatastore(new_pnad_op, [], pnad.option_spec)
        del pnad  # unused from here
        # Recompute datastore using the add_effects semantics.
        self._recompute_datastores_from_segments([new_pnad],
                                                 semantics="add_effects")
        # Determine the preconditions.
        preconditions = induce_pnad_preconditions(new_pnad)
        # Update the preconditions of the new PNAD's operator.
        new_pnad.op = new_pnad.op.copy_with(preconditions=preconditions)
        return new_pnad

    @staticmethod
    def _finalize_pnad_delete_effects(pnad: PartialNSRTAndDatastore) -> None:
        """Update the given PNAD to change the delete effects to ones obtained
        by unioning all lifted images in the datastore.

        IMPORTANT NOTE: We want to do a union here because the most
        general delete effects are the ones that capture _any possible_
        deletion that occurred in a training transition. (This is
        contrast to preconditions, where we want to take an intersection
        over our training transitions.) However, we do not allow
        creating new variables when we create these delete effects.
        Instead, we filter out delete effects that include new
        variables. Therefore, even though it may seem on the surface
        like this procedure will cause all delete effects in the data to
        be modeled accurately, this is not actually true.
        """
        delete_effects = set()
        for segment, var_to_obj in pnad.datastore:
            obj_to_var = {o: v for v, o in var_to_obj.items()}
            atoms = {
                atom
                for atom in segment.delete_effects
                if all(o in obj_to_var for o in atom.objects)
            }
            lifted_atoms = {atom.lift(obj_to_var) for atom in atoms}
            delete_effects |= lifted_atoms
        pnad.op = pnad.op.copy_with(delete_effects=delete_effects)

    @staticmethod
    def _finalize_pnad_side_predicates(pnad: PartialNSRTAndDatastore) -> None:
        """Update the given PNAD to change the side predicates to ones that
        include every unmodeled add or delete effect seen in the data."""
        # First, strip out any existing side predicates so that the call
        # to apply_operator() cannot use them, which would defeat the purpose.
        pnad.op = pnad.op.copy_with(side_predicates=set())
        side_predicates = set()
        for (segment, var_to_obj) in pnad.datastore:
            objs = tuple(var_to_obj[param] for param in pnad.op.parameters)
            ground_op = pnad.op.ground(objs)
            next_atoms = utils.apply_operator(ground_op, segment.init_atoms)
            for atom in segment.final_atoms - next_atoms:
                side_predicates.add(atom.predicate)
            for atom in next_atoms - segment.final_atoms:
                side_predicates.add(atom.predicate)
        pnad.op = pnad.op.copy_with(side_predicates=side_predicates)
