"""Algorithms for STRIPS learning that start from the most general operators,
then specialize them based on the data."""

import itertools
from typing import Dict, List, Optional, Set

from predicators.src import utils
from predicators.src.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.src.structs import GroundAtom, ParameterizedOption, \
    PartialNSRTAndDatastore, Segment, STRIPSOperator, _GroundSTRIPSOperator


class GeneralToSpecificSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a general-to-specific STRIPS learner."""

    def _initialize_general_pnad_for_option(
            self, parameterized_option: ParameterizedOption
    ) -> PartialNSRTAndDatastore:
        """Create the most general PNAD for the given option."""
        # Create the parameters, which are determined solely from the option
        # types, since the most general operator has no add/delete effects.
        parameters = utils.create_new_variables(parameterized_option.types)
        option_spec = (parameterized_option, parameters)

        # In the most general operator, the side predicates contain ALL
        # predicates.
        side_predicates = self._predicates.copy()

        # There are no add effects or delete effects. The preconditions
        # are initialized to be trivial. They will be recomputed next.
        op = STRIPSOperator(parameterized_option.name, parameters, set(),
                            set(), set(), side_predicates)
        pnad = PartialNSRTAndDatastore(op, [], option_spec)

        # Recompute datastore. This simply clusters by option, since the
        # side predicates contain all predicates, and effects are trivial.
        self._recompute_datastores_from_segments([pnad])

        # Determine the initial preconditions via a lifted intersection.
        preconditions = self._induce_preconditions_via_intersection(pnad)
        pnad.op = pnad.op.copy_with(preconditions=preconditions)

        return pnad


class BackchainingSTRIPSLearner(GeneralToSpecificSTRIPSLearner):
    """Learn STRIPS operators by backchaining."""

    def _learn(self) -> List[PartialNSRTAndDatastore]:
        # Initialize the most general PNADs by merging self._initial_pnads.
        # As a result, we will have one very general PNAD per option.
        param_opt_to_general_pnad = {}
        param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                     List[PartialNSRTAndDatastore]] = {}
        # Extract all parameterized options from the data.
        parameterized_options = set()
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            for segment in seg_traj:
                parameterized_options.add(segment.get_option().parent)

        # Set up the param_opt_to_general_pnad and param_opt_to_nec_pnads
        # dictionaries.
        for param_opt in parameterized_options:
            pnad = self._initialize_general_pnad_for_option(param_opt)
            param_opt_to_general_pnad[param_opt] = pnad
            param_opt_to_nec_pnads[param_opt] = []
        self._assert_all_data_in_exactly_one_datastore(
            list(param_opt_to_general_pnad.values()))

        # Pass over the demonstrations multiple times. Each time, backchain
        # to learn PNADs. Repeat until a fixed point is reached.
        nec_pnad_set_changed = True
        while nec_pnad_set_changed:
            # Before each pass, clear the poss_keep_effects and
            # seg_to_keep_effects_sub of all the PNADs. We do this because
            # we only want the poss_keep_effects of the final pass, where
            # the PNADs did not change.
            for pnads in param_opt_to_nec_pnads.values():
                for pnad in pnads:
                    pnad.poss_keep_effects.clear()
                    pnad.seg_to_keep_effects_sub.clear()
            # Run one pass of backchaining.
            nec_pnad_set_changed = self._backchain_one_pass(
                param_opt_to_nec_pnads, param_opt_to_general_pnad)

        final_pnads = self._finish_learning(param_opt_to_nec_pnads)
        self._assert_all_data_in_exactly_one_datastore(final_pnads)
        return final_pnads

    def _backchain_one_pass(
        self, param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                           List[PartialNSRTAndDatastore]],
        param_opt_to_general_pnad: Dict[ParameterizedOption,
                                        PartialNSRTAndDatastore]
    ) -> bool:
        """Take one pass through the demonstrations.

        Go through each one from the end back to the start, making the
        PNADs more specific whenever needed. Return whether any PNAD was
        changed.
        """
        nec_pnad_set_changed = False
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
                # Find the necessary PNADs associated with this option.
                # If there are none, then use the general PNAD
                # associated with this option.
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
                            param_opt_to_nec_pnads[option.parent].append(pnad)
                        break
                # If we weren't able to find a substitution (i.e, the above
                # for loop did not break), we need to try specializing each
                # of our PNADs.
                else:
                    nec_pnad_set_changed = True
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
                    # to spawn from the most general PNAD and make a new PNAD
                    # to cover these necessary add effects.
                    else:
                        new_pnad = self._try_specializing_pnad(
                            necessary_add_effects,
                            param_opt_to_general_pnad[option.parent], segment)
                        assert new_pnad is not None

                    pnad = new_pnad
                    del new_pnad  # unused from here
                    param_opt_to_nec_pnads[option.parent].append(pnad)

                    # Recompute datastores for ALL PNADs associated with this
                    # option. We need to do this because the new PNAD may now
                    # be a better match for some transition that we previously
                    # matched to another PNAD.
                    self._recompute_datastores_from_segments(
                        param_opt_to_nec_pnads[option.parent])
                    # Recompute all preconditions, now that we have recomputed
                    # the datastores.
                    for nec_pnad in param_opt_to_nec_pnads[option.parent]:
                        pre = self._induce_preconditions_via_intersection(
                            nec_pnad)
                        nec_pnad.op = nec_pnad.op.copy_with(preconditions=pre)

                    # After all this, the unification call that failed earlier
                    # (leading us into the current else statement) should work.
                    ground_op = self._find_unification(necessary_add_effects,
                                                       pnad, segment)
                    assert ground_op is not None
                    obj_to_var = dict(
                        zip(ground_op.objects, pnad.op.parameters))

                # Every atom in the necessary_image that wasn't in the
                # necessary_add_effects is a possible keep effect. This
                # may add new variables, whose mappings for this segment
                # we keep track of in the seg_to_keep_effects_sub dict.
                for atom in necessary_image - necessary_add_effects:
                    keep_eff_sub = {}
                    for obj in atom.objects:
                        if obj in obj_to_var:
                            continue
                        new_var = utils.create_new_variables(
                            [obj.type], obj_to_var.values())[0]
                        obj_to_var[obj] = new_var
                        keep_eff_sub[new_var] = obj
                    pnad.poss_keep_effects.add(atom.lift(obj_to_var))
                    if segment not in pnad.seg_to_keep_effects_sub:
                        pnad.seg_to_keep_effects_sub[segment] = {}
                    pnad.seg_to_keep_effects_sub[segment].update(keep_eff_sub)

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
        return nec_pnad_set_changed

    def _finish_learning(
        self, param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                           List[PartialNSRTAndDatastore]]
    ) -> List[PartialNSRTAndDatastore]:
        """Given the current PNADs where add effects and preconditions are
        correct, learn the remaining components."""
        # Make a list of all final PNADs. Note
        # that these final PNADs only come from the
        # param_opt_to_nec_pnads dict, since we can be assured
        # that our backchaining process ensured that the
        # PNADs in this dict cover all of the data!
        nec_pnads = []
        for pnad_list in sorted(param_opt_to_nec_pnads.values(), key=str):
            for i, pnad in enumerate(pnad_list):
                pnad.op = pnad.op.copy_with(name=(pnad.op.name + str(i)))
                nec_pnads.append(pnad)

        # At this point, all PNADs have correct parameters, preconditions,
        # and add effects. We now compute the delete effects, side
        # predicates, and keep effects. The order is important: each
        # of these depends on the previous one.
        pnads_with_keep_effects = set()
        for pnad in nec_pnads:
            self._compute_pnad_delete_effects(pnad)
            self._compute_pnad_side_predicates(pnad)
            pnads_with_keep_effects |= self._get_pnads_with_keep_effects(pnad)
        for pnad in pnads_with_keep_effects:
            nec_pnads.append(pnad)

        # Recompute datastores and induce preconditions one final time.
        # Filter out PNADs that have an empty datastore.
        self._recompute_datastores_from_segments(nec_pnads)
        final_pnads = []
        for pnad in nec_pnads:
            if pnad.datastore:
                pre = self._induce_preconditions_via_intersection(pnad)
                pnad.op = pnad.op.copy_with(preconditions=pre)
                final_pnads.append(pnad)

        return final_pnads

    @classmethod
    def get_name(cls) -> str:
        return "backchaining"

    @staticmethod
    def _find_unification(
        necessary_add_effects: Set[GroundAtom],
        pnad: PartialNSRTAndDatastore,
        segment: Segment,
        ground_eff_subset_necessary_eff: bool = False
    ) -> Optional[_GroundSTRIPSOperator]:
        """Find a mapping from the variables in the PNAD add effects and option
        to the objects in necessary_add_effects and the segment's option.

        If one exists, we don't need to modify this PNAD. Otherwise, we
        must make its add effects more specific. Note that we are
        assuming all variables in the parameters of the PNAD will appear
        in either the option arguments or the add effects. This is in
        contrast to strips_learning.py, where delete effect variables
        also contribute to parameters. If
        ground_eff_subset_necessary_eff is True, we want to find a
        grounding that achieves some subset of the
        necessary_add_effects. Else, we want to find a grounding that is
        some superset of the necessary_add_effects and also such that
        the ground operator's add effects are always true in the
        segment's final atoms.
        """
        objects = list(segment.states[0])
        option_objs = segment.get_option().objects
        isub = dict(zip(pnad.option_spec[1], option_objs))
        # Loop over all groundings.
        for ground_op in utils.all_ground_operators_given_partial(
                pnad.op, objects, isub):
            if not ground_op.preconditions.issubset(segment.init_atoms):
                continue
            if ground_eff_subset_necessary_eff:
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
        partially satisfy the necessary add effects, then returns None.
        If check_datastore_change is set to True, then additionally
        checks whether the newly created PNAD covers a smaller set of
        datapoints than the original, and returns None in this case.
        """

        # Get an arbitrary grounding of the PNAD's operator whose
        # preconditions hold in segment.init_atoms and whose add
        # effects are a subset of necessary_add_effects.
        ground_op = self._find_unification(
            necessary_add_effects,
            pnad,
            segment,
            ground_eff_subset_necessary_eff=True)
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

        # Create a new PNAD with the given parameters and add effects. Set
        # the preconditions to be trivial. They will be recomputed later.
        new_pnad_op = pnad.op.copy_with(parameters=updated_params,
                                        preconditions=set(),
                                        add_effects=updated_add_effects)
        new_pnad = PartialNSRTAndDatastore(new_pnad_op, [], pnad.option_spec)
        # Note: we don't need to copy anything related to keep effects into
        # new_pnad here, because we only care about keep effects on the final
        # iteration of backchaining, where this function is never called.

        return new_pnad

    @staticmethod
    def _compute_pnad_delete_effects(pnad: PartialNSRTAndDatastore) -> None:
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
    def _compute_pnad_side_predicates(pnad: PartialNSRTAndDatastore) -> None:
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

    @staticmethod
    def _get_pnads_with_keep_effects(
            pnad: PartialNSRTAndDatastore) -> Set[PartialNSRTAndDatastore]:
        """Return a new set of PNADs that include keep effects into the given
        PNAD."""
        # The keep effects that we want are the subset of possible keep
        # effects which are not already in the PNAD's add effects, and
        # whose predicates were determined to be side predicates.
        keep_effects = {
            eff
            for eff in pnad.poss_keep_effects if eff not in pnad.op.add_effects
            and eff.predicate in pnad.op.side_predicates
        }
        new_pnads_with_keep_effects = set()
        # Given these keep effects, we need to create a combinatorial number of
        # PNADs, one for each unique combination of keep effects. Moreover, we
        # need to ensure that they are named differently from each other. Some
        # of these PNADs will be filtered out later if they are not useful to
        # cover any datapoints.
        i = 0
        for r in range(1, len(keep_effects) + 1):
            for keep_effects_subset in itertools.combinations(keep_effects, r):
                # These keep effects (keep_effects_subset) could involve new
                # variables, which we need to add to the PNAD parameters.
                params_set = set(pnad.op.parameters)
                for eff in keep_effects_subset:
                    for var in eff.variables:
                        params_set.add(var)
                parameters = sorted(params_set)
                # The keep effects go into both the PNAD preconditions and the
                # PNAD add effects.
                preconditions = pnad.op.preconditions | set(
                    keep_effects_subset)
                add_effects = pnad.op.add_effects | set(keep_effects_subset)
                # Create the new PNAD.
                new_pnad_op = pnad.op.copy_with(name=f"{pnad.op.name}-KEEP{i}",
                                                parameters=parameters,
                                                preconditions=preconditions,
                                                add_effects=add_effects)
                new_pnad = PartialNSRTAndDatastore(new_pnad_op, [],
                                                   pnad.option_spec)
                # Remember to copy seg_to_keep_effects_sub into the new_pnad!
                new_pnad.seg_to_keep_effects_sub = pnad.seg_to_keep_effects_sub
                new_pnads_with_keep_effects.add(new_pnad)
                i += 1

        return new_pnads_with_keep_effects

    def _assert_all_data_in_exactly_one_datastore(
            self, pnads: List[PartialNSRTAndDatastore]) -> None:
        """Assert that every demo datapoint appears in exactly one datastore
        among the given PNADs' datastores."""
        all_segs_in_data_lst = [
            seg for pnad in pnads for seg, _ in pnad.datastore
        ]
        all_segs_in_data = set(all_segs_in_data_lst)
        assert len(all_segs_in_data_lst) == len(all_segs_in_data)
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:  # ignore non-demo data
                continue
            for segment in seg_traj:
                assert segment in all_segs_in_data
