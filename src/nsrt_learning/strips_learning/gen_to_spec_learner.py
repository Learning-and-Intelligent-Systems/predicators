"""Algorithms for STRIPS learning that start from the most general operators,
then specialize them based on the data."""

import itertools
from typing import Dict, List, Set

from predicators.src import utils
from predicators.src.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.src.structs import ParameterizedOption, \
    PartialNSRTAndDatastore, Segment, STRIPSOperator


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

        prev_itr_pnads: Set[PartialNSRTAndDatastore] = set()

        # We loop until the harmless PNADs induced by our procedure
        # converge to a fixed point (i.e, they don't change after two
        # subsequent iterations).
        while True:
            # Run multiple passes of backchaining over the data until
            # convergence to a fixed point. Note that this process creates
            # operators with only parameters, preconditions, and add effects.
            self._backchain_multipass(param_opt_to_nec_pnads,
                                      param_opt_to_general_pnad)

            # Induce delete effects, side predicates and potentially
            # keep effects.
            self._induce_delete_side_keep(param_opt_to_nec_pnads)

            # Recompute datastores and preconditions for all PNADs.
            # Filter out PNADs that don't have datastores.
            cur_itr_pnads_unfiltered = [
                pnad for pnads in param_opt_to_nec_pnads.values()
                for pnad in pnads
            ]
            self._recompute_datastores_from_segments(cur_itr_pnads_unfiltered)
            cur_itr_pnads_filtered = []
            for pnad in cur_itr_pnads_unfiltered:
                if len(pnad.datastore) > 0:
                    new_pre = self._induce_preconditions_via_intersection(pnad)
                    # NOTE: this implicitly changes param_opt_to_nec_pnads
                    # as well, since we're directly modifying the PNAD objects.
                    pnad.op = pnad.op.copy_with(preconditions=new_pre)
                    cur_itr_pnads_filtered.append(pnad)
                else:
                    param_opt_to_nec_pnads[pnad.option_spec[0]].remove(pnad)
            del cur_itr_pnads_unfiltered  # should be unused after this

            # Check if the PNAD set has converged. If so, break.
            if set(cur_itr_pnads_filtered) == prev_itr_pnads:
                break

            prev_itr_pnads = set(cur_itr_pnads_filtered)

        # Assign a unique name to each PNAD.
        final_pnads = self._get_uniquely_named_nec_pnads(
            param_opt_to_nec_pnads)
        # Assert data has been correctly partitioned amongst PNADs.
        self._assert_all_data_in_exactly_one_datastore(final_pnads)
        return final_pnads

    def _backchain_multipass(
        self, param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                           List[PartialNSRTAndDatastore]],
        param_opt_to_general_pnad: Dict[ParameterizedOption,
                                        PartialNSRTAndDatastore]
    ) -> None:
        """Take multiple passes through the demonstrations, running
        self._backchain_one_pass() each time.

        Keep going until the PNADs reach a fixed point. Note that this
        process creates operators with only parameters, preconditions,
        and add effects.
        """
        while True:
            # Before each pass, clear the poss_keep_effects
            # of all the PNADs. We do this because we only want the
            # poss_keep_effects of the final pass, where the PNADs did
            # not change. However, we cannot simply clear the
            # pnad.seg_to_keep_effects_sub because some of these
            # substitutions might be necessary if this happens to be
            # a PNAD that already has keep effects. Thus, we call a
            # method that handles this correctly.
            for pnads in param_opt_to_nec_pnads.values():
                for pnad in pnads:
                    pnad.poss_keep_effects.clear()
                    self._clear_unnecessary_keep_effs_sub(pnad)
            # Run one pass of backchaining.
            nec_pnad_set_changed = self._backchain_one_pass(
                param_opt_to_nec_pnads, param_opt_to_general_pnad)
            if not nec_pnad_set_changed:
                break

    def _backchain_one_pass(
        self, param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                           List[PartialNSRTAndDatastore]],
        param_opt_to_general_pnad: Dict[ParameterizedOption,
                                        PartialNSRTAndDatastore]
    ) -> bool:
        """Take one pass through the demonstrations in the given order.

        Go through each one from the end back to the start, making the
        PNADs more specific whenever needed. Return whether any PNAD was
        changed.
        """
        # Reset all segments' necessary_add_effects so that they aren't
        # accidentally used from a previous iteration of backchaining.
        self._reset_all_segment_add_effs()
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
                # Update the segment's necessary_add_effects.
                segment.necessary_add_effects = necessary_add_effects

                # We start by checking if any of the PNADs associated with the
                # demonstrated option are able to match this transition.
                objects = set(segment.states[0])
                pnad, var_to_obj = self._find_best_matching_pnad_and_sub(
                    segment, objects, pnads_for_option)
                if pnad is not None:
                    assert var_to_obj is not None
                    obj_to_var = {v: k for k, v in var_to_obj.items()}
                    assert len(var_to_obj) == len(obj_to_var)
                    ground_op = pnad.op.ground(
                        tuple(var_to_obj[var] for var in pnad.op.parameters))
                    if len(param_opt_to_nec_pnads[option.parent]) == 0:
                        param_opt_to_nec_pnads[option.parent].append(pnad)
                # If we weren't able to find a substitution (i.e, the above
                # _find_best_matching call didn't yield a PNAD), we need to
                # spawn a new PNAD from the most general PNAD to cover
                # these necessary add effects.
                else:
                    nec_pnad_set_changed = True
                    pnad = self._spawn_new_pnad(
                        param_opt_to_general_pnad[option.parent], segment)
                    param_opt_to_nec_pnads[option.parent].append(pnad)

                    # Recompute datastores for ALL PNADs associated with this
                    # option. We need to do this because the new PNAD may now
                    # be a better match for some transition that we previously
                    # matched to another PNAD.
                    self._recompute_datastores_from_segments(
                        param_opt_to_nec_pnads[option.parent])
                    # Recompute all preconditions, now that we have recomputed
                    # the datastores. While doing this, keep track of any
                    # PNADs that get empty datastores.
                    pnads_to_remove = []
                    for nec_pnad in param_opt_to_nec_pnads[option.parent]:
                        if len(nec_pnad.datastore) > 0:
                            pre = self._induce_preconditions_via_intersection(
                                nec_pnad)
                            nec_pnad.op = nec_pnad.op.copy_with(
                                preconditions=pre)
                        else:
                            pnads_to_remove.append(nec_pnad)

                    # Remove PNADs that are no longer necessary because they
                    # have no data in their datastores.
                    for rem_pnad in pnads_to_remove:
                        param_opt_to_nec_pnads[option.parent].remove(rem_pnad)

                    # After all this, the unification call that failed earlier
                    # (leading us into the current else statement) should work.
                    best_score_pnad, var_to_obj = \
                        self._find_best_matching_pnad_and_sub(
                        segment, objects,
                        param_opt_to_nec_pnads[option.parent])
                    assert var_to_obj is not None
                    assert best_score_pnad == pnad
                    obj_to_var = {v: k for k, v in var_to_obj.items()}
                    assert len(var_to_obj) == len(obj_to_var)
                    ground_op = pnad.op.ground(
                        tuple(var_to_obj[var] for var in pnad.op.parameters))

                # Every atom in the necessary_image that wasn't in the
                # ground_op's add effects is a possible keep effect. This
                # may add new variables, whose mappings for this segment
                # we keep track of in the seg_to_keep_effects_sub dict.
                for atom in necessary_image - ground_op.add_effects:
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
                necessary_image -= {
                    a.ground(var_to_obj)
                    for a in pnad.op.add_effects
                }
                necessary_image |= {
                    a.ground(var_to_obj)
                    for a in pnad.op.preconditions
                }
        return nec_pnad_set_changed

    def _induce_delete_side_keep(
        self, param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                           List[PartialNSRTAndDatastore]]
    ) -> None:
        """Given the current PNADs where add effects and preconditions are
        correct, learn the remaining components: delete effects, side
        predicates, and keep_effects.

        Note that this may require spawning new PNADs with keep effects.
        """
        for option, nec_pnad_list in sorted(param_opt_to_nec_pnads.items(),
                                            key=str):
            pnads_with_keep_effects = set()
            for pnad in nec_pnad_list:
                self._compute_pnad_delete_effects(pnad)
                self._compute_pnad_side_predicates(pnad)
                pnads_with_keep_effects |= self._get_pnads_with_keep_effects(
                    pnad)
            param_opt_to_nec_pnads[option].extend(
                list(pnads_with_keep_effects))

    def _reset_all_segment_add_effs(self) -> None:
        """Reset all segments' necessary_add_effects to None."""
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            for segment in seg_traj:
                segment.necessary_add_effects = None

    @staticmethod
    def _clear_unnecessary_keep_effs_sub(
            pnad: PartialNSRTAndDatastore) -> None:
        """Clear unnecessary substitution values from the PNAD's
        seg_to_keep_effects_sub_dict.

        A substitution is unnecessary if it concerns a variable that
        isn't in the PNAD's op parameters.
        """
        for segment, keep_eff_sub in pnad.seg_to_keep_effects_sub.items():
            new_keep_eff_sub_dict = {}
            for var, obj in keep_eff_sub.items():
                if var in pnad.op.parameters:
                    new_keep_eff_sub_dict[var] = obj
            pnad.seg_to_keep_effects_sub[segment] = new_keep_eff_sub_dict

    @staticmethod
    def _get_uniquely_named_nec_pnads(
        param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                     List[PartialNSRTAndDatastore]]
    ) -> List[PartialNSRTAndDatastore]:
        """Given the param_opt_to_nec_pnads dict, return a list of PNADs that
        have unique names and can be used for planning."""
        uniquely_named_nec_pnads: List[PartialNSRTAndDatastore] = []
        for pnad_list in sorted(param_opt_to_nec_pnads.values(), key=str):
            for i, pnad in enumerate(pnad_list):
                pnad.op = pnad.op.copy_with(name=(pnad.op.name + str(i)))
                uniquely_named_nec_pnads.append(pnad)
        return uniquely_named_nec_pnads

    @classmethod
    def get_name(cls) -> str:
        return "backchaining"

    def _spawn_new_pnad(self, pnad: PartialNSRTAndDatastore,
                        segment: Segment) -> PartialNSRTAndDatastore:
        """Given a general PNAD and some segment with necessary add effects
        that the PNAD must achieve, create a new PNAD ("spawn" from the most
        general one) so that it has these necessary add effects.

        Returns the newly constructed PNAD, without modifying the
        original.
        """
        # Assert that the segment contains necessary_add_effects.
        necessary_add_effects = segment.necessary_add_effects
        assert necessary_add_effects is not None
        # Assert that this really is a general PNAD.
        assert len(pnad.op.add_effects) == 0, \
            "Can't spawn from non-general PNAD"

        # Get an arbitrary grounding of the PNAD's operator whose
        # preconditions hold in segment.init_atoms.
        objects = set(segment.states[0])
        _, var_to_obj = self._find_best_matching_pnad_and_sub(
            segment, objects, [pnad], check_only_add_effects=True)
        # Assert that such a grounding exists - this must be the case
        # since we only ever call this method with the most general
        # PNAD for the option.
        assert var_to_obj is not None
        obj_to_var = {v: k for k, v in var_to_obj.items()}
        assert len(var_to_obj) == len(obj_to_var)
        # Before we can lift the necessary_add_effects, we need to add new
        # entries to obj_to_var, since necessary_add_effects may
        # contain objects that were not in the ground operator's
        # parameters.
        all_objs = {o for eff in necessary_add_effects for o in eff.objects}
        missing_objs = sorted(all_objs - set(obj_to_var))
        new_vars = utils.create_new_variables([o.type for o in missing_objs],
                                              existing_vars=pnad.op.parameters)
        obj_to_var.update(dict(zip(missing_objs, new_vars)))
        # Finally, we can lift necessary_add_effects.
        updated_params = sorted(obj_to_var.values())
        updated_add_effects = {
            a.lift(obj_to_var)
            for a in necessary_add_effects
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
                new_pnad_op = pnad.op.copy_with(parameters=parameters,
                                                preconditions=preconditions,
                                                add_effects=add_effects)
                new_pnad = PartialNSRTAndDatastore(new_pnad_op, [],
                                                   pnad.option_spec)
                # Remember to copy seg_to_keep_effects_sub into the new_pnad!
                new_pnad.seg_to_keep_effects_sub = pnad.seg_to_keep_effects_sub
                new_pnads_with_keep_effects.add(new_pnad)

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
