"""Algorithms for STRIPS learning that start from the most general operators,
then specialize them based on the data."""

from typing import Dict, List, Optional, Set

from predicators.src import utils
from predicators.src.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.src.structs import GroundAtom, LiftedAtom, \
    ParameterizedOption, PartialNSRTAndDatastore, Segment, STRIPSOperator, \
    Variable, _GroundSTRIPSOperator


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
        total_datastore_len = 0
        for param_opt in parameterized_options:
            pnad = self._initialize_general_pnad_for_option(param_opt)
            param_opt_to_general_pnad[param_opt] = pnad
            param_opt_to_nec_pnads[param_opt] = []
            total_datastore_len += len(pnad.datastore)
        # Assert that all data is in some PNAD's datastore.
        assert total_datastore_len == sum(
            len(seg_traj) for seg_traj in self._segmented_trajs)

        # Go through each demonstration from the end back to the start,
        # making the PNADs more specific whenever needed.
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
                            param_opt_to_general_pnad[option.parent], segment,
                            check_datastore_change=False)
                        assert new_pnad is not None

                    pnad = new_pnad
                    del new_pnad  # unused from here
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
        for pnad_list in sorted(param_opt_to_nec_pnads.values(), key=str):
            for i, pnad in enumerate(pnad_list):
                pnad.op = pnad.op.copy_with(name=pnad.op.name + str(i))
                all_pnads.append(pnad)

        # At this point, all PNADs have correct parameters, preconditions,
        # and add effects. We now finalize the delete effects and side
        # predicates. Note that we have to do delete effects first, and
        # then side predicates, because the latter rely on the former.
        for pnad in all_pnads:
            self._finalize_pnad_delete_effects(pnad)
            self._finalize_pnad_side_predicates(pnad)

        # Finally, recompute the datastores.
        self._recompute_datastores_from_segments(all_pnads)

        return all_pnads

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
        check_datastore_change: bool = True
    ) -> Optional[PartialNSRTAndDatastore]:
        """Given a PNAD and some necessary add effects that the PNAD must
        achieve, try to make the PNAD's add effects more specific
        ("specialize") so that they cover these necessary add effects.

        Returns the new constructed PNAD, without modifying the
        original. If the PNAD does not have a grounding that can even
        partially satisfy the necessary add effects, then returns None. 
        If check_datastore_change is set to True, then additionally
        checks whether the newly created PNAD covers a different set
        of datapoints than the original, and returns None in this case.
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
        # Create a new PNAD with the updated parameters and add effects.
        new_pnad = self._create_new_pnad_with_params_and_add_effects(
            pnad, updated_params, updated_add_effects)

        if check_datastore_change:
            # If the new PNAD has a datastore size that's not the same
            # as that of the original PNAD, then we've potentially lost some
            # data by specializing, which might do harm!
            if len(new_pnad.datastore) < len(pnad.datastore):
                return None

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
        preconditions = self._induce_preconditions_via_intersection(new_pnad)
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
