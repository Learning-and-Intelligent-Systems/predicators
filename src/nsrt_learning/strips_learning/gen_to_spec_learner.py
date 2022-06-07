"""Algorithms for STRIPS learning that start from the most general operators,
then specialize them based on the data."""

import itertools
from typing import Dict, List, Optional, Set, Tuple

from predicators.src import utils
from predicators.src.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.src.structs import GroundAtom, Object, ParameterizedOption, \
    PartialNSRTAndDatastore, Segment, STRIPSOperator, Variable, \
    _GroundSTRIPSOperator


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

        # Induce all components of the pnad given this datastore.
        self._induce_pnad_components_from_datastore(pnad)

        return pnad

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
            # Run one iteration of constraint satisfaction.
            nec_pnad_set_changed = self._satisfy_constraint(
                param_opt_to_nec_pnads, param_opt_to_general_pnad)
            # Run one iteration of objective optimization.
            # Add pnads that satisfy harmlessness constraints to list
            harmless_pnads = []
            for pnads in param_opt_to_nec_pnads.values():
                for pnad in pnads:
                    harmless_pnads.append(pnad)
            # Recompute datastore, preconditions, and effects for each pnad.
            self._recompute_datastores_from_segments(harmless_pnads, check_necessary_image=True)
            # Induce all components of the pnad given this datastore.
            for pnads in harmless_pnads:
                if len(pnad.datastore) == 0:
                    param_opt_to_nec_pnads[pnad.option_spec[0]].remove(pnad)
                    continue
                self._induce_pnad_components_from_datastore(pnad)
                    
            # TODO: think hard to come up with an actual termination condition
            # for this 2-step case.

        # Induce delete effects, side predicates, and keep effects if
        # necessary to finish learning.
        final_pnads: List[PartialNSRTAndDatastore] = []
        for pnad_list in param_opt_to_nec_pnads.values():
            for i, pnad in enumerate(pnad_list):
                pnad.op = pnad.op.copy_with(name=pnad.op.name + str(i))
                final_pnads.append(pnad)
        self._assert_all_data_in_exactly_one_datastore(final_pnads)
        return final_pnads

    def _satisfy_constraint(
        self, param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                           List[PartialNSRTAndDatastore]],
        param_opt_to_general_pnad: Dict[ParameterizedOption,
                                        PartialNSRTAndDatastore]
    ) -> bool:
        """Take one pass through the demonstrations in the given order.

        Go through each one from the end back to the start, inducing PNADs
        to cover any data that is currently uncovered. Return whether any
        PNAD was changed.
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
                first_violating_atoms, first_violating_pnad, first_violating_obj_to_var = None, None, None
                for pnad in pnads_for_option:
                    ground_op = self._find_unification(necessary_add_effects,
                                                       pnad, segment, check_add_effects=True, necessary_image=necessary_image)
                    if ground_op is not None:
                        # Check that this grounding is indeed harmless
                        # w.r.t the necessary_image.
                        is_harmless, violating_atoms = self._is_unification_harmless(
                            necessary_image, ground_op, segment)
                        obj_to_var = dict(
                            zip(ground_op.objects, pnad.op.parameters))
                        if is_harmless:
                            if len(param_opt_to_nec_pnads[option.parent]) == 0:
                                param_opt_to_nec_pnads[option.parent].append(
                                    pnad)
                            break
                        if not first_violating_atoms:
                            first_violating_atoms = violating_atoms
                            first_violating_pnad = pnad
                            first_violating_obj_to_var = obj_to_var

                # If we weren't able to find a substitution (i.e, the above
                # for loop did not break), we need to spawn a new PNAD from
                # the most general PNAD to cover these necessary add effects.
                else:
                    nec_pnad_set_changed = True
                    # In this case, we need to spawn a new operator to
                    # cover some necessary add effects.
                    if first_violating_atoms is None:
                        pnad = self._spawn_new_pnad(
                            necessary_add_effects,
                            param_opt_to_general_pnad[option.parent], segment)
                        # Add this newly-created PNAD to our necessary dictionary and
                        # recompute components of all PNADs associated with the same option.
                        # TODO: Make sure to pass in the necessary image here correctly
                        # so all the keep effects are ground correctly!
                        param_opt_to_nec_pnads[option.parent].append(pnad)
                        self._recompute_datastores_from_segments(param_opt_to_nec_pnads[option.parent])
                        # Compute the preconditions, delete effects and side predicates
                        # for the new PNAD.
                        self._induce_pnad_components_from_datastore(pnad)
                        # After all this, the unification call that failed earlier
                        # (leading us into the current else statement) should work.
                        ground_op = self._find_unification(necessary_add_effects,
                                                        pnad, segment, check_add_effects=True, necessary_image=necessary_image)
                        assert ground_op is not None
                        is_harmless, violating_atoms = self._is_unification_harmless(
                            necessary_image, ground_op, segment)
                        if not is_harmless:
                            first_violating_pnad = pnad
                            first_violating_atoms = violating_atoms
                            first_violating_obj_to_var = dict(
                                zip(ground_op.objects, pnad.op.parameters))
                            # Delete the PNAD from the necessary dictionary.
                            param_opt_to_nec_pnads[option.parent].remove(pnad)

                    # In this case, there is already a PNAD that matches the
                    # necessary_add_effects, but it unfortunately conflicts
                    # with the necessary_image. We need to spawn a new PNAD
                    # that doesn't have this problem, so we add the conflicting
                    # atoms to the necessary add effects.
                    if first_violating_atoms is not None:
                        assert first_violating_pnad is not None
                        assert first_violating_obj_to_var is not None
                        pnad = self._spawn_to_fix_nec_image_violation(
                            first_violating_pnad, first_violating_obj_to_var,
                            first_violating_atoms)
                        param_opt_to_nec_pnads[option.parent].append(pnad)
                        self._recompute_datastores_from_segments(param_opt_to_nec_pnads[option.parent])
                        # Compute the preconditions, delete effects and side predicates
                        # for the new PNAD.
                        self._induce_pnad_components_from_datastore(pnad)                        
                    
                    # After all this, the unification call that failed earlier
                    # (leading us into the current else statement) should work.
                    ground_op = self._find_unification(necessary_add_effects,
                                                    pnad, segment, check_add_effects=True, necessary_image=necessary_image)
                    assert ground_op is not None
                    is_harmless, violating_atoms = self._is_unification_harmless(
                        necessary_image, ground_op, segment)
                    assert is_harmless
                    obj_to_var = dict(
                        zip(ground_op.objects, pnad.op.parameters))

                # For every segement we will also save the necessary image.
                # This will be used in the optimization phase.
                segment.necessary_image = necessary_image.copy()

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

    def _induce_pnad_components_from_datastore(
            self, pnad: PartialNSRTAndDatastore) -> None:
        """Given an input PNAD with a non-empty datastore, this method induces
        (1) preconditions, (2) delete effects and (3) side predicates for that
        PNAD based on its datastore.

        Note that this method modifies the input PNAD in-place.
        """
        pre = self._induce_preconditions_via_intersection(pnad)
        pnad.op = pnad.op.copy_with(preconditions=pre)
        self._compute_pnad_delete_effects(pnad)
        self._compute_pnad_side_predicates(pnad)

    @classmethod
    def get_name(cls) -> str:
        return "backchaining"

    @staticmethod
    def _find_unification(
            necessary_add_effects: Set[GroundAtom],
            pnad: PartialNSRTAndDatastore,
            segment: Segment,
            check_add_effects: bool = True,
            necessary_image: Set[GroundAtom] = set()) -> Optional[_GroundSTRIPSOperator]:
        """Find a mapping from the variables in the PNAD add effects and option
        to the objects in necessary_add_effects and the segment's option.

        If one exists, we don't need to modify this PNAD. Otherwise, we
        must make its add effects more specific. Note that we are
        assuming all variables in the parameters of the PNAD will appear
        in either the option arguments or the add effects. This is in
        contrast to strips_learning.py, where delete effect variables
        also contribute to parameters. If check_add_effects is True, we
        want to find a grounding that is some superset of the
        necessary_add_effects and also such that the ground operator's
        add effects are always true in the segment's final atoms.
        Otherwise, we want to disregard any checks on add effects
        entirely (we will only make this call when spawning from a
        general PNAD, which has no add effects).
        """
        objects = list(segment.states[0])
        option_objs = segment.get_option().objects
        isub = dict(zip(pnad.option_spec[1], option_objs))
        # Loop over all groundings.
        for ground_op in utils.all_ground_operators_given_partial(
                pnad.op, objects, isub):
            if not ground_op.preconditions.issubset(segment.init_atoms):
                continue
            if check_add_effects:
                if not ground_op.add_effects.issubset(segment.final_atoms):
                    continue
                if not necessary_add_effects.issubset(ground_op.add_effects):
                    continue
                if len(necessary_image) > 0:
                    if not ground_op.add_effects.issubset(necessary_image):
                        continue
            return ground_op
        return None

    @staticmethod
    def _is_unification_harmless(
            necessary_image: Set[GroundAtom], ground_op: _GroundSTRIPSOperator,
            segment: Segment) -> Tuple[bool, Optional[Set[GroundAtom]]]:
        """Given a particular ground operator, check whether it will conflict
        with the necessary_image or not.

        If it does not, return (True, None). Otherwise, return (False,
        conflicting_atoms).
        """
        state_after_op = utils.apply_operator(ground_op, segment.init_atoms)
        if necessary_image.issubset(state_after_op):
            return (True, None)
        conflicting_atoms = necessary_image - state_after_op
        return (False, conflicting_atoms)

    def _spawn_new_pnad(
        self,
        necessary_add_effects: Set[GroundAtom],
        pnad: PartialNSRTAndDatastore,
        segment: Segment,
    ) -> PartialNSRTAndDatastore:
        """Given a general PNAD and some necessary add effects that the PNAD
        must achieve, create a new PNAD ("spawn" from the most general one) so
        that it has these necessary add effects.

        Returns the newly constructed PNAD, without modifying the
        original.
        """
        # Assert that this really is a general PNAD.
        assert len(pnad.op.add_effects) == 0, \
            "Can't spawn from non-general PNAD"

        # Get an arbitrary grounding of the PNAD's operator whose
        # preconditions hold in segment.init_atoms.
        ground_op = self._find_unification(necessary_add_effects,
                                           pnad,
                                           segment,
                                           check_add_effects=False)
        # Assert that such a grounding exists - this must be the case
        # since we only ever call this method with the most general
        # PNAD for the option.
        assert ground_op is not None

        obj_to_var = dict(zip(ground_op.objects, pnad.op.parameters))
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
    def _spawn_to_fix_nec_image_violation(
            pnad: PartialNSRTAndDatastore, obj_to_var: Dict[Object, Variable],
            violating_atoms: Set[GroundAtom]) -> PartialNSRTAndDatastore:
        """Return a new PNAD that is a more-specific version of the input PNAD
        that fixes the violating atoms."""
        keep_effects = set()
        # We need to lift the conflicting atoms to induce
        # operators with keep effects.
        for atom in violating_atoms:
            for obj in atom.objects:
                if obj in obj_to_var:
                    continue
                new_var = utils.create_new_variables([obj.type],
                                                     obj_to_var.values())[0]
                obj_to_var[obj] = new_var
            keep_effects.add(atom.lift(obj_to_var))
        params_set = set(pnad.op.parameters)
        for eff in keep_effects:
            for var in eff.variables:
                params_set.add(var)
        parameters = sorted(params_set)
        # The keep effects go into both the PNAD preconditions and the
        # PNAD add effects.
        preconditions = pnad.op.preconditions | set(keep_effects)
        add_effects = pnad.op.add_effects | set(keep_effects)
        # Create the new PNAD.
        new_pnad_op = pnad.op.copy_with(parameters=parameters,
                                        preconditions=preconditions,
                                        add_effects=add_effects)
        new_pnad = PartialNSRTAndDatastore(new_pnad_op, [], pnad.option_spec)

        return new_pnad

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
