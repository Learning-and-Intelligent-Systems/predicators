"""Algorithms for STRIPS learning that start from the most general operators,
then specialize them based on the data."""

import functools
import itertools
from typing import Dict, List, Set, Optional, Sequence

from predicators import utils
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import PNAD, GroundAtom, Object, \
    ParameterizedOption, Segment, STRIPSOperator, Variable, \
    _GroundSTRIPSOperator, _Atom, LowLevelTrajectory, Predicate, Type, Action, LiftedAtom, NSRT
from predicators.planning import task_plan, task_plan_grounding, _SkeletonSearchTimeout
import re

name_to_actions = {
    "Move": 0,
    "Actions.pickup_0": 3,
    "Actions.pickup_1": 4,
    "Actions.pickup_2": 5,
    "Actions.drop_0": 6,
    "Actions.drop_1": 7,
    "Actions.drop_2": 8,
    "Actions.drop_in": 9,
    "Actions.toggle": 10,
    "Actions.close": 11,
    "Actions.open": 12,
    "Actions.cook": 13,
    "Actions.slice": 14
}

class GeneralToSpecificSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a general-to-specific STRIPS learner."""

    @functools.lru_cache(maxsize=None)
    def _create_general_pnad_for_option(
            self, parameterized_option: ParameterizedOption) -> PNAD:
        """Create the most general PNAD for the given option."""
        # Create the parameters, which are determined solely from the option
        # types, since the most general operator has no add/delete effects.
        parameters = utils.create_new_variables(parameterized_option.types)
        option_spec = (parameterized_option, parameters)

        # In the most general operator, the ignore effects contain ALL
        # predicates.
        ignore_effects = self._predicates.copy()

        # There are no add effects or delete effects. The preconditions
        # are initialized to be trivial. They will be recomputed next.
        op = STRIPSOperator(parameterized_option.name, parameters, set(),
                            set(), set(), ignore_effects)
        pnad = PNAD(op, [], option_spec)

        # Recompute datastore. This simply clusters by option, since the
        # ignore effects contain all predicates, and effects are trivial.
        self._recompute_datastores_from_segments([pnad])

        return pnad

    def spawn_new_pnad(self, segment: Segment) -> PNAD:
        """Given some segment with necessary add effects that a new PNAD must
        achieve, create such a PNAD ("spawn" from the most general one
        associated with the segment's option) so that it has the necessary add
        effects contained in the given segment."""
        # Create a general PNAD for the segment's option.
        pnad = self._create_general_pnad_for_option(
            segment.get_option().parent)
        # Assert that this really is a general PNAD.
        assert len(pnad.op.add_effects) == 0, \
            "Can't spawn from non-general PNAD"
        # Assert that the segment contains necessary_add_effects.
        necessary_add_effects = segment.necessary_add_effects
        assert necessary_add_effects is not None

        # Get an arbitrary grounding of the PNAD's operator whose
        # preconditions hold in segment.init_atoms.
        objects = set(segment.states[0])
        _, var_to_obj = self._find_best_matching_pnad_and_sub(
            segment, objects, [pnad], check_only_preconditions=True)
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
        
        # #######
        # # Check if adding missing objects would exceed max_operator_arity
        # total_params = len(obj_to_var) + len(missing_objs)
        # if total_params > CFG.max_operator_arity:
        #     # Strategy: Filter necessary_add_effects to stay within limit
        #     # Priority 1: Keep effects involving option objects
        #     option_objs = set(segment.get_option().objects)
        #     filtered_effects = {eff for eff in necessary_add_effects 
        #                       if set(eff.objects).issubset(option_objs | set(obj_to_var.keys()))}
            
        #     # If still too many, prioritize effects with fewer objects
        #     if filtered_effects:
        #         all_objs = {o for eff in filtered_effects for o in eff.objects}
        #         missing_objs_filtered = sorted(all_objs - set(obj_to_var))
                
        #         if len(obj_to_var) + len(missing_objs_filtered) > CFG.max_operator_arity:
        #             # Take effects with fewest new objects first
        #             effects_by_new_objs = sorted(filtered_effects, 
        #                                         key=lambda eff: len(set(eff.objects) - set(obj_to_var.keys())))
                    
        #             # Greedily add effects until we hit the parameter limit
        #             kept_effects = set()
        #             current_objs = set(obj_to_var.keys())
        #             for eff in effects_by_new_objs:
        #                 new_objs = set(eff.objects) - current_objs
        #                 if len(current_objs) + len(new_objs) <= CFG.max_operator_arity:
        #                     kept_effects.add(eff)
        #                     current_objs.update(new_objs)
                    
        #             necessary_add_effects = kept_effects
        #         else:
        #             necessary_add_effects = filtered_effects
        #     else:
        #         # If no effects involve option objects, take first N effects
        #         effects_sorted = sorted(necessary_add_effects, 
        #                                key=lambda eff: len(eff.objects))
        #         kept_effects = set()
        #         current_objs = set(obj_to_var.keys())
        #         for eff in effects_sorted:
        #             new_objs = set(eff.objects) - current_objs
        #             if len(current_objs) + len(new_objs) <= CFG.max_operator_arity:
        #                 kept_effects.add(eff)
        #                 current_objs.update(new_objs)
        #         necessary_add_effects = kept_effects
            
        #     # Recalculate after filtering
        #     all_objs = {o for eff in necessary_add_effects for o in eff.objects}
        #     missing_objs = sorted(all_objs - set(obj_to_var))
        # #######

        new_vars = utils.create_new_variables([o.type for o in missing_objs],
                                              existing_vars=pnad.op.parameters)
        obj_to_var.update(dict(zip(missing_objs, new_vars)))
        # Finally, we can lift necessary_add_effects.
        updated_params = sorted(obj_to_var.values())
        
        # # TODO Assert that we stay within the arity limit
        # assert len(updated_params) <= CFG.max_operator_arity, \
        #     f"Operator would have {len(updated_params)} parameters " \
        #     f"(max {CFG.max_operator_arity} allowed). Filtering failed."
        
        updated_add_effects = {
            a.lift(obj_to_var)
            for a in necessary_add_effects
        }

        # Create a new PNAD with the given parameters and add effects. Set
        # the preconditions to be trivial. They will be recomputed later.
        new_pnad_op = pnad.op.copy_with(parameters=updated_params,
                                        preconditions=set(),
                                        add_effects=updated_add_effects)
        new_pnad = PNAD(new_pnad_op, [], pnad.option_spec)
        # Note: we don't need to copy anything related to keep effects into
        # new_pnad here, because we only care about keep effects on the final
        # iteration of backchaining, where this function is never called.

        return new_pnad

    @staticmethod
    def get_pnads_with_keep_effects(pnad: PNAD) -> Set[PNAD]:
        """Return a new set of PNADs that include keep effects into the given
        PNAD."""
        # The keep effects that we want are the subset of possible keep
        # effects which are not already in the PNAD's add effects, and
        # whose predicates were either (i) determined to be ignore effects,
        # or (ii) in the delete effects.
        keep_effects = {
            eff
            for eff in pnad.poss_keep_effects
            if eff not in pnad.op.add_effects and (
                eff.predicate in pnad.op.ignore_effects
                or eff in pnad.op.delete_effects)
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
                new_pnad = PNAD(new_pnad_op, [], pnad.option_spec)
                # Remember to copy seg_to_keep_effects_sub into the new_pnad!
                new_pnad.seg_to_keep_effects_sub = pnad.seg_to_keep_effects_sub
                new_pnads_with_keep_effects.add(new_pnad)
        return new_pnads_with_keep_effects

    def _reset_all_segment_necessary_add_effs(self) -> None:
        """Reset all segments' necessary_add_effects to None."""
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            for segment in seg_traj:
                segment.necessary_add_effects = None

    def _update_pnad_seg_to_keep_effs(self, pnad: PNAD,
                                      necessary_image: Set[GroundAtom],
                                      ground_op: _GroundSTRIPSOperator,
                                      obj_to_var: Dict[Object, Variable],
                                      segment: Segment) -> None:
        """Updates the pnad's seg_to_keep_effs_sub dictionary, which is
        necesssary for correctly grounding keep effects to data."""
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

    @staticmethod
    def clear_unnecessary_keep_effs(pnad: PNAD) -> None:
        """Clear the poss_keep_effects, as well as unnecessary substitution
        values from the PNAD's seg_to_keep_effects_sub_dict.

        A substitution is unnecessary if it concerns a variable that
        isn't in the PNAD's op parameters.
        """
        pnad.poss_keep_effects.clear()
        for segment, keep_eff_sub in pnad.seg_to_keep_effects_sub.items():
            new_keep_eff_sub_dict = {}
            for var, obj in keep_eff_sub.items():
                if var in pnad.op.parameters:
                    new_keep_eff_sub_dict[var] = obj
            pnad.seg_to_keep_effects_sub[segment] = new_keep_eff_sub_dict


class BackchainingSTRIPSLearner(GeneralToSpecificSTRIPSLearner):
    """Learn STRIPS operators by backchaining."""

    def _learn(self) -> List[PNAD]:
        # Initialize the most general PNADs by merging self._initial_pnads.
        # As a result, we will have one very general PNAD per option.
        param_opt_to_nec_pnads: Dict[ParameterizedOption, List[PNAD]] = {}
        # Extract all parameterized options from the data.
        parameterized_options = set()
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            for segment in seg_traj:
                parameterized_options.add(segment.get_option().parent)

        # Set up the param_opt_to_nec_pnads dictionary.
        for param_opt in parameterized_options:
            param_opt_to_nec_pnads[param_opt] = []

        prev_itr_ops: Set[STRIPSOperator] = set()

        # We loop until the harmless PNADs induced by our procedure
        # converge to a fixed point (i.e, they don't change after two
        # subsequent iterations).
        while True:
            # Run multiple passes of backchaining over the data until
            # convergence to a fixed point. Note that this process creates
            # operators with only parameters, preconditions, and add effects.
            self._backchain_multipass(param_opt_to_nec_pnads)

            # Induce delete effects, ignore effects and potentially
            # keep effects.
            self._induce_delete_side_keep(param_opt_to_nec_pnads)

            # Harmlessness should now hold, but it's slow to check.
            if CFG.backchaining_check_intermediate_harmlessness:
                assert self._check_harmlessness(
                    self._get_uniquely_named_nec_pnads(param_opt_to_nec_pnads))

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
            if {pnad.op for pnad in cur_itr_pnads_filtered} == prev_itr_ops:
                break

            prev_itr_ops = {pnad.op for pnad in cur_itr_pnads_filtered}

        # Assign a unique name to each PNAD.
        final_pnads = self._get_uniquely_named_nec_pnads(
            param_opt_to_nec_pnads)
        # Assert data has been correctly partitioned amongst PNADs.
        self._assert_all_data_in_exactly_one_datastore(final_pnads)
        return final_pnads

    def _backchain_multipass(
            self, param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                               List[PNAD]]) -> None:
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
                    self.clear_unnecessary_keep_effs(pnad)
            # Run one pass of backchaining.
            nec_pnad_set_changed = self._backchain_one_pass(
                param_opt_to_nec_pnads)
            if not nec_pnad_set_changed:
                break

    def _backchain_one_pass(
            self, param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                               List[PNAD]]) -> bool:
        """Take one pass through the demonstrations in the given order.

        Go through each one from the end back to the start, making the
        PNADs more specific whenever needed. Return whether any PNAD was
        changed.
        """
        # Reset all segments' necessary_add_effects so that they aren't
        # accidentally used from a previous iteration of backchaining.
        self._reset_all_segment_necessary_add_effs()
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
                # Find the necessary PNADs associated with this option. If
                # there are none, then use the general PNAD associated with
                # this option. (But make sure to use a copy of it, because we
                # don't want the general PNAD to get mutated when we mutate
                # necessary PNADs!)
                if len(param_opt_to_nec_pnads[option.parent]) == 0:
                    general_pnad = self._create_general_pnad_for_option(
                        option.parent)
                    pnads_for_option = [
                        PNAD(general_pnad.op, list(general_pnad.datastore),
                             general_pnad.option_spec)
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
                    segs_in_pnad = {
                        datapoint[0]
                        for datapoint in pnad.datastore
                    }
                    # In this case, we want to move the segment from
                    # another PNAD into the current PNAD. Note that
                    # we don't have to recompute the PNAD's add
                    # effects or preconditions because of the fact that
                    # this PNAD was found by the _find_best_matching
                    # function (which internally checks that the
                    # preconditions and add effects are all correct).
                    if segment not in segs_in_pnad:
                        # Find PNAD that the segment is currently in.
                        for seg_pnad in pnads_for_option:
                            segs_in_seg_pnad = [
                                datapoint[0]
                                for datapoint in seg_pnad.datastore
                            ]
                            if segment in set(segs_in_seg_pnad):
                                seg_idx = segs_in_seg_pnad.index(segment)
                                seg_pnad.datastore.pop(seg_idx)
                                break
                        pnad.datastore.append((segment, var_to_obj))
                        self._remove_empty_datastore_pnads(
                            param_opt_to_nec_pnads, option.parent)

                # If we weren't able to find a substitution (i.e, the above
                # _find_best_matching call didn't yield a PNAD), we need to
                # spawn a new PNAD from the most general PNAD to cover
                # these necessary add effects.
                else:
                    nec_pnad_set_changed = True
                    pnad = self.spawn_new_pnad(segment)
                    param_opt_to_nec_pnads[option.parent].append(pnad)

                    # Recompute datastores for ALL PNADs associated with this
                    # option. We need to do this because the new PNAD may now
                    # be a better match for some transition that we previously
                    # matched to another PNAD.
                    self._recompute_datastores_from_segments(
                        param_opt_to_nec_pnads[option.parent])
                    # Now that we have done this, certain PNADs may be
                    # left with empty datastores. Remove these.
                    self._remove_empty_datastore_pnads(param_opt_to_nec_pnads,
                                                       option.parent)

                    # Recompute all preconditions, now that we have recomputed
                    # the datastores.
                    for nec_pnad in param_opt_to_nec_pnads[option.parent]:
                        if len(nec_pnad.datastore) > 0:
                            pre = self._induce_preconditions_via_intersection(
                                nec_pnad)
                            nec_pnad.op = nec_pnad.op.copy_with(
                                preconditions=pre)

                    # After all this, the unification call that failed earlier
                    # (leading us into the current else statement) should work.
                    best_score_pnad, var_to_obj = \
                        self._find_best_matching_pnad_and_sub(
                        segment, objects,
                        param_opt_to_nec_pnads[option.parent])
                    assert var_to_obj is not None
                    assert best_score_pnad == pnad
                    # Also, since this segment caused us to induce the new
                    # PNAD, it should appear in this new PNAD's datastore.
                    segs_in_pnad = {
                        datapoint[0]
                        for datapoint in pnad.datastore
                    }
                    assert segment in segs_in_pnad
                    obj_to_var = {v: k for k, v in var_to_obj.items()}
                    assert len(var_to_obj) == len(obj_to_var)
                    ground_op = pnad.op.ground(
                        tuple(var_to_obj[var] for var in pnad.op.parameters))

                self._update_pnad_seg_to_keep_effs(pnad, necessary_image,
                                                   ground_op, obj_to_var,
                                                   segment)

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

    @staticmethod
    def _remove_empty_datastore_pnads(param_opt_to_nec_pnads: Dict[
        ParameterizedOption, List[PNAD]],
                                      param_opt: ParameterizedOption) -> None:
        """Removes all PNADs associated with the given param_opt that have
        empty datastores from the input param_opt_to_nec_pnads dict."""
        pnads_to_rm = []
        for pnad in param_opt_to_nec_pnads[param_opt]:
            if len(pnad.datastore) == 0:
                pnads_to_rm.append(pnad)
        for rm_pnad in pnads_to_rm:
            param_opt_to_nec_pnads[param_opt].remove(rm_pnad)

    def _induce_delete_side_keep(
            self, param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                               List[PNAD]]) -> None:
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
                self._compute_pnad_ignore_effects(pnad)
                pnads_with_keep_effects |= self.get_pnads_with_keep_effects(
                    pnad)
            param_opt_to_nec_pnads[option].extend(
                list(pnads_with_keep_effects))

    @classmethod
    def get_name(cls) -> str:
        return "backchaining"

    def _assert_all_data_in_exactly_one_datastore(self,
                                                  pnads: List[PNAD]) -> None:
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

class BackwardForwardSTRIPSLearner(GeneralToSpecificSTRIPSLearner):
    """Learn STRIPS operators by backchaining and forward search."""

    def _learn(self) -> List[PNAD]:
        # Initialize the most general PNADs by merging self._initial_pnads.
        # As a result, we will have one very general PNAD per option.
        param_opt_to_nec_pnads: Dict[ParameterizedOption, List[PNAD]] = {}
        # Extract all parameterized options from the data.
        parameterized_options = set()
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            for segment in seg_traj:
                parameterized_options.add(segment.get_option().parent)

        # Set up the param_opt_to_nec_pnads dictionary.
        for param_opt in parameterized_options:
            param_opt_to_nec_pnads[param_opt] = []

        prev_itr_ops: Set[STRIPSOperator] = set()

        # Load initial pnad set
        if CFG.backward_forward_load_initial:
            with open("test_saved.NSRTs.txt", "r") as file:
                content = file.read()
            nsrt_strs = ["NSRT-" + nsrt_str for nsrt_str in content.split("NSRT-") if nsrt_str != '']
            pnads = [self.parse_nsrt_block(nsrt_str) for nsrt_str in nsrt_strs]
            self._recompute_datastores_from_segments(pnads)
            for pnad in pnads:
                param_opt_to_nec_pnads[pnad.option_spec[0]].append(pnad)
                # TODO
                # if pnad.option_spec[0] in param_opt_to_nec_pnads:
                #     param_opt_to_nec_pnads[pnad.option_spec[0]].append(pnad)
                # else:
                #     print(f"Warning: PNAD with option {pnad.option_spec[0]} not in param_opt_to_nec_pnads")
        ###

        # We loop until the harmless PNADs induced by our procedure
        # converge to a fixed point (i.e, they don't change after two
        # subsequent iterations).
        while True:
            # Run multiple passes of backchaining over the data until
            # convergence to a fixed point. Note that this process creates
            # operators with only parameters, preconditions, and add effects.
            print("Backward-Forward STRIPS Learning Iteration")

            # Step 1: Run backchaining
            self._backchain_multipass(param_opt_to_nec_pnads)
            print("Backchaining multipass completed")

            # Induce delete effects, ignore effects and potentially
            # keep effects.
            self._induce_delete_side_keep(param_opt_to_nec_pnads)
            print("Inducing delete, ignore, and keep effects")

            # Harmlessness should now hold, but it's slow to check.
            if CFG.backchaining_check_intermediate_harmlessness:
                assert self._check_harmlessness(
                    self._get_uniquely_named_nec_pnads(param_opt_to_nec_pnads))
                print("Intermediate harmlessness check passed")
            print("Recomputing datastores and filtering out PNADs that don't have datastores")

            # Recompute datastores and filter out PNADs that don't have datastores.
            cur_itr_pnads_unfiltered = [
                pnad for pnads in param_opt_to_nec_pnads.values()
                for pnad in pnads
            ]
            self._recompute_datastores_from_segments(cur_itr_pnads_unfiltered)
            print("Finished recomputing datastores", len(cur_itr_pnads_unfiltered))
            cur_itr_pnads_filtered = []
            for pnad in cur_itr_pnads_unfiltered:
                if len(pnad.datastore) > 0:
                    # new_pre = self._induce_preconditions_via_intersection(pnad)
                    # NOTE: this implicitly changes param_opt_to_nec_pnads
                    # as well, since we're directly modifying the PNAD objects.
                    # nad.op = pnad.op.copy_with(preconditions=new_pre)
                    cur_itr_pnads_filtered.append(pnad)
                else:
                    param_opt_to_nec_pnads[pnad.option_spec[0]].remove(pnad)
            del cur_itr_pnads_unfiltered  # should be unused after this
            print("Current iteration PNADs filtered:", len(cur_itr_pnads_filtered))

            # Check if the PNAD set has converged. If so, break.
            if {pnad.op for pnad in cur_itr_pnads_filtered} == prev_itr_ops:
                print("No changes in this pass, backchaining has reached a fixed point")
                break

            prev_itr_ops = {pnad.op for pnad in cur_itr_pnads_filtered}

        ######
        # Step 2 & 3: Fixed forward refinement (strips and re-adds preconditions/ignore_effects)
        self._fixed_forward_one_pass(param_opt_to_nec_pnads)

        # # Recompute datastores.
        # cur_itr_pnads_unfiltered = [
        #     pnad for pnads in param_opt_to_nec_pnads.values()
        #     for pnad in pnads
        # ]
        # self._recompute_datastores_from_segments(cur_itr_pnads_unfiltered, check_only_preconditions=True, check_assertion=False)
        ######

        # Assign a unique name to each PNAD.
        final_pnads = self._get_uniquely_named_nec_pnads(
            param_opt_to_nec_pnads)
        # Assert data has been correctly partitioned amongst PNADs.
        # self._assert_all_data_in_exactly_one_datastore(final_pnads)
        return final_pnads
    
    def parse_nsrt_block(self, block: str) -> PNAD:
        """Parses a single NSRT block into an PNAD object."""
        lines = block.strip().split("\n")
        
        name_match = re.match(r"(\S+):", lines[0])
        name = name_match.group(1) if name_match else ""

        parameters = re.findall(r"\?x\d+:\w+", lines[1])
        
        def extract_effects(label: str) -> Set[str]:
            """Extracts a list of predicates from labeled sections."""
            for line in lines:
                if line.strip().startswith(label):
                    return set(re.findall(r"\w+\(.*?\)", line))
            return set()
        
        preconditions = extract_effects("Preconditions")
        add_effects = extract_effects("Add Effects")
        delete_effects = extract_effects("Delete Effects")
        ignore_effects = extract_effects("Ignore Effects")

        option_spec_match = re.search(r"Option Spec:\s*(.*)", block)
        option_spec = option_spec_match.group(1) if option_spec_match else ""

        objects = set()
        atoms = set()
        option_specs = {}
        for traj in self._segmented_trajs:
            for segment in traj:
                for state in segment.states:
                    for k, v in state.items():
                        objects.add(k)
                atoms |= segment.init_atoms | segment.final_atoms
                option_specs[segment.get_option().parent.name] = segment.get_option().parent
        all_predicates_list = [(atom.predicate.name,atom.predicate) for atom in atoms]
        def get_predicate(name, entities):
            for pred_name, pred in all_predicates_list:
                if pred_name == pred_name and pred.arity == len(entities):
                    valid_types = True
                    for i, ent in enumerate(entities):
                        if ent.type != pred.types[i]:
                            valid_types = False
                    if valid_types:
                        return pred
            raise NotImplementedError
            
        types = {obj.type.name:obj.type for obj in objects}

        def extract_parameters(predicate: str) -> Set[str]:
            parameter_pattern = re.compile(r"\?x\d+:\w+")  # Matches variables like ?x0:obj_type
            matches = parameter_pattern.findall(predicate)
            return matches
        
        parameters = [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in parameters]
        preconditions = set([LiftedAtom(get_predicate(pre.split("(")[0], [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(pre)]), [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(pre)]) for pre in preconditions])
        add_effects = set([LiftedAtom(get_predicate(add.split("(")[0], [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(add)]), [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(add)]) for add in add_effects])
        delete_effects = set([LiftedAtom(get_predicate(dle.split("(")[0], [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(dle)]), [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(dle)]) for dle in delete_effects])
        ignore_effects = set([get_predicate(ige, None) for ige in ignore_effects])
        if option_spec.split("(")[0] in option_specs:
            option_spec = (option_specs[option_spec.split("(")[0]], [])
        else:
            a_name = option_spec.split("(")[0]
            option_spec = utils.SingletonParameterizedOption(
                a_name, lambda s, m, o, p: Action(name_to_actions[a_name]))
            print("ADDED OPTION", a_name)

        nsrt = NSRT(name, parameters, preconditions, add_effects, delete_effects, ignore_effects, option_spec, [], None)
        return PNAD(nsrt.op, [], option_spec)
    
    def _forward_one_pass(
            self, param_opt_to_nec_pnads: Dict[ParameterizedOption, List[PNAD]]
        ) -> None:
        """Perform one forward search passes to refine PNAD preconditions
        """

        for ll_traj, seg_traj in zip(self._trajectories, self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            task = self._train_tasks[ll_traj.train_task_idx]

            # Get initial atoms and object list
            objects, _, _, ground_atoms_traj, _ = parse_objs_preds_and_options(
                ll_traj, train_task_idx=ll_traj.train_task_idx)

            while True:
                # TODO continue until plans match demo from start
                init_atoms = ground_atoms_traj[1][0]

                # plan with current nsrts
                nsrts = [pnad.op for pnads in param_opt_to_nec_pnads.values()
                        for pnad in pnads]
                nsrt_to_option = {pnad.op:pnad.option_spec for pnads in param_opt_to_nec_pnads.values() for pnad in pnads}
                predicates = self._predicates

                # Plan using current operators
                ground_nsrts, reachable_atoms = task_plan_grounding(
                    init_atoms, objects, nsrts, allow_noops=True)
                heuristic = utils.create_task_planning_heuristic(
                    "hadd", init_atoms, task.goal, ground_nsrts,
                    predicates, objects)
                task_plan_generator = task_plan(
                    init_atoms, task.goal, ground_nsrts,
                    reachable_atoms, heuristic,
                    timeout=100, seed=123, max_skeletons_optimized=3)
                
                skeleton, _, _ = next(task_plan_generator)

                # Check if plan matches the actual low-level trajectory
                planned_options = []
                for step in skeleton:
                    planned_options.append(nsrt_to_option[step.parent][0])

                last_mistakes = set()
                for i, planned_option in enumerate(planned_options):
                    curr_traj = seg_traj[i]
                    print("GT vs Our Plan")
                    print(i, curr_traj.get_option().name, "=?=", planned_option.name)
                    if curr_traj.get_option().name != planned_option.name:
                        # skip repeated mistakes
                        print(i, curr_traj.get_option().name, "is not", planned_option.name)
                        if (i, curr_traj.get_option().name, planned_option.name) in last_mistakes:
                            print("Skipping repeated mistake")
                            continue
                        last_mistakes.add((i, curr_traj.get_option().name, planned_option.name))
                        # TODO should not just be the first
                        pnad = None
                        for option_pnad in param_opt_to_nec_pnads[planned_option]:
                            if pnad is None:
                                pnad = option_pnad
                            if len(option_pnad.op.preconditions) < len(pnad.op.preconditions):
                                pnad = option_pnad
                        positive_data = pnad.datastore
                        diff_atoms = []
                        diff_preds = []
                        non_nec_diff_atoms = []
                        non_nec_diff_preds = []
                        necessary_effects = set.union(*[seg.necessary_add_effects for seg in seg_traj])

                        ####
                        # Lift atoms from each positive example using their substitutions
                        lifted_atoms_list = []
                        for pos_seg in positive_data:
                            segment, var_to_obj = pos_seg
                            obj_to_var = {v: k for k, v in var_to_obj.items()}
                            
                            # Lift the init_atoms by substituting objects with variables
                            lifted_atoms = set()
                            for atom in segment.init_atoms:
                                #print(atom)
                                lifted_objs = [obj_to_var.get(obj, obj) for obj in atom.objects]
                                # Only include if all objects were successfully mapped to variables
                                if all(isinstance(o, Variable) for o in lifted_objs):
                                    lifted_atoms.add(LiftedAtom(atom.predicate, lifted_objs))
                            lifted_atoms_list.append(lifted_atoms)
                        
                        # Find intersection of lifted atoms across all positive examples
                        if lifted_atoms_list:
                            common_lifted_atoms = set.intersection(*lifted_atoms_list) if lifted_atoms_list else set()
                            
                            # Separate into necessary and non-necessary based on predicates
                            necessary_lifted = {atom for atom in common_lifted_atoms 
                                              if any(atom.predicate == nec_atom.predicate for nec_atom in necessary_effects)}
                            non_necessary_lifted = common_lifted_atoms - necessary_lifted
                            
                            diff_atoms.append(necessary_lifted)
                            diff_preds.append({atom.predicate for atom in necessary_lifted})
                            non_nec_diff_atoms.append(non_necessary_lifted)
                            non_nec_diff_preds.append({atom.predicate for atom in non_necessary_lifted})

                        ####

                        new_pre = set()
                        new_params = []
                        print()
                        print(planned_option, set.intersection(*[s for s in diff_preds]))
                        new_preds = set.intersection(*[s for s in diff_preds])
                        if len(new_preds) <= 0:
                            new_preds = set.intersection(*[s for s in non_nec_diff_preds])
                        if new_preds != set():
                            for pred in new_preds:
                                best_pnad, best_sub = self._find_best_matching_pnad_and_sub(positive_data[0][0], objects, param_opt_to_nec_pnads[planned_option], check_only_preconditions=True, check_assertion=False, any_matching=True)
                                pred_objs = [atom.objects for atom in positive_data[0][0].init_atoms if atom.predicate == pred][0]
                                print(pred_objs)
                                obj_vars = {v:k for k,v in best_sub.items()}
                                if best_pnad is not None:
                                    params = []
                                    for obj in pred_objs:
                                        if obj in obj_vars:
                                            params.append(obj_vars[obj])
                                        else:
                                            params.append(Variable("?x" + str(len(obj_vars.keys())), obj.type))
                                    new_pre.add(LiftedAtom(pred, params))
                                    new_params += params
                                print(params)
                                print(new_params)
                                print(pnad)
                            if len(new_pre) > len(pnad.op.preconditions):
                                # randomly/incrementally add one of the different predicates to new pnad
                                import random
                                single_new_pre = random.choice(list(new_pre - pnad.op.preconditions))
                                updated_params = list(set(pnad.op.parameters + single_new_pre.variables))
                                updated_preconditions = set(list(pnad.op.preconditions) + [single_new_pre])
                                pnad.op = pnad.op.copy_with(parameters=updated_params,preconditions=updated_preconditions)
                        else:
                            # TODO No new predicates to differentiate
                            pass
                        print("Updated PNAD:", pnad)
                else:
                    break
                                

                # # Check for convergence
                # cur_op_set = {pnad.op for pnads in param_opt_to_nec_pnads.values()
                #             for pnad in pnads}
                # if cur_op_set == prev_op_set:
                #     break
                # prev_op_set = cur_op_set

    def _fixed_forward_one_pass(
            self, param_opt_to_nec_pnads: Dict[ParameterizedOption, List[PNAD]]
        ) -> None:
        """Simplified forward pass: strips preconditions and adds them back 
        until replanned trajectories match demos.
        """
        import random

        # Step 1: Save original preconditions and strip all preconditions
        original_pnads = {}
        for option, pnads in param_opt_to_nec_pnads.items():
            original_pnads[option] = []
            for pnad in pnads:
                original_pnads[option].append({
                    'preconditions': set(pnad.op.preconditions),
                    'ignore_effects': set(pnad.op.ignore_effects)
                })
                pnad.op = pnad.op.copy_with(preconditions=set())
        
        # Step 2: Ensure all operators have at least one precondition
        print("\n=== Ensuring all operators have at least one precondition ===")
        for option, pnads in param_opt_to_nec_pnads.items():
            for idx, pnad in enumerate(pnads):
                if len(pnad.op.preconditions) == 0:
                    original = original_pnads[option][idx]
                    if len(original['preconditions']) > 0:
                        new_pre = random.choice(list(original['preconditions']))
                        updated_params = list(set(pnad.op.parameters + list(new_pre.variables)))
                        updated_preconditions = {new_pre}
                        pnad.op = pnad.op.copy_with(
                            parameters=updated_params,
                            preconditions=updated_preconditions
                        )
                        print(f"  Added minimal precondition {new_pre} to {option.name}")
        
        # Step 3: Iteratively add preconditions back until all plans match demos
        max_iterations = 100
        print(f"\n=== Forward Refinement: Adding preconditions until plans match demos ===")
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}")
            all_match = True
            
            # Check each demo trajectory
            for traj_idx, (ll_traj, seg_traj) in enumerate(zip(self._trajectories, self._segmented_trajs)):
                if not ll_traj.is_demo:
                    continue
                
                task = self._train_tasks[ll_traj.train_task_idx]
                objects, _, _, ground_atoms_traj, _ = parse_objs_preds_and_options(
                    ll_traj, train_task_idx=ll_traj.train_task_idx)
                
                init_atoms = ground_atoms_traj[1][0]

                # Recompute datastores and filter out PNADs that don't have datastores.
                cur_itr_pnads_unfiltered = [
                    pnad for pnads in param_opt_to_nec_pnads.values()
                    for pnad in pnads
                ]
                self._recompute_datastores_from_segments(cur_itr_pnads_unfiltered)
                cur_itr_pnads_filtered = []
                for pnad in cur_itr_pnads_unfiltered:
                    if len(pnad.datastore) > 0:
                        # new_pre = self._induce_preconditions_via_intersection(pnad)
                        # NOTE: this implicitly changes param_opt_to_nec_pnads
                        # as well, since we're directly modifying the PNAD objects.
                        # nad.op = pnad.op.copy_with(preconditions=new_pre)
                        cur_itr_pnads_filtered.append(pnad)
                    else:
                        param_opt_to_nec_pnads[pnad.option_spec[0]].remove(pnad)
                del cur_itr_pnads_unfiltered  # should be unused after this
                #
                
                # Plan from initial state to goal
                nsrts = [pnad.op for pnads in param_opt_to_nec_pnads.values()
                        for pnad in pnads]
                nsrt_to_pnad = {pnad.op: pnad for pnads in param_opt_to_nec_pnads.values() 
                               for pnad in pnads}
                nsrt_to_option = {pnad.op: pnad.option_spec[0] 
                                 for pnads in param_opt_to_nec_pnads.values() 
                                 for pnad in pnads}
                
                try:
                    ground_nsrts, reachable_atoms = task_plan_grounding(
                        init_atoms, objects, nsrts, allow_noops=True)
                    heuristic = utils.create_task_planning_heuristic(
                        "hadd", init_atoms, task.goal, ground_nsrts,
                        self._predicates, objects)
                    task_plan_generator = task_plan(
                        init_atoms, task.goal, ground_nsrts,
                        reachable_atoms, heuristic,
                        timeout=100, seed=123, max_skeletons_optimized=3)
                    skeleton, _, _ = next(task_plan_generator)
                except (StopIteration, Exception) as e:
                    print(f"  Trajectory {traj_idx}: Failed to plan - {e}")
                    all_match = False
                    continue
                
                # Compare plan to demo
                planned_options = [nsrt_to_option[ground_nsrt.parent] for ground_nsrt in skeleton]
                demo_options = [seg.get_option().parent for seg in seg_traj]
                
                # Find first mismatch
                mismatch_idx = None
                for i, (planned_opt, demo_opt) in enumerate(zip(planned_options, demo_options)):
                    if planned_opt != demo_opt:
                        mismatch_idx = i
                        break
                
                if mismatch_idx is not None or len(planned_options) != len(demo_options):
                    all_match = False
                    
                    # Add a precondition to the wrongly chosen operator
                    if mismatch_idx is not None and mismatch_idx < len(skeleton):
                        wrong_ground_nsrt = skeleton[mismatch_idx]
                        wrong_pnad = nsrt_to_pnad[wrong_ground_nsrt.parent]
                        wrong_option = nsrt_to_option[wrong_ground_nsrt.parent]
                        demo_option = demo_options[mismatch_idx]
                        
                        print(f"  Trajectory {traj_idx}, Step {mismatch_idx}: {wrong_option.name} != {demo_option.name}")
                        
                        # Get original preconditions for this operator
                        option_idx = list(param_opt_to_nec_pnads[wrong_option]).index(wrong_pnad)
                        original = original_pnads[wrong_option][option_idx]
                        
                        # Find preconditions to add (ones that aren't already added)
                        available_pres = original['preconditions'] - wrong_pnad.op.preconditions
                        
                        if available_pres:
                            # Add one random precondition
                            new_pre = random.choice(list(available_pres))
                            updated_params = list(set(wrong_pnad.op.parameters + list(new_pre.variables)))
                            updated_preconditions = wrong_pnad.op.preconditions | {new_pre}
                            wrong_pnad.op = wrong_pnad.op.copy_with(
                                parameters=updated_params,
                                preconditions=updated_preconditions
                            )
                            print(f"    Added precondition {new_pre} to {wrong_option.name}")
                            # self._induce_delete_side_keep(param_opt_to_nec_pnads)

                            break  # Only fix one mismatch per iteration
                        else:
                            print(f"    No more preconditions available for {wrong_option.name}")
                            print(f"    Restoring all original preconditions for all operators")
                            # Restore original preconditions for all PNADs
                            for option, pnads in param_opt_to_nec_pnads.items():
                                for idx, pnad in enumerate(pnads):
                                    orig = original_pnads[option][idx]
                                    # Get all variables from original preconditions
                                    all_vars = set(pnad.op.parameters)
                                    for pre in orig['preconditions']:
                                        all_vars.update(pre.variables)
                                    pnad.op = pnad.op.copy_with(
                                        parameters=sorted(all_vars),
                                        preconditions=orig['preconditions'],
                                        ignore_effects=orig['ignore_effects']
                                    )
                            # self._induce_delete_side_keep(param_opt_to_nec_pnads)
                            all_match = True  # Exit loop since we've restored originals
                            break
                    break  # Move to next iteration after finding first trajectory mismatch
            
            if all_match:
                print(f"\n✓ All trajectories match demos after {iteration + 1} iterations!")
                break
        
        # # Final verification: replan from init to goal and assert equivalence to demos
        # print("\n=== Final Verification: Checking plans match demos ===")
        # for ll_traj, seg_traj in zip(self._trajectories, self._segmented_trajs):
        #     if not ll_traj.is_demo:
        #         continue
            
        #     task = self._train_tasks[ll_traj.train_task_idx]
        #     objects, _, _, ground_atoms_traj, _ = parse_objs_preds_and_options(
        #         ll_traj, train_task_idx=ll_traj.train_task_idx)
            
        #     init_atoms = ground_atoms_traj[1][0]
            
        #     # Plan with final operators
        #     nsrts = [pnad.op for pnads in param_opt_to_nec_pnads.values()
        #             for pnad in pnads]
        #     nsrt_to_option = {pnad.op: pnad.option_spec[0] 
        #                      for pnads in param_opt_to_nec_pnads.values() 
        #                      for pnad in pnads}
            
        #     try:
        #         ground_nsrts, reachable_atoms = task_plan_grounding(
        #             init_atoms, objects, nsrts, allow_noops=True)
        #         heuristic = utils.create_task_planning_heuristic(
        #             "hadd", init_atoms, task.goal, ground_nsrts,
        #             self._predicates, objects)
        #         task_plan_generator = task_plan(
        #             init_atoms, task.goal, ground_nsrts,
        #             reachable_atoms, heuristic,
        #             timeout=100, seed=123, max_skeletons_optimized=3)
        #         skeleton, _, _ = next(task_plan_generator)
        #     except (StopIteration, Exception) as e:
        #         print(f"Failed to plan for trajectory: {e}")
        #         assert False, f"Could not generate plan for demo trajectory"
            
        #     # Compare planned options to demo options
        #     planned_options = [nsrt_to_option[ground_nsrt.parent] for ground_nsrt in skeleton]
        #     demo_options = [seg.get_option().parent for seg in seg_traj]
            
        #     print(f"\nDemo trajectory {ll_traj.train_task_idx}:")
        #     print(f"  Demo options:   {[opt.name for opt in demo_options]}")
        #     print(f"  Planned options: {[opt.name for opt in planned_options]}")
            
        #     # Assert equivalence
        #     assert len(planned_options) == len(demo_options), \
        #         f"Plan length mismatch: {len(planned_options)} vs {len(demo_options)}"
            
        #     for i, (planned_opt, demo_opt) in enumerate(zip(planned_options, demo_options)):
        #         assert planned_opt == demo_opt, \
        #             f"Step {i}: planned {planned_opt.name} != demo {demo_opt.name}"
            
        #     print(f"  ✓ Plan matches demo!")
        
        # print("\n=== All plans match demos successfully! ===\n")

    def _try_lift_atom(self, ground_atom: GroundAtom, ground_objects: Sequence[Object],
                      parameters: Sequence[Variable]) -> Optional[LiftedAtom]:
        """Try to lift a ground atom using a mapping from objects to parameters."""
        # Create object to variable mapping
        obj_to_var = {}
        for i, (obj, param) in enumerate(zip(ground_objects, parameters)):
            if obj.type == param.type:
                obj_to_var[obj] = param
        
        # Try to lift the atom
        lifted_objs = []
        for obj in ground_atom.objects:
            if obj in obj_to_var:
                lifted_objs.append(obj_to_var[obj])
            else:
                # Can't lift this atom with current parameters
                return None
        
        return LiftedAtom(ground_atom.predicate, lifted_objs)

    def _backchain_multipass(
            self, param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                               List[PNAD]]) -> None:
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
                    self.clear_unnecessary_keep_effs(pnad)
            # Run one pass of backchaining.
            nec_pnad_set_changed = self._backchain_one_pass(
                param_opt_to_nec_pnads)
            
            print("inner pass of backchaining")
            if not nec_pnad_set_changed:
                print("no changes in this pass, backchaining has reached a fixed point")
                break

    def _backchain_one_pass(
            self, param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                               List[PNAD]]) -> bool:
        """Take one pass through the demonstrations in the given order.

        Go through each one from the end back to the start, making the
        PNADs more specific whenever needed. Return whether any PNAD was
        changed.
        """
        # Reset all segments' necessary_add_effects so that they aren't
        # accidentally used from a previous iteration of backchaining.
        self._reset_all_segment_necessary_add_effs()
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
                # Find the necessary PNADs associated with this option. If
                # there are none, then use the general PNAD associated with
                # this option. (But make sure to use a copy of it, because we
                # don't want the general PNAD to get mutated when we mutate
                # necessary PNADs!)
                if len(param_opt_to_nec_pnads[option.parent]) == 0:
                    general_pnad = self._create_general_pnad_for_option(
                        option.parent)
                    pnads_for_option = [
                        PNAD(general_pnad.op, list(general_pnad.datastore),
                             general_pnad.option_spec)
                    ]
                else:
                    pnads_for_option = param_opt_to_nec_pnads[option.parent]

                # Compute the ground atoms that must be added on this timestep.
                # They must be a subset of the current PNAD's add effects.
                necessary_add_effects = necessary_image - atoms_seq[t]
                necessary_objects = set()
                if len(necessary_add_effects) > 0:
                    necessary_objects = set.union(*[set(a.objects) for a in (list(necessary_add_effects))])
                if len(necessary_objects) > CFG.max_operator_arity:
                    from collections import Counter
                    new_necessary_objects = set([item for item, count in Counter([next(iter(a.objects)) for a in necessary_add_effects]).most_common(CFG.max_operator_arity)])
                    necessary_add_effects = set([a for a in necessary_add_effects if set(a.objects).issubset(new_necessary_objects)])
                if not necessary_add_effects.issubset(segment.add_effects):
                    necessary_add_effects = segment.add_effects & necessary_add_effects
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
                    segs_in_pnad = {
                        datapoint[0]
                        for datapoint in pnad.datastore
                    }
                    # In this case, we want to move the segment from
                    # another PNAD into the current PNAD. Note that
                    # we don't have to recompute the PNAD's add
                    # effects or preconditions because of the fact that
                    # this PNAD was found by the _find_best_matching
                    # function (which internally checks that the
                    # preconditions and add effects are all correct).
                    if segment not in segs_in_pnad:
                        # Find PNAD that the segment is currently in.
                        for seg_pnad in pnads_for_option:
                            segs_in_seg_pnad = [
                                datapoint[0]
                                for datapoint in seg_pnad.datastore
                            ]
                            if segment in set(segs_in_seg_pnad):
                                seg_idx = segs_in_seg_pnad.index(segment)
                                seg_pnad.datastore.pop(seg_idx)
                                break
                        pnad.datastore.append((segment, var_to_obj))
                        self._remove_empty_datastore_pnads(
                            param_opt_to_nec_pnads, option.parent)

                # If we weren't able to find a substitution (i.e, the above
                # _find_best_matching call didn't yield a PNAD), we need to
                # spawn a new PNAD from the most general PNAD to cover
                # these necessary add effects.
                else:
                    nec_pnad_set_changed = True
                    pnad = self.spawn_new_pnad(segment)
                    param_opt_to_nec_pnads[option.parent].append(pnad)

                    # Recompute datastores for ALL PNADs associated with this
                    # option. We need to do this because the new PNAD may now
                    # be a better match for some transition that we previously
                    # matched to another PNAD.
                    self._recompute_datastores_from_segments(
                        param_opt_to_nec_pnads[option.parent])
                    # Now that we have done this, certain PNADs may be
                    # left with empty datastores. Remove these.
                    self._remove_empty_datastore_pnads(param_opt_to_nec_pnads,
                                                       option.parent)

                    # Recompute all preconditions, now that we have recomputed
                    # the datastores.
                    for nec_pnad in param_opt_to_nec_pnads[option.parent]:
                        if len(nec_pnad.datastore) > 0:
                            pre = self._induce_preconditions_via_intersection(
                                nec_pnad)
                            nec_pnad.op = nec_pnad.op.copy_with(
                                preconditions=pre)
                            
                    # # assert that op arity is less than max_arity
                    # assert len(pnad.op.parameters) <= CFG.max_operator_arity 

                    # After all this, the unification call that failed earlier
                    # (leading us into the current else statement) should work.
                    best_score_pnad, var_to_obj = \
                        self._find_best_matching_pnad_and_sub(
                        segment, objects,
                        param_opt_to_nec_pnads[option.parent])

                    assert var_to_obj is not None
                    # TODO #assert best_score_pnad == pnad
                    # Also, since this segment caused us to induce the new
                    # PNAD, it should appear in this new PNAD's datastore.
                    segs_in_pnad = {
                        datapoint[0]
                        for datapoint in pnad.datastore
                    }
                    if segment not in segs_in_pnad:
                        import ipdb; ipdb.set_trace()
                    assert segment in segs_in_pnad
                    obj_to_var = {v: k for k, v in var_to_obj.items()}
                    assert len(var_to_obj) == len(obj_to_var)
                    ground_op = pnad.op.ground(
                        tuple(var_to_obj[var] for var in pnad.op.parameters))
                    

                self._update_pnad_seg_to_keep_effs(pnad, necessary_image,
                                                   ground_op, obj_to_var,
                                                   segment)

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

    @staticmethod
    def _remove_empty_datastore_pnads(param_opt_to_nec_pnads: Dict[
        ParameterizedOption, List[PNAD]],
                                      param_opt: ParameterizedOption) -> None:
        """Removes all PNADs associated with the given param_opt that have
        empty datastores from the input param_opt_to_nec_pnads dict."""
        pnads_to_rm = []
        for pnad in param_opt_to_nec_pnads[param_opt]:
            if len(pnad.datastore) == 0:
                pnads_to_rm.append(pnad)
        for rm_pnad in pnads_to_rm:
            param_opt_to_nec_pnads[param_opt].remove(rm_pnad)

    def _induce_delete_side_keep(
            self, param_opt_to_nec_pnads: Dict[ParameterizedOption,
                                               List[PNAD]]) -> None:
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
                self._compute_pnad_ignore_effects(pnad)
                pnads_with_keep_effects |= self.get_pnads_with_keep_effects(
                    pnad)
            param_opt_to_nec_pnads[option].extend(
                list(pnads_with_keep_effects))

    @classmethod
    def get_name(cls) -> str:
        return "backward-forward"

    def _assert_all_data_in_exactly_one_datastore(self,
                                                  pnads: List[PNAD]) -> None:
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

def parse_objs_preds_and_options(trajectory, train_task_idx=0, all_atoms=None):
    objs = set()
    preds = set()
    options = set()
    state = None
    states = []
    actions = []
    ground_atoms_traj = []
    obj_types = {"obj_type": Type("obj_type", ["is_obj"]), "surface_type": Type("surface_type", ["is_obj"])}
    
    for i, s in enumerate(trajectory.states):
        ground_atoms = set()
        for pred_str in s:
            pred = None
            choice = []
            pattern = re.compile(r"(\w+)\((.*?)\)")
            match = pattern.match(pred_str)
            if match:
                func_name = match.group(1)
                args = match.group(2).split(',') if match.group(2) else []
                for arg in args:
                    base_name = arg.strip().split("_")[0]
                    if base_name in ['box','cabinet','table','sink','bucket', 'ashcan']:
                        obj_types[base_name] = Type("surface_type", ["is_obj"])
                    else:
                        obj_types[base_name] = Type("obj_type", ["is_obj"]) #Type(base_name, ["is_obj"])
                    obj = obj_types[base_name](arg.strip())
                    choice.append(obj)
                    objs.add(obj)
                if len(args) == 1:
                    base_name = args[0].strip().split("_")[0]
                    pred = Predicate(func_name, [obj_types[base_name]], lambda s, o: True)
                    preds.add(pred)
                elif len(args) == 2:
                    base_name1 = args[0].strip().split("_")[0]
                    base_name2 = args[1].strip().split("_")[0]
                    pred = Predicate(func_name, [obj_types[base_name1], obj_types[base_name2]], lambda s, o: True)
                    if not(func_name == 'atsamelocation' and base_name1 == base_name2):
                        preds.add(pred)
                else:
                    NotImplementedError("")
            ground_atoms.add(GroundAtom(pred, choice))
        states.append(state)
        ground_atoms_traj.append(ground_atoms)

        if i < len(trajectory.actions):
            a_name = trajectory.actions[i]

            param_option = utils.SingletonParameterizedOption(
                a_name, lambda s, m, o, p: Action(name_to_actions[a_name]))
            options.add(param_option)
            option = param_option.ground([], [])
            action = option.policy(state)
            action.set_option(option)
            actions.append(action)

    def get_all_atoms_in_traj(ground_atoms_traj):
        all_atoms = set()
        for timestep_atoms in ground_atoms_traj:
            all_atoms.update(timestep_atoms)
        return all_atoms
    
    def add_neg_atoms(preds, lltraj, all_atoms):
        ground_atoms = []
        neg_pred_table = {str(atom):GroundAtom(Predicate("~" + atom.predicate.name, atom.predicate.types, lambda s, o: True), atom.objects) for atom in all_atoms}
        neg_pred_table["HandEmpty"] = GroundAtom(Predicate("handempty", [], lambda s, o: True), [])
        for timestep_atoms in lltraj[1]:
            missing_atoms = all_atoms - timestep_atoms
            neg_atoms = set([neg_pred_table[str(atom)] for atom in missing_atoms])
            handempty = True
            for atom in timestep_atoms:
                if "inhandofrobot" in str(atom):
                    handempty = False
            if handempty:
                neg_atoms |= set([neg_pred_table["HandEmpty"]])
            ground_atoms.append(timestep_atoms | neg_atoms)
        lltraj = (lltraj[0], ground_atoms)
        return preds | set([v.predicate for v in neg_pred_table.values()]) | set([atom.predicate for atom in all_atoms]), lltraj
    
    lltraj = (LowLevelTrajectory([{obj:[0.0] for obj in objs} for _ in states], actions, _is_demo=True, _train_task_idx=train_task_idx), ground_atoms_traj)
    if all_atoms is None:
        all_atoms = get_all_atoms_in_traj(ground_atoms_traj)
        preds, lltraj = add_neg_atoms(preds, lltraj, all_atoms)
    else:
        preds, lltraj = add_neg_atoms(preds, lltraj, all_atoms)
    
    return objs, preds, options, lltraj, all_atoms
