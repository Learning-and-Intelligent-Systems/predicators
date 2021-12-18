"""Algorithms for learning the various components of NSRT objects.
"""

import functools
from typing import Dict, Set, Tuple, List, Sequence, FrozenSet
from predicators.src.structs import Dataset, STRIPSOperator, NSRT, \
    GroundAtom, LiftedAtom, Variable, Predicate, ObjToVarSub, \
    LowLevelTrajectory, Segment, Partition, Object, GroundAtomTrajectory, \
    DefaultOption, ParameterizedOption, State, Action
from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.sampler_learning import learn_samplers
from predicators.src.option_learning import create_option_learner


def learn_nsrts_from_data(dataset: Dataset, predicates: Set[Predicate],
                          do_sampler_learning: bool) -> Set[NSRT]:
    """Learn NSRTs from the given dataset of transitions.
    States are abstracted using the given set of predicates.
    """
    print(f"\nLearning NSRTs on {len(dataset)} trajectories...")

    # Apply predicates to dataset.
    ground_atom_dataset = utils.create_ground_atom_dataset(dataset, predicates)

    # Segment transitions based on changes in predicates.
    demo_traj_segments, nondemo_segments, demo_goals = [], [], []
    for traj in ground_atom_dataset:
        segmented_traj = segment_trajectory(traj)
        ll_traj = traj[0]
        if ll_traj.is_demo:
            demo_traj_segments.append(segmented_traj)
            demo_goals.append(ll_traj.goal)
        else:
            nondemo_segments.extend(segmented_traj)

    # Learn strips operators.
    strips_ops, partitions = learn_strips_operators_from_demos(
        demo_traj_segments, demo_goals, nondemo_segments,
        verbose=CFG.do_option_learning)
    assert len(strips_ops) == len(partitions)

    # Learn option specs, or if known, just look them up. The order of
    # the options corresponds to the strips_ops. Each spec is a
    # (ParameterizedOption, Sequence[Variable]) tuple with the latter
    # holding the option_vars. After learning the specs, update the
    # segments to include which option is being executed within each
    # segment, so that sampler learning can utilize this.
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, partitions)
    assert len(option_specs) == len(strips_ops)
    # Seed the new parameterized option parameter spaces.
    for parameterized_option, _ in option_specs:
        parameterized_option.params_space.seed(CFG.seed)
    # Update the segments to include which option is being executed.
    for partition, spec in zip(partitions, option_specs):
        for (segment, _) in partition:
            # Modifies segment in-place.
            option_learner.update_segment_from_option_spec(segment, spec)

    # For the impatient, print out the STRIPSOperators with their option specs.
    print("\nLearned operators with option specs:")
    for strips_op, (option, option_vars) in zip(strips_ops, option_specs):
        print(strips_op)
        option_var_str = ", ".join([str(v) for v in option_vars])
        print(f"    Option Spec: {option.name}({option_var_str})")

    # Learn samplers.
    # The order of the samplers also corresponds to strips_ops.
    samplers = learn_samplers(strips_ops, partitions, option_specs,
                              do_sampler_learning)
    assert len(samplers) == len(strips_ops)

    # Create final NSRTs.
    nsrts = []
    for op, option_spec, sampler in zip(strips_ops, option_specs, samplers):
        param_option, option_vars = option_spec
        nsrt = op.make_nsrt(param_option, option_vars, sampler)
        nsrts.append(nsrt)

    print("\nLearned NSRTs:")
    for nsrt in sorted(nsrts):
        print(nsrt)
    print()

    return set(nsrts)


def segment_trajectory(trajectory: GroundAtomTrajectory) -> List[Segment]:
    """Segment a ground atom trajectory according to abstract state changes.

    If options are available, also use them to segment.
    """
    segments = []
    traj, all_atoms = trajectory
    assert len(traj.states) == len(all_atoms)
    current_segment_states : List[State] = []
    current_segment_actions : List[Action] = []
    for t in range(len(traj.actions)):
        current_segment_states.append(traj.states[t])
        current_segment_actions.append(traj.actions[t])
        switch = all_atoms[t] != all_atoms[t+1]
        # Segment based on option specs if we are assuming that options are
        # known. If we do not do this, it can lead to a bug where an option
        # has object arguments that do not appear in the strips operator
        # parameters. Note also that we are segmenting based on option specs,
        # rather than option changes. This distinction is subtle but important.
        # For example, in Cover, there is just one parameterized option, which
        # is PickPlace() with no object arguments. If we segmented based on
        # option changes, then segmentation would break up trajectories into
        # picks and places. Then, when operator learning, it would appear
        # that no predicates are necessary to distinguish between picking
        # and placing, since the option changes and segmentation have already
        # made the distinction. But we want operator learning to use predicates
        # like Holding, Handempty, etc., because when doing symbolic planning,
        # we only have predicates, and not the continuous parameters that would
        # be used to distinguish between a PickPlace that is a pick vs a place.
        if traj.actions[t].has_option():
            # Check for a change in option specs.
            if t < len(traj.actions) - 1:
                option_t = traj.actions[t].get_option()
                option_t1 = traj.actions[t+1].get_option()
                option_t_spec = (option_t.parent, option_t.objects)
                option_t1_spec = (option_t1.parent, option_t1.objects)
                if option_t_spec != option_t1_spec:
                    switch = True
            # Special case: if the final option terminates in the state, we
            # can safely segment without using any continuous info. Note that
            # excluding the final option from the data is highly problematic
            # when using demo+replay with the default 1 option per replay
            # because the replay data which causes no change in the symbolic
            # state would get excluded.
            elif traj.actions[t].get_option().terminal(traj.states[t]):
                switch = True
        if switch:
            # Include the final state as the end of this segment.
            current_segment_states.append(traj.states[t+1])
            current_segment_traj = LowLevelTrajectory(
                current_segment_states, current_segment_actions)
            if traj.actions[t].has_option():
                segment = Segment(current_segment_traj,
                                  all_atoms[t], all_atoms[t+1],
                                  traj.actions[t].get_option())
            else:
                # If option learning, include the default option here; replaced
                # during option learning.
                segment = Segment(current_segment_traj,
                                  all_atoms[t], all_atoms[t+1])
            segments.append(segment)
            current_segment_states = []
            current_segment_actions = []
    # Don't include the last current segment because it didn't result in
    # an abstract state change. (E.g., the option may not be terminating.)
    return segments


def learn_strips_operators_from_demos(
    trajectory_segments: List[List[Segment]],
    trajectory_goals: List[Set[GroundAtom]],
    nondemo_segments: List[Segment],
    verbose: bool = True
    ) -> Tuple[List[STRIPSOperator], List[Partition]]:
    """Learn operators given the segmented transitions.
    """
    assert len(trajectory_segments) == len(trajectory_goals)
    # Partition the segments according to common effects.
    params: List[Sequence[Variable]] = []
    parameterized_options: List[ParameterizedOption] = []
    option_vars: List[Tuple[Variable, ...]] = []
    add_effects: List[Set[LiftedAtom]] = []
    delete_effects: List[Set[LiftedAtom]] = []
    preconds: List[Set[LiftedAtom]] = []
    partitions: List[Partition] = []
    # Learn operators from end of trajectories to beginning.
    steps_to_goal = 0
    # Map from trajectory index to preimages. Initialize to goals.
    trajectory_preimages: Dict[int, Set[GroundAtom]] = \
        dict(enumerate(trajectory_goals))
    # Map from trajectory index to partition index.
    trajectory_partitions: Dict[int, int] = {}
    # Map from trajectory index to substitution.
    trajectory_subs: Dict[int, ObjToVarSub] = {}
    while True:
        learned_new_operator = False
        for traj_idx, trajectory in enumerate(trajectory_segments):
            # Get the segment for this trajectory that is steps_to_goal away.
            idx = len(trajectory) - steps_to_goal - 1
            if idx < 0:
                continue
            segment = trajectory[idx]
            # Get option spec for segment.
            if segment.has_option():
                segment_option = segment.get_option()
                segment_param_option = segment_option.parent
                segment_option_objs = tuple(segment_option.objects)
            else:
                segment_param_option = DefaultOption.parent
                segment_option_objs = tuple()
            # Get the current preimage for this trajectory.
            preimage = trajectory_preimages[traj_idx]
            # Calculate add effects.
            seg_add_effects = preimage - segment.init_atoms
            # Infer important objects based on those in the add effects
            # and option objs.
            important_objects = set(segment_option_objs) | \
                {o for a in seg_add_effects for o in a.objects}
            # Calculate delete effects from important objects.
            seg_delete_effects = {a for a in segment.delete_effects \
                if all(o in important_objects for o in a.objects)}
            for i in range(len(partitions)):
                # Try to unify this transition with existing effects.
                # Note that both add and delete effects must unify,
                # and also the objects that are arguments to the options.
                part_param_option = parameterized_options[i]
                part_option_vars = option_vars[i]
                part_add_effects = add_effects[i]
                part_delete_effects = delete_effects[i]
                suc, sub = unify_effects_and_options(
                    frozenset(seg_add_effects),
                    frozenset(part_add_effects),
                    frozenset(seg_delete_effects),
                    frozenset(part_delete_effects),
                    segment_param_option,
                    part_param_option,
                    segment_option_objs,
                    part_option_vars)
                if suc:
                    # Add to this partition.
                    assert set(sub.values()) == set(params[i])
                    partitions[i].add((segment, sub))
                    trajectory_partitions[traj_idx] = i
                    trajectory_subs[traj_idx] = sub
                    break

            # Otherwise, create a new group.
            else:
                # Get new lifted effects.
                objects = {o for atom in seg_add_effects |
                           seg_delete_effects for o in atom.objects} | \
                          set(segment_option_objs)
                objects_lst = sorted(objects)
                variables = [Variable(f"?x{i}", o.type)
                             for i, o in enumerate(objects_lst)]
                sub = dict(zip(objects_lst, variables))
                params.append(variables)
                parameterized_options.append(segment_param_option)
                option_vars.append(tuple(sub[o] for o in segment_option_objs))
                add_effects.append({atom.lift(sub) for atom
                                    in seg_add_effects})
                delete_effects.append({atom.lift(sub) for atom
                                       in seg_delete_effects})
                new_partition = Partition([(segment, sub)])
                partitions.append(new_partition)
                trajectory_partitions[traj_idx] = len(partitions) - 1
                trajectory_subs[traj_idx] = sub
                learned_new_operator = True

        if not learned_new_operator:
            break

        # Learn preconditions and update preimages.
        preconds = [_learn_preconditions(p) for p in partitions]
        for traj_idx, trajectory in enumerate(trajectory_segments):
            part_idx = trajectory_partitions[traj_idx]
            preimage = trajectory_preimages[traj_idx]
            lifted_preconds = preconds[part_idx]
            lifted_add_effs = add_effects[part_idx]
            sub = trajectory_subs[traj_idx]
            inv_sub = {v: k for k, v in sub.items()}
            ground_preconds = {a.ground(inv_sub) for a in lifted_preconds}
            ground_add_effects = {a.ground(inv_sub) for a in lifted_add_effs}
            # TODO: is this right? Not using delete effects...
            new_preimage = preimage.copy()
            new_preimage |= ground_preconds
            new_preimage -= ground_add_effects
            trajectory_preimages[traj_idx] = new_preimage
        steps_to_goal += 1

    kept_idxs = []
    for idx, partition in enumerate(partitions):
        # Prune partitions with not enough data.
        if len(partition) >= CFG.min_data_for_nsrt:
            kept_idxs.append(idx)
    params = [params[i] for i in kept_idxs]
    add_effects = [add_effects[i] for i in kept_idxs]
    delete_effects = [delete_effects[i] for i in kept_idxs]
    partitions = [partitions[i] for i in kept_idxs]

    # Create the operators.
    ops = []
    for i in range(len(params)):
        name = f"Op{i}"
        op = STRIPSOperator(name, params[i], preconds[i], add_effects[i],
                            delete_effects[i])
        ops.append(op)


    # Add nondemo data to partitions.
    for segment in nondemo_segments:
        # Get option spec for segment.
        if segment.has_option():
            segment_option = segment.get_option()
            segment_param_option = segment_option.parent
            segment_option_objs = tuple(segment_option.objects)
        else:
            segment_param_option = DefaultOption.parent
            segment_option_objs = tuple()
        for idx, op in enumerate(ops):
            # See about adding this segment to partitions[idx].
            if segment_param_option != parameterized_options[idx]:
                continue
            partial_sub = dict(zip(option_vars[idx], segment_option_objs))
            ground_ops = utils.all_ground_operators_given_partial(
                op, set(segment.states[0]),  partial_sub)
            for ground_op in utils.get_applicable_operators(
                ground_ops, segment.init_atoms):
                if not ground_op.add_effects.issubset(segment.add_effects):
                    continue
                if not ground_op.delete_effects.issubset(
                    segment.delete_effects):
                    continue
                sub = dict(zip(ground_op.objects, op.parameters))
                partitions[idx].add((segment, sub))

    # TODO: none of the below may actually be needed... it may just be
    # a question of data size.

    # Go through operators and filter out ones that are "dominated",
    # meaning in every state where the preconditions hold and the effects
    # follow, there is another operators whose preconditions also hold and
    # whose effects follow, and whose effects contain the original.
    for idx in range(len(partitions)-1, -1, -1):
        op = ops[idx]
        is_dominated = False
        for other_idx in range(len(partitions)):
            if is_dominated:
                break
            if idx == other_idx:
                continue
            other_op = ops[other_idx]
            # Check if other_op dominates op.
            for segment, sub in partition:
                inv_sub = {v: k for k, v in sub.items()}
                try:
                    op_predicted_add_effects = {a.ground(inv_sub)
                                                for a in op.add_effects}
                    op_predicted_delete_effects = {a.ground(inv_sub)
                                                   for a in op.delete_effects}
                except AssertionError:
                    # Other op definitely does not dominate because there is
                    # a mismatch in the variables.
                    break
                # Check if this segment is also covered by other_idx.
                # If the preconditions don't hold, that means that this
                # other_idx doesn't dominate.
                segment_covered_by_other = False
                ground_ops = utils.all_ground_operators(other_op,
                                                        set(segment.states[0]))
                for ground_op in utils.get_applicable_operators(
                    ground_ops, segment.init_atoms):
                    # Check if the ground effects predicted actually occur.
                    if not ground_op.add_effects.issubset(segment.add_effects):
                        continue
                    if not ground_op.delete_effects.issubset(
                        segment.delete_effects):
                        continue
                    # Check if the ground effects are a superset of the ones
                    # predicted by op.
                    if not ground_op.add_effects.issuperset(
                        op_predicted_add_effects):
                        continue
                    if not ground_op.delete_effects.issuperset(
                        op_predicted_delete_effects):
                        continue
                    # We have a match.
                    segment_covered_by_other = True
                    break
                # This segment was not covered by other_op, so other_op
                # does not dominate op.
                if not segment_covered_by_other:
                    break
            else:
                # All the segments for op were dominated by other op!
                is_dominated = True
                break
        # If this op is dominated, remove it.
        # TODO: add the data to the partition that dominated it! This will
        # require unifying the variables between op and other_op.
        if is_dominated:
            del ops[idx]
            del partitions[idx]

    return ops, partitions



def learn_strips_operators(segments: Sequence[Segment], verbose: bool = True,
        ) -> Tuple[List[STRIPSOperator], List[Partition]]:
    """Learn operators given the segmented transitions.
    """
    # Partition the segments according to common effects.
    params: List[Sequence[Variable]] = []
    parameterized_options: List[ParameterizedOption] = []
    option_vars: List[Tuple[Variable, ...]] = []
    add_effects: List[Set[LiftedAtom]] = []
    delete_effects: List[Set[LiftedAtom]] = []
    partitions: List[Partition] = []
    for segment in segments:
        if segment.has_option():
            segment_option = segment.get_option()
            segment_param_option = segment_option.parent
            segment_option_objs = tuple(segment_option.objects)
        else:
            segment_param_option = DefaultOption.parent
            segment_option_objs = tuple()
        for i in range(len(partitions)):
            # Try to unify this transition with existing effects.
            # Note that both add and delete effects must unify,
            # and also the objects that are arguments to the options.
            part_param_option = parameterized_options[i]
            part_option_vars = option_vars[i]
            part_add_effects = add_effects[i]
            part_delete_effects = delete_effects[i]
            suc, sub = unify_effects_and_options(
                frozenset(segment.add_effects),
                frozenset(part_add_effects),
                frozenset(segment.delete_effects),
                frozenset(part_delete_effects),
                segment_param_option,
                part_param_option,
                segment_option_objs,
                part_option_vars)
            if suc:
                # Add to this partition.
                assert set(sub.values()) == set(params[i])
                partitions[i].add((segment, sub))
                break
        # Otherwise, create a new group.
        else:
            # Get new lifted effects.
            objects = {o for atom in segment.add_effects |
                       segment.delete_effects for o in atom.objects} | \
                      set(segment_option_objs)
            objects_lst = sorted(objects)
            variables = [Variable(f"?x{i}", o.type)
                         for i, o in enumerate(objects_lst)]
            sub = dict(zip(objects_lst, variables))
            params.append(variables)
            parameterized_options.append(segment_param_option)
            option_vars.append(tuple(sub[o] for o in segment_option_objs))
            add_effects.append({atom.lift(sub) for atom
                                in segment.add_effects})
            delete_effects.append({atom.lift(sub) for atom
                                   in segment.delete_effects})
            new_partition = Partition([(segment, sub)])
            partitions.append(new_partition)

    # We don't need option_vars anymore; we'll recover them later when we call
    # `learn_option_specs`. The only reason to include them here is to make sure
    # that params include the option_vars when options are available.
    del option_vars

    assert len(params) == len(add_effects) == \
           len(delete_effects) == len(partitions)

    # Prune partitions with not enough data.
    kept_idxs = []
    for idx, partition in enumerate(partitions):
        if len(partition) >= CFG.min_data_for_nsrt:
            kept_idxs.append(idx)
    params = [params[i] for i in kept_idxs]
    add_effects = [add_effects[i] for i in kept_idxs]
    delete_effects = [delete_effects[i] for i in kept_idxs]
    partitions = [partitions[i] for i in kept_idxs]

    # Learn preconditions.
    preconds = [_learn_preconditions(p) for p in partitions]

    # Finalize the operators.
    ops = []
    for i in range(len(params)):
        name = f"Op{i}"
        op = STRIPSOperator(name, params[i], preconds[i], add_effects[i],
                            delete_effects[i])
        if verbose:
            print("Learned STRIPSOperator:")
            print(op)
        ops.append(op)

    return ops, partitions


def  _learn_preconditions(partition: Partition) -> Set[LiftedAtom]:
    for i, (segment, sub) in enumerate(partition):
        atoms = segment.init_atoms
        objects = set(sub.keys())
        atoms = {atom for atom in atoms if
                 all(o in objects for o in atom.objects)}
        lifted_atoms = {atom.lift(sub) for atom in atoms}
        if i == 0:
            variables = sorted(set(sub.values()))
        else:
            assert variables == sorted(set(sub.values()))
        if i == 0:
            preconditions = lifted_atoms
        else:
            preconditions &= lifted_atoms
    return preconditions


@functools.lru_cache(maxsize=None)
def unify_effects_and_options(
        ground_add_effects: FrozenSet[GroundAtom],
        lifted_add_effects: FrozenSet[LiftedAtom],
        ground_delete_effects: FrozenSet[GroundAtom],
        lifted_delete_effects: FrozenSet[LiftedAtom],
        ground_param_option: ParameterizedOption,
        lifted_param_option: ParameterizedOption,
        ground_option_args: Tuple[Object, ...],
        lifted_option_args: Tuple[Variable, ...]
) -> Tuple[bool, ObjToVarSub]:
    """Wrapper around utils.unify() that handles option arguments, add effects,
    and delete effects. Changes predicate names so that all are treated
    differently by utils.unify().
    """
    # Can't unify if the parameterized options are different.
    # Note, of course, we could directly check this in the loop above. But we
    # want to keep all the unification logic in one place, even if it's trivial
    # in this case.
    if ground_param_option != lifted_param_option:
        return False, {}
    ground_opt_arg_pred = Predicate("OPT-ARGS",
                                    [a.type for a in ground_option_args],
                                    _classifier=lambda s, o: False)  # dummy
    f_ground_option_args = frozenset({GroundAtom(ground_opt_arg_pred,
                                                 ground_option_args)})
    new_ground_add_effects = utils.wrap_atom_predicates_ground(
        ground_add_effects, "ADD-")
    f_new_ground_add_effects = frozenset(new_ground_add_effects)
    new_ground_delete_effects = utils.wrap_atom_predicates_ground(
        ground_delete_effects, "DEL-")
    f_new_ground_delete_effects = frozenset(new_ground_delete_effects)

    lifted_opt_arg_pred = Predicate("OPT-ARGS",
                                    [a.type for a in lifted_option_args],
                                    _classifier=lambda s, o: False)  # dummy
    f_lifted_option_args = frozenset({LiftedAtom(lifted_opt_arg_pred,
                                                 lifted_option_args)})
    new_lifted_add_effects = utils.wrap_atom_predicates_lifted(
        lifted_add_effects, "ADD-")
    f_new_lifted_add_effects = frozenset(new_lifted_add_effects)
    new_lifted_delete_effects = utils.wrap_atom_predicates_lifted(
        lifted_delete_effects, "DEL-")
    f_new_lifted_delete_effects = frozenset(new_lifted_delete_effects)
    return utils.unify(
        f_ground_option_args | f_new_ground_add_effects | \
            f_new_ground_delete_effects,
        f_lifted_option_args | f_new_lifted_add_effects | \
            f_new_lifted_delete_effects)
