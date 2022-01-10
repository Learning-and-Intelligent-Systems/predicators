"""Algorithms for learning the various components of NSRT objects.
"""

from collections import defaultdict
import functools
from typing import Set, Tuple, List, Sequence, FrozenSet, Dict
from predicators.src.structs import Dataset, STRIPSOperator, NSRT, \
    GroundAtom, LiftedAtom, Variable, Predicate, ObjToVarSub, \
    LowLevelTrajectory, Segment, Partition, Object, GroundAtomTrajectory, \
    DummyOption, ParameterizedOption, State, Action, OptionSpec
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

    # Learn strips operators.
    strips_ops, partitions = learn_strips_operators(
        ground_atom_dataset, verbose=CFG.do_option_learning)
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


def learn_strips_operators(ground_atom_dataset: Sequence[GroundAtomTrajectory],
                           verbose: bool = True,
                           ) -> Tuple[List[STRIPSOperator], List[Partition]]:
    """Learn STRIPSOperators given a dataset of ground atoms. These
    STRIPSOperators include side predicates. Also return the associated
    partitions (data stores) in a one-to-one list.
    """
    # Segment transitions based on changes in predicates and options.
    segmented_trajectories = [segment_trajectory(traj)
                              for traj in ground_atom_dataset]
    all_segments = [seg for segs in segmented_trajectories for seg in segs]

    # Learn operators without side predicates.
    ops_without_sides, option_specs = _learn_operators_no_side_predicates(
        all_segments)

    if True:#verbose:
        print("Learned operators without side predicates:")
        for op, (option, option_vars) in zip(ops_without_sides, option_specs):
            option_var_str = ", ".join([str(v) for v in option_vars])
            print(op)
            print(f"    Option Spec: {option.name}({option_var_str})")

    # Find skeletons for all trajectories.
    init_atoms = []
    final_relevant_atoms = []
    skeletons = []
    for segs, (ll_traj, _) in zip(segmented_trajectories, ground_atom_dataset):
        # Only demonstration trajectories are currently annotated with final
        # relevant atoms.
        if not ll_traj.is_demo:
            continue
        init_atoms.append(segs[0].init_atoms)
        final_relevant_atoms.append(ll_traj.goal)
        skeleton = _find_skeleton(segs, ops_without_sides, option_specs)
        skeletons.append(skeleton)

    # Convert effects to side predicates.
    name_to_strips_op = {op.name: op for op in ops_without_sides}
    # We need to remember the original strips operators because some of the
    # parameters might get lost during pruning. TODO: there's probably a more
    # clear way to code this.
    name_to_original_strips_op = name_to_strips_op.copy()
    for op, (_, option_vars) in zip(ops_without_sides, option_specs):
        # Consider converting each add effect.
        # TODO refactor to avoid redundant code.
        print("pnad:",op)
        verbose=True
        for effect in op.add_effects:
            if verbose:
                print(f"Considering add effect: {effect} from {op.name}")
            current_op = name_to_strips_op[op.name]
            assert current_op.name == op.name
            # Tentatively replace the current operator.
            remaining_params = {p for atom in current_op.preconditions | \
                (current_op.add_effects - {effect}) | \
                current_op.delete_effects
                for p in atom.variables} | set(option_vars)
            next_params = [p for p in op.parameters if p in remaining_params]
            name_to_strips_op[op.name] = STRIPSOperator(
                op.name, next_params,
                current_op.preconditions,
                current_op.add_effects - {effect},
                current_op.delete_effects,
                current_op.side_predicates | {effect.predicate})
            # Check if operators would still cover skeletons.
            if not all(_skeleton_covered(skeleton, inits, finals,
                                         name_to_strips_op,
                                         name_to_original_strips_op)
                       for (skeleton, inits, finals) in \
                        zip(skeletons, init_atoms, final_relevant_atoms)):
                # If not, revert the change.
                name_to_strips_op[op.name] = current_op
                if verbose:
                    print("Skeletons not covered; reverting conversion.")
            elif verbose:
                print("Skeletons still covered; keeping conversion.")
        # Consider converting each delete effect.
        for effect in op.delete_effects:
            if verbose:
                print(f"Considering delete effect: {effect} from {op.name}")
            current_op = name_to_strips_op[op.name]
            assert current_op.name == op.name
            # Tentatively replace the current operator.
            remaining_params = {p for atom in current_op.preconditions | \
                (current_op.add_effects - {effect}) | \
                current_op.delete_effects
                for p in atom.variables} | set(option_vars)
            next_params = [p for p in op.parameters if p in remaining_params]
            name_to_strips_op[op.name] = STRIPSOperator(
                op.name, next_params,
                current_op.preconditions,
                current_op.add_effects,
                current_op.delete_effects - {effect},
                current_op.side_predicates | {effect.predicate})
            # Check if operators would still cover skeletons.
            if not all(_skeleton_covered(skeleton, inits, finals,
                                         name_to_strips_op,
                                         name_to_original_strips_op)
                       for (skeleton, inits, finals) in \
                        zip(skeletons, init_atoms, final_relevant_atoms)):
                # If not, revert the change.
                name_to_strips_op[op.name] = current_op
                if verbose:
                    print("Skeletons not covered; reverting conversion.")
            elif verbose:
                print("Skeletons still covered; keeping conversion.")
        print("final pnad",name_to_strips_op[op.name])
        # input("!!")

    # Replace old operators.
    strips_ops = [name_to_strips_op[op.name] for op in ops_without_sides]

    if verbose:
        print("Learned operators with side effects:")
        for op in strips_ops:
            print(op)

    # Re-partition the data with the new operators.
    # We need to do this because now that we have side predicates, each
    # transition may be assigned to *multiple* partitions.
    partitions, partition_segment_indices = _partition_segments(
        segmented_trajectories, strips_ops, option_specs)
    assert len(partitions) == len(partition_segment_indices) == len(strips_ops)

    # Prune partitions that are redundant or that don't have enough data.
    final_strips_ops = []
    final_partitions = []
    seen_identifiers = set()
    for op, partition, part_ids in zip(strips_ops, partitions,
                                       partition_segment_indices):
        if len(partition) < CFG.min_data_for_nsrt:
            continue
        frozen_part_ids = frozenset(part_ids)
        if frozen_part_ids in seen_identifiers:
            continue
        final_strips_ops.append(op)
        final_partitions.append(partition)
        seen_identifiers.add(frozen_part_ids)

    if verbose:
        print("\nFinal operators after pruning:")
        for op in final_strips_ops:
            print(op)

    return final_strips_ops, final_partitions


def _learn_operators_no_side_predicates(segments: Sequence[Segment]
        ) -> Tuple[List[STRIPSOperator], List[OptionSpec]]:
    """Learn STRIPSOperators with no side effects. The option specs
    that are returned are either the ground truth options (if we are
    NOT doing option learning), or DummyOption specs (if we ARE doing
    option learning). In the option learning case, the actual options
    will be learned later in the pipeline.
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
            segment_param_option = DummyOption.parent
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

    option_specs = [(param_opt, list(opt_vars)) for param_opt, opt_vars in \
                    zip(parameterized_options, option_vars)]

    assert len(params) == len(add_effects) == len(delete_effects) == \
           len(option_specs) == len(partitions)

    # Learn preconditions.
    preconds = [_learn_preconditions(p) for p in partitions]

    # We don't need the partitions anymore, we'll re-partition the data after
    # learning operators with side effects.
    del partitions

    # Finalize the operators.
    ops = []
    for i in range(len(params)):
        name = f"Op{i}"
        op = STRIPSOperator(name, params[i], preconds[i], add_effects[i],
                            delete_effects[i], set())
        ops.append(op)

    return ops, option_specs


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
    new_ground_add_effects = utils.wrap_atom_predicates(
        ground_add_effects, "ADD-")
    f_new_ground_add_effects = frozenset(new_ground_add_effects)
    new_ground_delete_effects = utils.wrap_atom_predicates(
        ground_delete_effects, "DEL-")
    f_new_ground_delete_effects = frozenset(new_ground_delete_effects)

    lifted_opt_arg_pred = Predicate("OPT-ARGS",
                                    [a.type for a in lifted_option_args],
                                    _classifier=lambda s, o: False)  # dummy
    f_lifted_option_args = frozenset({LiftedAtom(lifted_opt_arg_pred,
                                                 lifted_option_args)})
    new_lifted_add_effects = utils.wrap_atom_predicates(
        lifted_add_effects, "ADD-")
    f_new_lifted_add_effects = frozenset(new_lifted_add_effects)
    new_lifted_delete_effects = utils.wrap_atom_predicates(
        lifted_delete_effects, "DEL-")
    f_new_lifted_delete_effects = frozenset(new_lifted_delete_effects)
    return utils.unify(
        f_ground_option_args | f_new_ground_add_effects | \
            f_new_ground_delete_effects,
        f_lifted_option_args | f_new_lifted_add_effects | \
            f_new_lifted_delete_effects)


def _find_skeleton(segment_traj: Sequence[Segment],
                   strips_ops: Sequence[STRIPSOperator],
                   option_specs: Sequence[OptionSpec]
                   ) -> List[Tuple[str, Tuple[Object, ...]]]:
    """A skeleton here is a list of (op name, object parameters).

    Only the operator names are used because the operators will change in the
    course of learning side predicates (and STRIPSOperators are frozen).
    """
    assert len(strips_ops) == len(option_specs)
    skeleton = []
    objects = set(segment_traj[0].states[0])
    for segment in segment_traj:
        if segment.has_option():
            segment_option = segment.get_option()
            segment_param_option = segment_option.parent
            segment_option_objs = tuple(segment_option.objects)
        else:
            segment_param_option = DummyOption.parent
            segment_option_objs = tuple()
        # Get ground operators given these objects and option objs.
        for op, (param_opt, opt_vars) in zip(strips_ops, option_specs):
            if param_opt != segment_param_option:
                continue
            op_idx = strips_ops.index(op)
            isub = dict(zip(opt_vars, segment_option_objs))
            # Consider adding this segment to each of the partitions.
            for ground_op in utils.all_ground_operators_given_partial(
                op, objects, isub):
                # Check if preconditions hold.
                if not ground_op.preconditions.issubset(segment.init_atoms):
                    continue
                # Check if effects match.
                if ground_op.add_effects != segment.add_effects or \
                   ground_op.delete_effects != segment.delete_effects:
                    continue
                # Operator covers the segment.
                skeleton.append((ground_op.operator.name,
                                 tuple(ground_op.objects)))
                break
            else:
                continue
            break
        else:
            raise Exception("Could not find operator that matches segment.")

    return skeleton


def _skeleton_covered(skeleton: Sequence[Tuple[str, Tuple[Object, ...]]],
                      init_atoms: Set[GroundAtom],
                      relevant_final_atoms: Set[GroundAtom],
                      name_to_strips_op: Dict[str, STRIPSOperator],
                      name_to_original_strips_op: Dict[str, STRIPSOperator]
                      ) -> bool:
    """A skeleton is covered if all preconditions hold and relevant final atoms
    are predicted from the init atoms.
    """
    # Check preconditions.
    current_atoms = init_atoms
    print("trying",skeleton)
    print(init_atoms)
    print(relevant_final_atoms)
    for (op_name, original_objects) in skeleton:
        # Some parameters may have changed.
        op = name_to_strips_op[op_name]
        original_vars = name_to_original_strips_op[op_name].parameters
        objects = tuple(o for o, v in zip(original_objects, original_vars)
                        if v in op.parameters)
        ground_op = op.ground(objects)
        if not ground_op.preconditions.issubset(current_atoms):
            print("failed pre")
            return False
        current_atoms = utils.apply_operator(ground_op, current_atoms)
    # Check final relevant atoms.
    print("eff success? ", relevant_final_atoms.issubset(current_atoms))
    return relevant_final_atoms.issubset(current_atoms)


def _partition_segments(segmented_trajectories: Sequence[Sequence[Segment]],
                        strips_ops: Sequence[STRIPSOperator],
                        option_specs: Sequence[OptionSpec]
                        ) -> Tuple[List[Partition], List[Set[Tuple[int, int]]]]:
    """The second list returned is a set of segment indices, which will be used
    to determine whether an operator is redundant.
    """
    partition_lsts : List[List[Tuple[Segment, ObjToVarSub]]] = \
        [[] for _ in strips_ops]
    partition_segment_indices : List[Set[Tuple[int, int]]] = \
        [set() for _ in partition_lsts]

    for traj_idx, segment_traj in enumerate(segmented_trajectories):

        objects = set(segment_traj[0].states[0])

        for segment_idx, segment in enumerate(segment_traj):
            identifier = (traj_idx, segment_idx)

            if segment.has_option():
                segment_option = segment.get_option()
                segment_param_option = segment_option.parent
                segment_option_objs = tuple(segment_option.objects)
            else:
                segment_param_option = DummyOption.parent
                segment_option_objs = tuple()

            # Get ground operators given these objects and option objs.
            for op, (param_opt, opt_vars) in zip(strips_ops, option_specs):
                if param_opt != segment_param_option:
                    continue
                op_idx = strips_ops.index(op)
                isub = dict(zip(opt_vars, segment_option_objs))
                # Consider adding this segment to each of the partitions.
                for ground_op in utils.all_ground_operators_given_partial(
                    op, objects, isub):
                    # Check if preconditions hold.
                    if not ground_op.preconditions.issubset(segment.init_atoms):
                        continue
                    # Check if effects match. Note that we're using the side
                    # predicates semantics here!
                    atoms = utils.apply_operator(ground_op, segment.init_atoms)
                    if not atoms.issubset(segment.final_atoms):
                        continue
                    # This segment belongs in this partition.
                    # TODO: this is an edge case we need to fix by changing
                    # ObjToVarSubs in the partition to VarToObjSubs.
                    if len(set(ground_op.objects)) != len(ground_op.objects):
                        continue
                    sub = dict(zip(ground_op.objects, op.parameters))
                    partition_lsts[op_idx].append((segment, sub))
                    partition_segment_indices[op_idx].add(identifier)

    # TODO refactor to avoid this.
    partitions = [Partition(elm) for elm in partition_lsts]

    return partitions, partition_segment_indices
