"""Algorithms for learning the various components of NSRT objects.
"""

import functools
from typing import Set, Tuple, List, Sequence, FrozenSet
from predicators.src.structs import Dataset, STRIPSOperator, NSRT, \
    GroundAtom, ParameterizedOption, LiftedAtom, Variable, Predicate, \
    ObjToVarSub, ActionTrajectory, Segment, Partition, Object
from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.sampler_learning import learn_samplers


def learn_nsrts_from_data(dataset: Dataset, predicates: Set[Predicate],
                          do_sampler_learning: bool) -> Set[NSRT]:
    """Learn NSRTs from the given dataset of transitions.
    States are abstracted using the given set of predicates.
    """
    print(f"\nLearning NSRTs on {len(dataset)} trajectories...")

    # Segment transitions based on changes in predicates.
    segments = [seg for traj in dataset
                for seg in segment_trajectory(traj, predicates)]

    # Learn strips operators.
    strips_ops, partitions = learn_strips_operators(segments)
    assert len(strips_ops) == len(partitions)

    # Learn options, or if known, just look them up.
    # The order of the options corresponds to the strips_ops.
    # Each item is a (ParameterizedOption, Sequence[Variable])
    # with the latter holding the option_vars.
    option_specs = learn_option_specs(strips_ops, partitions)
    assert len(option_specs) == len(strips_ops)

    # Now that options are learned, we can update the segments to include
    # which option is being executed within each segment.
    for partition, option_spec in zip(partitions, option_specs):
        for (segment, _) in partition:
            # Modifies segment in place.
            _update_segment_from_option_specs(segment, option_spec)

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
    for nsrt in nsrts:
        print(nsrt)
    print()

    return set(nsrts)


def segment_trajectory(trajectory: ActionTrajectory,
                       predicates: Set[Predicate]) -> List[Segment]:
    """Segment an action trajectory according to abstract state changes.
    """
    segments = []
    states, actions = trajectory
    assert len(states) == len(actions) + 1
    all_atoms = [utils.abstract(s, predicates) for s in states]
    current_segment_traj : ActionTrajectory = ([states[0]], [])
    for t in range(len(actions)):
        current_segment_traj[0].append(states[t])
        current_segment_traj[1].append(actions[t])
        if all_atoms[t] != all_atoms[t+1]:
            # Include the final state as both the end of this segment
            # and the start of the next segment.
            # Include the default option here; replaced during option learning.
            current_segment_traj[0].append(states[t+1])
            segment = Segment(current_segment_traj,
                              all_atoms[t], all_atoms[t+1])
            segments.append(segment)
            current_segment_traj = ([states[t+1]], [])
    # Don't include the last current segment because it didn't result in
    # an abstract state change. (E.g., the option may not be terminating.)
    return segments


def learn_strips_operators(segments: Sequence[Segment]
        ) -> Tuple[List[STRIPSOperator], List[Partition]]:
    """Learn operators given the segmented transitions.
    """
    # Partition the segments according to common effects.
    params: List[Sequence[Variable]] = []
    add_effects: List[Set[LiftedAtom]] = []
    delete_effects: List[Set[LiftedAtom]] = []
    partitions: List[Partition] = []
    for segment in segments:
        for i in range(len(partitions)):
            # Try to unify this transition with existing effects.
            # Note that both add and delete effects must unify.
            part_add_effects = add_effects[i]
            part_delete_effects = delete_effects[i]
            suc, sub = unify_effects_and_options(
                frozenset(segment.add_effects),
                frozenset(part_add_effects),
                frozenset(segment.delete_effects),
                frozenset(part_delete_effects))
            if suc:
                # Add to this partition
                assert set(sub.values()) == set(params[i])
                partitions[i].add((segment, sub))
                break
        # Otherwise, create a new group
        else:
            # Get new lifted effects
            objects = {o for atom in segment.add_effects |
                       segment.delete_effects for o in atom.objects}
            objects_lst = sorted(objects)
            variables = [Variable(f"?x{i}", o.type)
                         for i, o in enumerate(objects_lst)]
            sub = dict(zip(objects_lst, variables))
            params.append(variables)
            add_effects.append({atom.lift(sub) for atom
                                in segment.add_effects})
            delete_effects.append({atom.lift(sub) for atom
                                   in segment.delete_effects})
            new_partition = Partition([(segment, sub)])
            partitions.append(new_partition)

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
        name = f"Operator{i}"
        op = STRIPSOperator(name, params[i], preconds[i], add_effects[i],
                            delete_effects[i])
        print("Learned STRIPSOperator:")
        print(op)
        ops.append(op)

    return ops, partitions


def learn_option_specs(
    strips_ops: List[STRIPSOperator],
    partitions: List[Partition],
    ) -> List[Tuple[ParameterizedOption, List[Variable]]]:
    """Learn options for segments, or just look them up if they're given.
    """
    assert not CFG.do_option_learning, "TODO: implement option learning."
    del strips_ops  # unused
    return _extract_options_from_data(partitions)


def _extract_options_from_data(
    partitions: List[Partition],
    ) -> List[Tuple[ParameterizedOption, List[Variable]]]:
    """Look up the options from the data.
    """
    option_specs = []
    for partition in partitions:
        for i, (segment, sub) in enumerate(partition):
            option = segment.actions[0].get_option()
            if i == 0:
                param_option = option.parent
                option_vars = [sub[o] for o in option.objects]
            else:
                assert param_option == option.parent
                assert option_vars == [sub[o] for o in option.objects]
            # Make sure the option is consistent within a trajectory.
            for a in segment.actions:
                option_a = a.get_option()
                assert param_option == option_a.parent
                assert option_vars == [sub[o] for o in option_a.objects]
        option_specs.append((param_option, option_vars))
    return option_specs


def _update_segment_from_option_specs(segment: Segment,
        option_spec: Tuple[ParameterizedOption, Sequence[Variable]]) -> None:
    """Figure out which option was executed within the segment.

    At this point, we know which ParameterizedOption was used in the segment,
    and we know the option_vars, but we don't know what parameters were used.

    Modifies segment in place.
    """
    assert not CFG.do_option_learning, "TODO: implement option learning."
    assert not segment.has_option()
    segment.set_option_from_trajectory()
    option = segment.get_option()
    param_option, opt_vars = option_spec
    assert option.parent == param_option
    assert [o.type for o in option.objects] == \
           [v.type for v in opt_vars]


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
        ground_option_args: Tuple[Object, ...] = tuple(),
        lifted_option_args: Tuple[Variable, ...] = tuple()
) -> Tuple[bool, ObjToVarSub]:
    """Wrapper around utils.unify() that handles option arguments, add effects,
    and delete effects. Changes predicate names so that all are treated
    differently by utils.unify().
    """
    opt_arg_pred = Predicate("OPT-ARGS",
                             [a.type for a in ground_option_args],
                             _classifier=lambda s, o: False)  # dummy
    f_ground_option_args = frozenset({GroundAtom(opt_arg_pred,
                                                 ground_option_args)})
    new_ground_add_effects = utils.wrap_atom_predicates_ground(
        ground_add_effects, "ADD-")
    f_new_ground_add_effects = frozenset(new_ground_add_effects)
    new_ground_delete_effects = utils.wrap_atom_predicates_ground(
        ground_delete_effects, "DEL-")
    f_new_ground_delete_effects = frozenset(new_ground_delete_effects)

    f_lifted_option_args = frozenset({LiftedAtom(opt_arg_pred,
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
