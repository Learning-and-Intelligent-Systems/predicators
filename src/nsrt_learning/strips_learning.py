"""Code for learning the STRIPS operators within NSRTs."""

from typing import Set, List, Sequence, cast
from predicators.src.structs import STRIPSOperator, LiftedAtom, Variable, \
    Predicate, ObjToVarSub, LowLevelTrajectory, Segment, State, Action, \
    PartialNSRTAndDatastore, GroundAtomTrajectory, DummyOption
from predicators.src import utils
from predicators.src.settings import CFG


def segment_trajectory(trajectory: GroundAtomTrajectory) -> List[Segment]:
    """Segment a ground atom trajectory according to abstract state changes.

    If options are available, also use them to segment.
    """
    segments = []
    traj, all_atoms = trajectory
    assert len(traj.states) == len(all_atoms)
    current_segment_states: List[State] = []
    current_segment_actions: List[Action] = []
    for t in range(len(traj.actions)):
        current_segment_states.append(traj.states[t])
        current_segment_actions.append(traj.actions[t])
        switch = all_atoms[t] != all_atoms[t + 1]
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
        if not switch and traj.actions[t].has_option():
            # Check for a change in option specs.
            if t < len(traj.actions) - 1:
                option_t = traj.actions[t].get_option()
                option_t1 = traj.actions[t + 1].get_option()
                option_t_spec = (option_t.parent, option_t.objects)
                option_t1_spec = (option_t1.parent, option_t1.objects)
                if option_t_spec != option_t1_spec:
                    switch = True
            # Special case: if the final option terminates in the final state,
            # we can safely segment without using any continuous info. Note that
            # excluding the final option from the data is highly problematic
            # when using demo+replay with the default 1 option per replay
            # because the replay data which causes no change in the symbolic
            # state would get excluded.
            elif traj.actions[t].get_option().terminal(traj.states[t + 1]):
                switch = True
        if switch:
            # Include the final state as the end of this segment.
            current_segment_states.append(traj.states[t + 1])
            current_segment_traj = LowLevelTrajectory(current_segment_states,
                                                      current_segment_actions)
            if traj.actions[t].has_option():
                segment = Segment(current_segment_traj, all_atoms[t],
                                  all_atoms[t + 1],
                                  traj.actions[t].get_option())
            else:
                # If we're in option learning mode, include the default option
                # here; replaced later during option learning.
                segment = Segment(current_segment_traj, all_atoms[t],
                                  all_atoms[t + 1])
            segments.append(segment)
            current_segment_states = []
            current_segment_actions = []
    # Don't include the last current segment because it didn't result in
    # an abstract state change. (E.g., the option may not be terminating.)
    return segments


def learn_strips_operators(
    segments: Sequence[Segment],
    verbose: bool = True,
) -> List[PartialNSRTAndDatastore]:
    """Learn strips operators on the given data segments.

    Return a list of PNADs with op (STRIPSOperator), datastore, and
    option_spec fields filled in.
    """
    # Cluster the segments according to common effects.
    pnads: List[PartialNSRTAndDatastore] = []
    for segment in segments:
        if segment.has_option():
            segment_option = segment.get_option()
            segment_param_option = segment_option.parent
            segment_option_objs = tuple(segment_option.objects)
        else:
            segment_param_option = DummyOption.parent
            segment_option_objs = tuple()
        for pnad in pnads:
            # Try to unify this transition with existing effects.
            # Note that both add and delete effects must unify,
            # and also the objects that are arguments to the options.
            (pnad_param_option, pnad_option_vars) = pnad.option_spec
            suc, ent_to_ent_sub = utils.unify_preconds_effects_options(
                frozenset(),  # no preconditions
                frozenset(),  # no preconditions
                frozenset(segment.add_effects),
                frozenset(pnad.op.add_effects),
                frozenset(segment.delete_effects),
                frozenset(pnad.op.delete_effects),
                segment_param_option,
                pnad_param_option,
                segment_option_objs,
                tuple(pnad_option_vars))
            sub = cast(ObjToVarSub, ent_to_ent_sub)
            if suc:
                # Add to this PNAD.
                assert set(sub.values()) == set(pnad.op.parameters)
                pnad.add_to_datastore((segment, sub))
                break
        else:
            # Otherwise, create a new PNAD.
            objects = {o for atom in segment.add_effects |
                       segment.delete_effects for o in atom.objects} | \
                      set(segment_option_objs)
            objects_lst = sorted(objects)
            params = [
                Variable(f"?x{i}", o.type) for i, o in enumerate(objects_lst)
            ]
            preconds: Set[LiftedAtom] = set()  # will be learned later
            sub = dict(zip(objects_lst, params))
            add_effects = {atom.lift(sub) for atom in segment.add_effects}
            delete_effects = {
                atom.lift(sub)
                for atom in segment.delete_effects
            }
            side_predicates: Set[Predicate] = set()  # will be learned later
            op = STRIPSOperator(f"Op{len(pnads)}", params, preconds,
                                add_effects, delete_effects, side_predicates)
            datastore = [(segment, sub)]
            option_vars = [sub[o] for o in segment_option_objs]
            option_spec = (segment_param_option, option_vars)
            pnads.append(PartialNSRTAndDatastore(op, datastore, option_spec))

    # Prune PNADs with not enough data.
    pnads = [
        pnad for pnad in pnads if len(pnad.datastore) >= CFG.min_data_for_nsrt
    ]

    # Learn the preconditions of the operators in the PNADs via intersection.
    for pnad in pnads:
        for i, (segment, sub) in enumerate(pnad.datastore):
            objects = set(sub.keys())
            atoms = {
                atom
                for atom in segment.init_atoms
                if all(o in objects for o in atom.objects)
            }
            lifted_atoms = {atom.lift(sub) for atom in atoms}
            if i == 0:
                variables = sorted(set(sub.values()))
            else:
                assert variables == sorted(set(sub.values()))
            if i == 0:
                preconditions = lifted_atoms
            else:
                preconditions &= lifted_atoms
        # Replace the operator with one that contains the newly learned
        # preconditions. We do this because STRIPSOperator objects are
        # frozen, so their fields cannot be modified.
        pnad.op = STRIPSOperator(pnad.op.name, pnad.op.parameters,
                                 preconditions, pnad.op.add_effects,
                                 pnad.op.delete_effects,
                                 pnad.op.side_predicates)

    # Print and return the PNADs.
    if verbose:
        print("\nLearned operators (before side predicate & option learning):")
        for pnad in pnads:
            print(pnad)
    return pnads
