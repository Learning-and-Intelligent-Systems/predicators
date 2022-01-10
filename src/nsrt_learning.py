"""The core algorithm for learning a collection of NSRT data structures."""

from __future__ import annotations
from typing import Set, List, Sequence, cast, Tuple
from predicators.src.structs import Dataset, STRIPSOperator, NSRT, \
    LiftedAtom, Variable, Predicate, ObjToVarSub, LowLevelTrajectory, \
    Segment, PartialNSRTAndDatastore, GroundAtomTrajectory, State, \
    Action, Object, _GroundSTRIPSOperator, GroundAtom
from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.sampler_learning import learn_samplers
from predicators.src.option_learning import create_option_learner


def learn_nsrts_from_data(dataset: Dataset, predicates: Set[Predicate],
                          sampler_learner: str) -> Set[NSRT]:
    """Learn NSRTs from the given dataset of low-level transitions, using the
    given set of predicates."""
    print(f"\nLearning NSRTs on {len(dataset)} trajectories...")

    # STEP 1: Apply predicates to data, producing a dataset of abstract states.
    ground_atom_dataset = utils.create_ground_atom_dataset(dataset, predicates)

    # STEP 2: Segment each trajectory in the dataset based on changes in
    #         either predicates or options. If we are doing option learning,
    #         then the data will not contain options, so this segmenting
    #         procedure only uses the predicates.
    segmented_trajs = [
        segment_trajectory(traj) for traj in ground_atom_dataset
    ]
    segments = [seg for segs in segmented_trajs for seg in segs]

    # STEP 3: Cluster the data by effects, jointly producing one STRIPSOperator,
    #         Datastore, and OptionSpec per cluster. These items are then
    #         used to initialize PartialNSRTAndDatastore objects (PNADs).
    #         Note: The OptionSpecs here are extracted directly from the data.
    #         If we are doing option learning, then the data will not contain
    #         options, and so the option_spec fields are just the specs of a
    #         DummyOption. We need a default dummy because future steps require
    #         the option_spec field to be populated, even if just with a dummy.
    pnads = learn_strips_operators(segments,
                                   verbose=(CFG.option_learner != "no_learning"
                                            or CFG.learn_side_predicates))

    # STEP 4: Learn side predicates for the operators and update PNADs. These
    #         are predicates whose truth value becomes unknown (for *any*
    #         grounding not explicitly in effects) upon operator application.
    if CFG.learn_side_predicates:
        pnads = _learn_pnad_side_predicates(
            pnads,
            segmented_trajs,
            ground_atom_dataset,
            verbose=(CFG.option_learner != "no_learning"))

    # STEP 5: Prune PNADs with not enough data.
    pnads = [
        pnad for pnad in pnads if len(pnad.datastore) >= CFG.min_data_for_nsrt
    ]

    # STEP 6: Learn options (option_learning.py) and update PNADs.
    _learn_pnad_options(pnads)  # in-place update

    # STEP 7: Learn samplers (sampler_learning.py) and update PNADs.
    _learn_pnad_samplers(pnads, sampler_learner)  # in-place update

    # STEP 8: Print and return the NSRTs.
    nsrts = [pnad.make_nsrt() for pnad in pnads]
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
        if traj.actions[t].has_option():
            # Check for a change in option specs.
            if t < len(traj.actions) - 1:
                option_t = traj.actions[t].get_option()
                option_t1 = traj.actions[t + 1].get_option()
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
            current_segment_states.append(traj.states[t + 1])
            current_segment_traj = LowLevelTrajectory(current_segment_states,
                                                      current_segment_actions)
            if traj.actions[t].has_option():
                segment = Segment(current_segment_traj, all_atoms[t],
                                  all_atoms[t + 1],
                                  traj.actions[t].get_option())
            else:
                # If option learning, include the default option here; replaced
                # during option learning.
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
        segment_param_option, segment_option_objs = segment.get_option_spec()
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


def _learn_pnad_side_predicates(
        pnads: List[PartialNSRTAndDatastore],
        segmented_trajs: List[List[Segment]],
        ground_atom_dataset: List[GroundAtomTrajectory],
        verbose: bool) -> List[PartialNSRTAndDatastore]:
    print("\nDoing side predicate learning...")
    # For each demonstration in the data, determine its skeleton under
    # the operators stored in the PNADs.
    assert len(segmented_trajs) == len(ground_atom_dataset)
    all_init_atoms = []
    all_final_atoms = []
    skeletons = []
    for seg_traj, (ll_traj, _) in zip(segmented_trajs, ground_atom_dataset):
        if not ll_traj.is_demo:
            continue
        all_init_atoms.append(seg_traj[0].init_atoms)
        all_final_atoms.append(ll_traj.goal)
        skeleton = []
        for segment in seg_traj:
            ground_op = _get_ground_operator_for_segment(segment, pnads)
            skeleton.append((ground_op.operator.name,
                             tuple(ground_op.objects)))
        skeletons.append(skeleton)
    # Try converting each effect in each PNAD to a side predicate.
    orig_pnad_params = [pnad.op.parameters.copy() for pnad in pnads]
    for pnad in pnads:
        print("pnad:",pnad)
        _, option_vars = pnad.option_spec
        for effect_set, add_or_delete in [
                (pnad.op.add_effects, "add"),
                (pnad.op.delete_effects, "delete")]:
            for effect in effect_set:
                orig_op = pnad.op
                pnad.op = pnad.op.effect_to_side_predicate(
                    effect, option_vars, add_or_delete)
                print("effect:",effect)
                # If the new operator with this effect turned into a side
                # predicate does not cover every skeleton, revert the change.
                if not all(_skeleton_covered(
                        skeleton, init_atoms, final_atoms,
                        pnads, orig_pnad_params)
                       for skeleton, init_atoms, final_atoms in zip(
                               skeletons, all_init_atoms, all_final_atoms)):
                    pnad.op = orig_op
                    print("revert")
                else:
                    print("ACCEPT")
        print("final pnad:",pnad)
        # input("!!")
    # Recompute the datastores in the PNADs. We need to do this
    # because now that we have side predicates, each transition may be
    # assigned to *multiple* datastores.
    all_indices = _recompute_datastores_from_segments(segmented_trajs, pnads)
    # Prune redundant PNADs.
    final_pnads = []
    seen_identifiers = set()
    assert len(pnads) == len(all_indices)
    for pnad, indices in zip(pnads, all_indices):
        frozen_indices = frozenset(indices)
        if frozen_indices in seen_identifiers:
            continue
        final_pnads.append(pnad)
        seen_identifiers.add(frozen_indices)
    if verbose:
        print("\nLearned operators with side predicates:")
        for pnad in final_pnads:
            print(pnad)
    return final_pnads


def _get_ground_operator_for_segment(
    segment: Segment, pnads: List[PartialNSRTAndDatastore]
) -> _GroundSTRIPSOperator:
    """Helper for side predicate learning.

    Finds a grounding of any operator stored in the PNADs that
    induces the given segment.
    """
    objects = set(segment.states[0])
    segment_param_option, segment_option_objs = segment.get_option_spec()
    # Find a matching ground operator.
    for pnad in pnads:
        param_opt, opt_vars = pnad.option_spec
        if param_opt != segment_param_option:
            continue
        isub = dict(zip(opt_vars, segment_option_objs))
        for ground_op in utils.all_ground_operators_given_partial(
                pnad.op, objects, isub):
            # Check if preconditions hold.
            if not ground_op.preconditions.issubset(segment.init_atoms):
                continue
            # Check if effects match.
            if ground_op.add_effects != segment.add_effects or \
               ground_op.delete_effects != segment.delete_effects:
                continue
            # Ground operator covers the segment, we're done.
            return ground_op
    raise Exception("Could not find ground operator that matches segment.")


def _skeleton_covered(skeleton: Sequence[Tuple[str, Tuple[Object, ...]]],
                      init_atoms: Set[GroundAtom],
                      final_atoms: Set[GroundAtom],
                      pnads: List[PartialNSRTAndDatastore],
                      orig_pnad_params: List[List[Variable]]) -> bool:
    """A skeleton is covered if all preconditions hold and final atoms
    are predicted from the init atoms.
    """
    assert len(pnads) == len(orig_pnad_params)
    print("checking")
    print(skeleton)
    print(init_atoms)
    print(final_atoms)
    name_to_idx = {pnad.op.name: idx for idx, pnad in enumerate(pnads)}
    current_atoms = init_atoms
    for (op_name, orig_objects) in skeleton:
        idx = name_to_idx[op_name]
        pnad = pnads[idx]
        orig_params = orig_pnad_params[idx]
        # Some parameters may have been removed, but order is preserved.
        objects = tuple(o for o, v in zip(orig_objects, orig_params)
                        if v in pnad.op.parameters)
        ground_op = pnad.op.ground(objects)
        if not ground_op.preconditions.issubset(current_atoms):
            print("pre fail")
            return False
        current_atoms = utils.apply_operator(ground_op, current_atoms)
    print("did eff succeed? ", final_atoms.issubset(current_atoms))
    return final_atoms.issubset(current_atoms)


def _recompute_datastores_from_segments(
        segmented_trajs: List[List[Segment]],
        pnads: List[PartialNSRTAndDatastore]) -> List[Set[Tuple[int, int]]]:
    for pnad in pnads:
        pnad.datastore = []  # reset all PNAD datastores
    all_indices: List[Set[Tuple[int, int]]] = [set() for _ in pnads]
    for traj_idx, seg_traj in enumerate(segmented_trajs):
        objects = set(seg_traj[0].states[0])
        for segment_idx, segment in enumerate(seg_traj):
            identifier = (traj_idx, segment_idx)
            (segment_param_option,
             segment_option_objs) = segment.get_option_spec()
            # Get ground operators given these objects and option objs.
            for pnad_idx, pnad in enumerate(pnads):
                param_opt, opt_vars = pnad.option_spec
                if param_opt != segment_param_option:
                    continue
                isub = dict(zip(opt_vars, segment_option_objs))
                # Consider adding this segment to each datastore.
                for ground_op in utils.all_ground_operators_given_partial(
                        pnad.op, objects, isub):
                    # Check if preconditions hold.
                    if not ground_op.preconditions.issubset(segment.init_atoms):
                        continue
                    # Check if effects match. Note that we're using the side
                    # predicates semantics here!
                    # TODO: this is wrong. subset diff must only contain side predicates
                    atoms = utils.apply_operator(ground_op, segment.init_atoms)
                    if not atoms.issubset(segment.final_atoms):
                        continue
                    # This segment belongs in this datastore, so add it.
                    # TODO: this is an edge case we need to fix by changing
                    # ObjToVarSubs in the partition to VarToObjSubs.
                    if len(set(ground_op.objects)) != len(ground_op.objects):
                        continue
                    sub = dict(zip(ground_op.objects, pnad.op.parameters))
                    pnad.add_to_datastore((segment, sub))
                    all_indices[pnad_idx].add(identifier)
    return all_indices


def _learn_pnad_options(pnads: List[PartialNSRTAndDatastore]) -> None:
    print("\nDoing option learning...")
    option_learner = create_option_learner()
    strips_ops = []
    datastores = []
    for pnad in pnads:
        strips_ops.append(pnad.op)
        datastores.append(pnad.datastore)
    option_specs = option_learner.learn_option_specs(strips_ops, datastores)
    assert len(option_specs) == len(pnads)
    # Replace the option_specs in the PNADs.
    for pnad, option_spec in zip(pnads, option_specs):
        pnad.option_spec = option_spec
    # Seed the new parameterized option parameter spaces.
    for parameterized_option, _ in option_specs:
        parameterized_option.params_space.seed(CFG.seed)
    # Update the segments to include which option is being executed.
    for datastore, spec in zip(datastores, option_specs):
        for (segment, _) in datastore:
            # Modifies segment in-place.
            option_learner.update_segment_from_option_spec(segment, spec)
    print("\nLearned operators with option specs:")
    for pnad in pnads:
        print(pnad)


def _learn_pnad_samplers(pnads: List[PartialNSRTAndDatastore],
                         sampler_learner: str) -> None:
    print("\nDoing sampler learning...")
    strips_ops = []
    datastores = []
    option_specs = []
    for pnad in pnads:
        strips_ops.append(pnad.op)
        datastores.append(pnad.datastore)
        option_specs.append(pnad.option_spec)
    samplers = learn_samplers(strips_ops, datastores, option_specs,
                              sampler_learner)
    assert len(samplers) == len(strips_ops)
    # Replace the samplers in the PNADs.
    for pnad, sampler in zip(pnads, samplers):
        pnad.sampler = sampler
