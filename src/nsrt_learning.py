"""The core algorithm for learning a collection of NSRT data structures.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import functools
from typing import Set, Tuple, List, Sequence, FrozenSet, Optional
from predicators.src.structs import Dataset, STRIPSOperator, NSRT, \
    GroundAtom, LiftedAtom, Variable, Predicate, ObjToVarSub, \
    LowLevelTrajectory, Segment, Partition, Object, GroundAtomTrajectory, \
    DummyOption, ParameterizedOption, State, Action, OptionSpec, NSRTSampler
from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.sampler_learning import learn_samplers
from predicators.src.option_learning import create_option_learner


def learn_nsrts_from_data(dataset: Dataset, predicates: Set[Predicate],
                          do_sampler_learning: bool) -> Set[NSRT]:
    """Learn NSRTs from the given dataset of low-level transitions,
    using the given set of predicates. If do_sampler_learning is False,
    the NSRTs have random samplers rather than learned neural ones.
    """
    print(f"\nLearning NSRTs on {len(dataset)} trajectories...")

    # STEP 1: Apply predicates to data, producing a dataset of abstract states.
    ground_atom_dataset = utils.create_ground_atom_dataset(dataset, predicates)

    # STEP 2: Segment each trajectory in the dataset based on changes in
    #         either predicates or options. If we are doing option learning,
    #         then the data will not contain options, so this segmenting
    #         procedure only uses the predicates.
    segmented_trajs = [segment_trajectory(traj) for traj in ground_atom_dataset]
    segments = [seg for segs in segmented_trajs for seg in segs]

    # STEP 3: Cluster the data by effects, jointly producing one STRIPSOperator,
    #         OptionSpec, and Partition (data store) per cluster. These items
    #         then used to initialize _NSRTIntermediateData objects (NIDs).
    #         Note: The OptionSpecs here are extracted directly from the data.
    #         If we are doing option learning, then the data will not contain
    #         options, and so the option_spec fields are just their default
    #         values. We need a default value because future steps require
    #         the option_spec field to be populated, even if just with a dummy.
    nids = _get_initial_nids(segments, verbose=CFG.do_option_learning)

    # STEP 4: Learn side predicates for the operators and update NIDs. These
    #         are predicates whose truth value becomes unknown (for *any*
    #         grounding not explicitly in effects) upon operator application.
    if CFG.learn_side_predicates:
        _learn_nid_side_predicates(nids)

    # STEP 5: Learn options (option_learning.py) and update NIDs.
    _learn_nid_options(nids)

    # STEP 6: Learn samplers (sampler_learning.py) and update NIDs.
    _learn_nid_samplers(nids, do_sampler_learning)

    # STEP 7: Finalize and return the NSRTs.
    nsrts = []
    for nid in nids:
        nid.finalize()
        assert nid.nsrt is not None
        nsrts.append(nid.nsrt)
    print("\nLearned NSRTs:")
    for nsrt in sorted(nsrts):
        print(nsrt)
    print()
    return set(nsrts)


def learn_strips_operators(segments: Sequence[Segment], verbose: bool = True,
                           ) -> Tuple[List[STRIPSOperator], List[Partition]]:
    """A light wrapper around _get_initial_nids() that returns the operators
    and partitions themselves, rather than exposing the internal data structure.
    """
    nids = _get_initial_nids(segments, verbose=verbose)
    ops = []
    partitions = []
    for nid in nids:
        ops.append(nid.op)
        partitions.append(nid.partition)
    return ops, partitions


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


def _get_initial_nids(segments: Sequence[Segment], verbose: bool
                      ) -> List[_NSRTIntermediateData]:
    # Partition the segments according to common effects.
    nids = []
    for segment in segments:
        if not CFG.do_option_learning:
            segment_option = segment.get_option()
            segment_param_option = segment_option.parent
            segment_option_objs = tuple(segment_option.objects)
        else:
            segment_param_option = DummyOption.parent
            segment_option_objs = tuple()
        for nid in nids:
            # Try to unify this transition with existing effects.
            # Note that both add and delete effects must unify,
            # and also the objects that are arguments to the options.
            (nid_param_option, nid_option_vars) = nid.option_spec
            nid_add_effects = nid.op.add_effects
            nid_delete_effects = nid.op.delete_effects
            suc, sub = unify_effects_and_options(
                frozenset(segment.add_effects),
                frozenset(nid_add_effects),
                frozenset(segment.delete_effects),
                frozenset(nid_delete_effects),
                segment_param_option,
                nid_param_option,
                segment_option_objs,
                tuple(nid_option_vars))
            if suc:
                # Add to this NID.
                assert set(sub.values()) == set(nid.op.parameters)
                nid.partition.add((segment, sub))
                break
        else:
            # Otherwise, create a new NID.
            objects = {o for atom in segment.add_effects |
                       segment.delete_effects for o in atom.objects} | \
                      set(segment_option_objs)
            objects_lst = sorted(objects)
            params = [Variable(f"?x{i}", o.type)
                      for i, o in enumerate(objects_lst)]
            preconds: Set[LiftedAtom] = set()  # will be learned later
            sub = dict(zip(objects_lst, params))
            add_effects = {atom.lift(sub) for atom in segment.add_effects}
            delete_effects = {atom.lift(sub) for atom in segment.delete_effects}
            side_predicates: Set[Predicate] = set()  # will be learned later
            op = STRIPSOperator(f"Op{len(nids)}", params, preconds,
                                add_effects, delete_effects, side_predicates)
            partition = Partition([(segment, sub)])
            option_vars = [sub[o] for o in segment_option_objs]
            option_spec = (segment_param_option, option_vars)
            nids.append(_NSRTIntermediateData(op, partition, option_spec))

    # Prune NIDs with not enough data.
    nids = [nid for nid in nids if len(nid.partition) >= CFG.min_data_for_nsrt]

    # Learn the preconditions of the operators in the NIDs via intersection.
    for nid in nids:
        for i, (segment, sub) in enumerate(nid.partition):
            objects = set(sub.keys())
            atoms = {atom for atom in segment.init_atoms if
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
        nid.op = STRIPSOperator(
            nid.op.name, nid.op.parameters, preconditions, nid.op.add_effects,
            nid.op.delete_effects, nid.op.side_predicates)

    # Print and return the NIDs.
    if verbose:
        print("\nLearned operators (before option learning):")
        for nid in nids:
            print(nid)
    return nids


def _learn_nid_side_predicates(nids: List[_NSRTIntermediateData]) -> None:
    raise NotImplementedError  # TODO


def _learn_nid_options(nids: List[_NSRTIntermediateData]) -> None:
    print("\nDoing option learning...")
    option_learner = create_option_learner()
    strips_ops = []
    partitions = []
    for nid in nids:
        strips_ops.append(nid.op)
        partitions.append(nid.partition)
    option_specs = option_learner.learn_option_specs(strips_ops, partitions)
    assert len(option_specs) == len(nids)
    # Replace the option_specs in the NIDs.
    for nid, option_spec in zip(nids, option_specs):
        nid.option_spec = option_spec
    # Seed the new parameterized option parameter spaces.
    for parameterized_option, _ in option_specs:
        parameterized_option.params_space.seed(CFG.seed)
    # Update the segments to include which option is being executed.
    for partition, spec in zip(partitions, option_specs):
        for (segment, _) in partition:
            # Modifies segment in-place.
            option_learner.update_segment_from_option_spec(segment, spec)
    print("\nLearned operators with option specs:")
    for nid in nids:
        print(nid)


def _learn_nid_samplers(nids: List[_NSRTIntermediateData],
                        do_sampler_learning: bool) -> None:
    print("\nDoing sampler learning...")
    strips_ops = []
    partitions = []
    option_specs = []
    for nid in nids:
        strips_ops.append(nid.op)
        partitions.append(nid.partition)
        option_specs.append(nid.option_spec)
    samplers = learn_samplers(strips_ops, partitions, option_specs,
                              do_sampler_learning)
    assert len(samplers) == len(strips_ops)
    # Replace the samplers in the NIDs.
    for nid, sampler in zip(nids, samplers):
        nid.sampler = sampler


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


@dataclass(repr=False)
class _NSRTIntermediateData:
    """NID: An internal helper class for NSRT learning that contains information
    useful to maintain throughout the learning procedure. Each object of this
    class corresponds to an NSRT, and at the end of learning will ultimately
    contain a finalized NSRT object.
    """
    # The symbolic components of the NSRT are the first thing that get learned.
    op: STRIPSOperator
    # The data store (i.e., partition) describing which segments in the
    # dataset are covered by the STRIPSOperator self.op.
    partition: Partition
    # The OptionSpec of this NSRT, which is a tuple of (option, option_vars).
    # See the NSRT class and definition of OptionSpec in structs.py.
    option_spec: OptionSpec
    # The sampler for this NSRT.
    sampler: Optional[NSRTSampler] = field(init=False, default=None)
    # The finalized NSRT produced at the end of the learning procedure.
    nsrt: Optional[NSRT] = field(init=False, default=None)

    def finalize(self) -> None:
        """Set self.nsrt to the finalized NSRT object.
        """
        assert self.nsrt is None
        assert self.sampler is not None
        param_option, option_vars = self.option_spec
        self.nsrt = self.op.make_nsrt(param_option, option_vars, self.sampler)

    def __repr__(self) -> str:
        param_option, option_vars = self.option_spec
        vars_str = ", ".join(str(v) for v in option_vars)
        return f"{self.op}\n    Option Spec: {param_option.name}({vars_str})"

    def __str__(self) -> str:
        return repr(self)
