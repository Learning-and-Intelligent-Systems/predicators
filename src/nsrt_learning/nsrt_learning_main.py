"""The core algorithm for learning a collection of NSRT data structures."""

from __future__ import annotations
from typing import Set, List, Sequence, Iterator, Tuple
from predicators.src.structs import NSRT, Predicate, LowLevelTrajectory, \
    Segment, PartialNSRTAndDatastore, GroundAtomTrajectory, Task, DummyOption
from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.nsrt_learning.strips_learning import segment_trajectory, \
    learn_strips_operators
from predicators.src.nsrt_learning.sampler_learning import learn_samplers
from predicators.src.nsrt_learning.option_learning import create_option_learner
from predicators.src.predicate_search_score_functions import \
    _PredictionErrorScoreFunction


def learn_nsrts_from_data(trajectories: Sequence[LowLevelTrajectory],
                          train_tasks: List[Task], predicates: Set[Predicate],
                          sampler_learner: str) -> Set[NSRT]:
    """Learn NSRTs from the given dataset of low-level transitions, using the
    given set of predicates."""
    print(f"\nLearning NSRTs on {len(trajectories)} trajectories...")

    # STEP 1: Apply predicates to data, producing a dataset of abstract states.
    ground_atom_dataset = utils.create_ground_atom_dataset(
        trajectories, predicates)

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
        pnads = _learn_pnad_side_predicates(pnads, ground_atom_dataset,
                                            train_tasks, predicates, segments,
                                            segmented_trajs)

    # STEP 5: Learn options (option_learning.py) and update PNADs.
    _learn_pnad_options(pnads)  # in-place update

    # STEP 6: Learn samplers (sampler_learning.py) and update PNADs.
    _learn_pnad_samplers(pnads, sampler_learner)  # in-place update

    # STEP 7: Print and return the NSRTs.
    nsrts = [pnad.make_nsrt() for pnad in pnads]
    print("\nLearned NSRTs:")
    for nsrt in nsrts:
        print(nsrt)
    print()
    return set(nsrts)


def _learn_pnad_side_predicates(
        pnads: List[PartialNSRTAndDatastore],
        ground_atom_dataset: List[GroundAtomTrajectory],
        train_tasks: List[Task], predicates: Set[Predicate],
        segments: List[Segment],
        segmented_trajs: List[List[Segment]]) -> List[PartialNSRTAndDatastore]:
    def _check_goal(s: Tuple[PartialNSRTAndDatastore, ...]) -> bool:
        del s  # unused
        # There are no goal states for this search; run until exhausted.
        return False

    def _get_successors(
        s: Tuple[PartialNSRTAndDatastore, ...],
    ) -> Iterator[Tuple[None, Tuple[PartialNSRTAndDatastore, ...], float]]:
        # For each PNAD/operator...
        for i in range(len(s)):
            pnad = s[i]
            _, option_vars = pnad.option_spec
            # ...consider changing each of its effects to a side predicate.
            for effect_set, add_or_delete in [(pnad.op.add_effects, "add"),
                                              (pnad.op.delete_effects,
                                               "delete")]:
                for effect in effect_set:
                    new_pnad = PartialNSRTAndDatastore(
                        pnad.op.effect_to_side_predicate(
                            effect, option_vars, add_or_delete),
                        pnad.datastore, pnad.option_spec)
                    sprime = list(s)
                    sprime[i] = new_pnad
                    yield (None, tuple(sprime), 1.0)
            # ...consider removing it.
            sprime = list(s)
            del sprime[i]
            yield (None, tuple(sprime), 1.0)

    score_func = _PredictionErrorScoreFunction(predicates, [], {}, train_tasks)

    def _evaluate(s: Tuple[PartialNSRTAndDatastore, ...]) -> float:
        # Score function for search. Lower is better.
        strips_ops = [pnad.op for pnad in s]
        option_specs = [pnad.option_spec for pnad in s]
        score = score_func.evaluate_with_operators(frozenset(),
                                                   ground_atom_dataset,
                                                   segments, strips_ops,
                                                   option_specs)
        return score

    # Run the search, starting from original PNADs.
    path, _, _ = utils.run_hill_climbing(tuple(pnads), _check_goal,
                                         _get_successors, _evaluate)
    # The last state in the search holds the final PNADs.
    pnads = list(path[-1])
    # Recompute the datastores in the PNADs. We need to do this
    # because now that we have side predicates, each transition may be
    # assigned to *multiple* datastores.
    _recompute_datastores_from_segments(segmented_trajs, pnads)
    return pnads


def _recompute_datastores_from_segments(
        segmented_trajs: List[List[Segment]],
        pnads: List[PartialNSRTAndDatastore]) -> None:
    for pnad in pnads:
        pnad.datastore = []  # reset all PNAD datastores
    for seg_traj in segmented_trajs:
        objects = set(seg_traj[0].states[0])
        for segment in seg_traj:
            if segment.has_option():
                segment_option = segment.get_option()
                segment_param_option = segment_option.parent
                segment_option_objs = tuple(segment_option.objects)
            else:
                segment_param_option = DummyOption.parent
                segment_option_objs = tuple()
            # Get ground operators given these objects and option objs.
            for pnad in pnads:
                param_opt, opt_vars = pnad.option_spec
                if param_opt != segment_param_option:
                    continue
                isub = dict(zip(opt_vars, segment_option_objs))
                # Consider adding this segment to each datastore.
                for ground_op in utils.all_ground_operators_given_partial(
                        pnad.op, objects, isub):
                    # Check if preconditions hold.
                    if not ground_op.preconditions.issubset(
                            segment.init_atoms):
                        continue
                    # Check if effects match. Note that we're using the side
                    # predicates semantics here!
                    atoms = utils.apply_operator(ground_op, segment.init_atoms)
                    if not atoms.issubset(segment.final_atoms):
                        continue
                    # This segment belongs in this datastore, so add it.
                    # TODO: this is an edge case we need to fix by changing
                    # ObjToVarSubs in the partition to VarToObjSubs.
                    if len(set(ground_op.objects)) != len(ground_op.objects):
                        continue
                    sub = dict(zip(ground_op.objects, pnad.op.parameters))
                    pnad.add_to_datastore((segment, sub),
                                          check_consistency=False)


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
