"""process_learning_main module."""
import logging
from pprint import pformat
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, cast

from gym.spaces import Box

from predicators import utils
from predicators.nsrt_learning.nsrt_learning_main import _learn_pnad_options, \
    _learn_pnad_samplers
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning import learn_strips_operators
from predicators.nsrt_learning.strips_learning.clustering_learner import \
    ClusterAndSearchProcessLearner
from predicators.settings import CFG
from predicators.structs import CausalProcess, DerivedPredicate, DummyOption, \
    EndogenousProcess, ExogenousProcess, GroundAtomTrajectory, LiftedAtom, \
    LowLevelTrajectory, ParameterizedOption, Predicate, Segment, Task


def learn_processes_from_data(
    trajectories: List[LowLevelTrajectory],
    train_tasks: List[Task],
    predicates: Set[Predicate],
    known_options: Optional[Set[ParameterizedOption]] = None,
    action_space: Optional[Box] = None,
    ground_atom_dataset: Optional[List[GroundAtomTrajectory]] = None,
    sampler_learner: Optional[str] = None,
    annotations: Optional[List[Any]] = None,
    current_processes: Optional[Set[CausalProcess]] = None,
    log_all_processes: bool = True,
    online_learning_cycle: Optional[int] = None,
) -> Tuple[Set[CausalProcess], Dict[str, List]]:
    """Learn CausalProcesses from the given dataset of low-level transitions,
    using the given set of predicates."""
    logging.info(f"\nLearning CausalProcesses on {len(trajectories)} "
                 "trajectories...")
    # remember to reset at the end
    initial_segmentation_method = CFG.segmenter

    # We will probably learn endogenous and exogenous processes separately.
    if CFG.only_learn_exogenous_processes:
        endogenous_processes = [
            p for p in (current_processes or [])
            if isinstance(p, EndogenousProcess)
        ]
    else:
        assert sampler_learner is not None, \
            "Sampler learner must be specified for action model learning."
        # -- Learn the endogenous processes ---
        CFG.segmenter = "option_changes"
        # STEP 1: Segment the trajectory by options. (don't currently consider
        #         segmenting by predicates).
        #         Segment each trajectory in the dataset based on changes in
        #         either predicates or options. If we are doing option learning,
        #         then the data will not contain options, so this segmenting
        #         procedure only uses the predicates.
        #         If we know the option segmentations this is pretty similar to
        #         learning NSRTs.
        if ground_atom_dataset is None:
            segmented_trajs = [
                segment_trajectory(traj, predicates) for traj in trajectories
            ]
        else:
            segmented_trajs = [
                segment_trajectory(traj, predicates, ground_atom_dataset[i][1])
                for i, traj in enumerate(trajectories)
            ]

        # STEP 2: Learn STRIPS operators on the given data segments as for
        # NSRTs.
        pnads = learn_strips_operators(
            trajectories,
            train_tasks,
            predicates,
            segmented_trajs,
            verify_harmlessness=
            False,  # these processes are in principal 'harmful'
            # because they should leave some atoms to be explained by exogenous
            # processes.
            verbose=(CFG.option_learner != "no_learning"),
            annotations=annotations)

        # STEP 3: Learn options and update PNADs
        if CFG.strips_learner != "oracle" or \
                CFG.sampler_learner != "oracle" or \
                CFG.option_learner != "no_learning":
            assert action_space is not None, \
                "Action space must be provided for option learning."
            assert known_options is not None, \
                "Known options must be provided for option learning."
            # Updates the endo_papads in-place.
            _learn_pnad_options(pnads, known_options, action_space)

        # STEP 4 (currently skipped): Learn samplers and update PNADs
        _learn_pnad_samplers(pnads, sampler_learner)

        # STEP 5: Convert PNADs to endogenous processes. (Maybe also make rough
        #         parameter estimates.)
        endogenous_processes = [
            pnad.make_endogenous_process() for pnad in pnads
        ]
        # for proc in endogenous_processes:
        #     logging.debug(f"{proc}")
        # logging.debug("")

    # --- Learn the exogenous processes. ---
    # STEP 1: Segment the trajectory by atom_changes, and filter out the ones
    #         that are explained by the endogenous processes.
    CFG.segmenter = "atom_changes"
    CFG.strips_learner = CFG.exogenous_process_learner

    segmented_trajs = [
        segment_trajectory(traj, predicates, verbose=False)
        for traj in trajectories
    ]
    # Filter out segments explained by endogenous processes.
    remaining_segmented_trajs = filter_explained_segment(
        segmented_trajs,
        cast(List[CausalProcess], endogenous_processes),
        remove_options=False)

    # STEP 2: Learn the exogenous processes based on unexplained processes.
    #         This is different from STRIPS/endogenous processes, where these
    #         don't have options and samplers.
    num_unexplaned_segments = sum(
        len(sugments) for sugments in remaining_segmented_trajs)
    if num_unexplaned_segments == 0:
        new_exogenous_processes = []
    else:
        process_learner = ClusterAndSearchProcessLearner(
            trajectories,
            train_tasks,
            predicates,
            remaining_segmented_trajs,
            verify_harmlessness=False,
            verbose=(CFG.option_learner != "no_learning"),
            annotations=annotations,
            endogenous_processes=set(endogenous_processes),
            online_learning_cycle=online_learning_cycle,
        )
        exogenous_processes_pnad = process_learner.learn()
        new_exogenous_processes = [
            pnad.make_exogenous_process() for pnad in exogenous_processes_pnad
        ]
        # Get the other conditions' scores through class attributes.
        proc_name_to_results: Dict[str, List[
            Tuple[float, FrozenSet[LiftedAtom], Tuple, ExogenousProcess]]] =\
                process_learner.proc_name_to_results
        logging.info(
            f"Learned {len(new_exogenous_processes)} exogenous processes:\n"
            f"{pformat(new_exogenous_processes)}")
    if CFG.pause_after_process_learning_for_inspection:
        input("Press Enter to continue...")  # pause for user inspection

    # STEP 3: Make, log, and return the endogenous and exogenous processes.
    processes = endogenous_processes + new_exogenous_processes
    if log_all_processes:
        logging.info(f"\nLearned CausalProcesses:\n{pformat(processes)}")

    CFG.segmenter = initial_segmentation_method
    return set(processes), proc_name_to_results


def is_endogenous_process_list(processes: List) -> bool:
    """Check if all elements in the list are EndogenousProcess."""
    return all(isinstance(p, EndogenousProcess) for p in processes)


def is_exogenous_process_list(processes: List) -> bool:
    """Check if all elements in the list are ExogenousProcess."""
    return all(isinstance(p, ExogenousProcess) for p in processes)


def filter_explained_segment(
    segmented_trajs: List[List[Segment]],
    processes: List[CausalProcess],
    remove_options: bool = False,
    log_remaining_trajs: bool = False,
) -> List[List[Segment]]:
    """Filter out segments that are explained by the given PNADs."""
    num_segments = sum(len(traj) for traj in segmented_trajs)
    if is_endogenous_process_list(processes):
        processes_type_str = "endogenous"
    elif is_exogenous_process_list(processes):
        processes_type_str = "exogenous"
    else:
        raise NotImplementedError("Currently don't support "
                                  "mixed process types.")
    logging.debug(f"\nNum of segments before filtering the ones explained by "
                  f"{processes_type_str} procs: {num_segments}, from "
                  f"{len(segmented_trajs)} trajs.")
    remaining_trajs = []
    for traj in segmented_trajs:
        objects = set(traj[0].trajectory.states[0])
        remaining_segments = []
        for segment in traj:
            # Note: is this kind of like "cover"?
            if processes_type_str == "endogenous":
                relevant_procs = [
                    p for p in processes
                    if segment.get_option().parent == cast(
                        EndogenousProcess, p).option
                ]
            else:
                # all exogenous; mixed cases all handle at the top.
                relevant_procs = processes
            add_atoms = {
                a
                for a in segment.add_effects
                if not isinstance(a.predicate, DerivedPredicate)
            }
            delete_atoms = {
                a
                for a in segment.delete_effects
                if not isinstance(a.predicate, DerivedPredicate)
            }
            # if not explained by any; consider explained if the atom change is
            # a subset of the add_effects and delete_effects of any compatible
            # ground process.
            not_explained_by_any = True
            for proc in relevant_procs:
                if processes_type_str == "endogenous":
                    endo_proc = cast(EndogenousProcess, proc)
                    option_vars = endo_proc.option_vars
                    ignore_effects = endo_proc.ignore_effects
                else:
                    option_vars = []
                    ignore_effects = set()
                var_to_obj = dict(
                    zip(option_vars,
                        segment.get_option().objects))
                for g_proc in utils.all_ground_operators_given_partial(
                        proc, objects, var_to_obj):  # type: ignore[arg-type]
                    _add_atoms = add_atoms.copy()
                    _delete_atoms = delete_atoms.copy()
                    if ignore_effects:
                        _add_atoms = {
                            a
                            for a in add_atoms
                            if a.predicate not in ignore_effects
                        }
                        _delete_atoms = {
                            a
                            for a in delete_atoms
                            if a.predicate not in ignore_effects
                        }
                    if _add_atoms.issubset(g_proc.add_effects) and \
                        _delete_atoms.issubset(g_proc.delete_effects):
                        not_explained_by_any = False
                        break
            if not_explained_by_any:
                if remove_options:
                    segment.set_option(DummyOption)
                remaining_segments.append(segment)
        remaining_trajs.append(remaining_segments)

    num_remaining_segments = sum(len(traj) for traj in remaining_trajs)
    logging.debug(f"Num of leftover segments: {num_remaining_segments}")
    if log_remaining_trajs:
        for j, seg_traj in enumerate(remaining_trajs):
            logging.debug(f"Trajectory {j}:")
            for i, segment in enumerate(seg_traj):
                logging.debug(f"Segment {i}. Init atoms: "
                              f"{sorted(segment.init_atoms)}")
                logging.debug(f"Add effects: {sorted(segment.add_effects)}")
                logging.debug(
                    f"Delete effects: {sorted(segment.delete_effects)}")
                logging.debug(f"Option: {segment.get_option()}\n")
    return remaining_trajs
