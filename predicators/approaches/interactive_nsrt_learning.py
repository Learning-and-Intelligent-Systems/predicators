# """An approach that learns one new operator from a demonstration + exploration."""

# import logging
# from typing import Any, Callable, Dict, List, Optional, Sequence, Set

# import dill as pkl
# import numpy as np
# from gym.spaces import Box

# from predicators import utils
# from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
# from predicators.explorers import create_explorer
# from predicators.ground_truth_models import get_gt_nsrts
# from predicators.ml_models import BinaryClassifierEnsemble, \
#     KNeighborsClassifier, LearnedPredicateClassifier, MLPBinaryClassifier
# from predicators.nsrt_learning.nsrt_learning_main import learn_new_nsrts_from_data
# from predicators.settings import CFG
# from predicators.structs import Dataset, GroundAtom, GroundAtomsHoldQuery, \
#     GroundAtomsHoldResponse, InteractionRequest, InteractionResult, \
#     LowLevelTrajectory, NSRT, ParameterizedOption, Predicate, Query, State, Task, \
#     Type

# class InteractiveNSRTLearningApproach(NSRTLearningApproach):

#     def __init__(self, initial_predicates: Set[Predicate],
#                  initial_options: Set[ParameterizedOption], types: Set[Type],
#                  action_space: Box, train_tasks: List[Task]) -> None:
#         super().__init__(initial_predicates, initial_options, types,
#                          action_space, train_tasks)

#     @classmethod
#     def get_name(cls) -> str:
#         return "interactive_nsrt_learning"

#     @property
#     def is_learning_based(self) -> bool:
#         return True

#     def _get_current_nsrts(self) -> Set[NSRT]:
#         return self._gt_nsrts | self._learned_nsrts

#     def load(self, online_learning_cycle: Optional[int]) -> None:
#         raise NotImplementedError

#     def learn_from_offline_dataset(self, dataset: Dataset) -> None:
#         gt_nsrts = get_gt_nsrts(CFG.env, self._initial_predicates,
#                                     self._initial_options)
#         self._gt_nsrts = self._keep_only_config_included_nsrts(gt_nsrts)
#         assert len(dataset.trajectories) == 1

#         self._learn_nsrts(dataset.trajectories,
#                           online_learning_cycle=None,
#                           annotations=(dataset.annotations
#                                       if dataset.has_annotations else None))

#     def _learn_nsrts(self, trajectories: List[LowLevelTrajectory],
#                      online_learning_cycle: Optional[int],
#                      annotations: Optional[List[Any]]) -> None:
#         dataset_fname, _ = utils.create_dataset_filename_str(
#             saving_ground_atoms=True,
#             online_learning_cycle=online_learning_cycle)
#         # If CFG.load_atoms is set, then try to create a GroundAtomTrajectory
#         # by loading sets of GroundAtoms directly from a saved file.
#         if CFG.load_atoms:
#             os.makedirs(CFG.data_dir, exist_ok=True)
#             # Check that the dataset file was previously saved.
#             if os.path.exists(dataset_fname):
#                 # Load the ground atoms dataset.
#                 with open(dataset_fname, "rb") as f:
#                     ground_atom_dataset_atoms = pkl.load(f)
#                 assert len(trajectories) == len(ground_atom_dataset_atoms)
#                 logging.info("\n\nLOADED GROUND ATOM DATASET")

#                 # The saved ground atom dataset consists only of sequences
#                 # of sets of GroundAtoms, we need to recombine this with
#                 # the LowLevelTrajectories to create a GroundAtomTrajectory.
#                 ground_atom_dataset = []
#                 for i, traj in enumerate(trajectories):
#                     ground_atom_seq = ground_atom_dataset_atoms[i]
#                     ground_atom_dataset.append(
#                         (traj, [set(atoms) for atoms in ground_atom_seq]))
#             else:
#                 raise ValueError(f"Cannot load ground atoms: {dataset_fname}")
#         else:
#             # Apply predicates to data, producing a dataset of abstract states.
#             ground_atom_dataset = utils.create_ground_atom_dataset(
#                 trajectories, self._get_current_predicates())
#             # Save ground atoms dataset to file. Note that a
#             # GroundAtomTrajectory contains a normal LowLevelTrajectory and a
#             # list of sets of GroundAtoms, so we only save the list of
#             # GroundAtoms (the LowLevelTrajectories are saved separately).
#             ground_atom_dataset_to_pkl = []
#             for gt_traj in ground_atom_dataset:
#                 trajectory = []
#                 for ground_atom_set in gt_traj[1]:
#                     trajectory.append(ground_atom_set)
#                 ground_atom_dataset_to_pkl.append(trajectory)
#             with open(dataset_fname, "wb") as f:
#                 pkl.dump(ground_atom_dataset_to_pkl, f)

#         self._nsrts, self._segmented_trajs, self._seg_to_nsrt = \
#             learn_new_nsrts_from_data(trajectories,
#                                   self._gt_nsrts,
#                                   self._train_tasks,
#                                   self._get_current_predicates(),
#                                   self._initial_options,
#                                   self._action_space,
#                                   ground_atom_dataset,
#                                   sampler_learner=CFG.sampler_learner,
#                                   annotations=annotations)
#         save_path = utils.get_approach_save_path_str()
#         with open(f"{save_path}_{online_learning_cycle}.NSRTs", "wb") as f:
#             pkl.dump(self._nsrts, f)
#         if CFG.compute_sidelining_objective_value:
#             self._compute_sidelining_objective_value(trajectories)



#     def _keep_only_config_included_nsrts(self, nsrts):
#         if not CFG.excluded_nsrts:
#             return set()
#         excluded_names = set(CFG.excluded_nsrts.split(","))
#         assert excluded_names.issubset({n.name for n in nsrts}), "Unrecognized NSRT in excluded_nsrts!"
#         included_nsrts = {n for n in nsrts if n.name not in excluded_names}
#         return included_nsrts

