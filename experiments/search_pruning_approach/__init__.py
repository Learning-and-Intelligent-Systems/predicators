import cProfile
from copy import deepcopy
import dataclasses
from dataclasses import dataclass
from itertools import cycle
from gym.spaces import Box
import logging
import numpy as np
from numpy import typing as npt
import pickle
import time
import torch
from tqdm import tqdm
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple, cast

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("tkagg")

from experiments.search_pruning_approach.learning import FeasibilityClassifier, FeasibilityDatapoint, NeuralFeasibilityClassifier, StaticFeasibilityClassifier
from experiments.search_pruning_approach.low_level_planning import run_low_level_search, run_backtracking_with_previous_states
from experiments.shelves2d import Shelves2DEnv

from predicators import utils
from predicators.approaches.base_approach import ApproachFailure, ApproachTimeout
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.nsrt_learning.sampler_learning import _LearnedSampler
from predicators.option_model import _OptionModelBase
from predicators.planning import PlanningTimeout, task_plan, task_plan_grounding
from predicators.settings import CFG
from predicators.structs import NSRT, _GroundNSRT, _Option, Dataset, GroundAtom, Metrics, ParameterizedOption, Predicate, Segment, State, Task, Type, Variable


__all__ = ["SearchPruningApproach"]

class PicklableGroundNSRT:
    """Used for compression of an NSRT to a form suitable for saving to a file.
    When decompressing it's necessary to provide a mapping from nsrt's str
    to nsrts with those names to properly decode them. The NSRTs need to have
    also been learned with the same parameters.
    """
    def __init__(self, ground_nsrt: _GroundNSRT):
        self.parent_str = str(ground_nsrt.parent)
        self.objects = ground_nsrt.objects

    def to_ground_nsrt(self, nsrts_dict: Dict[str, NSRT]) -> _GroundNSRT:
        return nsrts_dict[self.parent_str].ground(self.objects)

class PicklableFeasibilityDatapoint:
    """Used for a compression of a feasibility datapoint to a form suitable for
    saving to a file. When decompressing it's necessary to provide a mapping from
    nsrt names to nsrts with those names to properly decode them. The NSRTs need
    to have also been learned with the same parameters.
    """
    def __init__(self, datapoint: FeasibilityDatapoint):
        self.states = datapoint.states
        self.skeleton = [PicklableGroundNSRT(ground_nsrt) for ground_nsrt in datapoint.skeleton]

    def to_feasibility_datapoint(self, nsrts: Set[NSRT]) -> _GroundNSRT:
        nsrts_dict = {str(nsrt): nsrt for nsrt in nsrts}
        return FeasibilityDatapoint(self.states, [
            picklable_ground_nsrt.to_ground_nsrt(nsrts_dict)
            for picklable_ground_nsrt in self.skeleton
        ])

@dataclass(frozen=True)
class InterleavedBacktrackingDatapoint():
    states: List[State]
    atoms_sequence: List[Set[GroundAtom]]
    horizons: npt.NDArray
    skeleton: List[_GroundNSRT]

    def __post_init__(self):
        assert len(self.states) + 1 == len(self.skeleton) == len(self.atoms_sequence) == len(self.horizons)

    def substitute_nsrts(self, nsrts_dict: Dict[str, NSRT]) -> 'InterleavedBacktrackingDatapoint':
        return dataclasses.replace(self, skeleton=[
            nsrts_dict[str(ground_nsrt.parent)].ground(ground_nsrt.objects)
            for ground_nsrt in self.skeleton
        ])

class SearchPruningApproach(NSRTLearningApproach):
    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types, action_space, train_tasks)
        self._positive_feasibility_dataset: List[FeasibilityDatapoint] = []
        self._negative_feasibility_dataset: List[FeasibilityDatapoint] = []
        self._feasibility_classifier = StaticFeasibilityClassifier(lambda states, skeleton: True, 1.0)

    @classmethod
    def get_name(cls) -> str:
        return "search_pruning"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Generate the base NSRTs
        super().learn_from_offline_dataset(dataset)

        # Make sure we have direct access to the regressors
        assert all(type(nsrt.sampler) == _LearnedSampler for nsrt in self._nsrts)

        # Make sure the trajectories are entirely covered by the learned NSRTs (we can easily generate skeletons)
        assert all(
            segment in self._seg_to_ground_nsrt
            for segmented_traj in self._segmented_trajs
            for segment in segmented_traj
        )

        # Preparing data collection and training
        self._positive_feasibility_dataset: List[FeasibilityDatapoint] = []
        self._negative_feasibility_dataset: List[FeasibilityDatapoint] = []
        dataset_path: str = utils.create_dataset_filename_str(["feasibility_dataset"])[0]

        # Running data collection and training
        seed = self._seed + 696969
        assert CFG.feasibility_learning_strategy in {
            "backtracking", "generated_data", "ground_truth_data", "ground_truth_classifier", "load_data"
        }

        if CFG.feasibility_learning_strategy == "load_data":
            self._positive_feasibility_dataset, self._negative_feasibility_dataset = self._load_data(dataset_path, self._nsrts)
        elif CFG.feasibility_learning_strategy == "ground_truth_classifier":
            self._set_ground_truth_classifier()
        elif CFG.feasibility_learning_strategy == "ground_truth_data":
            self._collect_data_ground_truth()
        elif CFG.feasibility_learning_strategy == "generated_data":
            self._collect_data_generated(seed)
        else: # Main strategy; the other ones are for testing/debugging
            self._collect_data_interleaved_backtracking(seed)

        # Saving the generated dataset (if applicable)
        if CFG.feasibility_learning_strategy not in {"load_data", "ground_truth_classifier"}:
            self._save_data(dataset_path, self._positive_feasibility_dataset, self._negative_feasibility_dataset)

        if CFG.feasibility_learning_strategy not in {"ground_truth_classifier"}:
            self._learn_neural_feasibility_classifier(CFG.horizon)

    @classmethod
    def _load_data(
        cls, path: str, nsrts: Iterable[NSRT]
    ) -> Tuple[List[FeasibilityDatapoint], List[FeasibilityDatapoint]]:
        """Loads the feasibility datasets saved under a path.
        """
        picklable_positive_dataset, picklable_negative_dataset = pickle.load(open(path, 'rb'))
        assert len(picklable_negative_dataset) >= CFG.feasibility_num_negative_loaded_datapoints
        positive_dataset = [
            picklable_datapoint.to_feasibility_datapoint(nsrts)
            for picklable_datapoint in picklable_positive_dataset[:CFG.feasibility_num_negative_loaded_datapoints]
        ]
        negative_dataset = [
            picklable_datapoint.to_feasibility_datapoint(nsrts)
            for picklable_datapoint in picklable_negative_dataset[:CFG.feasibility_num_negative_loaded_datapoints]
        ]
        logging.info(f"loaded feasibility dataset of {len(positive_dataset)} positive and {len(negative_dataset)} negative datapoints")
        return positive_dataset, negative_dataset

    @classmethod
    def _save_data(
        cls,
        path: str,
        positive_dataset: List[FeasibilityDatapoint],
        negative_dataset: List[FeasibilityDatapoint]
    ) -> None:
        """Saves the feasibility datasets under a path.
        """
        pickle.dump((
            [PicklableFeasibilityDatapoint(datapoint) for datapoint in positive_dataset],
            [PicklableFeasibilityDatapoint(datapoint) for datapoint in negative_dataset],
        ), open(path, "wb"))
        logging.info(f"saved generated feasibility datasets to {path}")

    def _set_ground_truth_classifier(self) -> None:
        """Sets a ground truth feasibility classifier.
        Used for debugging and testing purposes only.

        NOTE: works only for the Shelves2D environment
        """
        assert CFG.env == "shelves2d"
        def classifier(states: Sequence[State], skeleton: Sequence[_GroundNSRT]) -> Tuple[bool, float]:
            current_nsrt = skeleton[len(states) - 2]
            final_nsrt = skeleton[-1]

            assert final_nsrt.name in {"MoveCoverToBottom", "MoveCoverToTop"}
            if current_nsrt.name != "InsertBox":
                return True, 1.0

            box, shelf, _, _ = current_nsrt.objects
            _, box_y, _, box_h = Shelves2DEnv.get_shape_data(states[-1], box)
            shelf_x, shelf_y, shelf_w, shelf_h = Shelves2DEnv.get_shape_data(states[-1], shelf)

            if final_nsrt.name == "MoveCoverToTop":
                return box_y + box_h <= shelf_y + shelf_h, 1.0
            else:
                return box_y >= shelf_y, 1.0
        self._feasibility_classifier = StaticFeasibilityClassifier(classifier)

    def _collect_data_ground_truth(self) -> None:
        """Collects ground truth data for training the neural feasibility classifier.
        Used for debugging and testing purposes only. Not suitable for executing the environment
        since it does not produce the full distribution of negative samples that can be generated by
        diffusion models.

        NOTE: works only for the Shelves2D environment
        """
        assert CFG.env == "shelves2d"

        logging.info("Generating ground truth data...")

        # Collecting positive data based on the recordings
        logging.info("Generating positive data")
        def gen_positive_datapoints(segmented_traj: List[Segment]):
            skeleton = [self._seg_to_ground_nsrt[segment] for segment in segmented_traj]
            states = [segment.states[0] for segment in segmented_traj] + [segmented_traj[-1].states[-1]]
            return [
                FeasibilityDatapoint(
                    states = states[:-suffix_length],
                    skeleton = skeleton,
                )
                for suffix_length in range(1, len(skeleton))
            ]
        self._positive_feasibility_dataset: List[FeasibilityDatapoint] = sum(
            (gen_positive_datapoints(segmented_traj) for segmented_traj in self._segmented_trajs),
            []
        )

        # Collecting negative data by augmenting the positive data
        logging.info("Generating negative data")
        def gen_negative_datapoint(positive_datapoint: FeasibilityDatapoint):
            assert len(positive_datapoint.states) >= 2
            current_nsrt = positive_datapoint.skeleton[len(positive_datapoint.states) - 2]
            next_state = positive_datapoint.states[-1]
            assert current_nsrt.name == "InsertBox"

            box, shelf, _, _ = current_nsrt.objects
            _, box_y, _, box_h = Shelves2DEnv.get_shape_data(next_state, box)
            _, shelf_y, _, shelf_h = Shelves2DEnv.get_shape_data(next_state, shelf)

            new_next_state = next_state.copy()
            new_next_state.set(box, "pose_y", 2 * shelf_y + shelf_h - box_h - box_y)

            return dataclasses.replace(positive_datapoint, states=positive_datapoint.states[:-1] + [new_next_state])
        self._negative_feasibility_dataset: List[FeasibilityDatapoint] = [
            gen_negative_datapoint(positive_datapoint)
            for positive_datapoint in self._positive_feasibility_dataset
        ]
        logging.info(f"Collected ground truth feasibility dataset of {len(self._positive_feasibility_dataset)} positive and {len(self._negative_feasibility_dataset)} negative datapoints")

    def _collect_data_generated(self, seed: int) -> None:
        """Collects data for training the neural feasibility classifier by generating negative samples
        using the learned diffusion models. Used for debugging and testing purposes only.

        NOTE: works only for the Shelves2D environment
        """
        assert CFG.env == "shelves2d"

        logging.info("Generating ground truth data...")

        # Collecting positive data based on the recordings
        def gen_positive_datapoints(segmented_traj: List[Segment]):
            skeleton = [self._seg_to_ground_nsrt[segment] for segment in segmented_traj]
            states = [segment.states[0] for segment in segmented_traj] + [segmented_traj[-1].states[-1]]
            return [
                FeasibilityDatapoint(
                    states = states[:-suffix_length],
                    skeleton = skeleton,
                )
                for suffix_length in range(1, len(skeleton))
            ]

        positive_datapoints_goals: List[Tuple[FeasibilityDatapoint, Set[Predicate]]] = [
            (datapoint, segmented_traj[-1].final_atoms)
            for segmented_traj in tqdm(self._segmented_trajs[:800], "Generating positive data")
            for datapoint in gen_positive_datapoints(segmented_traj)
        ]

        self._positive_feasibility_dataset: List[FeasibilityDatapoint] = [
            datapoint for datapoint, _ in positive_datapoints_goals
        ]

        # Collecting negative data using learned samplers and by augmenting the positive data
        preds = self._initial_predicates | {
            lifted_atom.predicate
            for nsrt in self._nsrts
            for lifted_atom in nsrt.add_effects | nsrt.delete_effects | nsrt.preconditions
        } | {
            predicate
            for nsrt in self._nsrts
            for predicate in nsrt.ignore_effects
        }
        num_invalid = 0
        def gen_negative_datapoints(positive_datapoint: FeasibilityDatapoint, goal: Set[GroundAtom]):
            nonlocal num_invalid
            assert len(positive_datapoint.states) >= 2

            last_nsrt = positive_datapoint.skeleton[-1]
            current_nsrt = positive_datapoint.skeleton[len(positive_datapoint.states) - 2]
            assert current_nsrt.name == "InsertBox"

            current_state, next_state = positive_datapoint.states[-2:]

            # Collecting negative datapoints based on the positive datapoint
            datapoints = []
            for _ in range(3):
                # Running the sampler and the option
                option = current_nsrt.sample_option(current_state, goal, self._rng)
                new_next_state, _ = self._option_model.get_next_state_and_num_actions(current_state, option)

                # Checking if the sampling was successful
                if utils.abstract(new_next_state, preds) != utils.abstract(next_state, preds):
                    continue

                # Checking if the sample is negative but within bounds of the shelf
                box, shelf, _, _ = current_nsrt.objects
                box_x, box_y, box_w, box_h = Shelves2DEnv.get_shape_data(new_next_state, box)
                shelf_x, shelf_y, shelf_w, shelf_h = Shelves2DEnv.get_shape_data(new_next_state, shelf)
                if last_nsrt.name == "MoveCoverToBottom":
                    if box_y >= shelf_y and shelf_x <= box_x and box_x + box_w <= shelf_x + shelf_w:
                        continue
                    if not box_y + box_h <= shelf_y + shelf_h:
                        num_invalid += 1
                else:
                    if box_y + box_h <= shelf_y + shelf_h and shelf_x <= box_x and box_x + box_w <= shelf_x + shelf_w:
                        continue
                    if not box_y >= shelf_y:
                        num_invalid += 1
                datapoints.append(dataclasses.replace(positive_datapoint, states=positive_datapoint.states[:-1] + [new_next_state]))
                if len(datapoints) >= 2:
                    break
            return datapoints

        self._negative_feasibility_dataset: List[FeasibilityDatapoint] = [
            negative_datapoint
            for positive_datapoint, goal in tqdm(positive_datapoints_goals, "Generating negative data")
            for negative_datapoint in gen_negative_datapoints(positive_datapoint, goal)
        ]
        logging.info(f"Generated classifier-based feasibility dataset of {len(self._positive_feasibility_dataset)} positive and {len(self._negative_feasibility_dataset)} negative datapoints")

    def _collect_data_interleaved_backtracking(self, seed: int) -> None:
        """Collects data by interleaving data collection for a certain suffix length and learning on that data
        """
        # TODO: add an option to limit the number of tasks explored and handling around that
        assert CFG.feasibility_device in {'cpu', 'cuda'}

        # To make sure we are able to use the Pool without having to copy the entire model we reset the classifier
        self._feasibility_classifier = StaticFeasibilityClassifier(lambda states, skeleton: True, 1.0)

        # Precomputing the datapoints for interleaved backtracking
        search_datapoints: List[InterleavedBacktrackingDatapoint] = [
            InterleavedBacktrackingDatapoint(
                states = [segment.states[0] for segment in segmented_traj] + [segmented_traj[-1].states[-1]],
                atoms_sequence = [segment.init_atoms for segment in segmented_traj] + [segmented_traj[-1].final_atoms],
                horizons = np.cumsum(len(segment.action) for segment in segmented_traj),
                skeleton = [self._seg_to_ground_nsrt[segment] for segment in segmented_traj]
            ) for segmented_traj in self._segmented_trajs
        ]

        # Precomputing the nsrts on different devices
        if CFG.feasibility_device == 'cpu':
            nsrts_dicts: List[Dict[str, NSRT]] = [{str(nsrt): nsrt for nsrt in self._nsrts}]
            for nsrt in self._nsrts:
                nsrt.sampler.to('cpu')
        else:
            nsrts_dicts: List[Dict[str, NSRT]] = [{str(nsrt): nsrt for nsrt in self._nsrts}] + \
                [{str(nsrt): deepcopy(nsrt) for nsrt in self._nsrts} for _ in range(1, torch.cuda.device_count())]
            for id, nsrts_dict in enumerate(nsrts_dicts):
                device_name = f'cuda{id}'
                for nsrt in nsrts_dict.values():
                    nsrt.to(device_name)

        # Main data generation loop
        logging.info("Generating data with interleaved learning...")

        max_skeleton_length = max(map(len, self._segmented_trajs))
        for suffix_length in range(1, max_skeleton_length):
            logging.info(f"Collecting data for suffix length {suffix_length}...")

            # Moving the feasibility classifier to devices
            if isinstance(self._feasibility_classifier, NeuralFeasibilityClassifier) and CFG.feasibility_device == 'cuda':
                feasibility_classifiers = [self._feasibility_classifier] + \
                    [deepcopy(self._feasibility_classifier) for _ in range(1, torch.cuda.device_count())]
                for id, feasibility_classifier in enumerate(feasibility_classifiers):
                    feasibility_classifier.to(f'cuda{id}')
            else:
                feasibility_classifiers = [self._feasibility_classifier]
                if isinstance(self._feasibility_classifier, NeuralFeasibilityClassifier):
                    self._feasibility_classifier.to('cpu')

            # Creating the datapoints to search over and moving them to different devices
            indices = self._rng.choice(len(search_datapoints), CFG.feasibility_num_datapoints_per_iter)
            chosen_search_datapoints = [
                search_datapoints[idx].substitute_nsrts(nsrts_dict)
                for idx, nsrts_dict in zip(indices, cycle(nsrts_dicts))
            ]

            # Collecting positive examples
            self._positive_feasibility_dataset.extend(
                FeasibilityDatapoint(
                    states = search_datapoint.states[:-suffix_length],
                    skeleton = search_datapoint.skeleton,
                )
                for search_datapoint in tqdm(chosen_search_datapoints, "Collecting positive examples")
                if len(search_datapoint.skeleton) > suffix_length
            )

            # Collecting negative examples
            with torch.multiprocessing.pool.Pool(CFG.feasibility_num_data_collection_threads) as pool:
                self._negative_feasibility_dataset.extend(
                    datapoint
                    for datapoints in pool.imap(
                        lambda inputs: self._negative_data_collection_datapoint(
                            suffix_length,
                            self._option_model,
                            *inputs,
                        ), tqdm(zip(
                            cycle(feasibility_classifiers),
                            range(seed, seed + len(chosen_search_datapoints)),
                            chosen_search_datapoints
                        ), "Collecting negative examples")
                    )
                    for datapoint in datapoints
                )
                seed += len(chosen_search_datapoints)

            if suffix_length < max_skeleton_length - 1:
                self._learn_neural_feasibility_classifier(suffix_length)

        logging.info("Generated interleaving-based feasibility dataset of "
                     f"{len(self._positive_feasibility_dataset)} positive and {len(self._negative_feasibility_dataset)} negative datapoints")

    @classmethod
    def _negative_data_collection_datapoint(
        cls,
        idx,
        suffix_length: int,
        option_model: _OptionModelBase,
        feasibility_classifier: FeasibilityClassifier,
        seed: int,
        search_datapoint: InterleavedBacktrackingDatapoint,
    ) -> List[FeasibilityDatapoint]:
        """ Running data collection for a single suffix length and task
        """
        states, atoms_sequence, horizons, skeleton = dataclasses.astuple(search_datapoint)
        if len(skeleton) <= suffix_length:
            return []
        backtracking, _ = run_backtracking_with_previous_states(
            previous_states = states[:-suffix_length - 1],
            goal = atoms_sequence[-1],
            option_model = option_model,
            skeleton = skeleton,
            feasibility_classifier = feasibility_classifier,
            atoms_sequence = atoms_sequence,
            seed = seed,
            timeout = float('inf'),
            metrics = {},
            max_horizon = CFG.horizon - horizons[-suffix_length - 1],
        )
        next_states = [
            mb_subtree.state
            for _, mb_subtree in backtracking.failed_tries if mb_subtree is not None
        ]
        return [
            FeasibilityDatapoint(
                states = states[:-suffix_length - 1] + [next_state],
                skeleton = skeleton,
            ) for next_state in next_states
        ]

    def _run_sesame_plan(
        self,
        task: Task,
        nsrts: Set[NSRT],
        preds: Set[Predicate],
        timeout: float,
        seed: int
    ) -> Tuple[List[_Option], List[_GroundNSRT], Metrics]:
        end = time.perf_counter() + timeout

        init_atoms = utils.abstract(task.init, preds)
        objects = set(task.init)

        ground_nsrts, reachable_atoms = task_plan_grounding(init_atoms, objects, nsrts)
        heuristic = utils.create_task_planning_heuristic(
            heuristic_name = CFG.sesame_task_planning_heuristic,
            init_atoms = init_atoms,
            goal = task.goal,
            ground_ops = ground_nsrts,
            predicates = preds,
            objects = objects,
        )
        generator = task_plan(
            init_atoms = utils.abstract(task.init, preds),
            goal = task.goal,
            ground_nsrts = ground_nsrts,
            reachable_atoms = reachable_atoms,
            heuristic = heuristic,
            seed = seed,
            timeout = timeout,
            max_skeletons_optimized = CFG.horizon,
        )
        partial_refinements = []
        print("Starting sesame loop")
        for _ in range(CFG.sesame_max_skeletons_optimized):
            print("Still in the loop")
            skeleton, backtracking, timed_out = None, None, False
            try:
                skeleton, atoms_seq, metrics = next(generator)
                backtracking, is_success = run_low_level_search(
                    task = task,
                    option_model =self._option_model,
                    skeleton = skeleton,
                    feasibility_classifier = self._feasibility_classifier,
                    atoms_sequence = atoms_seq,
                    seed = seed, timeout = end - time.perf_counter(),
                    metrics = metrics,
                    max_horizon = CFG.horizon
                )
                if is_success:
                    _, options = backtracking.successful_trajectory
                    return options, skeleton, metrics
            except StopIteration:
                break
            except PlanningTimeout as e:
                backtracking = e.info.get('backtracking_tree')
                timed_out = True
            if skeleton is not None and backtracking is not None:
                _, plan = backtracking.longest_failuire
                partial_refinements.append((skeleton, plan))
            if timed_out:
                raise ApproachTimeout(
                    "Planning timed out!",
                    info = {"partial_refinements": partial_refinements}
                )
        raise ApproachFailure("Failed to find a successful backtracking")

    def _learn_neural_feasibility_classifier(
        self, max_inference_suffix: int
    ) -> None:
        """Running training on a fresh classifier

        Params:
            max_inference_suffix - what is the longest suffix (number of decoder NSRTs)
                which the classifier will attempt to classify
            shared_memory - whether to make the classifier movable between processes
                (using the torch.multiprocessing library)
        """
        neural_feasibility_classifier = NeuralFeasibilityClassifier(
            seed = CFG.seed,
            featurizer_hidden_sizes = CFG.feasibility_featurizer_hid_sizes,
            classifier_feature_size = CFG.feasibility_feature_size,
            positional_embedding_size = CFG.feasibility_embedding_size,
            positional_embedding_concat = CFG.feasibility_embedding_concat,
            transformer_num_heads = CFG.feasibility_num_heads,
            transformer_encoder_num_layers = CFG.feasibility_enc_num_layers,
            transformer_decoder_num_layers = CFG.feasibility_dec_num_layers,
            transformer_ffn_hidden_size = CFG.feasibility_ffn_hid_size,
            max_train_iters = CFG.feasibility_max_itr,
            general_lr = CFG.feasibility_general_lr,
            transformer_lr = CFG.feasibility_transformer_lr,
            max_inference_suffix = max_inference_suffix,
            cls_style = CFG.feasibility_cls_style,
            embedding_horizon = CFG.feasibility_embedding_max_idx,
            batch_size = CFG.feasibility_batch_size,
            threshold_recalibration_percentile = CFG.feasibility_threshold_recalibration_percentile,
            device = CFG.feasibility_device
        )
        neural_feasibility_classifier.fit(self._positive_feasibility_dataset, self._negative_feasibility_dataset)
        neural_feasibility_classifier.freeze()
        self._feasibility_classifier = neural_feasibility_classifier