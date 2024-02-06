import cProfile
import dataclasses
from dataclasses import dataclass
import pickle
import time
from typing import Dict, Iterable, List, Sequence, Set, Tuple
from gym.spaces import Box
from experiments.search_pruning_approach.learning import FeasibilityDatapoint, NeuralFeasibilityClassifier
from experiments.search_pruning_approach.low_level_planning import run_low_level_search, run_backtracking_with_previous_states
from experiments.shelves2d import Shelves2DEnv
from predicators import utils
from predicators.approaches.base_approach import ApproachFailure
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.nsrt_learning.sampler_learning import _LearnedSampler
from predicators.planning import task_plan, task_plan_grounding
from predicators.settings import CFG
from predicators.structs import NSRT, _GroundNSRT, _Option, Dataset, GroundAtom, Metrics, ParameterizedOption, Predicate, Segment, State, Task, Type, Variable
import numpy as np
from numpy import typing as npt
import logging

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("tkagg")

__all__ = ["SearchPruningApproach"]

class PicklableGroundNSRT:
    def __init__(self, ground_nsrt: _GroundNSRT):
        self.parent_name = ground_nsrt.parent.name
        self.objects = ground_nsrt.objects

    def to_ground_nsrt(self, nsrts: Dict[str, NSRT]) -> _GroundNSRT:
        return nsrts[self.parent_name].ground(self.objects)

class PicklableFeasibilityDatapoint:
    def __init__(self, datapoint: FeasibilityDatapoint):
        self.states = datapoint.states
        self.skeleton = [PicklableGroundNSRT(ground_nsrt) for ground_nsrt in datapoint.skeleton]

    def to_feasibility_datapoint(self, nsrts: Dict[str, NSRT]) -> _GroundNSRT:
        return FeasibilityDatapoint(self.states, [
            picklable_ground_nsrt.to_ground_nsrt(nsrts)
            for picklable_ground_nsrt in self.skeleton
        ])


class SearchPruningApproach(NSRTLearningApproach):
    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types, action_space, train_tasks)
        self._positive_feasibility_dataset: List[FeasibilityDatapoint] = []
        self._negative_feasibility_dataset: List[FeasibilityDatapoint] = []
        self._learn_neural_feasibility_classifier(0)

    @classmethod
    def get_name(cls) -> str:
        return "search_pruning"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Generate the base NSRTs
        super().learn_from_offline_dataset(dataset)

        # Make sure we have direct access to the regressors
        assert all(type(nsrt._sampler) == _LearnedSampler for nsrt in self._nsrts)

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
        assert CFG.feasibility_learning_strategy in {"backtracking", "generated_data", "ground_truth_data", "ground_truth_classifier", "load_data"}
        if CFG.feasibility_learning_strategy == "load_data":
            self._positive_feasibility_dataset, self._negative_feasibility_dataset = self._load_data(dataset_path, self._nsrts)
        elif CFG.feasibility_learning_strategy == "ground_truth_classifier":
            self._set_ground_truth_classifier()
        elif CFG.feasibility_learning_strategy == "ground_truth_data":
            self._collect_data_ground_truth()
        elif CFG.feasibility_learning_strategy == "generated_data":
            self._collect_data_generated(seed)
        else: # Main strategy; the other ones are for testing/debugging
            self._collect_data_interleaved_backtracking(seed, {})

        # Saving the generated dataset (if applicable)
        if CFG.feasibility_learning_strategy not in {"load_data", "ground_truth_classifier"}:
            self._save_data(dataset_path, self._positive_feasibility_dataset, self._negative_feasibility_dataset)

        if CFG.feasibility_learning_strategy not in {"ground_truth_classifier", "backtracking"}:
            self._learn_neural_feasibility_classifier(CFG.horizon)

    @classmethod
    def _load_data(cls, path: str, nsrts: Iterable[NSRT]) -> Tuple[List[FeasibilityDatapoint], List[FeasibilityDatapoint]]:
        nsrts_dict = {nsrt.name: nsrt for nsrt in nsrts}
        picklable_positive_dataset, picklable_negative_dataset = pickle.load(open(path, 'rb'))
        return ([
                picklable_datapoint.to_feasibility_datapoint(nsrts_dict)
                for picklable_datapoint in picklable_positive_dataset
            ], [
                picklable_datapoint.to_feasibility_datapoint(nsrts_dict)
                for picklable_datapoint in picklable_negative_dataset
            ]
        )

    @classmethod
    def _save_data(cls, path: str, positive_dataset: List[FeasibilityDatapoint], negative_dataset: List[FeasibilityDatapoint]):
        pickle.dump((
            [PicklableFeasibilityDatapoint(datapoint) for datapoint in positive_dataset],
            [PicklableFeasibilityDatapoint(datapoint) for datapoint in negative_dataset],
        ), open(path, "wb"))

    def _set_ground_truth_classifier(self) -> None:
        assert CFG.env == "shelves2d"
        def feasibility_classifier(states: Sequence[State], skeleton: Sequence[_GroundNSRT]) -> bool:
            current_nsrt = skeleton[len(states) - 2]
            final_nsrt = skeleton[-1]
            if current_nsrt.name != "InsertBox":
                return True
            assert final_nsrt.name in {"MoveCoverToBottom", "MoveCoverToTop"}
            box, shelf, _, _ = current_nsrt.objects
            _, box_y, _, box_h = Shelves2DEnv.get_shape_data(states[-1], box)
            shelf_x, shelf_y, shelf_w, shelf_h = Shelves2DEnv.get_shape_data(states[-1], shelf)
            if final_nsrt.name == "MoveCoverToTop":
                return box_y + box_h <= shelf_y + shelf_h
            else:
                return box_y >= shelf_y
        self._feasibility_classifier = feasibility_classifier

    def _collect_data_ground_truth(self) -> None:
        assert CFG.env == "shelves2d"

        logging.info("Generating ground truth data...")
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

    def _collect_data_generated(self, seed: int) -> None:
        assert CFG.env == "shelves2d"

        logging.info("Generating ground truth data...")
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

        positive_datapoints_goals: List[Tuple[FeasibilityDatapoint, Set[GroundAtom]]] = [
            (datapoint, segmented_traj[-1].final_atoms, preds)
            for segmented_traj in self._segmented_trajs[:800]
            for preds in [{
                atom.predicate for segment in segmented_traj
                for atoms in [segment.init_atoms, segment.final_atoms]
                for atom in atoms
            }]
            for datapoint in gen_positive_datapoints(segmented_traj)
        ]

        self._positive_feasibility_dataset: List[FeasibilityDatapoint] = [
            datapoint for datapoint, _, _ in positive_datapoints_goals
        ]

        logging.info("Generating negative data")
        rng = np.random.default_rng(seed)
        num_invalid = 0
        def gen_negative_datapoints(positive_datapoint: FeasibilityDatapoint, goal: Set[GroundAtom], preds: Set[GroundAtom]):
            nonlocal num_invalid
            assert len(positive_datapoint.states) >= 2

            last_nsrt = positive_datapoint.skeleton[-1]
            current_nsrt = positive_datapoint.skeleton[len(positive_datapoint.states) - 2]
            assert current_nsrt.name == "InsertBox"

            current_state, next_state = positive_datapoint.states[-2:]
            datapoints = []
            for _ in range(3):
                option = current_nsrt.sample_option(current_state, goal, rng)
                new_next_state, _ = self._option_model.get_next_state_and_num_actions(current_state, option)

                if utils.abstract(new_next_state, preds) != utils.abstract(next_state, preds):
                    continue

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
            print(len(datapoints))
            return datapoints

        self._negative_feasibility_dataset: List[FeasibilityDatapoint] = [
            negative_datapoint
            for positive_datapoint, goal, preds in positive_datapoints_goals
            for negative_datapoint in gen_negative_datapoints(positive_datapoint, goal, preds)
        ]


        print(len(self._positive_feasibility_dataset), len(self._negative_feasibility_dataset), num_invalid)

    def _collect_data_interleaved_backtracking(self, seed: int, metrics: Metrics) -> None:
        # TODO: add metric collection (how many samples needed for backtracking)
        states_data: List[List[State]] = [[
            segment.states[0] for segment in segmented_traj
        ] + [segmented_traj[-1].states[-1]] for segmented_traj in self._segmented_trajs]
        skeletons: List[List[_GroundNSRT]] = [[
            self._seg_to_ground_nsrt[segment] for segment in segmented_traj
        ] for segmented_traj in self._segmented_trajs]
        atom_sequences: List[List[Set[GroundAtom]]] = [[
            segment.init_atoms for segment in segmented_traj
        ] + [segmented_traj[-1].final_atoms] for segmented_traj in self._segmented_trajs]
        horizons: List[npt.NDArray[npt.int32]] = [
            np.cumsum(len(segment.action) for segment in segmented_traj)
            for segmented_traj in self._segmented_trajs
        ]

        for suffix_length in range(1, 2):#range(1, max(map(len, self._segmented_trajs))): # length of the suffix of states
            for states, skeleton, atom_sequence, horizon, idx in zip(states_data, skeletons, atom_sequences, horizons, range(len(horizons))):
                if suffix_length >= len(atom_sequences):
                    continue

                # Including positive example
                self._positive_feasibility_dataset.append(
                    FeasibilityDatapoint(
                        states = states[:-suffix_length],
                        skeleton = skeleton,
                    )
                )

                print(f"Backtracking nr {idx}; suffix_length {suffix_length}")

                # Calculating negative examples if needed
                if len(self._negative_feasibility_dataset) > 2*len(self._positive_feasibility_dataset):
                    continue
                seed += 1
                backtracking, _ = run_backtracking_with_previous_states(
                    previous_states = states[:-suffix_length - 1],
                    goal = atom_sequence[-1],
                    option_model = self._option_model,
                    skeleton = skeleton,
                    feasibility = None,#self._feasibility_classifier,
                    atoms_sequence = atom_sequence,
                    seed = seed,
                    timeout = float('inf'),
                    metrics = metrics,
                    max_horizon = CFG.horizon - horizon[:-suffix_length - 1],
                )
                next_states = [
                    mb_subtree.state
                    for _, mb_subtree in backtracking.failed_tries if mb_subtree is not None
                ]
                self._negative_feasibility_dataset.extend(
                    FeasibilityDatapoint(
                        states = states[:-suffix_length - 1] + [next_state],
                        skeleton = skeleton,
                    ) for next_state in next_states
                )
                print(f"Total number of negative examples: {len(self._negative_feasibility_dataset)}")
            # self._learn_neural_feasibility_classifier(suffix_length)
        for idx in np.random.default_rng().choice(len(self._negative_feasibility_dataset), 16, replace=False):
            datapoint = self._negative_feasibility_dataset[idx]
            next_state = datapoint.states[-1]
            fig = Shelves2DEnv.render_state_plt(next_state, None)
            fig.suptitle(datapoint.skeleton[-1].name)
        plt.show()

    def _run_sesame_plan(
            self, task: Task, nsrts: Set[NSRT], preds: Set[Predicate],
            timeout: float, seed: int) -> Tuple[List[_Option], List[_GroundNSRT], Metrics]:
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

        for skeleton, atoms_seq, metrics in generator:
            backtracking, is_success = run_low_level_search(
                task = task,
                option_model =self._option_model,
                skeleton = skeleton,
                feasibility = self._feasibility_classifier,
                atoms_sequence = atoms_seq,
                seed = seed, timeout = end - time.perf_counter(),
                metrics = metrics,
                max_horizon = CFG.horizon
            )
            if is_success:
                _, options = backtracking.successful_trajectory
                return options, skeleton, metrics
        raise ApproachFailure("Failed to find a successful backtracking")

    def _learn_neural_feasibility_classifier(self, max_inference_suffix: int) -> None:
        neural_feasibility_classifier = NeuralFeasibilityClassifier(
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
            embedding_horizon = CFG.horizon,
            batch_size = CFG.feasibility_batch_size,
        )
        neural_feasibility_classifier.fit(self._positive_feasibility_dataset, self._negative_feasibility_dataset)

        self._feasibility_classifier = neural_feasibility_classifier.classify