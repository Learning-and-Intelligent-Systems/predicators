from collections import defaultdict
import dataclasses
import pickle
import time
from typing import List, Sequence, Set, Tuple
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
from predicators.structs import NSRT, _GroundNSRT, _Option, Dataset, Metrics, ParameterizedOption, Predicate, Segment, State, Task, Type, Variable
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("tkagg")

__all__ = ["SearchPruningApproach"]

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
        dataset_path: str = utils.create_dataset_filename_str(["feasibility_dataset"])

        # Running data collection and training
        assert CFG.feasibility_learning_strategy in {"backtracking", "ground_truth_data", "ground_truth_classifier", "load_data"}
        if CFG.feasibility_learning_strategy == "load_data":
            self._positive_feasibility_dataset, self._negative_feasibility_dataset = pickle.load(open(dataset_path, 'rb'))
        elif CFG.feasibility_learning_strategy == "ground_truth_classifier":
            self._set_ground_truth_classifier()
        elif CFG.feasibility_learning_strategy == "ground_truth_data":
            self._collect_data_ground_truth()
        else: # Main strategy; the other ones are for testing/debugging
            seed = self._seed + 696969
            self._collect_data_interleaved_backtracking(seed, {})

        # # Saving the generated dataset (if applicable)
        # if CFG.feasibility_learning_strategy not in {"load_data", "ground_truth_classifier"}:
        #     pickle.dump((self._positive_feasibility_dataset, self._negative_feasibility_dataset), open(dataset_path, "wb"))

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

        self._learn_neural_feasibility_classifier(CFG.horizon)
        print(len(self._positive_feasibility_dataset), len(self._negative_feasibility_dataset))
        eval = [
            self._feasibility_classifier(
                datapoint.previous_states + [datapoint.next_state],
                datapoint.encoder_nsrts + datapoint.decoder_nsrts,
            ) for datapoint in self._positive_feasibility_dataset[:100]
        ] + [
            not self._feasibility_classifier(
                datapoint.previous_states + [datapoint.next_state],
                datapoint.encoder_nsrts + datapoint.decoder_nsrts,
            ) for datapoint in self._negative_feasibility_dataset[:100]
        ]
        print(f"Feasibility Classifier Accuracy: {sum(eval) / len(eval)}")

    def _collect_data_interleaved_backtracking(self, seed: int, metrics: Metrics) -> None: # TODO: reimplement datapoint generation
        # TODO: add metric collection (how many samples needed for backtracking)
        init_states_data: List[List[State]] = [[
            segment.states[0] for segment in segmented_traj
        ] for segmented_traj in self._segmented_trajs]
        skeletons: List[List[_GroundNSRT]] = [[
            self._seg_to_ground_nsrt[segment] for segment in segmented_traj
        ] for segmented_traj in self._segmented_trajs]
        for suffix_length in range(1, max(map(len, self._segmented_trajs))): # length of the suffix of states
            for init_states, skeleton, segmented_traj in zip(init_states_data, skeletons, self._segmented_trajs):
                if suffix_length >= len(segmented_traj):
                    continue

                # Including positive example
                encoder_nsrts: List[_GroundNSRT] = skeleton[:-suffix_length]
                decoder_nsrts: List[_GroundNSRT] = skeleton[-suffix_length:]
                prev_states: List[State] = init_states[:-suffix_length]
                next_good_state: State = segmented_traj[-suffix_length].states[0]
                self._positive_feasibility_dataset.append(
                    FeasibilityDatapoint(
                        prev_states,
                        next_good_state,
                        encoder_nsrts,
                        decoder_nsrts
                    )
                )

                # Calculating negative examples
                seed += 1
                backtracking, success = run_backtracking_with_previous_states(
                    previous_states = prev_states,
                    goal = segmented_traj[-1].final_atoms,
                    option_model = self._option_model,
                    skeleton = skeleton,
                    feasibility_classifier = self._feasibility_classifier, # TODO: try it without the feasibility classifier
                    min_classifier_depth = len(prev_states),
                    atoms_sequence = [segment.final_atoms for segment in segmented_traj],
                    seed = seed,
                    timeout = float('inf'),
                    metrics = metrics,
                    max_horizon = CFG.horizon - sum(len(segment.actions) for segment in segmented_traj[:-suffix_length - 1]),
                )
                if not success:
                    continue
                next_states = [
                    mb_subtree.state
                    for _, mb_subtree in backtracking.failed_tries if mb_subtree is not None
                ]
                self._negative_feasibility_dataset.expand(
                    FeasibilityDatapoint(
                        prev_states,
                        next_state,
                        encoder_nsrts,
                        decoder_nsrts
                    ) for next_state in next_states
                )
            self._learn_neural_feasibility_classifier(suffix_length) # TODO: change the suffix length for the next suffix length

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
            lr = CFG.learning_rate,
            max_inference_suffix = max_inference_suffix,
        )
        neural_feasibility_classifier.fit(self._positive_feasibility_dataset, self._negative_feasibility_dataset)

        self._feasibility_classifier = neural_feasibility_classifier.classify