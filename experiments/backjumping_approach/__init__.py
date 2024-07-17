

from collections import defaultdict
from dataclasses import dataclass
import functools
import itertools
import logging
import os
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import dill as pkl

from matplotlib.dviread import Box
import numpy as np
import torch
from torch import nn
import torch.utils.data
from experiments.backjumping_approach.imitation import Imitation
from experiments.search_pruning_approach.low_level_planning import BacktrackingTree, run_backtracking_for_data_generation, run_low_level_search
from predicators import utils
from predicators.approaches.gnn_approach import GNNApproach
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.gnn.gnn_utils import graph_batch_collate
from predicators.ml_models import _get_torch_device
from predicators.nsrt_learning.sampler_learning import _LearnedSampler
from predicators.option_model import _OptionModelBase
from predicators.planning import PlanningTimeout
from predicators.settings import CFG
from predicators.structs import NSRT, _GroundNSRT, _Option, Dataset, GroundAtom, Metrics, ParameterizedOption, Predicate, State, Task, Type

def binary_search_by_result(low: int, high: int, f: Callable[[int], bool]):
    """Assuming `f' outputs True for early inputs and False for later ones, searches for the last True input in the range [low, high)
    """
    while low + 1 < high:
        test_point = (low + high)//2
        if f(test_point):
            low = test_point
        else:
            high = test_point
    return high - 1

@dataclass
class BackjumpingDataGenerationDatapoint:
    task: Task
    skeleton: List[_GroundNSRT]
    atoms_sequence: List[Set[GroundAtom]]

@dataclass
class BackjumpingDatapoint:
    next_states: List[State]
    step: int
    goal: Set[GroundAtom]

class BackjumpingGraphDataset(list, torch.utils.data.Dataset):
    def __init__(self, data: Iterable):
        super().__init__(data)

class BackjumpingFeaturizer:
    def __init__(self, types: Iterable[Type], predicates: Iterable[Predicate]) -> None:
        predicates = self._predicates = set(predicates)

        obj_types_set = set(types)
        obj_attrs_set = {feat for t in types for feat in t.feature_names}
        unary_predicates_set: Set[Predicate] = set()
        binary_predicates_set: Set[Predicate] = set()

        for predicate in predicates:
            assert predicate.arity in {1, 2}
            (unary_predicates_set if predicate.arity == 1 else binary_predicates_set).add(predicate)

        obj_types = sorted(obj_types_set)
        obj_attrs = sorted(obj_attrs_set)
        unary_predicates = sorted(unary_predicates_set)
        binary_predicates = sorted(binary_predicates_set)

        # Initialize input node features.
        self._node_feature_to_index: Dict[Any, int] = {
            feat: idx for idx, feat in enumerate(itertools.chain(
                obj_types, unary_predicates, map(self.G, unary_predicates), obj_attrs
            ))
        }

        # Initialize input edge features.
        self._edge_feature_to_index: Dict[Any, int] = {
            feat: idx for idx, feat in enumerate(itertools.chain(
                binary_predicates,
                map(self.R, binary_predicates),
                map(self.G, binary_predicates),
                map(lambda p: self.G(self.R(p)), binary_predicates)
            ))
        }

    @property
    def node_dim(self) -> int:
        return len(self._node_feature_to_index)

    @property
    def edge_dim(self) -> int:
        return len(self._edge_feature_to_index)

    def featurize_datapoint(self, datapoint: BackjumpingDatapoint) -> Tuple[Dict, int]:
        return graph_batch_collate(
            [self._graphify_single_state(state, datapoint.goal) for state in datapoint.next_states],
            device=_get_torch_device(CFG.use_torch_gpu)
        ), datapoint.step

    def _graphify_single_state(self, state: State, goal: Set[GroundAtom]) -> Dict:
        atoms = utils.abstract(state, self._predicates)
        all_objects = list(state)
        object_to_node = {v: k for k, v in enumerate(all_objects)}
        num_objects = len(all_objects)
        num_node_features = len(self._node_feature_to_index)
        num_edge_features = len(self._edge_feature_to_index)

        # Input globals are empty since we don't support nullary predicates.
        graph = {"globals": np.zeros((0,))}

        # Add nodes (one per object) and node features.
        graph["n_node"] = np.array(num_objects)
        graph["nodes"] = node_features = np.zeros((num_objects, num_node_features))

        ## Add node features for obj types.
        for obj in state:
            node_index = object_to_node[obj]
            type_index = self._node_feature_to_index[obj.type]
            node_features[node_index, type_index] = 1

        ## Add node features for unary atoms.
        for atom in atoms:
            if atom.predicate.arity != 1:
                continue
            node_index = object_to_node[atom.objects[0]]
            atom_index = self._node_feature_to_index[atom.predicate]
            node_features[node_index, atom_index] = 1

        ## Add node features for unary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 1:
                continue
            node_index = object_to_node[atom.objects[0]]
            atom_index = self._node_feature_to_index[self.G(atom.predicate)]
            node_features[node_index, atom_index] = 1

        ## Add node features for state.
        for obj in state:
            node_index = object_to_node[obj]
            for feat, val in zip(obj.type.feature_names, state[obj]):
                feat_index = self._node_feature_to_index[feat]
                node_features[node_index, feat_index] = val

        # Deal with edge case (pun).
        assert num_edge_features >= 1

        # Add edges (one between each pair of objects) and edge features.
        all_edge_features = np.zeros((num_objects, num_objects, num_edge_features))

        ## Add edge features for binary atoms.
        for atom in atoms:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[atom.predicate]
            node0_index = object_to_node[atom.objects[0]]
            node1_index = object_to_node[atom.objects[1]]
            all_edge_features[node0_index, node1_index, pred_index] = 1

        ## Add edge features for reversed binary atoms.
        for atom in atoms:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[self.R(atom.predicate)]
            node0_index = object_to_node[atom.objects[0]]
            node1_index = object_to_node[atom.objects[1]]
            # Note: the next line is reversed on purpose!
            all_edge_features[node1_index, node0_index, pred_index] = 1

        ## Add edge features for binary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[self.G(atom.predicate)]
            node0_index = object_to_node[atom.objects[0]]
            node1_index = object_to_node[atom.objects[1]]
            all_edge_features[node0_index, node1_index, pred_index] = 1

        ## Add edge features for reversed binary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[self.G(self.R(atom.predicate))]
            node0_index = object_to_node[atom.objects[0]]
            node1_index = object_to_node[atom.objects[1]]
            # Note: the next line is reversed on purpose!
            all_edge_features[node1_index, node0_index, pred_index] = 1

        # Organize into expected representation.
        adjacency_mat = np.any(all_edge_features, axis=2)
        receivers, senders, edges = [], [], []
        for sender, receiver in np.argwhere(adjacency_mat):
            edge = all_edge_features[sender, receiver]
            senders.append(sender)
            receivers.append(receiver)
            edges.append(edge)

        n_edge = len(edges)
        graph["edges"] = np.reshape(edges, [n_edge, num_edge_features])
        graph["receivers"] = np.reshape(receivers, [n_edge]).astype(np.int64)
        graph["senders"] = np.reshape(senders, [n_edge]).astype(np.int64)
        graph["n_edge"] = np.reshape(n_edge, [1]).astype(np.int64)

        return graph

    @classmethod
    def G(cls, predicate: Predicate) -> Predicate:
        return Predicate(
            'GOAL-' + predicate.name,
            predicate.types,
            predicate._classifier
        )

    @classmethod
    def R(cls, predicate: Predicate) -> Predicate:
        assert predicate.arity == 2
        return Predicate(
            'REV-' + predicate.name,
            predicate.types,
            lambda s, os: predicate._classifier(s, list(reversed(os)))
        )

class BackjumpingApproach(NSRTLearningApproach):
    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types, action_space, train_tasks)
        self._backjumping_featurizer = BackjumpingFeaturizer(self._types, self._initial_predicates)

        self._dataset: Optional[Dataset] = None

        self._imitation = Imitation(
            gnn_node_dim = self._backjumping_featurizer.node_dim,
            gnn_edge_dim = self._backjumping_featurizer.edge_dim,
            gnn_global_dim = 0,
            gnn_node_layers = CFG.backjumping_gnn_layers,
            gnn_edge_layers = CFG.backjumping_gnn_layers,
            gnn_global_layers = CFG.backjumping_gnn_layers,
            gnn_num_steps = CFG.backjumping_gnn_num_steps,
            attention_size = CFG.backjumping_attention_size,
            attention_num_heads = CFG.backjumping_attention_num_heads,
            attention_num_blocks = CFG.backjumping_attention_num_blocks,
            attention_fc_sizes = CFG.backjumping_attention_fc_sizes,
            predictor_sizes = CFG.backjumping_predictor_sizes,
            lr = CFG.backjumping_lr,
            use_torch_gpu = CFG.use_torch_gpu,
        )

    @classmethod
    def get_name(cls) -> str:
        return "backjumping"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        assert self._dataset is None
        self._dataset = dataset

        # Setting up the logger properly
        logging.getLogger().setLevel(logging.DEBUG)

        # Generate the base NSRTs
        start = time.perf_counter()
        super().learn_from_offline_dataset(dataset)
        logging.info(f"TRAINING NSRTS TOOK {time.perf_counter() - start}")

        # Make sure the trajectories are entirely covered by the learned NSRTs (we can easily generate skeletons)
        assert all(
            segment in self._seg_to_ground_nsrt
            for segmented_traj in self._segmented_trajs
            for segment in segmented_traj
        )

        # Gather the data for imitation training
        save_path = utils.get_approach_load_path_str()
        dataset_save_path = os.path.join(save_path, "backjumping-dataset.pkl")
        if CFG.backjumping_load_dataset:
            backjumping_data = pkl.load(open(dataset_save_path, 'rb'))
        else:
            backjumping_data = self.generate_backjumping_datapoints(self.get_data_generation_datapoints())
            logging.info(f"Finished gathering backjumping data - {sum(len(dps) for dps in backjumping_data)} datapoints")
            for dps in backjumping_data:
                for dp in dps:
                    logging.info(f"{dp.step};{len(dp.next_states)}")
            pkl.dump(backjumping_data, open(dataset_save_path, 'wb'))

        # Train Imitation
        num_train_backjumping_tasks = len(backjumping_data) - int(len(backjumping_data)*(1-CFG.backjumping_validation_fraction))
        train_backjumping_graph_data = [self._backjumping_featurizer.featurize_datapoint(dp) for dps in backjumping_data[:num_train_backjumping_tasks] for dp in dps]
        validation_backjumping_graph_data = [self._backjumping_featurizer.featurize_datapoint(dp) for dps in backjumping_data[num_train_backjumping_tasks:] for dp in dps]

        train_dataset = BackjumpingGraphDataset(train_backjumping_graph_data)
        validation_dataset = BackjumpingGraphDataset(validation_backjumping_graph_data)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.backjumping_batch_size, collate_fn=self._datapoint_collate_fn, shuffle=True)
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=CFG.backjumping_batch_size, collate_fn=self._datapoint_collate_fn)

        best_model_savefile = os.path.join(save_path, "backjumping-model.pt")
        best_model_loss = np.inf
        self._imitation.save(best_model_savefile)

        for itr, (state_graph_trajs, backjumping_indices) in zip(
            reversed(range(CFG.backjumping_num_iters, -1, -1)),
            itertools.chain.from_iterable(itertools.repeat(train_dataloader))
        ):
            self._imitation.update(state_graph_trajs, backjumping_indices)
            if itr % 200 == 0 and len(validation_dataset) > 0:
                validation_loss = 0
                for validation_state_graph_trajs, validation_backjumping_indices in validation_dataloader:
                    validation_loss += self._imitation.valid(validation_state_graph_trajs, validation_backjumping_indices)["loss"].item() * len(validation_state_graph_trajs)
                validation_loss /= len(validation_dataset)
                print(f"Iter {itr}/{CFG.backjumping_num_iters}, Validation Loss: {validation_loss}")

                if validation_loss < best_model_loss:
                    self._imitation.save(best_model_savefile)

        self._imitation.load(best_model_savefile)

    def get_data_generation_datapoints(self) -> List[BackjumpingDataGenerationDatapoint]:
        assert self._dataset is not None
        assert self._segmented_trajs
        assert self._seg_to_ground_nsrt
        assert len(self._segmented_trajs) == len(self._dataset.trajectories)

        data_generation_datapoints = []
        for segmented_traj, ll_traj in zip(self._segmented_trajs, self._dataset.trajectories):
            assert ll_traj._train_task_idx is not None
            data_generation_datapoints.append(BackjumpingDataGenerationDatapoint(
                task = self._train_tasks[ll_traj._train_task_idx],
                skeleton = [self._seg_to_ground_nsrt[segment] for segment in segmented_traj],
                atoms_sequence = [segment.init_atoms for segment in segmented_traj] + [segmented_traj[-1].final_atoms],
            ))

        self._rng.shuffle(data_generation_datapoints)
        return data_generation_datapoints

    @classmethod
    def _get_necessary_cfg_namespace(cls) -> SimpleNamespace:
        return SimpleNamespace(
            sesame_max_samples_per_step=CFG.sesame_max_samples_per_step,
            sesame_propagate_failures=CFG.sesame_propagate_failures,
            sesame_check_expected_atoms=True,
            sesame_check_static_object_changes=CFG.sesame_check_static_object_changes,
            sesame_static_object_change_tol=CFG.sesame_static_object_change_tol,
            sampler_disable_classifier=CFG.sampler_disable_classifier,
            max_num_steps_option_rollout=CFG.max_num_steps_option_rollout,
            option_model_terminate_on_repeat=CFG.option_model_terminate_on_repeat,
            pybullet_control_mode=CFG.pybullet_control_mode,
            pybullet_max_vel_norm=CFG.pybullet_max_vel_norm,
            horizon=CFG.horizon,
            timeout=CFG.timeout
        )

    def generate_backjumping_datapoints(self, data_generation_datapoints: List[BackjumpingDataGenerationDatapoint]) -> List[List[BackjumpingDatapoint]]:
        debug_dir = os.path.join(CFG.feasibility_debug_directory, 'data-gathering')
        os.makedirs(debug_dir, exist_ok=True)
        torch.multiprocessing.set_start_method('forkserver')
        for nsrt in self._nsrts:
            if isinstance(nsrt.sampler, _LearnedSampler):
                nsrt.sampler.to('cpu').share_memory()
        with torch.multiprocessing.Pool(CFG.feasibility_num_data_collection_threads) as pool:
            return [
                dps for dps in pool.starmap(BackjumpingApproach._gather_single_backjumping_datapoint, zip(
                    data_generation_datapoints,
                    np.arange(len(data_generation_datapoints)) + CFG.seed,
                    itertools.repeat(self._option_model),
                    itertools.repeat(self._get_necessary_cfg_namespace()),
                    itertools.repeat(debug_dir),
                    itertools.repeat(time.time() + CFG.backjumping_data_gathering_timeout),
                )) if dps
            ]

    @classmethod
    def _gather_single_backjumping_datapoint(
        cls,
        data_generation_datapoint: BackjumpingDataGenerationDatapoint,
        seed: int,
        option_model: _OptionModelBase,
        cfg: SimpleNamespace,
        debug_dir: str,
        stop_time: float,
    ) -> List[BackjumpingDatapoint]:
        if time.time() >= stop_time:
            return []

        global CFG
        CFG.__dict__.update(cfg.__dict__)

        logging.basicConfig(filename=os.path.join(debug_dir, f"{seed}.log"), force=True, level=logging.DEBUG)

        def search_stop_condition(states: List[State], tree: BacktrackingTree) -> int:
            current_depth = len(states) - 1
            if tree.is_successful:
                return -1
            return current_depth - 1 if tree.num_tries >= CFG.sesame_max_samples_per_step else current_depth

        try:
            backtracking, _ = run_backtracking_for_data_generation(
                previous_states = [data_generation_datapoint.task.init],
                goal = data_generation_datapoint.task.goal,
                option_model = option_model,
                skeleton = data_generation_datapoint.skeleton,
                feasibility_classifier = None,
                atoms_sequence = data_generation_datapoint.atoms_sequence,
                search_stop_condition = search_stop_condition,
                seed = seed,
                timeout = stop_time - time.time(),
                metrics = defaultdict(lambda: 0),
                max_horizon = CFG.horizon,
            )
        except Exception as e:
            logging.error(e)
            raise e

        return cls._generate_data_from_tree(backtracking, data_generation_datapoint.task.goal)

    @classmethod
    def _generate_data_from_tree(cls, root: BacktrackingTree, goal: Set[GroundAtom]) -> List[BackjumpingDatapoint]:
        datapoint_candidates = defaultdict(lambda: [])
        step: int = 0
        datapoints: List[BackjumpingDatapoint] = []
        def traverse_tree(tree: BacktrackingTree, states: List[State], steps: List[int]) -> None:
            nonlocal step, datapoints
            step += 1
            states.append(tree.state)
            steps.append(step)

            depth = len(states) - 1

            for _, mb_subtree in tree.tries:
                if mb_subtree is None:
                    datapoint_candidates[depth].append((states.copy(), steps.copy()))
                else:
                    for dp_states, dp_steps in datapoint_candidates[depth]:
                        assert len(steps) == len(dp_steps)
                        backjumping_depth = binary_search_by_result(0, len(steps), lambda i: steps[i] == dp_steps[i])
                        if backjumping_depth == depth:
                            continue
                        datapoints.append(BackjumpingDatapoint(
                            dp_states[1:],
                            backjumping_depth,
                            goal,
                        ))
                    datapoint_candidates[depth] = []
                    traverse_tree(mb_subtree, states, steps)
            states.pop()
            steps.pop()
        traverse_tree(root, [], [])
        return datapoints

    def _run_sesame_plan(
        self,
        task: Task,
        nsrts: Set[NSRT],
        preds: Set[Predicate],
        timeout: float,
        seed: int
    ) -> Tuple[List[_Option], List[_GroundNSRT], Metrics]:
        raise NotImplementedError("TODO")

    @classmethod
    def _datapoint_collate_fn(cls, batch: Iterable[Tuple[Dict, int]]) -> Tuple[List[Dict], torch.Tensor]:
        state_graph_trajs, backtracking_indices = map(list, zip(*batch))
        return state_graph_trajs, torch.tensor(backtracking_indices)

    def load(self, online_learning_cycle: Optional[int]) -> None:
        assert online_learning_cycle is None
        super().load(online_learning_cycle)
        self._imitation.load(os.path.join(utils.get_approach_load_path_str(), "backjumping-model.pt"))