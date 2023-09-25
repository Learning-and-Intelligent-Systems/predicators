"""A learning-based refinement cost estimator that trains a GNN regression
model mapping initial state, intermediate atoms, goal, and operator to cost,
and estimates refinement cost for a full skeleton by summing the model output
over individual actions in the skeleton."""

import functools
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import dill as pkl
import numpy as np
import torch
from torch.utils.data import DataLoader

from predicators import utils
from predicators.gnn.gnn import EncodeProcessDecode, setup_graph_net
from predicators.gnn.gnn_utils import GraphDictDataset, compute_normalizers, \
    get_graph_batch_collate_with_device, get_single_model_prediction, \
    normalize_graph, train_model
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.refinement_estimators import BaseRefinementEstimator
from predicators.settings import CFG
from predicators.structs import NSRT, GroundAtom, NDArray, Predicate, \
    RefinementDatapoint, State, Task, _GroundNSRT


class GNNRefinementEstimator(BaseRefinementEstimator):
    """A refinement cost estimator that uses a GNN to predict refinement cost
    from an initial state, intermediate atoms, goal, and abstract action."""

    def __init__(self) -> None:
        super().__init__()
        self._gnn: Optional[EncodeProcessDecode] = None
        self._data_exemplar: Tuple[Dict, Dict] = ({}, {})
        self._nsrts: List[NSRT] = []
        self._max_nsrt_objects = 0
        self._node_feature_to_index: Dict[Any, int] = {}
        self._edge_feature_to_index: Dict[Any, int] = {}
        self._nullary_predicates: List[Predicate] = []
        self._input_normalizers: Dict = {}
        self._target_normalizers: Dict = {}
        self._mse_loss = torch.nn.MSELoss()
        self._device = torch.device("cuda:0" if CFG.use_torch_gpu
                                    and torch.cuda.is_available() else "cpu")
        self._setup_fields()

    @classmethod
    def get_name(cls) -> str:
        return "gnn"

    @property
    def is_learning_based(self) -> bool:
        return True

    def get_cost(self, initial_task: Task, skeleton: List[_GroundNSRT],
                 atoms_sequence: List[Set[GroundAtom]]) -> float:
        assert self._gnn is not None, "Need to train"
        cost = 0
        state, goal = initial_task.init, initial_task.goal
        # Run each step of the skeleton through the GNN model to estimate cost
        for i, action in enumerate(skeleton):
            atoms = atoms_sequence[i]
            in_graph = self._graphify_single_input(state, atoms, goal, action)
            if CFG.gnn_do_normalization:
                in_graph = normalize_graph(in_graph, self._input_normalizers)
            out_graph = get_single_model_prediction(self._gnn,
                                                    in_graph,
                                                    device=self._device)
            if CFG.gnn_do_normalization:
                out_graph = normalize_graph(out_graph,
                                            self._target_normalizers,
                                            invert=True)
            refinement_time, low_level_count = out_graph["globals"]
            cost += refinement_time
            if CFG.refinement_data_include_execution_cost:
                cost += (low_level_count *
                         CFG.refinement_data_low_level_execution_cost)
        return cost

    def train(self, data: List[RefinementDatapoint]) -> None:
        """Split up each RefinementDatapoint into distinct training data points
        for the per-action GNN, and train the GNN regressor."""
        graph_inputs = []
        graph_targets = []
        for (task, skeleton, atoms_sequence, succeeded, refinement_time,
             low_level_count) in data:
            state, goal = task.init, task.goal
            for i, action in enumerate(skeleton):
                atoms = atoms_sequence[i]
                target_time = refinement_time[i]
                # Add failed penalty to the value if failure occurred
                if not succeeded:
                    target_time += CFG.refinement_data_failed_refinement_penalty
                # Convert input and target to graphs
                graph_inputs.append(
                    self._graphify_single_input(state, atoms, goal, action))
                graph_targets.append(
                    self._graphify_single_target(
                        target_time, low_level_count[i] if succeeded else 0))
        assert len(graph_inputs) and len(graph_targets), "No usable data"
        self._data_exemplar = (graph_inputs[0], graph_targets[0])

        # Normalize if needed
        if CFG.gnn_do_normalization:
            # Update normalization constants. Note that we do this for both
            # the input graph and the target graph.
            self._input_normalizers = compute_normalizers(graph_inputs)
            self._target_normalizers = compute_normalizers(
                graph_targets,
                normalize_nodes=False,
                normalize_edges=False,
            )
            graph_inputs = [
                normalize_graph(g, self._input_normalizers)
                for g in graph_inputs
            ]
            graph_targets = [
                normalize_graph(g, self._target_normalizers)
                for g in graph_targets
            ]
        # Run training.
        if CFG.gnn_use_validation_set:
            ## Split data, using 10% for validation.
            num_validation = max(1, int(len(graph_inputs) * 0.1))
        else:
            num_validation = 0
        shuffled_indices = self._rng.permutation(len(graph_inputs))
        graph_inputs = [graph_inputs[i] for i in shuffled_indices]
        graph_targets = [graph_targets[i] for i in shuffled_indices]
        train_inputs = graph_inputs[num_validation:]
        train_targets = graph_targets[num_validation:]
        val_inputs = graph_inputs[:num_validation]
        val_targets = graph_targets[:num_validation]
        train_dataset = GraphDictDataset(train_inputs, train_targets)
        val_dataset = GraphDictDataset(val_inputs, val_targets)
        # Set up model
        self._gnn = setup_graph_net(train_dataset,
                                    num_steps=CFG.gnn_num_message_passing,
                                    layer_size=CFG.gnn_layer_size).to(
                                        self._device)
        # Set up Adam optimizer and dataloaders.
        optimizer = torch.optim.Adam(self._gnn.parameters(),
                                     lr=CFG.gnn_learning_rate,
                                     weight_decay=CFG.gnn_weight_decay)
        graph_batch_collate = get_graph_batch_collate_with_device(self._device)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=CFG.gnn_batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=graph_batch_collate)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=CFG.gnn_batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=graph_batch_collate)
        dataloaders = {"train": train_dataloader, "val": val_dataloader}
        ## Launch training code.
        logging.info(f"Training GNN on {len(train_inputs)} examples")
        best_model_dict = train_model(self._gnn,
                                      dataloaders,
                                      optimizer=optimizer,
                                      criterion=None,
                                      global_criterion=self._global_criterion,
                                      num_epochs=CFG.gnn_num_epochs,
                                      do_validation=CFG.gnn_use_validation_set,
                                      device=self._device)
        self._gnn.load_state_dict(best_model_dict)

    def _global_criterion(self, output: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        """Global criterion function for training GNN."""
        return self._mse_loss(output, target)

    def _graphify_single_input(self, state: State, atoms: Set[GroundAtom],
                               goal: Set[GroundAtom],
                               action: _GroundNSRT) -> Dict:
        """Convert (initial state, atoms, goal, action) to graph."""
        all_objects = list(state)
        object_to_node = {obj: i for i, obj in enumerate(all_objects)}
        num_objects = len(all_objects)
        num_node_features = len(self._node_feature_to_index)
        num_edge_features = max(len(self._edge_feature_to_index), 1)

        G = functools.partial(utils.wrap_predicate, prefix="GOAL-")
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        # Add 1 node per object and create node features array
        graph: Dict[str, NDArray[np.float64]] = {
            "n_node": np.reshape(num_objects, [1]).astype(np.int64)
        }
        node_features = np.zeros((num_objects, num_node_features))
        # Handle each object's state features
        for obj in state:
            obj_index = object_to_node[obj]
            for feat, val in zip(obj.type.feature_names, state[obj]):
                feat_index = self._node_feature_to_index[f"feat_{feat}"]
                node_features[obj_index, feat_index] = val

        # Initialize feature vectors for nullary/binary predicates
        edge_features_dict: DefaultDict[
            Tuple[int, int],
            np.ndarray] = defaultdict(lambda: np.zeros(num_edge_features))
        atoms_globals = np.zeros(len(self._nullary_predicates), dtype=np.int64)
        goal_globals = np.zeros(len(self._nullary_predicates), dtype=np.int64)

        # Handle atoms
        for atom in atoms:
            arity = atom.predicate.arity
            if arity == 0:
                atoms_globals[self._nullary_predicates.index(
                    atom.predicate)] = 1
                continue
            obj0_index = object_to_node[atom.objects[0]]
            if arity == 1:
                atom_index = self._node_feature_to_index[atom.predicate]
                node_features[obj0_index, atom_index] = 1
            elif arity == 2:
                obj1_index = object_to_node[atom.objects[1]]
                atom_index = self._edge_feature_to_index[atom.predicate]
                edge_features_dict[(obj0_index, obj1_index)][atom_index] = 1
                rev_index = self._edge_feature_to_index[R(atom.predicate)]
                edge_features_dict[(obj1_index, obj0_index)][rev_index] = 1

        # Handle goal atoms
        for atom in goal:
            arity = atom.predicate.arity
            if arity == 0:
                goal_globals[self._nullary_predicates.index(
                    atom.predicate)] = 1
                continue
            obj0_index = object_to_node[atom.objects[0]]
            if arity == 1:
                atom_index = self._node_feature_to_index[G(atom.predicate)]
                node_features[obj0_index, atom_index] = 1
            elif arity == 2:
                obj1_index = object_to_node[atom.objects[1]]
                atom_index = self._edge_feature_to_index[G(atom.predicate)]
                edge_features_dict[(obj0_index, obj1_index)][atom_index] = 1
                rev_index = self._edge_feature_to_index[G(R(atom.predicate))]
                edge_features_dict[(obj1_index, obj0_index)][rev_index] = 1

        # Handle action globals
        action_globals = np.zeros(len(self._nsrts), dtype=np.int64)
        action_globals[self._nsrts.index(action.parent)] = 1
        for i, action_obj in enumerate(action.objects):
            obj_index = object_to_node[action_obj]
            feat_index = self._node_feature_to_index[f"nsrt-{i}"]
            node_features[obj_index, feat_index] = 1

        # Organize
        graph["nodes"] = node_features.astype(np.float32)
        graph["globals"] = np.r_[atoms_globals, goal_globals, action_globals]
        senders, receivers, edges = [], [], []
        for (sender, receiver), edge in edge_features_dict.items():
            senders.append(sender)
            receivers.append(receiver)
            edges.append(edge)
        n_edge = len(edges)
        graph["senders"] = np.reshape(senders, [n_edge]).astype(np.int64)
        graph["receivers"] = np.reshape(receivers, [n_edge]).astype(np.int64)
        graph["edges"] = np.reshape(edges, [n_edge, num_edge_features])
        graph["n_edge"] = np.reshape(n_edge, [1]).astype(np.int64)

        return graph

    @staticmethod
    def _graphify_single_target(refinement_time: float,
                                low_level_count: int) -> Dict:
        """Convert target cost into a graph."""
        graph = {
            "n_node": np.array([1], dtype=np.int64),
            "nodes": np.array([]),
            "n_edge": np.array([0], dtype=np.int64),
            "edges": np.array([]),
            "senders": np.array([]),
            "receivers": np.array([]),
            "globals": np.array([refinement_time, low_level_count]),
        }
        return graph

    def _setup_fields(self) -> None:
        """Assign indices to each node and edge feature, and also identify list
        of nullary predicates."""
        self._node_feature_to_index = {}
        self._edge_feature_to_index = {}
        node_feature_index = 0
        edge_feature_index = 0
        self._nullary_predicates = []

        G = functools.partial(utils.wrap_predicate, prefix="GOAL-")
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        # Identify object types
        obj_attrs_set = set()
        for obj_type in sorted(self._env.types):
            self._node_feature_to_index[
                f"type_{obj_type.name}"] = node_feature_index
            node_feature_index += 1
            # Also list object features to add to node features later
            for feat in obj_type.feature_names:
                obj_attrs_set.add(f"feat_{feat}")

        # Identify predicates
        for predicate in sorted(self._env.predicates):
            arity = predicate.arity
            assert arity <= 2, "Predicates with arity > 2 are not supported"
            if arity == 0:
                self._nullary_predicates.append(predicate)
            elif arity == 1:
                for feature in (predicate, G(predicate)):
                    self._node_feature_to_index[feature] = node_feature_index
                    node_feature_index += 1
            elif arity == 2:
                for feature in (predicate, R(predicate), G(predicate),
                                G(R(predicate))):
                    self._edge_feature_to_index[feature] = edge_feature_index
                    edge_feature_index += 1

        # Identify NSRTs
        gt_nsrts = get_gt_nsrts(CFG.env, self._env.predicates,
                                get_gt_options(self._env.get_name()))
        self._nsrts = sorted(gt_nsrts)
        max_nsrt_objects = 0
        for nsrt in self._nsrts:
            max_nsrt_objects = max(max_nsrt_objects, len(nsrt.parameters))
        self._max_nsrt_objects = max_nsrt_objects
        for i in range(max_nsrt_objects):
            self._node_feature_to_index[f"nsrt-{i}"] = node_feature_index
            node_feature_index += 1

        # Add object features
        for obj_attr in sorted(obj_attrs_set):
            self._node_feature_to_index[obj_attr] = node_feature_index
            node_feature_index += 1

    def save_model(self, filepath: Path) -> None:
        info = {
            "exemplar": self._data_exemplar,
            "state_dict": self._gnn.state_dict() if self._gnn else None,
            "input_normalizers": self._input_normalizers,
            "target_normalizers": self._target_normalizers,
        }
        with open(filepath, "wb") as f:
            pkl.dump(info, f)

    def load_model(self, filepath: Path) -> None:
        with open(filepath, "rb") as f:
            info = pkl.load(f)
        self._data_exemplar = info["exemplar"]
        ex_input, ex_target = self._data_exemplar
        example_dataset = GraphDictDataset([ex_input], [ex_target])
        self._gnn = setup_graph_net(example_dataset,
                                    num_steps=CFG.gnn_num_message_passing,
                                    layer_size=CFG.gnn_layer_size).to(
                                        self._device)
        state_dict = info["state_dict"]
        if state_dict is not None:
            self._gnn.load_state_dict(info["state_dict"])
        self._input_normalizers = info["input_normalizers"]
        self._target_normalizers = info["target_normalizers"]
        # Run GNN once to avoid the weird delay issue
        get_single_model_prediction(self._gnn, ex_input, device=self._device)
