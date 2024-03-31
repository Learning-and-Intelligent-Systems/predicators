"""An abstract approach that trains a GNN mapping states, atoms, and goals to
anything.

The input is always the same.
"""

import abc
import functools
import logging
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar

import dill as pkl
import numpy as np
import torch
import torch.nn
import torch.optim
from gym.spaces import Box
from torch.utils.data import DataLoader

from predicators import utils
from predicators.approaches import BaseApproach
from predicators.gnn.gnn import EncodeProcessDecode, setup_graph_net
from predicators.gnn.gnn_utils import GraphDictDataset, compute_normalizers, \
    get_single_model_prediction, graph_batch_collate, normalize_graph, \
    train_model
from predicators.settings import CFG
from predicators.structs import Dataset, GroundAtom, ParameterizedOption, \
    Predicate, State, Task, Type

_Output = TypeVar("_Output")  # a generic type for the output of this GNN


class GNNApproach(BaseApproach, Generic[_Output]):
    """Abstract GNNApproach definition."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Fields for the GNN.
        self._gnn: Optional[EncodeProcessDecode] = None
        self._nullary_predicates: List[Predicate] = []
        self._node_feature_to_index: Dict[Any, int] = {}
        self._edge_feature_to_index: Dict[Any, int] = {}
        self._input_normalizers: Dict = {}
        self._target_normalizers: Dict = {}
        self._data_exemplar: Tuple[Dict, Dict] = ({}, {})

    @abc.abstractmethod
    def _generate_data_from_dataset(
        self, dataset: Dataset
    ) -> List[Tuple[State, Set[GroundAtom], Set[GroundAtom], _Output]]:
        """Given a Dataset object, organize it into tuples of (state, atoms,
        goal, target).

        The inputs to the GNN are (state, atoms, goal). The target can
        be of any type.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _setup_output_specific_fields(
        self, data: List[Tuple[State, Set[GroundAtom], Set[GroundAtom],
                               _Output]]
    ) -> None:
        """Given the dataset of inputs and targets, set up any necessary
        output-specific fields."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _graphify_single_target(self, target: _Output, graph_input: Dict,
                                object_to_node: Dict) -> Dict:
        """Given a target output, return a target graph.

        We also provide the return values of graphify_single_input on
        the corresponding input, because some fields may be the same
        between the input and the target graphs, and so we can simply
        copy them over.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _criterion(self, output: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        """Define the criterion function for passing into train_model()."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _global_criterion(self, output: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        """Define the global criterion function for passing into
        train_model()."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _add_output_specific_fields_to_save_info(self, info: Dict) -> None:
        """Given a dict of info to be saved, add output-specific fields to
        it."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _load_output_specific_fields_from_save_info(self, info: Dict) -> None:
        """Given a dict of saved info, load output-specific fields from it."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _extract_output_from_graph(self, graph_output: Dict,
                                   object_to_node: Dict) -> _Output:
        """At evaluation time, given an output GNN, extract the actual
        output."""
        raise NotImplementedError("Override me!")

    @property
    def is_learning_based(self) -> bool:
        return True

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        data = self._generate_data_from_dataset(dataset)
        self._setup_fields(data)
        # Set up exemplar, which is just the first tuple in the data.
        example_input, example_object_to_node = self._graphify_single_input(
            data[0][0], data[0][1], data[0][2])
        example_target = self._graphify_single_target(data[0][3],
                                                      example_input,
                                                      example_object_to_node)
        self._data_exemplar = (example_input, example_target)
        example_dataset = GraphDictDataset([example_input], [example_target])
        self._gnn = setup_graph_net(example_dataset,
                                    num_steps=CFG.gnn_num_message_passing,
                                    layer_size=CFG.gnn_layer_size)
        # Set up all the graphs, now using *all* the data.
        inputs = [(d[0], d[1], d[2]) for d in data]
        targets = [d[3] for d in data]
        graph_inputs = []
        graph_targets = []
        for (state, atoms, goal), target in zip(inputs, targets):
            graph_input, object_to_node = self._graphify_single_input(
                state=state, atoms=atoms, goal=goal)
            graph_inputs.append(graph_input)
            graph_targets.append(
                self._graphify_single_target(target, graph_input,
                                             object_to_node))
        if CFG.gnn_do_normalization:
            # Update normalization constants. Note that we do this for both
            # the input graph and the target graph.
            self._input_normalizers = compute_normalizers(graph_inputs)
            self._target_normalizers = compute_normalizers(graph_targets)
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
            num_validation = max(1, int(len(inputs) * 0.1))
        else:
            num_validation = 0
        train_inputs = graph_inputs[num_validation:]
        train_targets = graph_targets[num_validation:]
        val_inputs = graph_inputs[:num_validation]
        val_targets = graph_targets[:num_validation]
        train_dataset = GraphDictDataset(train_inputs, train_targets)
        val_dataset = GraphDictDataset(val_inputs, val_targets)
        ## Set up Adam optimizer and dataloaders.
        optimizer = torch.optim.Adam(self._gnn.parameters(),
                                     lr=CFG.gnn_learning_rate)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=CFG.gnn_batch_size,
                                      shuffle=False,
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
                                      criterion=self._criterion,
                                      global_criterion=self._global_criterion,
                                      num_epochs=CFG.gnn_num_epochs,
                                      do_validation=CFG.gnn_use_validation_set)
        self._gnn.load_state_dict(best_model_dict)
        info = {
            "exemplar": self._data_exemplar,
            "state_dict": self._gnn.state_dict(),
            "nullary_predicates": self._nullary_predicates,
            "node_feature_to_index": self._node_feature_to_index,
            "edge_feature_to_index": self._edge_feature_to_index,
            "input_normalizers": self._input_normalizers,
            "target_normalizers": self._target_normalizers,
        }
        self._add_output_specific_fields_to_save_info(info)
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_None.gnn", "wb") as f:
            pkl.dump(info, f)

    def load(self, online_learning_cycle: Optional[int]) -> None:
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}_{online_learning_cycle}.gnn", "rb") as f:
            info = pkl.load(f)
        # Initialize fields from loaded dictionary.
        input_example, target_example = info["exemplar"]
        dataset = GraphDictDataset([input_example], [target_example])
        self._gnn = setup_graph_net(dataset,
                                    num_steps=CFG.gnn_num_message_passing,
                                    layer_size=CFG.gnn_layer_size)
        self._gnn.load_state_dict(info["state_dict"])
        self._nullary_predicates = info["nullary_predicates"]
        self._node_feature_to_index = info["node_feature_to_index"]
        self._edge_feature_to_index = info["edge_feature_to_index"]
        self._input_normalizers = info["input_normalizers"]
        self._target_normalizers = info["target_normalizers"]
        self._load_output_specific_fields_from_save_info(info)

    def _predict(self, state: State, atoms: Set[GroundAtom],
                 goal: Set[GroundAtom], memory: Dict) -> _Output:
        del memory  # unused
        # Get output graph.
        in_graph, object_to_node = self._graphify_single_input(
            state, atoms, goal)
        if CFG.gnn_do_normalization:
            in_graph = normalize_graph(in_graph, self._input_normalizers)
        out_graph = get_single_model_prediction(self._gnn, in_graph)
        if CFG.gnn_do_normalization:
            out_graph = normalize_graph(out_graph,
                                        self._target_normalizers,
                                        invert=True)
        # Extract the output from the output graph.
        return self._extract_output_from_graph(out_graph, object_to_node)

    def _setup_fields(
        self, data: List[Tuple[State, Set[GroundAtom], Set[GroundAtom],
                               _Output]]
    ) -> None:
        obj_types_set = set()
        nullary_predicates_set = set()
        unary_predicates_set = set()
        binary_predicates_set = set()
        obj_attrs_set = set()

        # Go through the data, identifying the types/predicates/attributes.
        for state, atoms, goal, _ in data:
            for atom in atoms | goal:
                arity = atom.predicate.arity
                assert arity <= 2, "Predicates with arity > 2 are not supported"
                if arity == 0:
                    nullary_predicates_set.add(atom.predicate)
                elif arity == 1:
                    unary_predicates_set.add(atom.predicate)
                elif arity == 2:
                    binary_predicates_set.add(atom.predicate)
            for obj in state:
                obj_types_set.add(f"type_{obj.type.name}")
                for feat in obj.type.feature_names:
                    obj_attrs_set.add(f"feat_{feat}")
        self._nullary_predicates = sorted(nullary_predicates_set)
        self._setup_output_specific_fields(data)

        obj_types = sorted(obj_types_set)
        unary_predicates = sorted(unary_predicates_set)
        binary_predicates = sorted(binary_predicates_set)
        obj_attrs = sorted(obj_attrs_set)

        G = functools.partial(utils.wrap_predicate, prefix="GOAL-")
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        # Initialize input node features.
        self._node_feature_to_index = {}
        index = 0
        for obj_type in obj_types:
            self._node_feature_to_index[obj_type] = index
            index += 1
        for unary_predicate in unary_predicates:
            self._node_feature_to_index[unary_predicate] = index
            index += 1
        for unary_predicate in unary_predicates:
            self._node_feature_to_index[G(unary_predicate)] = index
            index += 1
        for obj_attr in obj_attrs:
            self._node_feature_to_index[obj_attr] = index
            index += 1

        # Initialize input edge features.
        self._edge_feature_to_index = {}
        index = 0
        for binary_predicate in binary_predicates:
            self._edge_feature_to_index[binary_predicate] = index
            index += 1
        for binary_predicate in binary_predicates:
            self._edge_feature_to_index[R(binary_predicate)] = index
            index += 1
        for binary_predicate in binary_predicates:
            self._edge_feature_to_index[G(binary_predicate)] = index
            index += 1
        for binary_predicate in binary_predicates:
            self._edge_feature_to_index[G(R(binary_predicate))] = index
            index += 1

    def _graphify_single_input(self, state: State, atoms: Set[GroundAtom],
                               goal: Set[GroundAtom]) -> Tuple[Dict, Dict]:
        all_objects = list(state)
        node_to_object = dict(enumerate(all_objects))
        object_to_node = {v: k for k, v in node_to_object.items()}
        num_objects = len(all_objects)
        num_node_features = len(self._node_feature_to_index)
        num_edge_features = len(self._edge_feature_to_index)

        G = functools.partial(utils.wrap_predicate, prefix="GOAL-")
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        graph = {}

        # Input globals: nullary predicates in atoms and goal.
        atoms_globals = np.zeros(len(self._nullary_predicates), dtype=np.int64)
        for atom in atoms:
            if atom.predicate.arity != 0:
                continue
            atoms_globals[self._nullary_predicates.index(atom.predicate)] = 1
        goal_globals = np.zeros(len(self._nullary_predicates), dtype=np.int64)
        for atom in goal:
            if atom.predicate.arity != 0:
                continue
            goal_globals[self._nullary_predicates.index(atom.predicate)] = 1
        graph["globals"] = np.r_[atoms_globals, goal_globals]

        # Add nodes (one per object) and node features.
        graph["n_node"] = np.array(num_objects)
        node_features = np.zeros((num_objects, num_node_features))

        ## Add node features for obj types.
        for obj in state:
            obj_index = object_to_node[obj]
            type_index = self._node_feature_to_index[f"type_{obj.type.name}"]
            node_features[obj_index, type_index] = 1

        ## Add node features for unary atoms.
        for atom in atoms:
            if atom.predicate.arity != 1:
                continue
            obj_index = object_to_node[atom.objects[0]]
            atom_index = self._node_feature_to_index[atom.predicate]
            node_features[obj_index, atom_index] = 1

        ## Add node features for unary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 1:
                continue
            obj_index = object_to_node[atom.objects[0]]
            atom_index = self._node_feature_to_index[G(atom.predicate)]
            node_features[obj_index, atom_index] = 1

        ## Add node features for state.
        for obj in state:
            obj_index = object_to_node[obj]
            for feat, val in zip(obj.type.feature_names, state[obj]):
                feat_index = self._node_feature_to_index[f"feat_{feat}"]
                node_features[obj_index, feat_index] = val

        graph["nodes"] = node_features

        # Deal with edge case (pun).
        num_edge_features = max(num_edge_features, 1)

        # Add edges (one between each pair of objects) and edge features.
        all_edge_features = np.zeros(
            (num_objects, num_objects, num_edge_features))

        ## Add edge features for binary atoms.
        for atom in atoms:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[atom.predicate]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            all_edge_features[obj0_index, obj1_index, pred_index] = 1

        ## Add edge features for reversed binary atoms.
        for atom in atoms:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[R(atom.predicate)]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            # Note: the next line is reversed on purpose!
            all_edge_features[obj1_index, obj0_index, pred_index] = 1

        ## Add edge features for binary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[G(atom.predicate)]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            all_edge_features[obj0_index, obj1_index, pred_index] = 1

        ## Add edge features for reversed binary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[G(R(atom.predicate))]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            # Note: the next line is reversed on purpose!
            all_edge_features[obj1_index, obj0_index, pred_index] = 1

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

        return graph, object_to_node
