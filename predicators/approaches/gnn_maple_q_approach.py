"""A parameterized action reinforcement learning approach inspired by MAPLE,
(https://ut-austin-rpl.github.io/maple/) but where only a Q function is
learned. The Q-function here takes the form of a GNN so that it is able to
generalize to problems with a different number of objects.

Base samplers and applicable actions are used to perform the argmax.
"""

from __future__ import annotations

import abc
import functools
import logging
from collections import deque
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

import dill as pkl
import numpy as np
import torch
from torch.utils.data import DataLoader

from predicators import utils
from predicators.approaches.maple_q_approach import MapleQApproach
from predicators.gnn.gnn import EncodeProcessDecode, GraphDictDataset, \
    setup_graph_net
from predicators.gnn.gnn_utils import compute_normalizers, \
    get_single_model_prediction, graph_batch_collate, normalize_graph, \
    train_model
from predicators.ml_models import MapleQData, MapleQFunction
from predicators.settings import CFG
from predicators.structs import GroundAtom, Object, ParameterizedOption, \
    Predicate, State, Type, _GroundNSRT, _Option


class GNNMapleQApproach(MapleQApproach):
    """A parameterized action RL approach inspired by MAPLE."""

    def _initialize_q_function(self) -> MapleQFunction:
        # Store the Q function. Note that this implicitly
        # contains a replay buffer.
        return GNNMapleQFunction(
            seed=CFG.seed,
            predicates=self._get_current_predicates(),
            types=self._types,
            options=self._initial_options,
            num_lookahead_samples=CFG.
            active_sampler_learning_num_lookahead_samples,
            batch_size=CFG.active_sampler_learning_batch_size,
            replay_buffer_max_size=CFG.
            active_sampler_learning_replay_buffer_size,
        )

    @classmethod
    def get_name(cls) -> str:
        return "gnn_maple_q"


class GNNMapleQFunction(MapleQFunction):
    """A Maple Q function where the base model is a GNN."""

    def __init__(self,
                 seed: int,
                 predicates: Set[Predicate],
                 types: Set[Type],
                 options: Set[ParameterizedOption],
                 num_lookahead_samples: int,
                 discount: float = 0.99,
                 batch_size: int = 128,
                 replay_buffer_max_size: int = 1000000,
                 replay_buffer_sample_with_replacement: bool = True):

        # Several of these inputs are not used, set to trivial values.
        super().__init__(seed=seed,
                         hid_sizes=[],
                         max_train_iters=0,
                         clip_gradients=False,
                         clip_value=0.0,
                         learning_rate=0.0,
                         weight_decay=0.0,
                         use_torch_gpu=False,
                         train_print_every=0,
                         n_iter_no_change=0,
                         discount=discount,
                         num_lookahead_samples=num_lookahead_samples,
                         batch_size=batch_size,
                         replay_buffer_max_size=replay_buffer_max_size,
                         replay_buffer_sample_with_replacement=
                         replay_buffer_sample_with_replacement)

        # Store the predicates, types, and options for state abstraction and
        # graph construction.
        self._predicates = predicates
        self._sorted_types = sorted(types)
        self._sorted_options = sorted(options)

        # Fields for the GNN; we initially just initialize them to
        # default values.
        self._gnn: Optional[EncodeProcessDecode] = None
        self._nullary_predicates: List[Predicate] = []
        self._node_feature_to_index: Dict[Any, int] = {}
        self._edge_feature_to_index: Dict[Any, int] = {}
        self._input_normalizers: Dict = {}
        self._target_normalizers: Dict = {}
        self._data_exemplar: Tuple[Dict, Dict] = ({}, {})
        self._max_option_objects: int = 0

        # Set up the GNN fields.
        self._setup_gnn_fields()

        # Initialize the loss function (for the globals only).
        self._mse_loss = torch.nn.MSELoss()

    def __getnewargs__(self) -> Tuple:
        # TODO
        return ("TODO",)
    
    def __getstate__(self) -> Dict:
        return {}

    def set_grounding(self, init_states: Collection[State],
                      objects: Set[Object], goals: Collection[Set[GroundAtom]],
                      ground_nsrts: Collection[_GroundNSRT]) -> None:
        super().set_grounding(init_states, objects, goals, ground_nsrts)

        init_states_list = list(init_states)
        goals_list = list(goals)
        ground_nsrts_list = list(ground_nsrts)
        eg_init_state = init_states_list[0]
        eg_goal = goals_list[0]
        # Use the remaining inputs to initialize the GNN model.
        eg_atoms = utils.abstract(eg_init_state, self._predicates)
        # Set up exemplar, which is just the first tuple in the data.
        example_input_graph, _ = self._graphify_single_input(
            eg_init_state, eg_atoms, eg_goal,
            ground_nsrts_list[0].sample_option(eg_init_state, eg_goal,
                                               self._rng))
        # We can't estimate a q-value given inputs yet, since we don't have
        # the GNN instantiated. Thus, we will simply pick an arbitrary value.
        q_val = 0.0
        example_target = self._graphify_single_target(q_val,
                                                      example_input_graph)
        self._data_exemplar = (example_input_graph, example_target)
        example_dataset = GraphDictDataset([example_input_graph],
                                           [example_target])
        self._gnn = setup_graph_net(example_dataset,
                                    num_steps=CFG.gnn_num_message_passing,
                                    layer_size=CFG.gnn_layer_size)

    def _graphify_single_target(self, target_q_val: float,
                                graph_input: Dict) -> Dict:
        """Given a target output, return a target graph.

        We also provide the return values of graphify_single_input on
        the corresponding input, because some fields may be the same
        between the input and the target graphs, and so we can simply
        copy them over.
        """
        # First, copy over all unchanged fields.
        graph_target = {
            "n_node": graph_input["n_node"],
            "n_edge": graph_input["n_edge"],
            "edges": graph_input["edges"],
            "nodes": graph_input["nodes"],
            "senders": graph_input["senders"],
            "receivers": graph_input["receivers"],
        }
        graph_target["globals"] = np.array([target_q_val])
        return graph_target

    def _criterion(self, output: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        """Define the criterion function for passing into train_model()."""
        # Unused, since input and output nodes and edges are the same.
        del output, target
        return torch.tensor(0.0)

    def _global_criterion(self, output: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        """Define the global criterion function for passing into
        train_model()."""
        return self._mse_loss(output, target)

    def _setup_gnn_fields(self) -> None:
        """Use the initial predicates and options to setup the fields necessary
        to instantiate the GNN Q-network."""
        nullary_predicates_set = set(p for p in self._predicates
                                     if p.arity == 0)
        unary_predicates_set = set(p for p in self._predicates if p.arity == 1)
        binary_predicates_set = set(p for p in self._predicates
                                    if p.arity == 2)
        obj_attrs_set = set(f"feat_{f}" for t in self._sorted_types
                            for f in t.feature_names)
        self._nullary_predicates = sorted(nullary_predicates_set)
        self._max_option_objects = max(
            len(o.types) for o in self._sorted_options)

        # Seemingly, this below line is just something that sets up
        # class-specific variables that might be important for setting the output
        # I doubt we'll need any of this, since we're just outputting a single float.
        # self._setup_output_specific_fields(data)

        obj_types = [f"type_{t.name}" for t in self._sorted_types]
        unary_predicates = sorted(unary_predicates_set)
        binary_predicates = sorted(binary_predicates_set)
        obj_attrs = sorted(obj_attrs_set)
        option_index_features = [
            f"option_index_{i}" for i in range(self._max_option_objects)
        ]

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
        for option_idx_feature in option_index_features:
            self._node_feature_to_index[option_idx_feature] = index
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
                               goal: Set[GroundAtom],
                               option: _Option) -> Tuple[Dict, Dict]:
        all_objects = list(state)
        node_to_object = dict(enumerate(all_objects))
        object_to_node = {v: k for k, v in node_to_object.items()}
        num_objects = len(all_objects)
        num_node_features = len(self._node_feature_to_index)
        num_edge_features = len(self._edge_feature_to_index)

        G = functools.partial(utils.wrap_predicate, prefix="GOAL-")
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        graph = {}

        # Input globals: nullary predicates in atoms and goal, option index,
        # and option continuous params
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
        option_globals = np.zeros(len(self._sorted_options), dtype=np.int64)
        option_globals[self._sorted_options.index(option.parent)] = 1
        continuous_params_globals = np.zeros(self._max_num_params,
                                             dtype=np.float32)
        continuous_params_globals[:option.params.shape[0]] = option.params

        graph["globals"] = np.r_[atoms_globals, goal_globals, option_globals,
                                 continuous_params_globals]

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

        ## Finally, create node features for the objects involved in the
        ## currently-taken option.
        for i, obj in enumerate(option.objects):
            obj_index = object_to_node[obj]
            feat_index = self._node_feature_to_index[ f"option_index_{i}"]
            node_features[obj_index, feat_index] = 1

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

    def predict(self, in_graph: Any) -> float:
        if CFG.gnn_do_normalization:
            in_graph = normalize_graph(in_graph, self._input_normalizers)
        out_graph = get_single_model_prediction(self._gnn, in_graph)
        if CFG.gnn_do_normalization:
            out_graph = normalize_graph(out_graph,
                                        self._target_normalizers,
                                        invert=True)
        return out_graph

    def _get_q_function_input(self, state: State, goal: Set[GroundAtom],
                              option: _Option) -> Any:
        atoms = utils.abstract(state, self._predicates)
        in_graph, _  = self._graphify_single_input(state, atoms, goal, option)
        return in_graph

    def _get_q_value_from_prediction(self, prediction: Any) -> float:
        return prediction["globals"][0]

    def _get_prediction_from_q_value(self, value: float, q_function_input: Any) -> Any:
        # Copy the graph and substitute the globals.
        out_graph = {k: v.copy() for k, v in q_function_input.items()}
        out_graph["globals"] = np.array([value], dtype=np.float32)
        return out_graph

    def _fit_from_input_output_lists(self, inputs: List[Any],
                                     outputs: List[Any]) -> None:
        if CFG.gnn_do_normalization:
            # Update normalization constants. Note that we do this for both
            # the input graph and the target graph.
            self._input_normalizers = compute_normalizers(inputs)
            self._target_normalizers = compute_normalizers(outputs)
            inputs = [
                normalize_graph(g, self._input_normalizers) for g in inputs
            ]
            outputs = [
                normalize_graph(g, self._target_normalizers) for g in outputs
            ]
        # Run training.
        if CFG.gnn_use_validation_set:
            ## Split data, using 10% for validation.
            num_validation = max(1, int(len(inputs) * 0.1))
        else:
            num_validation = 0
        train_inputs = inputs[num_validation:]
        train_targets = outputs[num_validation:]
        val_inputs = inputs[:num_validation]
        val_targets = outputs[:num_validation]
        train_dataset = GraphDictDataset(train_inputs, train_targets)
        val_dataset = GraphDictDataset(val_inputs, val_targets)

        ## Set up Adam optimizer and dataloaders.
        optimizer = torch.optim.Adam(self._gnn.parameters(),
                                     lr=CFG.gnn_learning_rate)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self._batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      collate_fn=graph_batch_collate)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=self._batch_size,
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
