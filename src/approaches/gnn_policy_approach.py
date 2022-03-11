"""An approach that trains a GNN mapping states and goals to options."""

import functools
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import dill as pkl
import numpy as np
import torch
import torch.nn
import torch.optim
from gym.spaces import Box
from torch.utils.data import DataLoader

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, ApproachTimeout, \
    BaseApproach
from predicators.src.gnn.gnn import setup_graph_net
from predicators.src.gnn.gnn_utils import GraphDictDataset, \
    compute_normalizers, get_single_model_prediction, graph_batch_collate, \
    normalize_graph, train_model
from predicators.src.nsrt_learning.segmentation import segment_trajectory
from predicators.src.option_model import create_option_model
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, Dataset, DummyOption, \
    GroundAtom, Object, ParameterizedOption, Predicate, State, Task, Type, \
    _Option


class GNNPolicyApproach(BaseApproach):
    """Trains and uses a goal-conditioned GNN policy."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._sorted_options = sorted(self._initial_options,
                                      key=lambda o: o.name)
        self._option_model = create_option_model(CFG.option_model_name)
        # Fields for the GNN.
        self._gnn: Any = None
        self._nullary_predicates: List[Predicate] = []
        self._max_option_objects = 0
        self._max_option_params = 0
        self._node_feature_to_index: Dict[Any, int] = {}
        self._edge_feature_to_index: Dict[Any, int] = {}
        self._input_normalizers: Dict = {}
        self._target_normalizers: Dict = {}
        self._data_exemplar: Tuple[Dict, Dict] = ({}, {})
        # Seed torch.
        torch.manual_seed(self._seed)

    @property
    def is_learning_based(self) -> bool:
        return True

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        assert self._gnn is not None, "Learning hasn't happened yet!"
        if CFG.gnn_policy_solve_with_shooting:
            return self._solve_with_shooting(task, timeout)
        return self._solve_without_shooting(task)

    def _solve_without_shooting(self, task: Task) -> Callable[[State], Action]:
        cur_option = DummyOption

        def _policy(state: State) -> Action:
            atoms = utils.abstract(state, self._initial_predicates)
            nonlocal cur_option
            if cur_option is DummyOption or cur_option.terminal(state):
                # Just use the mean parameters to ground the option.
                param_opt, objects, params_mean = self._predict(
                    state, atoms, task.goal)
                cur_option = param_opt.ground(objects, params_mean)
                if not cur_option.initiable(state):
                    raise ApproachFailure("GNN policy chose a non-initiable "
                                          "option")
            act = cur_option.policy(state)
            return act

        return _policy

    def _solve_with_shooting(self, task: Task,
                             timeout: int) -> Callable[[State], Action]:
        start_time = time.time()
        # Keep trying until the timeout.
        while time.time() - start_time < timeout:
            total_num_act = 0
            state = task.init
            plan: List[_Option] = []
            # A single shooting try goes up to the environment's horizon.
            while total_num_act < CFG.horizon:
                if task.goal_holds(state):
                    return utils.option_plan_to_policy(plan)
                atoms = utils.abstract(state, self._initial_predicates)
                param_opt, objects, params_mean = self._predict(
                    state, atoms, task.goal)
                low = param_opt.params_space.low
                high = param_opt.params_space.high
                # Sample an initiable option.
                for _ in range(CFG.gnn_policy_shooting_num_samples):
                    params = np.array(self._rng.normal(
                        params_mean, CFG.gnn_policy_shooting_variance),
                                      dtype=np.float32)
                    params = np.clip(params, low, high)
                    opt = param_opt.ground(objects, params)
                    if opt.initiable(state):
                        break
                else:
                    break  # out of the while loop for this shooting try
                plan.append(opt)
                # Use the option model to determine the next state.
                try:
                    state, num_act = \
                        self._option_model.get_next_state_and_num_actions(
                            state, opt)
                except utils.EnvironmentFailure:
                    break
                total_num_act += num_act
        raise ApproachTimeout("Shooting timed out!")

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Organize data into (state, atoms, goal, option) tuples.
        ground_atom_dataset = utils.create_ground_atom_dataset(
            dataset.trajectories, self._initial_predicates)
        segmented_trajs = [
            segment_trajectory(traj) for traj in ground_atom_dataset
        ]
        data = []
        for segment_traj, (ll_traj, _) in zip(segmented_trajs,
                                              ground_atom_dataset):
            if not ll_traj.is_demo:
                continue
            goal = self._train_tasks[ll_traj.train_task_idx].goal
            for segment in segment_traj:
                state = segment.states[0]  # the segment's initial state
                atoms = segment.init_atoms  # the segment's initial atoms
                option = segment.get_option()  # the segment's option
                data.append((state, atoms, goal, option))
        self._setup_fields(data)
        # Set up exemplar, which is just the first tuple in the data.
        # Note that this call to graphify_data() will set the normalization
        # constants to undesired values, but it doesn't matter because in a
        # few lines, we'll call graphify_data() again with all the data,
        # which will set them in the desired way.
        example_inputs, example_targets = self._graphify_data(
            [(data[0][0], data[0][1], data[0][2])], [data[0][3]])
        self._data_exemplar = (example_inputs[0], example_targets[0])
        example_dataset = GraphDictDataset(example_inputs, example_targets)
        self._gnn = setup_graph_net(
            example_dataset,
            num_steps=CFG.gnn_policy_num_message_passing,
            layer_size=CFG.gnn_policy_layer_size)
        # Set up all the graphs, now using *all* the data.
        inputs, targets = self._graphify_data([(d[0], d[1], d[2])
                                               for d in data],
                                              [d[3] for d in data])
        # Run training.
        if CFG.gnn_policy_use_validation_set:
            ## Split data, using 10% for validation.
            num_validation = max(1, int(len(inputs) * 0.1))
        else:
            num_validation = 0
        train_inputs = inputs[num_validation:]
        train_targets = targets[num_validation:]
        val_inputs = inputs[:num_validation]
        val_targets = targets[:num_validation]
        train_dataset = GraphDictDataset(train_inputs, train_targets)
        val_dataset = GraphDictDataset(val_inputs, val_targets)
        ## Set up Adam optimizer and dataloaders.
        optimizer = torch.optim.Adam(self._gnn.parameters(),
                                     lr=CFG.gnn_policy_learning_rate)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=CFG.gnn_policy_batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      collate_fn=graph_batch_collate)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=CFG.gnn_policy_batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=graph_batch_collate)
        dataloaders = {"train": train_dataloader, "val": val_dataloader}
        ## Set up the optimization criteria.
        bce_loss = torch.nn.BCEWithLogitsLoss()
        crossent_loss = torch.nn.CrossEntropyLoss()
        mse_loss = torch.nn.MSELoss()

        def _criterion(output: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
            if self._max_option_objects == 0:
                return torch.tensor(0.0)
            return bce_loss(output, target)

        def _global_criterion(output: torch.Tensor,
                              target: torch.Tensor) -> torch.Tensor:
            # Combine losses from the one-hot option selection and
            # the continuous parameters.
            onehot_output, params_output = torch.split(  # type: ignore
                output, [len(self._sorted_options), self._max_option_params],
                dim=1)
            onehot_target, params_target = torch.split(  # type: ignore
                target, [len(self._sorted_options), self._max_option_params],
                dim=1)
            onehot_loss = crossent_loss(onehot_output,
                                        onehot_target.argmax(dim=1))
            if self._max_option_params > 0:
                params_loss = mse_loss(params_output, params_target)
            else:
                params_loss = torch.tensor(0.0)
            return onehot_loss + params_loss

        ## Launch training code.
        best_model_dict = train_model(
            self._gnn,
            dataloaders,
            optimizer=optimizer,
            criterion=_criterion,
            global_criterion=_global_criterion,
            num_epochs=CFG.gnn_policy_num_epochs,
            do_validation=CFG.gnn_policy_use_validation_set)
        self._gnn.load_state_dict(best_model_dict)
        info = {
            "exemplar": self._data_exemplar,
            "state_dict": self._gnn.state_dict(),
            "nullary_predicates": self._nullary_predicates,
            "max_option_objects": self._max_option_objects,
            "max_option_params": self._max_option_params,
            "node_feature_to_index": self._node_feature_to_index,
            "edge_feature_to_index": self._edge_feature_to_index,
            "input_normalizers": self._input_normalizers,
            "target_normalizers": self._target_normalizers,
        }
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_None.gnn", "wb") as f:
            pkl.dump(info, f)

    def load(self, online_learning_cycle: Optional[int]) -> None:
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.gnn", "rb") as f:
            info = pkl.load(f)
        # Initialize fields from loaded dictionary.
        input_example, target_example = info["exemplar"]
        dataset = GraphDictDataset([input_example], [target_example])
        self._gnn = setup_graph_net(
            dataset,
            num_steps=CFG.gnn_policy_num_message_passing,
            layer_size=CFG.gnn_policy_layer_size)
        self._gnn.load_state_dict(info["state_dict"])
        self._nullary_predicates = info["nullary_predicates"]
        self._max_option_objects = info["max_option_objects"]
        self._max_option_params = info["max_option_params"]
        self._node_feature_to_index = info["node_feature_to_index"]
        self._edge_feature_to_index = info["edge_feature_to_index"]
        self._input_normalizers = info["input_normalizers"]
        self._target_normalizers = info["target_normalizers"]

    def _predict(
        self, state: State, atoms: Set[GroundAtom], goal: Set[GroundAtom]
    ) -> Tuple[ParameterizedOption, List[Object], Array]:
        """Uses the GNN, which must already have been trained, to predict a
        parameterized option from self._sorted_options, discrete object
        arguments, and continuous arguments."""
        # Get output graph.
        in_graph, object_to_node = self._graphify_single_input(
            state, atoms, goal)
        node_to_object = {v: k for k, v in object_to_node.items()}
        type_to_node = defaultdict(set)
        for obj, node in object_to_node.items():
            type_to_node[obj.type.name].add(node)
        if CFG.gnn_policy_do_normalization:
            in_graph = normalize_graph(in_graph, self._input_normalizers)
        out_graph = get_single_model_prediction(self._gnn, in_graph)
        if CFG.gnn_policy_do_normalization:
            out_graph = normalize_graph(out_graph,
                                        self._target_normalizers,
                                        invert=True)
        # Extract parameterized option and continuous parameters.
        onehot_output, params = np.split(  # type: ignore
            out_graph["globals"], [len(self._sorted_options)])
        param_opt = self._sorted_options[np.argmax(onehot_output)]
        # Pad and clip parameters.
        params = params[:param_opt.params_space.shape[0]]
        params = params.clip(param_opt.params_space.low,
                             param_opt.params_space.high)
        # Extract objects, making sure to enforce the typing constraints.
        objects = []
        for i, obj_type in enumerate(param_opt.types):
            scores = out_graph["nodes"][:, i]
            allowed_idxs = type_to_node[obj_type.name]
            for j in range(len(scores)):
                if j not in allowed_idxs:
                    scores[j] = float("-inf")  # set its score to be really bad
            if np.max(scores) == float("-inf"):  # type: ignore
                # If all scores are -inf, we failed to select an object.
                raise ApproachFailure("GNN policy could not select an object")
            objects.append(node_to_object[np.argmax(scores)])
        return param_opt, objects, params

    def _setup_fields(
        self, data: List[Tuple[State, Set[GroundAtom], Set[GroundAtom],
                               _Option]]
    ) -> None:
        obj_types_set = set()
        nullary_predicates_set = set()
        unary_predicates_set = set()
        binary_predicates_set = set()
        obj_attrs_set = set()
        max_option_objects = 0
        max_option_params = 0

        # Go through the data, identifying the maximum number of option
        # objects and parameters, and the types/predicates/attributes.
        for state, atoms, goal, option in data:
            assert len(option.params.shape) == 1
            max_option_objects = max(max_option_objects, len(option.objects))
            max_option_params = max(max_option_params, option.params.shape[0])
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
        self._max_option_objects = max_option_objects
        self._max_option_params = max_option_params

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

    def _graphify_data(
            self, inputs: List[Tuple[State, Set[GroundAtom], Set[GroundAtom]]],
            targets: List[_Option]) -> Tuple[List[Dict], List[Dict]]:
        graph_inputs = []
        graph_targets = []
        assert len(inputs) == len(targets)

        for (state, atoms, goal), option in zip(inputs, targets):
            # Create input graph.
            graph_input, object_to_node = self._graphify_single_input(
                state=state, atoms=atoms, goal=goal)
            graph_inputs.append(graph_input)
            # Create target graph.
            ## First, copy over all unchanged fields.
            graph_target = {
                "n_node": graph_input["n_node"],
                "n_edge": graph_input["n_edge"],
                "edges": graph_input["edges"],
                "senders": graph_input["senders"],
                "receivers": graph_input["receivers"],
            }
            ## Next, set up the target node features.
            object_mask = np.zeros(
                (len(object_to_node), self._max_option_objects),
                dtype=np.int64)
            for i, obj in enumerate(option.objects):
                object_mask[object_to_node[obj], i] = 1
            graph_target["nodes"] = object_mask
            ## Finally, set up the target globals.
            option_index = self._sorted_options.index(option.parent)
            onehot_target = np.zeros(len(self._sorted_options))
            onehot_target[option_index] = 1
            assert len(option.params.shape) == 1
            params_target = np.zeros(self._max_option_params)
            params_target[:option.params.shape[0]] = option.params
            graph_target["globals"] = np.r_[onehot_target, params_target]
            graph_targets.append(graph_target)

        if CFG.gnn_policy_do_normalization:
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

        return graph_inputs, graph_targets

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
