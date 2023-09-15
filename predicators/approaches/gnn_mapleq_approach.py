"""A parameterized action reinforcement learning approach inspired by MAPLE,
(https://ut-austin-rpl.github.io/maple/) but where only a Q function is
learned. The Q-function here  takes the form of a GNN so that it is able
to generalize to problems with a different number of objects.

Base samplers and applicable actions are used to perform the argmax.
"""

from __future__ import annotations

import abc
import functools
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar
from torch.utils.data import DataLoader

import dill as pkl
import torch
from gym.spaces import Box
import numpy as np

from predicators import utils
from predicators.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.explorers import BaseExplorer, create_explorer
from predicators.gnn.gnn import EncodeProcessDecode, setup_graph_net, GraphDictDataset
from predicators.ml_models import MapleQData, MapleQFunction
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, InteractionRequest, \
    LowLevelTrajectory, ParameterizedOption, Predicate, State, Task, Type, \
    _GroundNSRT, _Option

_Output = TypeVar("_Output")  # a generic type for the output of this GNN


class GNNMapleQApproach(OnlineNSRTLearningApproach):
    """A parameterized action RL approach inspired by MAPLE."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)

        # The current implementation assumes that NSRTs are not changing.
        assert CFG.strips_learner == "oracle"
        # The base sampler should also be unchanging and from the oracle.
        assert CFG.sampler_learner == "oracle"

        # Log all transition data.
        self._interaction_goals: List[Set[GroundAtom]] = []
        self._last_seen_segment_traj_idx = -1

        # Store the Q function. Note that this implicitly
        # contains a replay buffer.
        self._q_function = MapleQFunction(
            seed=CFG.seed,
            hid_sizes=CFG.mlp_regressor_hid_sizes,
            max_train_iters=CFG.mlp_regressor_max_itr,
            clip_gradients=CFG.mlp_regressor_clip_gradients,
            clip_value=CFG.mlp_regressor_gradient_clip_value,
            learning_rate=CFG.learning_rate,
            weight_decay=CFG.weight_decay,
            use_torch_gpu=CFG.use_torch_gpu,
            train_print_every=CFG.pytorch_train_print_every,
            n_iter_no_change=CFG.active_sampler_learning_n_iter_no_change,
            num_lookahead_samples=CFG.
            active_sampler_learning_num_lookahead_samples)

        # Fields for the GNN Q function.
        self._gnn: Optional[EncodeProcessDecode] = None
        self._nullary_predicates: List[Predicate] = []
        self._node_feature_to_index: Dict[Any, int] = {}
        self._edge_feature_to_index: Dict[Any, int] = {}
        self._input_normalizers: Dict = {}
        self._target_normalizers: Dict = {}
        self._data_exemplar: Tuple[Dict, Dict] = ({}, {})
        self._sorted_options: List[ParameterizedOption] = sorted(
            self._initial_options, key=lambda o: o.name)
        # Seed torch.
        torch.manual_seed(self._seed)

    @classmethod
    def get_name(cls) -> str:
        return "maple_q"

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        def _option_policy(state: State) -> _Option:
            return self._q_function.get_option(
                state,
                task.goal,
                num_samples_per_ground_nsrt=CFG.
                active_sampler_learning_num_samples)

        return utils.option_policy_to_policy(
            _option_policy, max_option_steps=CFG.max_num_steps_option_rollout)

    def _create_explorer(self) -> BaseExplorer:
        """Create a new explorer at the beginning of each interaction cycle."""
        # Note that greedy lookahead is not yet supported.
        preds = self._get_current_predicates()
        assert CFG.explorer == "maple_q"
        explorer = create_explorer(CFG.explorer,
                                   preds,
                                   self._initial_options,
                                   self._types,
                                   self._action_space,
                                   self._train_tasks,
                                   self._get_current_nsrts(),
                                   self._option_model,
                                   maple_q_function=self._q_function)
        return explorer

    def load(self, online_learning_cycle: Optional[int]) -> None:
        super().load(online_learning_cycle)
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}_{online_learning_cycle}.DATA", "rb") as f:
            save_dict = pkl.load(f)
        self._q_function = save_dict["q_function"]
        self._last_seen_segment_traj_idx = save_dict[
            "last_seen_segment_traj_idx"]
        self._interaction_goals = save_dict["interaction_goals"]
        self._online_learning_cycle = CFG.skip_until_cycle + 1

    def _learn_nsrts(self, trajectories: List[LowLevelTrajectory],
                     online_learning_cycle: Optional[int],
                     annotations: Optional[List[Any]]) -> None:
        # Start by learning NSRTs in the usual way.
        super()._learn_nsrts(trajectories, online_learning_cycle, annotations)
        # Check the assumption that operators and options are 1:1.
        # This is just an implementation convenience.
        assert len({nsrt.option for nsrt in self._nsrts}) == len(self._nsrts)
        for nsrt in self._nsrts:
            assert nsrt.option_vars == nsrt.parameters
        # On the first cycle, we need to register the ground NSRTs, goals, and
        # objects in the Q function so that it can define its inputs.
        if not online_learning_cycle:
            all_ground_nsrts: Set[_GroundNSRT] = set()
            if CFG.sesame_grounder == "naive":
                for nsrt in self._nsrts:
                    all_objects = {
                        o
                        for t in self._train_tasks for o in t.init
                    }
                    all_ground_nsrts.update(
                        utils.all_ground_nsrts(nsrt, all_objects))
            elif CFG.sesame_grounder == "fd_translator":  # pragma: no cover
                all_objects = set()
                for t in self._train_tasks:
                    curr_task_objects = set(t.init)
                    curr_task_types = {o.type for o in t.init}
                    curr_init_atoms = utils.abstract(
                        t.init, self._get_current_predicates())
                    all_ground_nsrts.update(
                        utils.all_ground_nsrts_fd_translator(
                            self._nsrts, curr_task_objects,
                            self._get_current_predicates(), curr_task_types,
                            curr_init_atoms, t.goal))
                    all_objects.update(curr_task_objects)
            else:  # pragma: no cover
                raise ValueError(
                    f"Unrecognized sesame_grounder: {CFG.sesame_grounder}")
            goals = [t.goal for t in self._train_tasks]
            self._q_function.set_grounding(all_objects, goals,
                                           all_ground_nsrts)
        # Update the data using the updated self._segmented_trajs.
        self._update_maple_data()
        # Re-learn Q function.
        self._q_function.train_q_function()
        # Save the things we need other than the NSRTs, which were already
        # saved in the above call to self._learn_nsrts()
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.DATA", "wb") as f:
            pkl.dump(
                {
                    "q_function": self._q_function,
                    "last_seen_segment_traj_idx":
                    self._last_seen_segment_traj_idx,
                    "interaction_goals": self._interaction_goals,
                }, f)

    def _update_maple_data(self) -> None:
        start_idx = self._last_seen_segment_traj_idx + 1
        new_trajs = self._segmented_trajs[start_idx:]

        goal_offset = CFG.max_initial_demos
        assert len(self._segmented_trajs) == goal_offset + \
            len(self._interaction_goals)
        new_traj_goals = self._interaction_goals[goal_offset + start_idx:]

        for traj_i, segmented_traj in enumerate(new_trajs):
            self._last_seen_segment_traj_idx += 1
            for seg_i, segment in enumerate(segmented_traj):
                s = segment.states[0]
                goal = new_traj_goals[traj_i]
                o = segment.get_option()
                ns = segment.states[-1]
                reward = 1.0 if goal.issubset(segment.final_atoms) else 0.0
                terminal = reward > 0 or seg_i == len(segmented_traj) - 1
                self._q_function.add_datum_to_replay_buffer(
                    (s, goal, o, ns, reward, terminal))

    def get_interaction_requests(self) -> List[InteractionRequest]:
        # Save the goals for each interaction request so we can later associate
        # states, actions, and goals.
        requests = super().get_interaction_requests()
        for request in requests:
            goal = self._train_tasks[request.train_task_idx].goal
            self._interaction_goals.append(goal)
        return requests


class GNNMapleQFunction():
    def __init__(self, seed: int, n_iter_no_change: int,
                 num_lookahead_samples: int, max_replay_buffer_size: int,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 rng: np.random.Generator):
        self._seed = seed
        self._n_iter_no_change = n_iter_no_change
        self._num_lookahead_samples = num_lookahead_samples
        self._max_replay_buffer_size = max_replay_buffer_size
        self._replay_buffer = deque(maxlen=self._max_replay_buffer_size)
        self._initial_predicates = initial_predicates
        self._initial_options = initial_options
        self._sorted_options: List[ParameterizedOption] = sorted(
            self._initial_options, key=lambda o: o.name)
        self._rng = rng
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
        self._max_option_params: int = 0
        self._setup_gnn_fields()
        # Seed torch.
        torch.manual_seed(self._seed)

    def add_datum_to_replay_buffer(self, datum: MapleQData) -> None:
        """Add one datapoint to the replay buffer.

        If the buffer is full, data is appended in a FIFO manner.
        """
        self._replay_buffer.append(datum)

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


    def _graphify_single_target(self, target_q_val: float, graph_input: Dict) -> Dict:
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

    def _setup_gnn_fields(self) -> None:
        """Use the initial predicates and options to setup
        the fields necessary to instantiate the GNN Q-network."""
        all_obj_types = set(t for p in self._initial_predicates
                            for t in p.types) | set(
                                t for o in self._initial_options
                                for t in o.types)
        obj_types_set = set(f"type_{t.name}" for t in all_obj_types)
        nullary_predicates_set = set(p for p in self._initial_predicates
                                     if p.arity == 0)
        unary_predicates_set = set(p for p in self._initial_predicates
                                   if p.arity == 1)
        binary_predicates_set = set(p for p in self._initial_predicates
                                    if p.arity == 2)
        obj_attrs_set = set(f"feat_{f}" for t in all_obj_types
                            for f in t.feature_names)
        self._nullary_predicates = sorted(nullary_predicates_set)
        self._max_option_objects = max(
            len(o.types) for o in self._initial_options)
        self._max_option_params = max(o.params_space.shape[0]
                                      for o in self._initial_options)

        # Seemingly, this below line is just something that sets up
        # class-specific variables that might be important for setting the output
        # I doubt we'll need any of this, since we're just outputting a single float.
        # self._setup_output_specific_fields(data)

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
                               goal: Set[GroundAtom], option: _Option) -> Tuple[Dict, Dict]:
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
        continuous_params_globals = np.zeros(self._max_option_params, dtype=np.float32)
        continuous_params_globals[:option.params.shape[0]] = option.params
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

        ## Finally, create node features for the objects involved in the
        ## currently-taken option.
        option_objs_mask = np.zeros((num_objects, self._max_option_objects))
        for i, obj in enumerate(option.objects):
            option_objs_mask[object_to_node[obj], i] = 1
        
        node_features = np.concatenate((node_features, option_objs_mask), axis=1)
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


    def train_q_function(self):
        """Trains the GNN-based Q-function."""
        # If there's no data in the replay buffer, we can't train.
        if len(self._replay_buffer) == 0:
            return
        
        # If the GNN hasn't been instantiated yet, then setup some example
        # data to fix the structure of the GNN.        
        if self._gnn is None:
            state, goal, option, _, _, _ = self._replay_buffer[0]
            atoms = utils.abstract(state, self._initial_predicates)
            # Set up exemplar, which is just the first tuple in the data.
            example_input_graph, _ = self._graphify_single_input(
                state, atoms, goal, option)
            # We can't estimate a q-value given inputs yet, since we don't have
            # the GNN instantiated. Thus, we will simply pick a random value.
            q_val = self._rng.random()
            example_target = self._graphify_single_target(q_val,
                                                        example_input_graph)
            self._data_exemplar = (example_input_graph, example_target)
            example_dataset = GraphDictDataset([example_input_graph], [example_target])
            self._gnn = setup_graph_net(example_dataset,
                                        num_steps=CFG.gnn_num_message_passing,
                                        layer_size=CFG.gnn_layer_size)

        # Set up all the input and output graphs, now using *all* the data.
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
