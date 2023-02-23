import logging
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Set, Tuple
import functools

import numpy as np
from scipy.special import logsumexp
import torch
from gym.spaces import Box
from torch.utils.data import DataLoader

from predicators.approaches.sampler_learning_approach import \
    SamplerLearningApproach
from predicators.explorers import create_explorer
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.settings import CFG
from predicators.structs import Array, Dataset, GroundAtom, GroundAtomsHoldQuery, \
    GroundAtomsHoldResponse, InteractionRequest, InteractionResult, \
    LowLevelTrajectory, NSRT, NSRTSampler, Object, ParameterizedOption, Predicate, \
    Query, State, Task, Type, Variable, _Option
from predicators.ml_models import BinaryCNNEBM, BinaryEBM, MLPBinaryClassifier, NeuralGaussianRegressor
from predicators import utils
from predicators.envs import get_or_create_env
from predicators.gnn.gnn import EncodeProcessDecode, setup_graph_net
from predicators.gnn.gnn_utils import GraphDictDataset, compute_normalizers, \
    get_single_model_prediction, get_single_model_sample, graph_batch_collate, \
    normalize_graph, train_model

class GNNSamplerLearningApproach(SamplerLearningApproach):
    """A bilevel planning approach that uses hand-specified Operators
    but learns the samplers from interaction."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks, task_planning_heuristic,
                         max_skeletons_optimized)
        self._ebms = {}
        self._optimizers = {}
        self._replay = {}
        self._bce_loss = torch.nn.BCEWithLogitsLoss()

    @classmethod
    def get_name(cls) -> str:
        return "gnn_sampler_learning"

    def _initialize_ebm_samplers(self, trajectories):
        new_nsrts = []
        ground_atom_dataset = utils.create_ground_atom_dataset(
            trajectories, self._initial_predicates)
        # self._setup_fields(ground_atom_dataset)
        self._setup_fields()
        for nsrt in self._nsrts:
            if nsrt.name in self._ebms: # It's already initialized
                new_nsrts.append(nsrt)
                continue
            example_xa = None
            for traj, all_atoms in ground_atom_dataset:
                # goal = self._train_tasks[traj.train_task_idx].goal
                for t, (state, atoms, action) in enumerate(zip(traj.states[:-1], all_atoms[:-1], traj.actions)):
                    option = action.get_option()
                    if nsrt.option.name == option.name:
                        # example_xa, example_obj_to_node = self._graphify_single_state_action(state, atoms, goal, option)
                        example_xa, example_obj_to_node = self._graphify_single_state_action(state, atoms, option)
                        break
                if example_xa is not None:
                    break
            else:   # did not find any data points for this NSRT
                new_nsrts.append(nsrt)
                continue
            example_target = self._graphify_single_target(0.0, example_xa, example_obj_to_node)
            example_dataset = GraphDictDataset([example_xa], [example_target])
            ebm = setup_graph_net(example_dataset,
                                  num_steps=CFG.gnn_num_message_passing,
                                  layer_size=CFG.gnn_layer_size)


            # new_sampler = _LearnedSampler(ebm, nsrt.parameters, nsrt.option, self._nsrts, self._horizon, nsrt.sampler).sampler
            new_sampler = _LearnedSampler(ebm, nsrt.parameters, self._graphify_single_state_action, nsrt.option, nsrt.option_vars, nsrt.sampler, self._initial_predicates).sampler
            optimizer = torch.optim.Adam(ebm.parameters(),
                                         lr=CFG.gnn_learning_rate)
            self._ebms[nsrt.name] = ebm
            self._optimizers[nsrt.name] = optimizer
            new_nsrts.append(NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions,
                                  nsrt.add_effects, nsrt.delete_effects,
                                  nsrt.ignore_effects, nsrt.option, 
                                  nsrt.option_vars, new_sampler))

            self._replay[nsrt.name] = ([], [], [], [], [], [])  # state_actions_replay, object_to_node_replay, rewards_replay, next_states_replay, next_ground_nsrts_replay, terminals_replay

        self._nsrts = new_nsrts


    # TD-learning (SQL)
    def _update_samplers(self, trajectories: List[LowLevelTrajectory], annotations_list: List[Any], skeletons: List[Any]) -> None:
        """Learns the sampler in a self-supervised fashion."""
        logging.info("\nUpdating the samplers...")
        # Only initializes EBMs that don't exist yet
        self._initialize_ebm_samplers(trajectories)

        ground_atom_dataset = utils.create_ground_atom_dataset(
            trajectories, self._initial_predicates)
        for nsrt in self._nsrts:
            if nsrt.name not in self._ebms:
                continue
            ebm = self._ebms[nsrt.name]
            optimizer = self._optimizers[nsrt.name]
            replay = self._replay[nsrt.name]

            state_actions = []
            object_to_node_list = []
            rewards = []
            next_states = []
            next_ground_nsrts = []
            terminals = []

            for (traj, all_atoms), annotations, skeleton in zip(ground_atom_dataset, annotations_list, skeletons):
                # goal = self._train_tasks[traj.train_task_idx].goal
                for t, (state, atoms, action, annotation, ground_nsrt, next_state, next_atoms, next_ground_nsrt) in enumerate(zip(traj.states[:-1], all_atoms[:-1], traj.actions, annotations, skeleton, traj.states[1:], all_atoms[1:], (skeleton[1:] + [None]))):
                    option = action.get_option()
                    if nsrt.option.name == option.name:
                        # xa, obj_to_node = self._graphify_single_state_action(state, atoms, goal, option)
                        xa, obj_to_node = self._graphify_single_state_action(state, atoms, option)

                        state_actions.append(xa)
                        object_to_node_list.append(obj_to_node)
                        rewards.append(annotations[t] * self._reward_scale)
                        # next_states.append((next_state, next_atoms, goal))
                        next_states.append((next_state, next_atoms))
                        next_ground_nsrts.append(next_ground_nsrt)
                        terminals.append(next_ground_nsrt is None or not annotations[t])

            state_actions_replay, object_to_node_replay, rewards_replay, next_states_replay, next_ground_nsrts_replay, terminals_replay = replay

            state_actions_replay += state_actions
            object_to_node_replay += object_to_node_list
            rewards_replay += rewards
            next_states_replay += next_states
            next_ground_nsrts_replay += next_ground_nsrts
            terminals_replay += terminals

            if len(state_actions_replay) > 0:
                ebm_target = ebm
                if len(rewards) > 0:
                    print(nsrt.name, 'success rate:', sum(rewards)/len(rewards)/self._reward_scale)
                
                if not self._single_step:
                    raise NotImplementedError('Missing actual SQL implementation')
                else:
                    targets = rewards_replay
                graph_targets = []
                for xa, obj_to_node, target in zip(state_actions_replay, object_to_node_replay, targets):
                    graph_targets.append(
                        self._graphify_single_target(target, xa, obj_to_node))

                num_train = len(state_actions)#int(0.8*len(state_actions))
                state_actions_train = state_actions[:num_train]
                graph_targets_train = graph_targets[:num_train]
                # state_actions_val = state_actions[num_train:]
                # graph_targets_val = graph_targets[num_train:]
                train_dataset = GraphDictDataset(state_actions_train, graph_targets_train)
                train_dataloader = DataLoader(train_dataset,
                                        batch_size=CFG.gnn_batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        collate_fn=graph_batch_collate)
                # val_dataset = GraphDictDataset(state_actions_val, graph_targets_val)
                # val_dataloader = DataLoader(val_dataset,
                #                         batch_size=CFG.gnn_batch_size,
                #                         shuffle=True,
                #                         num_workers=0,
                #                         collate_fn=graph_batch_collate)
                logging.info(f"Training GNN on {len(train_dataset)} examples")
                best_model_dict = train_model(ebm,
                                              {"train":train_dataloader},#, "val": val_dataloader},
                                              optimizer=optimizer,
                                              # criterion=self._criterion,
                                              criterion=None,
                                              global_criterion=self._global_criterion,
                                              num_epochs=CFG.gnn_num_epochs,
                                              do_validation=False)
                ebm.load_state_dict(best_model_dict)
                # TODO: some saving stuff

                import matplotlib
                matplotlib.use('TkAgg')
                import matplotlib.pyplot as plt
                labels = np.array([t['globals'] for t in graph_targets_train])
                params = np.array([x['globals'] for x in state_actions_train])
                samples = []
                sample_predicted_labels = []
                for xa in state_actions_train:
                    xa['globals'] = nsrt.option.params_space.sample()
                    samples.append(get_single_model_sample(ebm, xa, self._rng))
                    xa['globals'] = samples[-1]
                    sample_predicted_labels.append(get_single_model_prediction(ebm, xa)['globals'] > 0)
                samples = np.array(samples)
                sample_predicted_labels = np.array(sample_predicted_labels)
                print(sample_predicted_labels.mean())
                plt.scatter(params[:, 0], params[:, 1], color=['red' if l == 0 else 'blue' for l in labels])
                plt.scatter(samples[:, 0], samples[:, 1], color='green', alpha=0.3)
                plt.show()
                exit()


    def _global_criterion(self, output: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        return self._bce_loss(output, target)    

    def _setup_fields(self) -> None:
        nullary_predicates_set = set()
        unary_predicates_set = set()
        binary_predicates_set = set()
        obj_attrs_set = set()
        env = get_or_create_env(CFG.env)

        for predicate in self._initial_predicates:
            arity = predicate.arity
            assert arity <= 2, "Predicates with arity > 2 are not supported"
            if arity == 0:
                nullary_predicates_set.add(predicate)
            elif arity == 1:
                unary_predicates_set.add(predicate)
            elif arity == 2:
                binary_predicates_set.add(predicate)
        obj_types_set = env.types
        for obj_type in obj_types_set:
            for feat in obj_type.feature_names:
                obj_attrs_set.add(f"feat_{feat}")
        option_set = env.options

        # Initialize option-specific features
        options = sorted(option_set, key=lambda x: x.name)
        self._option_parameters = {}
        self._node_feature_to_index_option = {}
        for option in options:
            ## Global features
            self._option_parameters[option.name] = np.arange(option.params_space.shape[0])
            ## Node features
            self._node_feature_to_index_option[option.name] = {}
            index = 0
            for obj_type in sorted(option.types):
                self._node_feature_to_index_option[option.name][f"option_obj_{obj_type.name}"] = index
                index += 1

        self._nullary_predicates = sorted(nullary_predicates_set)
        # self._setup_output_specific_fields(data)

        obj_types = sorted(obj_types_set)
        unary_predicates = sorted(unary_predicates_set)
        binary_predicates = sorted(binary_predicates_set)
        obj_attrs = sorted(obj_attrs_set)

        # G = functools.partial(utils.wrap_predicate, prefix="GOAL-")
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        # Initialize input node features.
        self._node_feature_to_index = {}
        index = 0
        for obj_type in obj_types:
            self._node_feature_to_index[f"type_{obj_type.name}"] = index
            index += 1
        for unary_predicate in unary_predicates:
            self._node_feature_to_index[unary_predicate] = index
            index += 1
        # for unary_predicate in unary_predicates:
        #     self._node_feature_to_index[G(unary_predicate)] = index
        #     index += 1
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
        # for binary_predicate in binary_predicates:
        #     self._edge_feature_to_index[G(binary_predicate)] = index
        #     index += 1
        # for binary_predicate in binary_predicates:
        #     self._edge_feature_to_index[G(R(binary_predicate))] = index
        #     index += 1

    def _graphify_single_target(self, target: float, graph_input: Dict,
                                object_to_node: Dict) -> Dict:
        # First, copy over all unchanged fields.
        graph_target = {
            "n_node": graph_input["n_node"],
            "n_edge": graph_input["n_edge"],
            "edges": graph_input["edges"],
            "senders": graph_input["senders"],
            "receivers": graph_input["receivers"],
        }
        graph_target["nodes"] = np.zeros((graph_target["n_node"], 0))
        graph_target["globals"] = np.array([target])
        return graph_target

    def _graphify_single_state_action(self, state: State, 
                                      atoms: Set[GroundAtom], 
                                      # goal: Set[GroundAtom], 
                                      option: _Option) -> Tuple[Dict, Dict]:
        all_objects = list(state)
        node_to_object = dict(enumerate(all_objects))
        object_to_node = {v: k for k, v in node_to_object.items()}
        num_objects = len(all_objects)
        num_node_features_option = len(self._node_feature_to_index_option[option.name])
        num_node_features = num_node_features_option + len(self._node_feature_to_index)
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
        # goal_globals = np.zeros(len(self._nullary_predicates), dtype=np.int64)
        # for atom in goal:
        #     if atom.predicate.arity != 0:
        #         continue
        #     goal_globals[self._nullary_predicates.index(atom.predicate)] = 1
        option_params_globals = option.params

        # graph["globals"] = np.r_[atoms_globals, goal_globals, option_params_globals]
        graph["globals"] = np.r_[atoms_globals, option_params_globals]

        # Add nodes (one per object) and node features.
        graph["n_node"] = np.array(num_objects)
        node_features = np.zeros((num_objects, num_node_features))

        ## Add node features for obj types.
        for obj in state:
            obj_index = object_to_node[obj]
            type_index = num_node_features_option + self._node_feature_to_index[f"type_{obj.type.name}"]
            node_features[obj_index, type_index] = 1

        ## Add node features for unary atoms.
        for atom in atoms:
            if atom.predicate.arity != 1:
                continue
            obj_index = object_to_node[atom.objects[0]]
            atom_index = num_node_features_option + self._node_feature_to_index[atom.predicate]
            node_features[obj_index, atom_index] = 1

        ## Add node features for unary atoms in goal.
        # for atom in goal:
        #     if atom.predicate.arity != 1:
        #         continue
        #     obj_index = object_to_node[atom.objects[0]]
        #     atom_index = num_node_features_option + self._node_feature_to_index[G(atom.predicate)]
        #     node_features[obj_index, atom_index] = 1

        ## Add node features for state.
        for obj in state:
            obj_index = object_to_node[obj]
            for feat, val in zip(obj.type.feature_names, state[obj]):
                feat_index = num_node_features_option + self._node_feature_to_index[f"feat_{feat}"]
                node_features[obj_index, feat_index] = val

        ## Add option object features.
        for i, obj in enumerate(option.objects):
            node_features[object_to_node[obj], i] = 1

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
        # for atom in goal:
        #     if atom.predicate.arity != 2:
        #         continue
        #     pred_index = self._edge_feature_to_index[G(atom.predicate)]
        #     obj0_index = object_to_node[atom.objects[0]]
        #     obj1_index = object_to_node[atom.objects[1]]
        #     all_edge_features[obj0_index, obj1_index, pred_index] = 1

        ## Add edge features for reversed binary atoms in goal.
        # for atom in goal:
        #     if atom.predicate.arity != 2:
        #         continue
        #     pred_index = self._edge_feature_to_index[G(R(atom.predicate))]
        #     obj0_index = object_to_node[atom.objects[0]]
        #     obj1_index = object_to_node[atom.objects[1]]
        #     # Note: the next line is reversed on purpose!
        #     all_edge_features[obj1_index, obj0_index, pred_index] = 1

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

@dataclass(frozen=True, eq=False, repr=False)
class _LearnedSampler:
    """A convenience class for holding the models underlying a learned
    sampler."""
    _ebm: BinaryEBM
    _nsrt_parameters: Sequence[Variable]
    _graphyfier: Callable[[State, Set[GroundAtom], _Option], Tuple[Dict, Dict]]
    _param_option: ParameterizedOption
    _option_vars: Sequence[Variable]
    _original_sampler: NSRTSampler
    _predicates: Collection[Predicate]

    def sampler(self, state: State, goal: Set[GroundAtom],
                rng: np.random.Generator, objects: Sequence[Object],
                skeleton: List[Any]) -> Array:
                                      # goal: Set[GroundAtom], 
                                      # option: _Option
        atoms = utils.abstract(state, self._predicates)
        sub = dict(zip(self._nsrt_parameters, objects))
        option_objs = [sub[v] for v in self._option_vars]
        option_params = self._param_option.params_space.sample()
        option = self._param_option.ground(option_objs, option_params)
        graph, _ = self._graphyfier(state, atoms, option)
        return np.array(get_single_model_sample(self._ebm, graph, rng),
                          dtype=self._param_option.params_space.dtype)
        return params