import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np
from scipy.special import logsumexp
import torch
from gym.spaces import Box

from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.explorers import create_explorer
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.settings import CFG
from predicators.structs import Array, Dataset, GroundAtom, GroundAtomsHoldQuery, \
    GroundAtomsHoldResponse, InteractionRequest, InteractionResult, \
    LowLevelTrajectory, NSRT, NSRTSampler, Object, ParameterizedOption, Predicate, \
    Query, State, Task, Type, Variable
from predicators.ml_models import BinaryCNNEBM, BinaryEBM, MLPBinaryClassifier, \
    NeuralGaussianRegressor, DiffusionRegressor, CNNDiffusionRegressor
from predicators import utils
from predicators.envs import get_or_create_env
from multiprocessing import Pool

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

def _featurize_state(all_args):
    nsrt_names, nsrt_parameters, horizon, state, ground_nsrt_objects, skeleton_names, skeleton_objects = all_args
    x = state.vec(ground_nsrt_objects)
    if CFG.use_full_state:
        # The full state is represented as the image observation of the env
        env = get_or_create_env(CFG.env)
        img = env.render_state(state, None)[0][::6,::6,:3].reshape(-1)
        # print(img.shape)
        # plt.imshow(img)
        # plt.show()
        # exit()
        # img = env.grid_state(state).reshape(-1)
        x = np.r_[img, x]
    if CFG.use_skeleton_state:
        # The skeleton representation is a series of self._horizon one-hot vectors
        # indicating which action is executed, plus a series of self._horizon * num_actions
        # vectors, where the chosen action per step contains the object features of the 
        # operator objects, while the other actions contain all-zeros
        num_nsrts = len(nsrt_names)
        skeleton_rep = np.zeros(0)
        for t in range(horizon):
            one_hot = np.zeros(num_nsrts)
            if t < len(skeleton_names):
                one_hot[nsrt_names.index(skeleton_names[t])] = 1
            nsrt_object_rep = np.zeros(0)
            for nsrt_tmp_name, nsrt_tmp_parameters in zip(nsrt_names, nsrt_parameters):
                if t < len(skeleton_names) and nsrt_tmp_name == skeleton_names[t]:
                    rep = state.vec(skeleton_objects[t])
                    assert state.vec(skeleton_objects[t]).shape[0] == sum(obj.type.dim for obj in nsrt_tmp_parameters), f'{state.vec(skeleton_objects[t_prime]).shape[0]}, {sum(obj.type.dim for obj in nsrt_tmp_parameters)}, {nsrt_tmp_name}, {skeleton_objects[t_prime]}, {nsrt_tmp._parameters}'
                else:
                    rep = np.zeros(sum(obj.type.dim for obj in nsrt_tmp_parameters))
                nsrt_object_rep = np.r_[nsrt_object_rep, rep]
            skeleton_rep = np.r_[skeleton_rep, one_hot, nsrt_object_rep]
        x = np.r_[x, skeleton_rep]
    return x



class SamplerLearningApproach(BilevelPlanningApproach):
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
        self._nsrts = get_gt_nsrts(CFG.env, self._initial_predicates,
                                   self._initial_options)
        self._online_learning_cycle = 0
        self._horizon = CFG.sampler_horizon
        self._gamma = 0.999
        self._reward_scale = CFG.sql_reward_scale
        self._single_step = CFG.sampler_learning_single_step
        self._ebm_class = CFG.ebm_class

    @classmethod
    def get_name(cls) -> str:
        return "sampler_learning"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._nsrts

    def _initialize_ebm_samplers(self) -> None:
        new_nsrts = []
        if CFG.use_ebm:
            self._ebms = []
        else:
            self._gaussians = []
            self._classifiers = []
        self._replay = []
        for nsrt in self._nsrts:
            if CFG.use_ebm:
                if self._ebm_class == 'ebm':
                    if CFG.use_full_state:
                        ebm_class = BinaryCNNEBM
                    else:
                        ebm_class = BinaryEBM
                    classifier = ebm_class(
                        seed=CFG.seed,
                        balance_data=CFG.mlp_classifier_balance_data,
                        max_train_iters=CFG.sampler_mlp_classifier_max_itr,
                        learning_rate=CFG.learning_rate,
                        n_iter_no_change=CFG.mlp_classifier_n_iter_no_change,
                        hid_sizes=CFG.mlp_classifier_hid_sizes,
                        n_reinitialize_tries=CFG.sampler_mlp_classifier_n_reinitialize_tries,
                        weight_init="default"
                    )
                elif self._ebm_class == 'diff':
                    if CFG.use_full_state:
                        ebm_class = CNNDiffusionRegressor
                    else:
                        ebm_class = DiffusionRegressor
                    classifier = ebm_class(
                        seed=CFG.seed,
                        hid_sizes=CFG.mlp_classifier_hid_sizes,
                        max_train_iters=CFG.sampler_mlp_classifier_max_itr,
                        timesteps=100,
                        learning_rate=CFG.learning_rate
                    )
                else:
                    raise ValueError('Can only use ebm class ebm or diff')
                new_sampler = _LearnedSampler(classifier, nsrt.parameters, nsrt.option, self._nsrts, self._horizon, nsrt.sampler).sampler
                self._ebms.append(classifier)
            else:
                gaussian = NeuralGaussianRegressor(
                    seed=CFG.seed,
                    hid_sizes=CFG.neural_gaus_regressor_hid_sizes,
                    max_train_iters=CFG.neural_gaus_regressor_max_itr,
                    clip_gradients=CFG.mlp_regressor_clip_gradients,
                    clip_value=CFG.mlp_regressor_gradient_clip_value,
                    learning_rate=CFG.learning_rate
                )
                classifier = MLPBinaryClassifier(
                    seed=CFG.seed,
                    balance_data=CFG.mlp_classifier_balance_data,
                    max_train_iters=CFG.sampler_mlp_classifier_max_itr,
                    learning_rate=CFG.learning_rate,
                    n_iter_no_change=CFG.mlp_classifier_n_iter_no_change,
                    hid_sizes=CFG.mlp_classifier_hid_sizes,
                    n_reinitialize_tries=CFG.sampler_mlp_classifier_n_reinitialize_tries,
                    weight_init="default"
                )
                new_sampler = _LearnedSampler2(gaussian, classifier, nsrt.parameters, nsrt.option, self._nsrts, self._horizon, nsrt.sampler).sampler
                self._gaussians.append(gaussian)
                self._classifiers.append(classifier)
            
            new_nsrts.append(NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions,
                                  nsrt.add_effects, nsrt.delete_effects,
                                  nsrt.ignore_effects, nsrt.option, 
                                  nsrt.option_vars, new_sampler))

            self._replay.append(([], [], [], [], [], [], []))   # states_replay, actions_replay, rewards_replay, next_states_replay, next_ground_nsrts_replay, terminals_replay, train_task_idx

        self._nsrts = new_nsrts

    def load(self, online_learning_cycle: Optional[int]) -> None:
        raise 'NotImplementedError'

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Haven't given this thought, so check that dataset is empty 
        # and do nothing
        assert len(dataset.trajectories) == 0

    def get_interaction_requests(self):
        requests = []
        # Create the explorer.
        explorer = create_explorer(
            "partial_planning_failures" if CFG.collect_failures else "partial_planning",
            self._initial_predicates,
            self._initial_options,
            self._types,
            self._action_space,
            self._train_tasks,
            self._get_current_nsrts(),
            self._option_model)
        self._request_task_idx = []
        total_requests = CFG.interactive_num_requests_per_cycle
        train_task_idxs = self._select_interaction_train_task_idxs()
        tasks_per_process = total_requests // CFG.data_collection_num_processes
        process_idx = CFG.data_collection_process_idx
        process_train_task_idxs = train_task_idxs[process_idx * tasks_per_process : (process_idx + 1) *tasks_per_process]
        logging.info(f"Collecting data for {len(process_train_task_idxs)} tasks ({process_idx * tasks_per_process} : {(process_idx + 1) * tasks_per_process})")
        for curr_request, train_task_idx in enumerate(process_train_task_idxs):
            if curr_request % 100 == 0:
                logging.info(f"\t{curr_request} / {tasks_per_process}")
            # Determine the action policy and termination function.
            act_policy, termination_fn, skeleton = explorer.get_exploration_strategy(
                train_task_idx, CFG.timeout)
            # Determine the query policy.
            # query_policy = self._create_goal_query_policy(train_task_idx)
            # query_policy = self._create_all_query_policy()
            query_policy = self._create_none_query_policy()
            if CFG.collect_failures:
                requests += [InteractionRequest(train_task_idx, act_policy[i] ,
                                                query_policy, termination_fn[i], skeleton[i])
                             for i in range(len(act_policy))]
                self._request_task_idx += [train_task_idx for _ in range(len(act_policy))]
            else:
                requests.append(InteractionRequest(train_task_idx, act_policy, query_policy, termination_fn, skeleton))
                self._request_task_idx.append(train_task_idx)
        # assert len(requests) == CFG.interactive_num_requests_per_cycle
        return requests
    
    def _create_none_query_policy(self) -> Callable[[State], Optional[Query]]:
        def _query_policy(s: State) -> Optional[GroundAtomsHoldQuery]:
            return None
        return _query_policy

    def _create_goal_query_policy(self,
            train_task_idx: int) -> Callable[[State], Optional[Query]]:
        """Query all goal atoms for the task."""
        def _query_policy(s: State) -> Optional[GroundAtomsHoldQuery]:
            return GroundAtomsHoldQuery(self._train_tasks[train_task_idx].goal)

        return _query_policy

    def _create_all_query_policy(self) -> Callable[[State], Optional[Query]]:
        """Query all possible atoms in this state."""
        def _query_policy(s: State) -> Optional[GroundAtomsHoldQuery]:
            ground_atoms = utils.all_possible_ground_atoms(
                s, self._initial_predicates)
            return GroundAtomsHoldQuery(ground_atoms)

        return _query_policy

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        traj_list: List[LowLevelTrajectory] = []
        annotations_list: List[Any] = []
        skeleton_list: List[Any] = []

        for task_idx, result in zip(self._request_task_idx, results):
            state_annotations = []
            atoms_prev_state = utils.abstract(result.states[0], self._initial_predicates)
            for state, ground_nsrt in zip(result.states[1:], result.skeleton):
                atoms_state = utils.abstract(state, self._initial_predicates)
                expected_atoms = utils.apply_operator(ground_nsrt, atoms_prev_state)
                state_annotations.append(all(a.holds(state) for a in expected_atoms))
                atoms_prev_state = atoms_state

            traj = LowLevelTrajectory(result.states, result.actions)
            traj_list.append(traj)
            annotations_list.append(state_annotations)
            skeleton_list.append(result.skeleton)
        self._update_samplers(traj_list, annotations_list, skeleton_list)
        self._online_learning_cycle += 1

    def _select_interaction_train_task_idxs(self) -> List[int]:
        # At the moment, we select train task indices uniformly at
        # random, with replacement. In the future, we may want to
        # try other strategies.
        return self._rng.choice(len(self._train_tasks),
                                size=CFG.interactive_num_requests_per_cycle)

    # TD-learning (SQL)
    def _update_samplers(self, trajectories: List[LowLevelTrajectory], annotations_list: List[Any], skeletons: List[Any]) -> None:
        """Learns the sampler in a self-supervised fashion."""
        assert len(trajectories) == len(self._request_task_idx)
        logging.info("\nUpdating the samplers...")
        if self._online_learning_cycle == 0:
            self._initialize_ebm_samplers()
        logging.info("Featurizing the samples...")
        cnt_featurized = 0
        total_samples = sum(len(traj.actions) for traj in trajectories)

        for ebm, nsrt, replay in zip(self._ebms, self._nsrts, self._replay):
            states = []
            actions = []
            rewards = []
            next_states = []
            next_ground_nsrts = []
            terminals = []
            train_task_idxs = []
            state_dicts = []

            for traj, annotations, skeleton, train_task_idx in zip(trajectories, annotations_list, skeletons, self._request_task_idx):
                for t, (state, action, annotation, ground_nsrt, next_state, next_ground_nsrt) in enumerate(zip(traj.states[:-1], traj.actions, annotations, skeleton, traj.states[1:], (skeleton[1:] + [None]))):
                    # Assume there's a single sampler per option
                    option = action.get_option()
                    if nsrt.option.name == option.name:
                        cnt_featurized += 1
                        x = _featurize_state(([nsrt.name for nsrt in self._nsrts], [nsrt.parameters for nsrt in self._nsrts], self._horizon, state, ground_nsrt.objects, [s.name for s in skeleton[t + 1:]], [s.objects for s in skeleton[t + 1:]]))
                        a = option.params
                        if next_ground_nsrt is not None:
                            next_x = _featurize_state(([nsrt.name for nsrt in self._nsrts], [nsrt.parameters for nsrt in self._nsrts], self._horizon, next_state, next_ground_nsrt.objects, [s.name for s in skeleton[t + 2:]], [s.objects for s in skeleton[t + 2:]]))
                        else:
                            next_x = np.empty(0)
                        
                        state_dicts.append(state)
                        states.append(x)
                        actions.append(a)
                        rewards.append(annotations[t] * self._reward_scale)
                        next_states.append(next_x)
                        next_ground_nsrts.append(next_ground_nsrt)
                        terminals.append(next_ground_nsrt is None or not annotations[t])
                        train_task_idxs.append(train_task_idx)
                        if cnt_featurized % 100 == 0:
                            logging.info(f"{cnt_featurized} / {total_samples}")

            states_replay, actions_replay, rewards_replay, next_states_replay, next_ground_nsrts_replay, terminals_replay, train_task_idxs_replay = replay
            states_replay += states
            actions_replay += actions
            rewards_replay += rewards
            next_states_replay += next_states
            next_ground_nsrts_replay += next_ground_nsrts
            terminals_replay += terminals
            train_task_idxs_replay += train_task_idxs

            if len(states_replay) > 0:
                ebm_target = ebm    # TODO: this should be a copy and done iteratively, but for now it'll do
                states_arr = np.array(states_replay)
                actions_arr = np.array(actions_replay)
                rewards_arr = np.array(rewards_replay, dtype=np.float32)
                next_states_arr = np.array(next_states_replay)
                terminals_arr = np.array(terminals_replay)
                train_task_idxs_arr = np.array(train_task_idxs_replay)
                print(states_arr.shape)
                torch.save({
                    'states': states_arr,
                    'state_dicts': state_dicts,
                    'actions': actions_arr,
                    'rewards': rewards_arr,
                    'next_states': next_states_arr,
                    'terminals': terminals_arr,
                    'train_task_idxs': train_task_idxs_arr
                    # }, f'data_img/data_obs/{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}split/{CFG.data_collection_process_idx:02}_{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks{"_random" if CFG.collect_failures else ""}.pt')
                    # }, f'data_sampler_viz/{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}split/{CFG.data_collection_process_idx:02}_singlestep_{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks{"_random" if CFG.collect_failures else ""}.pt')
                    # }, f'data_sampler_viz/{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}split/{CFG.data_collection_process_idx:02}_{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks{"_random" if CFG.collect_failures else ""}.pt')
                    }, f'data_sampler_viz2/{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}split/{CFG.data_collection_process_idx:02}_singlestep_{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks{"_random" if CFG.collect_failures else ""}.pt')
                    # }, f'data_sampler_viz2/{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}split/{CFG.data_collection_process_idx:02}_{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks{"_random" if CFG.collect_failures else ""}.pt')
                continue

        # ##### Changing for data load from file #####
        # for ebm, nsrt, replay in zip(self._ebms, self._nsrts, self._replay):
        #         if self._ebm_class == 'ebm':
        #             data_random = torch.load(f'data_obs/{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks_random_trimmed.pt')#.pt')#
        #             data_planner = torch.load(f'data_obs/{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt')
        #             # del data_planner['next_states']
        #             # del data_random['next_states']
        #             data = {k: np.concatenate((data_random[k], data_planner[k]), axis=0) for k in data_planner.keys()}
        #             # For fairness of comparison to random-only and planner-only data, sample half of the data
        #             # mask = np.random.choice(data['states'].shape[0], size=data['states'].shape[0] // 2, replace=False)
        #             # data = data_random
        #             states_arr = data['states']#[mask]
        #             actions_arr = data['actions']#[mask]
        #             rewards_arr = data['rewards']#[mask]
        #             # next_states_arr = data['next_states'][mask]
        #             terminals_arr = data['terminals']#[mask]
        #             train_task_idxs_arr = data['train_task_idxs']#[mask]
        #         else:
        #             # for i in range(1):
        #             #     logging.info(f'Loading {i}th file')
        #             #     tmp_d = torch.load(f'data_img/data_obs/5ksplit/{i:02}_{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt') 
        #             #     del tmp_d['next_states']
        #             #     if i == 0:
        #             #         data = {k: np.empty((tmp_d[k].shape[0] * 21, *tmp_d[k].shape[1:]), dtype=np.float32) for k in tmp_d}
        #             #         cnt = {k: 0 for k in tmp_d}
        #             #     for k in tmp_d:
        #             #         data[k][cnt[k] : cnt[k] + tmp_d[k].shape[0]] = tmp_d[k]
        #             #         cnt[k] += tmp_d[k].shape[0]
        #             #     del tmp_d
        #             # for k in data:
        #             #     data[k] = data[k][:cnt[k]]
        #             # data = torch.load(f'data_grid/data/{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt')
        #             print(f'data_{CFG.env}/data/{nsrt.name}_full_obs_50ktasks.pt')
        #             data = torch.load(f'data_{CFG.env}/data/{nsrt.name}_full_obs_50ktasks.pt')
        #             states_arr = data['states']
        #             actions_arr = data['actions']
        #             rewards_arr = data['rewards']
        #             # next_states_arr = data['next_states']
        #             terminals_arr = data['terminals']
        #             train_task_idxs_arr = data['train_task_idxs']
                    
        #             # Hack to make sure we get exactly num_train_tasks tasks
        #             if CFG.num_train_tasks < 50000:
        #                 mask = train_task_idxs_arr < 2 * CFG.num_train_tasks    # take twice as many takss
        #                 train_task_idxs_arr_tmp = train_task_idxs_arr[mask]
        #                 max_task_idx = np.unique(train_task_idxs_arr_tmp)[CFG.num_train_tasks - 1]  # take the (num_train_task)-th unique task idx
        #             else:
        #                 max_task_idx = 50000
        #             mask = train_task_idxs_arr < max_task_idx

        #             states_arr = states_arr[mask]
        #             actions_arr = actions_arr[mask]
        #             rewards_arr = rewards_arr[mask]
        #             terminals_arr = terminals_arr[mask]
        #             train_task_idxs_arr = train_task_idxs_arr[mask]
        #         # if not CFG.use_full_state:
        #         #     # states_arr = states_arr[:, 150*150*3:]
        #         #     states_arr = states_arr[:, 20*20*3:]
        #         #     # next_states_arr = next_states_arr[:, 20*20*3:]
        #         print(states_arr.shape)
        #         if len(rewards_arr) > 0:
        #             logging.info(f"\n{nsrt.name} success rate: {rewards_arr.mean()/self._reward_scale}")
        # ###########################################
                # if len(rewards) > 0:
                #     logging.info(f"\n{nsrt.name} success rate: {sum(rewards)/len(rewards)/self._reward_scale}")
                    # if ebm._x_dim != -1:
                    #     positive_samples = ebm.predict_probas(np.c_[states, actions]) > 0.5
                    #     logging.info(f"\tpredicted successes: {positive_samples.mean()}")
                assert terminals_arr[rewards_arr == 0].all(), 'all r=0 should be terminal {}'.format(terminals_arr[rewards_arr == 0])

                next_v = np.zeros(states_arr.shape[0])
                if not self._single_step:
                    for nsrt_tmp, ebm_tmp in zip(self._nsrts, self._ebms):
                        nsrt_mask = np.zeros(states_arr.shape[0], dtype=np.bool)
                        for i, gt_nsrt in enumerate(next_ground_nsrts_replay):
                            if gt_nsrt is not None and gt_nsrt.name == nsrt_tmp.name:
                                nsrt_mask[i] = True
                        if nsrt_mask.sum() > 0:
                            next_states_rep = np.stack(next_states_arr[nsrt_mask], axis=0).repeat(10, axis=0)
                            next_actions_arr = np.stack([nsrt_tmp.option.params_space.sample() for _ in range(10 * nsrt_mask.sum())], axis=0)
                            next_x = np.c_[next_states_rep, next_actions_arr]
                            if ebm_tmp._x_dim == -1:
                                next_q = np.zeros(next_x.shape[0])
                            else:
                                with torch.no_grad():
                                    next_q = ebm_tmp.predict_probas(next_x)

                            # log ( mean ( exp (q) ) ) = log ( sum ( exp (q) ) / len (q) ) =
                            # = log ( sum ( exp (q) ) ) - log ( len (q) )
                            next_v[nsrt_mask] = logsumexp(next_q.reshape(10, -1), axis=0) - np.log(10)

                    target = rewards_arr + self._gamma * next_v * (1 - terminals_arr)
                    target = np.clip(target, None, 1.0)
                else:
                    target = rewards_arr
                x = np.c_[states_arr, actions_arr]
                logging.info(f"max r: {rewards_arr.max()}")
                logging.info(f"max v: {next_v.max()}")
                logging.info(f"max target: {target.max()}")
                ######### Delete any repeated samples ##########
                logging.info(f"Dataset size before trimming: {x.shape[0]}")
                logging.info(f"Total failures before: {(1-target).sum()}")
                logging.info(f"Total successes before: {(target).sum()}")
                if False:#CFG.num_train_tasks <= 5000:
                    _, unique_idx = np.unique(x.round(decimals=6), axis=0, return_index=True)
                    x = x[unique_idx]
                    rewards_arr = rewards_arr[unique_idx]
                    # next_states_arr = next_states_arr[unique_idx]
                    terminals_arr = terminals_arr[unique_idx]
                    train_task_idxs_arr = train_task_idxs_arr[unique_idx]
                    target = target[unique_idx]
                logging.info(f"Dataset size after trimming: {x.shape[0]}")
                logging.info(f"{nsrt.name} success rate after trimming: {rewards_arr.mean()/self._reward_scale}")
                logging.info(f"Total failures after: {(1-target).sum()}")
                logging.info(f"Total successes after: {(target).sum()}")
                ################################################

                if CFG.ebm_aux_training == 'reconstruct':
                    aux_labels = x[:, :-actions_arr.shape[1]]
                elif CFG.ebm_aux_training == 'geometry':
                    # t_dim = ((x.shape[1] - actions_arr.shape[1]) // 2) * 2 
                    # aux_inputs = np.c_[x, np.zeros((x.shape[0], t_dim))]
                    if nsrt.name.startswith('NavigateTo'):
                        obj_x = x[:, 0]
                        obj_y = x[:, 1]
                        obj_w = x[:, 2]
                        obj_h = x[:, 3]
                        obj_yaw = x[:, 4]

                        offset_x = x[:, -2]
                        offset_y = x[:, -1]
                        pos_x = obj_x + obj_w * offset_x * np.cos(obj_yaw) - \
                                obj_h * offset_y * np.sin(obj_yaw)
                        pos_y = obj_y + obj_w * offset_x * np.sin(obj_yaw) + \
                                obj_h * offset_y * np.cos(obj_yaw)

                        pos_x = np.clip(pos_x, 0, 20 - 1e-6)
                        pos_y = np.clip(pos_y, 0, 20 - 1e-6)

                        aux_labels = np.empty((x.shape[0], 1))
                        for i in range(x.shape[0]):
                            rect = utils.Rectangle(obj_x[i], obj_y[i], obj_w[i], obj_h[i], obj_yaw[i])
                            if nsrt.name == 'NavigateToCup':
                                aux_labels[i] = rect.line_segments[0].distance_nearest_point(pos_x[i], pos_y[i])
                            elif nsrt.name == 'NavigateToTray':
                                x1 = obj_x[i]
                                y1 = obj_y[i]
                                x2 = x1 + (obj_w[i] - obj_h[i]) * np.cos(obj_yaw[i])
                                y2 = y1 + (obj_w[i] - obj_h[i]) * np.sin(obj_yaw[i])
                                rect1 = utils.Rectangle(x1, y1, obj_h[i], obj_h[i], obj_yaw[i])
                                rect2 = utils.Rectangle(x2, y2, obj_h[i], obj_h[i], obj_yaw[i])
                                aux_labels[i] = min(rect1.distance_nearest_point(pos_x[i], pos_y[i]),
                                                        rect2.distance_nearest_point(pos_x[i], pos_y[i]))
                            else:
                                aux_labels[i] = rect.distance_nearest_point(pos_x[i], pos_y[i])
                    elif nsrt.name.startswith('Pick'):
                        obj_x = x[:, 0]
                        obj_y = x[:, 1]
                        obj_w = x[:, 2]
                        obj_h = x[:, 3]
                        obj_yaw = x[:, 4]

                        robby_x = x[:, 6]
                        robby_y = x[:, 7]
                        robby_yaw = x[:, 8]

                        offset_gripper = x[:, -2]

                        tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
                        tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)

                        aux_labels = np.empty((x.shape[0], 1))
                        for i in range(x.shape[0]):
                            rect = utils.Rectangle(obj_x[i], obj_y[i], obj_w[i], obj_h[i], obj_yaw[i])
                            aux_labels[i] = rect.distance_nearest_point(tip_x[i], tip_y[i])
                    elif nsrt.name.startswith('Place'):
                        obj_x = x[:, 6]
                        obj_y = x[:, 7]
                        obj_w = x[:, 8]
                        obj_h = x[:, 9]
                        obj_yaw = x[:, 10]

                        robby_x = x[:, 12]
                        robby_y = x[:, 13]
                        robby_yaw = x[:, 14]

                        offset_gripper = x[:, -1]

                        tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
                        tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)

                        aux_labels = np.empty((x.shape[0], 1))
                        for i in range(x.shape[0]):
                            rect = utils.Rectangle(obj_x[i], obj_y[i], obj_w[i], obj_h[i], obj_yaw[i])
                            aux_labels[i] = rect.distance_nearest_point(tip_x[i], tip_y[i])
                    else:
                        raise ValueError('NSRT should be navigate, pick, or place')
                elif CFG.ebm_aux_training == 'geometry+':
                    if nsrt.name.startswith('NavigateTo'):
                        obj_x = x[:, 0]
                        obj_y = x[:, 1]
                        obj_w = x[:, 2]
                        obj_h = x[:, 3]
                        obj_yaw = x[:, 4]

                        offset_x = x[:, -2]
                        offset_y = x[:, -1]
                        pos_x = obj_x + obj_w * offset_x * np.cos(obj_yaw) - \
                                obj_h * offset_y * np.sin(obj_yaw)
                        pos_y = obj_y + obj_w * offset_x * np.sin(obj_yaw) + \
                                obj_h * offset_y * np.cos(obj_yaw)

                        pos_x = np.clip(pos_x, 0, 20 - 1e-6)
                        pos_y = np.clip(pos_y, 0, 20 - 1e-6)

                        aux_labels = np.empty((x.shape[0], 7))
                        for i in range(x.shape[0]):
                            rect = utils.Rectangle(obj_x[i], obj_y[i], obj_w[i], obj_h[i], obj_yaw[i])
                            if nsrt.name == 'NavigateToCup':
                                aux_labels[i, 0] = rect.line_segments[0].distance_nearest_point(pos_x[i], pos_y[i])
                            elif nsrt.name == 'NavigateToTray':
                                x1 = obj_x[i]
                                y1 = obj_y[i]
                                x2 = x1 + (obj_w[i] - obj_h[i]) * np.cos(obj_yaw[i])
                                y2 = y1 + (obj_w[i] - obj_h[i]) * np.sin(obj_yaw[i])
                                rect1 = utils.Rectangle(x1, y1, obj_h[i], obj_h[i], obj_yaw[i])
                                rect2 = utils.Rectangle(x2, y2, obj_h[i], obj_h[i], obj_yaw[i])
                                aux_labels[i, 0] = min(rect1.distance_nearest_point(pos_x[i], pos_y[i]),
                                                        rect2.distance_nearest_point(pos_x[i], pos_y[i]))
                            else:
                                aux_labels[i, 0] = rect.distance_nearest_point(pos_x[i], pos_y[i])
                            aux_labels[i, 1], aux_labels[i, 2] = rect.nearest_point(pos_x[i], pos_y[i])
                            aux_labels[i, 3], aux_labels[i, 4] = pos_x[i], pos_y[i]
                            aux_labels[i, 5], aux_labels[i, 6] = rect.relative_reoriented_coordinates(pos_x[i], pos_y[i])
                    elif nsrt.name.startswith('Pick'):
                        obj_x = x[:, 0]
                        obj_y = x[:, 1]
                        obj_w = x[:, 2]
                        obj_h = x[:, 3]
                        obj_yaw = x[:, 4]

                        robby_x = x[:, 6]
                        robby_y = x[:, 7]
                        robby_yaw = x[:, 8]

                        offset_gripper = x[:, -2]

                        tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
                        tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)

                        aux_labels = np.empty((x.shape[0], 4))
                        for i in range(x.shape[0]):
                            rect = utils.Rectangle(obj_x[i], obj_y[i], obj_w[i], obj_h[i], obj_yaw[i])
                            aux_labels[i, 0], aux_labels[i, 1] = tip_x[i], tip_y[i]
                            aux_labels[i, 2], aux_labels[i, 3] = rect.relative_reoriented_coordinates(tip_x[i], tip_y[i])
                    elif nsrt.name.startswith('Place'):
                        obj_relative_x = x[:, 0]
                        obj_relative_y = x[:, 1]
                        obj_w = x[:, 2]
                        obj_h = x[:, 3]
                        obj_relative_yaw = x[:, 4]

                        target_x = x[:, 6]
                        target_y = x[:, 7]
                        target_w = x[:, 8]
                        target_h = x[:, 9]
                        target_yaw = x[:, 10]

                        robby_x = x[:, 12]
                        robby_y = x[:, 13]
                        robby_yaw = x[:, 14]

                        offset_gripper = x[:, -1]

                        tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
                        tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)

                        place_x = tip_x + obj_relative_x * np.sin(
                            robby_yaw) + obj_relative_y * np.cos(robby_yaw)
                        place_y = tip_y + obj_relative_y * np.sin(
                            robby_yaw) - obj_relative_x * np.cos(robby_yaw)
                        place_yaw = obj_relative_yaw + robby_yaw

                        aux_labels = np.empty((x.shape[0], 4))
                        for i in range(x.shape[0]):
                            target_rect = utils.Rectangle(target_x[i], target_y[i], target_w[i], target_h[i], target_yaw[i])
                            
                            obj_yaw = place_yaw[i]
                            while obj_yaw > np.pi:
                                obj_yaw -= (2 * np.pi)
                            while obj_yaw < -np.pi:
                                obj_yaw += (2 * np.pi)
                            obj_rect = utils.Rectangle(place_x[i], place_y[i], obj_w[i], obj_h[i], obj_yaw)
                            com_x, com_y = obj_rect.center

                            aux_labels[i, 0], aux_labels[i, 1] = com_x, com_y
                            aux_labels[i, 2], aux_labels[i, 3] = target_rect.relative_reoriented_coordinates(com_x, com_y)
                    else:
                        raise ValueError('NSRT should be navigate, pick, or place')
                elif CFG.ebm_aux_training is None:
                    aux_labels = None
                else:
                    raise ValueError(f'ebm_aux_training cannot be {CFG.ebm_aux_training}')
                train_data_mask = train_task_idxs_arr < int(0.95 * CFG.num_train_tasks)

                x_train = x[train_data_mask]
                target_train = target[train_data_mask]
                # aux_inputs_train = aux_inputs[train_data_mask]
                if aux_labels is not None:
                    aux_labels_train = aux_labels[train_data_mask]
                x_val = x[~train_data_mask]
                target_val = target[~train_data_mask]
                # aux_inputs_val = aux_inputs[~train_data_mask]
                if aux_labels is not None:
                    aux_labels_val = aux_labels[~train_data_mask]
                if True:#((target_train > 0).any() and (target_train == 0).any()):#nsrt.name == 'NavigateTo':#
                    # logging.info(f"Train samples: {x_train.shape[0]}, validation samples: {x_val.shape[0]}")
                    # if os.path.exists(f'models_{CFG.env}/{nsrt.name}{self._ebm_class}{f"_{CFG.ebm_aux_training}" if CFG.ebm_aux_training else ""}{"cf" if CFG.classifier_free_guidance else ""}_obs_{CFG.use_full_state}_myopic_True_dropout_False_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt'):
                    #     continue
                    # ebm.train()
                    # if self._ebm_class == 'diff':
                    #     y_train = x_train[target_train > 0, -actions_arr.shape[1]:]
                    #     x_train = x_train[target_train > 0, :-actions_arr.shape[1]]
                    #     if aux_labels is not None:
                    #         aux_labels_train = aux_labels_train[target_train > 0]
                    #     else:
                    #         aux_labels_train = None
                    #     ebm.fit(x_train, y_train, aux_labels_train)
                    #     # ebm.fit(x_train[target_train > 0, :-actions_arr.shape[1]], x_train[target_train > 0, -actions_arr.shape[1]:], x_train[target_train == 0, :-actions_arr.shape[1]], x_train[target_train == 0, -actions_arr.shape[1]:])
                    # else:
                    #     ebm.fit(x_train, target_train, x_val, target_val)

                    # # print(ebm._x_dim, ebm._input_scale.shape, ebm._input_shift.shape, ebm._output_scale.shape, ebm._output_shift.shape)
                    # ebm.eval()
                    # if aux_labels is not None:
                    #     train_aux = ebm.predict_aux(x_train, y_train)

                    #     y_val = x_val[target_val > 0, -actions_arr.shape[1]:]
                    #     x_val = x_val[target_val > 0, :-actions_arr.shape[1]]
                    #     aux_labels_val = aux_labels_val[target_val > 0]
                    #     val_aux = ebm.predict_aux(x_val, y_val)
                    #     l2_train = np.linalg.norm(train_aux - aux_labels_train, axis=1)
                    #     l2_val = np.linalg.norm(val_aux - aux_labels_val, axis=1)
                    #     print(f'Aux loss\n\t mean -- train: {l2_train.mean()}, val: {l2_val.mean()}')
                    #     print(f'\t max -- train: {l2_train.max()}, val: {l2_val.max()}')
                    #     print(f'\t min -- train: {l2_train.min()}, val: {l2_val.min()}')
                    # torch.save({
                    #             'state_dict': ebm.state_dict(),
                    #             'x_dim': ebm._x_dim,
                    #             'y_dim': ebm._y_dim if hasattr(ebm, 'y_dim') else None,
                    #             't_dim': ebm._t_dim if hasattr(ebm, 't_dim') else None,
                    #             'input_scale': ebm._input_scale,
                    #             'input_shift': ebm._input_shift, 
                    #             'output_scale': ebm._output_scale if hasattr(ebm, '_output_scale') else None,
                    #             'output_shift': ebm._output_shift if hasattr(ebm, '_output_shift') else None,
                    #             'y_aux_dim': ebm._y_aux_dim if hasattr(ebm, '_y_aux_dim') else None,
                    #             'output_aux_scale': ebm._output_aux_scale if hasattr(ebm, '_output_aux_scale') else None,
                    #             'output_aux_shift': ebm._output_aux_shift if hasattr(ebm, '_output_aux_shift') else None,
                    #            },
                    #            f'models_{CFG.env}/{nsrt.name}{self._ebm_class}{f"_{CFG.ebm_aux_training}" if CFG.ebm_aux_training else ""}{"cf" if CFG.classifier_free_guidance else ""}_obs_{CFG.use_full_state}_myopic_True_dropout_False_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt')
                    # if not CFG.bookshelf_specialized_nsrts:
                    #     exit()
                    ##### Changing for model load from file #####
                    ebm.train()
                    idx_positive = target_train > 0
                    idx_negative = target_train == 0
                    x_dummy = np.r_[x_train[idx_positive][:1], x_train[idx_negative][:1]]
                    target_dummy = np.r_[target_train[idx_positive][:1], target_train[idx_negative][:1]]
                    # ebm.fit(x_dummy, target_dummy)
                    ebm.fit(x_dummy[target_dummy > 0, :-actions_arr.shape[1]], x_dummy[target_dummy > 0, -actions_arr.shape[1]:])
                    ebm.eval()
                    model_state = torch.load(f'models_{CFG.env}/{nsrt.name}{self._ebm_class}{f"_{CFG.ebm_aux_training}" if CFG.ebm_aux_training else ""}{"cf" if CFG.classifier_free_guidance else ""}_obs_{CFG.use_full_state}_myopic_True_dropout_False_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt', map_location='cpu' if not CFG.use_cuda else None)
                    ebm.load_state_dict(model_state['state_dict'])
                    ebm._x_dim = model_state['x_dim']
                    ebm._input_scale = model_state['input_scale']
                    ebm._input_shift = model_state['input_shift']
                    ebm._output_scale = model_state['output_scale']
                    ebm._output_shift = model_state['output_shift']
                    #############################################
                        
                # params = x[:, -2:]
                # for i in range(10):
                #     samples = []
                # # for single_x in x:
                # #     samples.append(ebm.predict_sample(single_x[:-2], self._rng))
                #     for _ in range(100):
                #         samples.append(ebm.predict_sample(x_train[i*1000, :-2], self._rng))
                #     samples = np.array(samples)
                #     observed_samples = x_train[np.isclose(x_train[:, :-2], x_train[i*1000, :-2]).all(axis=1), -2:]
                #     # sample_predicted_labels = (ebm.predict_probas(np.c_[x[:, :-2], samples]) > 0.5)
                #     # sample_predicted_labels = (ebm.predict_probas(np.c_[x_train[i*1000, :-2].reshape(1,-1).repeat(100, axis=0), samples]) > 0.5)
                #     # print(sample_predicted_labels.mean())
                #     plt.scatter(params[::100, 0], params[::100, 1], color=['red' if l == 0 else 'blue' for l in target[::100]])
                #     plt.scatter(samples[:, 0], samples[:, 1], color='green', alpha=0.3)
                #     plt.scatter(observed_samples[:, 0], observed_samples[:, 1], color='orange', alpha=0.3)
                #     plt.show()
        exit()

@dataclass(frozen=True, eq=False, repr=False)
class _LearnedSampler:
    """A convenience class for holding the models underlying a learned
    sampler."""
    _ebm: BinaryEBM
    _variables: Sequence[Variable]
    _param_option: ParameterizedOption
    _nsrts: List[NSRT]
    _horizon: int
    _original_sampler: NSRTSampler

    def sampler(self, state: State, goal: Set[GroundAtom],
                rng: np.random.Generator, objects: Sequence[Object],
                skeleton: List[Any]) -> Array:
        """The sampler corresponding to the given models.

        May be used as the _sampler field in an NSRT.
        """
        if not self._ebm.is_trained:
            return self._original_sampler(state, goal, rng, objects, skeleton)
        x = _featurize_state((None, None, None, state, objects, None, None))

    #     x_lst: List[Any] = []  
    #     sub = dict(zip(self._variables, objects))
    #     for var in self._variables:
    #         x_lst.extend(state[sub[var]])
    #     # if CFG.sampler_learning_use_goals:
    #     #     # For goal-conditioned sampler learning, we currently make the
    #     #     # extremely limiting assumption that there is one goal atom, with
    #     #     # one goal object. This will not be true in most cases. This is a
    #     #     # placeholder for better methods to come.
    #     #     assert len(goal) == 1
    #     #     goal_atom = next(iter(goal))
    #     #     assert len(goal_atom.objects) == 1
    #     #     goal_obj = goal_atom.objects[0]
    #     #     x_lst.extend(state[goal_obj])  # add goal state
    #     x = np.array(x_lst)
    #     assert (x == state.vec(objects)).all()
    #     env = get_or_create_env(CFG.env)
    #     if CFG.use_full_state:
    #         # img = env.render_state(state, None)[0][:,:,:3].mean(axis=2).reshape(-1)
    #         img = env.grid_state(state).reshape(-1)
    #         x = np.r_[img, x]
    #     if CFG.use_skeleton_state:
    #         # The skeleton representation is a series of self._horizon one-hot vectors
    #         # indicating which action is executed, plus a series of self._horizon * num_actions
    #         # vectors, where the chosen action per step contains the object features of the 
    #         # operator objects, while the other actions contain all-zeros
    #         nsrt_names = [nsrt.name for nsrt in self._nsrts]
    #         num_nsrts = len(self._nsrts)

    #         skeleton_rep = np.zeros(0)
    #         for t in range(self._horizon - 1):
    #             one_hot = np.zeros(num_nsrts)
    #             if t < len(skeleton):
    #                 one_hot[nsrt_names.index(skeleton[t].name)] = 1
    #             nsrt_object_rep = np.zeros(0)
    #             for nsrt in self._nsrts:
    #                 if t < len(skeleton) and nsrt.name == skeleton[t].name:
    #                     rep = state.vec(skeleton[t].objects)
    #                 else:
    #                     rep = np.zeros(sum(obj.type.dim for obj in nsrt.parameters))
    #                 nsrt_object_rep = np.r_[nsrt_object_rep, rep]
    #             skeleton_rep = np.r_[skeleton_rep, one_hot, nsrt_object_rep]
    #         x = np.r_[x, skeleton_rep]
        
        params = np.array(self._ebm.predict_sample(x, rng),
                          dtype=self._param_option.params_space.dtype)
        # print(params)
        # exit()
        return params

@dataclass(frozen=True, eq=False, repr=False)
class _LearnedSampler2:
    """A convenience class for holding the models underlying a learned
    sampler."""
    _regressor: NeuralGaussianRegressor
    _classifier: MLPBinaryClassifier
    _variables: Sequence[Variable]
    _param_option: ParameterizedOption
    _nsrts: List[NSRT]
    _horizon: int
    _original_sampler: NSRTSampler

    def sampler(self, state: State, goal: Set[GroundAtom],
                rng: np.random.Generator, objects: Sequence[Object],
                skeleton: List[Any]) -> Array:
        """The sampler corresponding to the given models.

        May be used as the _sampler field in an NSRT.
        """
        if self._regressor._x_dim == -1:
            return self._original_sampler(state, goal, rng, objects, skeleton)
        x_lst: List[Any] = []  
        sub = dict(zip(self._variables, objects))
        for var in self._variables:
            x_lst.extend(state[sub[var]])
        # if CFG.sampler_learning_use_goals:
        #     # For goal-conditioned sampler learning, we currently make the
        #     # extremely limiting assumption that there is one goal atom, with
        #     # one goal object. This will not be true in most cases. This is a
        #     # placeholder for better methods to come.
        #     assert len(goal) == 1
        #     goal_atom = next(iter(goal))
        #     assert len(goal_atom.objects) == 1
        #     goal_obj = goal_atom.objects[0]
        #     x_lst.extend(state[goal_obj])  # add goal state
        x = np.array(x_lst)
        assert (x == state.vec(objects)).all()
        env = get_or_create_env(CFG.env)
        if CFG.use_full_state:
            # img = env.render_state(state, None)[0][:,:,:3].mean(axis=2).reshape(-1)
            img = env.grid_state(state).reshape(-1)
            x = np.r_[img, x]
        if CFG.use_skeleton_state:
            # The skeleton representation is a series of self._horizon one-hot vectors
            # indicating which action is executed, plus a series of self._horizon * num_actions
            # vectors, where the chosen action per step contains the object features of the 
            # operator objects, while the other actions contain all-zeros
            nsrt_names = [nsrt.name for nsrt in self._nsrts]
            num_nsrts = len(self._nsrts)

            skeleton_rep = np.zeros(0)
            for t in range(self._horizon - 1):
                one_hot = np.zeros(num_nsrts)
                if t < len(skeleton):
                    one_hot[nsrt_names.index(skeleton[t].name)] = 1
                nsrt_object_rep = np.zeros(0)
                for nsrt in self._nsrts:
                    if t < len(skeleton) and nsrt.name == skeleton[t].name:
                        rep = state.vec(skeleton[t].objects)
                    else:
                        rep = np.zeros(sum(obj.type.dim for obj in nsrt.parameters))
                    nsrt_object_rep = np.r_[nsrt_object_rep, rep]
                skeleton_rep = np.r_[skeleton_rep, one_hot, nsrt_object_rep]
            x = np.r_[x, skeleton_rep]
        
        num_rejections = 0
        while num_rejections <= CFG.max_rejection_sampling_tries:
            params = np.array(self._regressor.predict_sample(x, rng),
                              dtype=self._param_option.params_space.dtype)
            if self._param_option.params_space.contains(params) and \
               self._classifier.classify(np.r_[x, params]):
                break
            num_rejections += 1
        return params