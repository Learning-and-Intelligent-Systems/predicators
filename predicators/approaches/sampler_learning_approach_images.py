import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np
from scipy.special import logsumexp
from scipy.signal import convolve
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
import matplotlib.pyplot as plt

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
        return "sampler_learning_images"#"sampler_learning"#

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
                    if CFG.ebm_input_mode != 'local':
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
                raise ValueError("This code is not written for Gaussian regressor")
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

        # for ebm, nsrt, replay in zip(self._ebms, self._nsrts, self._replay):
        #     states = []
        #     actions = []
        #     rewards = []
        #     next_states = []
        #     next_ground_nsrts = []
        #     terminals = []
        #     train_task_idxs = []

        #     for traj, annotations, skeleton, train_task_idx in zip(trajectories, annotations_list, skeletons, self._request_task_idx):
        #         for t, (state, action, annotation, ground_nsrt, next_state, next_ground_nsrt) in enumerate(zip(traj.states[:-1], traj.actions, annotations, skeleton, traj.states[1:], (skeleton[1:] + [None]))):
        #             # Assume there's a single sampler per option
        #             option = action.get_option()
        #             if nsrt.option.name == option.name:
        #                 cnt_featurized += 1
        #                 x = _featurize_state(([nsrt.name for nsrt in self._nsrts], [nsrt.parameters for nsrt in self._nsrts], self._horizon, state, ground_nsrt.objects, [s.name for s in skeleton[t + 1:]], [s.objects for s in skeleton[t + 1:]]))
        #                 a = option.params
        #                 if next_ground_nsrt is not None:
        #                     next_x = _featurize_state(([nsrt.name for nsrt in self._nsrts], [nsrt.parameters for nsrt in self._nsrts], self._horizon, next_state, next_ground_nsrt.objects, [s.name for s in skeleton[t + 2:]], [s.objects for s in skeleton[t + 2:]]))
        #                 else:
        #                     next_x = np.empty(0)
                        
        #                 states.append(x)
        #                 actions.append(a)
        #                 rewards.append(annotations[t] * self._reward_scale)
        #                 next_states.append(next_x)
        #                 next_ground_nsrts.append(next_ground_nsrt)
        #                 terminals.append(next_ground_nsrt is None or not annotations[t])
        #                 train_task_idxs.append(train_task_idx)
        #                 if cnt_featurized % 100 == 0:
        #                     logging.info(f"{cnt_featurized} / {total_samples}")

        #     states_replay, actions_replay, rewards_replay, next_states_replay, next_ground_nsrts_replay, terminals_replay, train_task_idxs_replay = replay
        #     states_replay += states
        #     actions_replay += actions
        #     rewards_replay += rewards
        #     next_states_replay += next_states
        #     next_ground_nsrts_replay += next_ground_nsrts
        #     terminals_replay += terminals
        #     train_task_idxs_replay += train_task_idxs

        #     if len(states_replay) > 0:
        #         ebm_target = ebm    # TODO: this should be a copy and done iteratively, but for now it'll do
        #         states_arr = np.array(states_replay)
        #         actions_arr = np.array(actions_replay)
        #         rewards_arr = np.array(rewards_replay, dtype=np.float32)
        #         next_states_arr = np.array(next_states_replay)
        #         terminals_arr = np.array(terminals_replay)
        #         train_task_idxs_arr = np.array(train_task_idxs_replay)
        #         print(states_arr.shape)
        #         torch.save({
        #             'states': states_arr,
        #             'actions': actions_arr,
        #             'rewards': rewards_arr,
        #             'next_states': next_states_arr,
        #             'terminals': terminals_arr,
        #             'train_task_idxs': train_task_idxs_arr
        #             }, f'data_{CFG.env}/data/{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}split/{CFG.data_collection_process_idx:02}_{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks{"_random" if CFG.collect_failures else ""}.pt')
        #         continue

        ##### Changing for data load from file #####
        for ebm, nsrt, replay in zip(self._ebms, self._nsrts, self._replay):
                if self._ebm_class == 'ebm':
                    for i in range(50):#range(1):#
                        logging.info(f'Loading {i}th file')
                        tmp_d = torch.load(f'data_img/data_obs/5ksplit/{i:02}_{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks_random.pt') 
                        del tmp_d['next_states']
                        if i == 0:
                            data_random = {k: np.empty((tmp_d[k].shape[0] * 55, *tmp_d[k].shape[1:]), dtype=np.float32) for k in tmp_d}
                            cnt = {k: 0 for k in tmp_d}
                        for k in tmp_d:
                            data_random[k][cnt[k] : cnt[k] + tmp_d[k].shape[0]] = tmp_d[k]
                            cnt[k] += tmp_d[k].shape[0]
                        del tmp_d
                    for k in data_random:
                        data_random[k] = data_random[k][:cnt[k]]

                    for i in range(20):#range(1):#
                        logging.info(f'Loading {i}th file')
                        tmp_d = torch.load(f'data_img/data_obs/5ksplit/{i:02}_{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt') 
                        del tmp_d['next_states']
                        if i == 0:
                            data_planner = {k: np.empty((tmp_d[k].shape[0] * 55, *tmp_d[k].shape[1:]), dtype=np.float32) for k in tmp_d}
                            cnt = {k: 0 for k in tmp_d}
                        for k in tmp_d:
                            data_planner[k][cnt[k] : cnt[k] + tmp_d[k].shape[0]] = tmp_d[k]
                            cnt[k] += tmp_d[k].shape[0]
                        del tmp_d
                    for k in data_planner:
                        data_planner[k] = data_planner[k][:cnt[k]]

                    # data_random = torch.load(f'data_obs/{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks_random_trimmed.pt')#.pt')#
                    # data_planner = torch.load(f'data_obs/{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt')
                    # del data_planner['next_states']
                    # del data_random['next_states']
                    data = {k: np.concatenate((data_random[k], data_planner[k]), axis=0) for k in data_planner.keys()}
                    # For fairness of comparison to random-only and planner-only data, sample half of the data
                    mask = np.random.choice(data['states'].shape[0], size=data['states'].shape[0] // 2, replace=False)
                    # data = data_random
                    states_arr = data['states'][mask]
                    actions_arr = data['actions'][mask]
                    rewards_arr = data['rewards'][mask]
                    # next_states_arr = data['next_states'][mask]
                    terminals_arr = data['terminals'][mask]
                    train_task_idxs_arr = data['train_task_idxs'][mask]
                else:
                    for i in range(1):#range(20):#
                        logging.info(f'Loading {i}th file')
                        tmp_d = torch.load(f'data_img/data_obs/5ksplit/{i:02}_{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt') 
                        del tmp_d['next_states']
                        if i == 0:
                            data = {k: np.empty((tmp_d[k].shape[0] * 55, *tmp_d[k].shape[1:]), dtype=np.float32) for k in tmp_d}
                            cnt = {k: 0 for k in tmp_d}
                        for k in tmp_d:
                            data[k][cnt[k] : cnt[k] + tmp_d[k].shape[0]] = tmp_d[k]
                            cnt[k] += tmp_d[k].shape[0]
                        del tmp_d
                    for k in data:
                        data[k] = data[k][:cnt[k]]
                    # data = torch.load(f'data_grid/data/{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt')
                    #mask = np.random.choice(data['states'].shape[0], size=data['states'].shape[0] // 2, replace=False)
                    states_arr = data['states']#[mask]
                    actions_arr = data['actions']#[mask]
                    rewards_arr = data['rewards']#[mask]
                    # next_states_arr = data['next_states']
                    terminals_arr = data['terminals']#[mask]
                    train_task_idxs_arr = data['train_task_idxs']#[mask]

                ####
                img_arr = states_arr[:, :150*150*3].reshape(states_arr.shape[0], 150, 150, 3) / 255
                padding = img_arr.min(axis=(0,3))
                rows = padding.min(axis=1)
                cols = padding.min(axis=0)
                img_arr = img_arr[:, rows < 1]
                img_arr = img_arr[:, :, cols < 1]
                # del rows
                # del cols
                n_rows = img_arr.shape[1]
                n_cols = img_arr.shape[2]
                if n_rows < n_cols:
                    img_arr = np.concatenate((img_arr, np.zeros((img_arr.shape[0], n_cols - n_rows, n_cols, img_arr.shape[3]))), axis=1)
                elif n_cols < n_rows:
                    img_arr = np.concatenate((img_arr, np.zeros((img_arr.shape[0], n_rows, n_rows - n_cols, img_arr.shape[3]))), axis=2)
                local_arr = states_arr[:, 150*150*3:]
                del states_arr

                def standardize(vec):
                    if len(vec.shape) == 2:
                        std = vec.std(axis=0)
                        mean = vec.mean(axis=0)
                    elif len(vec.shape) == 4:
                        std = vec.std(axis=(0,1,2))
                        mean = vec.mean(axis=(0,1,2))
                    std = np.where(std > 0, std, 1)
                    return (vec - mean) / std, mean, std

                def imagize_local(img_shape, local):
                    x,y = np.meshgrid(np.arange(img_shape[2]), np.arange(img_shape[1])[::-1])
                    x = x * 20 / img_shape[2]
                    y = y * 20 / img_shape[1]

                    # obj_x = local[:, 0]
                    # obj_y = local[:, 1]
                    # obj_w = local[:, 2]
                    # obj_h = local[:, 3]
                    # obj_yaw = local[:, 4]

                    local_img = np.zeros(img_shape[:3])
                    for i in range(local.shape[0]):
                        # rect = utils.Rectangle(obj_x[i], obj_y[i], obj_w[i], obj_h[i], obj_yaw[i])
                        rect = utils.Rectangle(local[i, 0], local[i, 1], local[i, 2], local[i, 3], local[i, 4])
                        obj_img = rect.contains_point_vec(x, y)
                        local_img[i,:,:] = obj_img 
                    return local_img

                def imagize_action(img_shape, local, actions):
                    obj_x = local[:, 0]
                    obj_y = local[:, 1]
                    obj_w = local[:, 2]
                    obj_h = local[:, 3]
                    obj_yaw = local[:, 4]

                    offset_x = actions[:, 0]
                    offset_y = actions[:, 1]
                    pos_x = obj_x + obj_w * offset_x * torch.cos(obj_yaw) - \
                            obj_h * offset_y * torch.sin(obj_yaw)
                    pos_y = obj_y + obj_w * offset_x * torch.sin(obj_yaw) + \
                            obj_h * offset_y * torch.cos(obj_yaw)

                    pos_x = torch.clip(pos_x, 0, 20 - 1e-6)
                    pos_y = torch.clip(pos_y, 0, 20 - 1e-6)

                    pos_x = (pos_x * img_shape[2] / 20).long()#.astype(int)
                    pos_y = img_shape[1] - 1 - (pos_y * img_shape[0] / 20).long()#.astype(int)
                    action_img = np.zeros(actions.shape[0], img_shape[1], img_shape[2], dtype=actions.dtype, device=actions.device)
                    action_img[np.arange(actions.shape[0]), pos_y, pos_x] = 1
                    return action_img


                def imagize_action_torch(actions, local, img_shape):
                    obj_x = local[:, 0]
                    obj_y = local[:, 1]
                    obj_w = local[:, 2]
                    obj_h = local[:, 3]
                    obj_yaw = local[:, 4]

                    offset_x = actions[:, 0]
                    offset_y = actions[:, 1]
                    pos_x = obj_x + obj_w * offset_x * torch.cos(obj_yaw) - \
                            obj_h * offset_y * torch.sin(obj_yaw)
                    pos_y = obj_y + obj_w * offset_x * torch.sin(obj_yaw) + \
                            obj_h * offset_y * torch.cos(obj_yaw)

                    pos_x = torch.clip(pos_x, 0, 20 - 1e-6)
                    pos_y = torch.clip(pos_y, 0, 20 - 1e-6)

                    pos_x = (pos_x * img_shape[1] / 20).long()#.astype(int)
                    pos_y = img_shape[0] - 1 - (pos_y * img_shape[0] / 20).long()#.astype(int)
                    action_img = torch.zeros(actions.shape[0], img_shape[0], img_shape[1], dtype=actions.dtype, device=actions.device)
                    action_img[torch.arange(actions.shape[0]), pos_y, pos_x] = 1
                    return action_img

                def unnormalize_action(local, actions):
                    obj_x = local[:, 0]
                    obj_y = local[:, 1]
                    obj_w = local[:, 2]
                    obj_h = local[:, 3]
                    obj_yaw = local[:, 4]

                    offset_x = actions[:, 0]
                    offset_y = actions[:, 1]
                    # pos_x = obj_x + obj_w * offset_x * np.cos(obj_yaw) - \
                    #         obj_h * offset_y * np.sin(obj_yaw)
                    # pos_y = obj_y + obj_w * offset_x * np.sin(obj_yaw) + \
                    #         obj_h * offset_y * np.cos(obj_yaw)
                    # pos_x = np.clip(pos_x, 0, 20 - 1e-6)
                    # pos_y = np.clip(pos_y, 0, 20 - 1e-6)

                    # return np.c_[pos_x, pos_y]

                    # logging.info('Using relative positions in place of actions')
                    # rel_pos_x = obj_w * offset_x * np.cos(obj_yaw) - \
                    #         obj_h * offset_y * np.sin(obj_yaw)
                    # rel_pos_y = obj_w * offset_x * np.sin(obj_yaw) + \
                    #         obj_h * offset_y * np.cos(obj_yaw)
                    # return np.c_[rel_pos_x, rel_pos_y]

                    logging.info('Using unnormalized offsets in place of actions')
                    unnormalized_offset_x = obj_w * offset_x
                    unnormalized_offset_y = obj_h * offset_y
                    return np.c_[unnormalized_offset_x, unnormalized_offset_y]


                def make_cspace(img_arr):
                    # Remove robot (gray) from image)
                    img_wo_robot = np.where(np.logical_and(img_arr[:, :, :, 0] == img_arr[:, :, :, 1], img_arr[:, :, :, 1] == img_arr[:, :, :, 2]).reshape(*(img_arr.shape[:3]), 1), 1, img_arr)
                    
                    # Create mask from robot
                    radius = int(np.ceil(2 / 20 * max(img_arr.shape[1], img_arr.shape[2])))
                    conv_filter = np.zeros((2 * radius, 2 * radius))
                    center_x = radius
                    center_y = radius
                    x, y = np.meshgrid(np.arange(2 * radius), np.arange(2 * radius))
                    conv_filter[(x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2] = 1

                    # Turn into occupancy grid
                    occupancy_wo_robot = (img_wo_robot.min(axis=3) < 1).astype(float)
                    
                    # Turn into C-space occupancy
                    cspace_img = convolve(occupancy_wo_robot, conv_filter.reshape(1, *(conv_filter.shape)), mode='same')
                    cspace_img /= conv_filter.sum()
                    cspace_img = (cspace_img > 1e-5).astype(float)
                    return cspace_img

                    # # TODO: remove below, used for debugging collisions in 3-space vs C-space
                    # radius = int(np.ceil(2 / 20 * max(img_arr.shape[1], img_arr.shape[2]))) - 1
                    # x, y = np.meshgrid(np.arange(img_arr.shape[2]), np.arange(img_arr.shape[1]))
                    # mask = (x[None] - pos_x[:, None, None]) ** 2 + (y[None] - pos_y[:, None, None]) ** 2 <= radius ** 2
                    # img_arr[mask, 4] = 1

                if CFG.ebm_input_mode == 'local':
                    local_arr = standardize(local_arr)
                    actions_arr = standardize(actions_arr)
                    x = np.c_[local_arr, actions_arr]
                elif CFG.ebm_input_mode == 'full-image-vector':
                    img_arr, _, _= standardize(img_arr)
                    img_shape = img_arr.shape[1:]

                    local_arr,_,_ = standardize(local_arr)
                    actions_arr,_,_ = standardize(actions_arr)
                    x = np.c_[img_arr.reshape(img_arr.shape[0], -1), local_arr, actions_arr]
                elif CFG.ebm_input_mode == 'full-image-action':
                    print('imagizing local')
                    img_arr = np.concatenate((img_arr, imagize_local(img_arr.shape, local_arr)[...,None]), axis=3)
                    print('imagized; standardizing img')
                    img_arr, mean, std = standardize(img_arr)
                    img_shape = img_arr.shape[1:]

                    # actions_arr = unnormalize_action(local_arr, actions_arr)
                    print('standardized; standardizing actions')
                    actions_arr, actions_mean, actions_std = standardize(actions_arr)
                    print('standardized; concatenating images')
                    x = np.c_[img_arr.reshape(img_arr.shape[0], -1), actions_arr]

                    # TODO: this is horrible and needs to be changed if it becomes part of the solution
                    ebm._HID_mean = mean
                    ebm._HID_std = std
                    ebm._HID_rows = rows
                    ebm._HID_cols = cols
                elif CFG.ebm_input_mode == 'full-image-only':
                    img_arr = np.concatenate((img_arr, 
                                              imagize_local(img_arr.shape, local_arr)[...,None], 
                                              imagize_action(img_arr.shape, local_arr, actions_arr)[...,None]), axis=3)
                    img_arr, mean, std = standardize(img_arr)
                    img_shape = img_arr.shape[1:]
                    x = img_arr.reshape(img_arr.shape[0], -1)

                    # TODO: this is horrible and needs to be changed if it becomes part of the solution
                    ebm._HID_mean = mean
                    ebm._HID_std = std
                    ebm._HID_rows = rows
                    ebm._HID_cols = cols
                elif CFG.ebm_input_mode == 'full-cspace-vector':
                    img_arr = make_cspace(img_arr)[...,None]
                    img_arr = standardize(img_arr)
                    img_shape = img_arr.shape[1:]

                    local_arr = standardize(local_arr)
                    actions_arr = standardize(actions_arr)
                    x = np.c_[img_arr.reshape(img_arr.shape[0], -1), local_arr, actions_arr]
                elif CFG.ebm_input_mode == 'full-cspace-action':
                    img_arr = np.concatenate((make_cspace(img_arr)[...,None], 
                                              imagize_local(img_arr.shape, local_arr)[...,None]), axis=3)
                    img_arr = standardize(img_arr)
                    img_shape = img_arr.shape[1:]

                    actions_arr = unnormalize_action(local_arr, actions_arr)
                    actions_arr = standardize(actions_arr)
                    x = np.c_[img_arr.reshape(img_arr.shape[0], -1), actions_arr]
                elif CFG.ebm_input_mode == 'full-cspace-only':
                    img_arr = np.concatenate((make_cspace(img_arr)[...,None], 
                                              imagize_local(img_arr.shape, local_arr)[...,None], 
                                              imagize_action(img_arr.shape, local_arr, actions_arr)[...,None]), axis=3)
                    img_arr = standardize(img_arr)
                    img_shape = img_arr.shape[1:]
                    x = img_arr.reshape(img_arr.shape[0], -1)
                else:
                    raise NotImplementedError('Please use a valid ebm_input_mode')
                # print(img_arr.max(axis=(0,1,2)), img_arr.min(axis=(0,1,2)))
                # img_arr_pos = img_arr[rewards_arr > 0]
                # img_arr_neg = img_arr[rewards_arr == 0]
                # print(f'pos: {img_arr_pos.shape[0]}, neg {img_arr_neg.shape[0]}')

                # action_mask = img_arr[:, :, :, 2] == 1
                # print(img_arr[:,:,:,0][action_mask].shape)

                # action_mask_pos = img_arr_pos[:, :, :, 2] == 1
                # # collisions_pos = img_arr_pos[:,:,:,0][action_mask_pos].mean()
                # collisions_pos = (img_arr_pos[:, :, :, 0] * action_mask_pos).max(axis=(1,2)).mean()
                # action_mask_neg = img_arr_neg[:,:,:,2] == 1
                # # collisions_neg = img_arr_neg[:,:,:,0][action_mask_neg].mean()
                # collisions_neg = (img_arr_neg[:, :, :, 0] * action_mask_neg).max(axis=(1,2)).mean()
                # print(f'C-space collisions. pos: {collisions_pos}, neg: {collisions_neg}')

                # collisions_pos_imgs = img_arr_pos[(img_arr_pos[:, :, :, 0] * action_mask_pos).max(axis=(1,2)).astype(bool)]

                # matplotlib.use('TkAgg')
                # for i in range(10):
                #     plt.figure()
                #     plt.imshow(collisions_pos_imgs[10 * i])
                #     plt.figure()
                #     plt.imshow(collisions_pos_imgs[10 * i, :, :, 0])
                #     plt.figure()
                #     plt.imshow(collisions_pos_imgs[10 * i, :, :, 1])
                #     plt.figure()
                #     plt.imshow(collisions_pos_imgs[10 * i, :, :, 2])
                #     plt.show()
                # exit()
                # if not CFG.use_full_state:
                #     # states_arr = states_arr[:, 150*150*3:]
                #     states_arr = states_arr[:, 20*20*3:]
                #     # next_states_arr = next_states_arr[:, 20*20*3:]
                logging.info(f'{x.shape}')
                if len(rewards_arr) > 0:
                    logging.info(f"\n{nsrt.name} success rate: {rewards_arr.mean()/self._reward_scale}")
        ###########################################
                # if len(rewards) > 0:
                #     logging.info(f"\n{nsrt.name} success rate: {sum(rewards)/len(rewards)/self._reward_scale}")
                    # if ebm._x_dim != -1:
                    #     positive_samples = ebm.predict_probas(np.c_[states, actions]) > 0.5
                    #     logging.info(f"\tpredicted successes: {positive_samples.mean()}")
                assert terminals_arr[rewards_arr == 0].all(), 'all r=0 should be terminal {}'.format(terminals_arr[rewards_arr == 0])

                # next_v = np.zeros(states_arr.shape[0])
                next_v = np.zeros(x.shape[0])
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
                train_data_mask = train_task_idxs_arr < int(0.95 * CFG.num_train_tasks)

                x_train = x[train_data_mask]
                target_train = target[train_data_mask]
                local_train = local_arr[train_data_mask]
                x_val = x[~train_data_mask]
                target_val = target[~train_data_mask]
                del x
                if True:#((target_train > 0).any() and (target_train == 0).any()):#nsrt.name == 'NavigateTo':#
                    logging.info(f"Train samples: {x_train.shape[0]}, validation samples: {x_val.shape[0]}")
                    ebm.train()
                    if self._ebm_class == 'diff':
                        assert CFG.ebm_input_mode == 'full-image-action'
                        y_train = x_train[target_train > 0, -actions_arr.shape[1]:]
                        x_train = x_train[target_train > 0, :-actions_arr.shape[1]]
                        local_train = local_train[target_train > 0]
                        ebm.fit(x_train, y_train, imagize_action_torch, local_train, actions_mean, actions_std, img_shape)
                        # if CFG.ebm_input_mode == 'full-image-only':
                        #     x_train = x_train[target_train > 0]
                        #     ebm.fit(x_train, img_shape)
                        # else:
                        #     y_train = x_train[target_train > 0, -actions_arr.shape[1]:]
                        #     x_train = x_train[target_train > 0, :-actions_arr.shape[1]]
                        #     ebm.fit(x_train, y_train, img_shape)
                        # ebm.fit(x_train[target_train > 0, :-actions_arr.shape[1]], x_train[target_train > 0, -actions_arr.shape[1]:], x_train[target_train == 0, :-actions_arr.shape[1]], x_train[target_train == 0, -actions_arr.shape[1]:])
                    else:
                        ebm.fit(x_train, target_train, x_val, target_val, img_shape)
                    # print(ebm._x_dim, ebm._input_scale.shape, ebm._input_shift.shape, ebm._output_scale.shape, ebm._output_shift.shape)
                    ebm.eval()
                    torch.save({
                                'state_dict': ebm.state_dict(),
                                'x_dim': ebm._x_dim if hasattr(ebm, 'x_dim') else None,
                                'y_dim': ebm._y_dim if hasattr(ebm, 'y_dim') else None,
                                't_dim': ebm._t_dim if hasattr(ebm, 't_dim') else None,
                                'input_scale': ebm._input_scale if hasattr(ebm, 'input_scale') else None,
                                'input_shift': ebm._input_shift if hasattr(ebm, 'input_shift') else None, 
                                'output_scale': ebm._output_scale if hasattr(ebm, '_output_scale') else None,
                                'output_shift': ebm._output_shift if hasattr(ebm, '_output_shift') else None,
                               },
                               f'models_obs/{nsrt.name}{self._ebm_class}_obs_{CFG.ebm_input_mode}_myopic_True_dropout_False_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt')
                    exit()
                    # ##### Changing for model load from file #####
                    # ebm.train()
                    # idx_positive = target_train > 0
                    # idx_negative = target_train == 0
                    # assert CFG.ebm_input_mode == 'full-image-action'
                    # if CFG.ebm_input_mode == 'full-image-action':
                    #     y_train = x_train[target_train > 0, -actions_arr.shape[1]:]
                    #     x_train = x_train[target_train > 0, :-actions_arr.shape[1]]
                    #     x_dummy = x_train[:1]
                    #     y_dummy = y_train[:1]
                    #     local_dummy = local_train[:1]
                    #     ebm.fit(x_dummy, y_dummy, imagize_action_torch, local_dummy, actions_mean, actions_std, img_shape)
                    # elif CFG.ebm_input_mode == 'full-image-only':
                    #     x_dummy = x_train[idx_positive][:1]
                    #     ebm.fit(x_dummy, img_shape)
                    # else:
                    #     x_dummy = np.r_[x_train[idx_positive][:1], x_train[idx_negative][:1]]
                    #     target_dummy = np.r_[target_train[idx_positive][:1], target_train[idx_negative][:1]]
                    #     ebm.fit(x_dummy[target_dummy > 0, :-actions_arr.shape[1]], x_dummy[target_dummy > 0, -actions_arr.shape[1]:])
                    # # ebm.fit(x_dummy, target_dummy)
                    # ebm.eval()
                    # model_state = torch.load(f'models_obs/{nsrt.name}{self._ebm_class}_obs_{CFG.ebm_input_mode}_myopic_True_dropout_False_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt')
                    # ebm.load_state_dict(model_state['state_dict'])
                    # ebm._x_dim = model_state['x_dim']
                    # ebm._input_scale = model_state['input_scale']
                    # ebm._input_shift = model_state['input_shift']
                    # ebm._output_scale = model_state['output_scale']
                    # ebm._output_shift = model_state['output_shift']
                    # break
                    # #############################################
                        
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
                # exit()

        # exit()

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
        if isinstance(self._ebm, CNNDiffusionRegressor):
            # assert CFG.ebm_input_mode == 'full-image-only'
            assert CFG.ebm_input_mode == 'full-image-action'
            img_arr = x[:150*150*3].reshape(150, 150, 3) / 255
            img_arr = img_arr[self._ebm._HID_rows < 1]
            img_arr = img_arr[:, self._ebm._HID_cols < 1]
            n_rows = img_arr.shape[0]
            n_cols = img_arr.shape[1]
            if n_rows < n_cols:
                img_arr = np.concatenate((img_arr, np.zeros((n_cols - n_rows, n_cols, img_arr.shape[2]))), axis=0)
            elif n_cols < n_rows:
                img_arr = np.concatenate((img_arr, np.zeros((n_rows, n_rows - n_cols, img_arr.shape[2]))), axis=1)
            local_arr = x[150*150*3:]
            def imagize_local(img_shape, local):
                x,y = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0])[::-1])
                x = x * 20 / img_shape[1]
                y = y * 20 / img_shape[0]

                obj_x = local[0]
                obj_y = local[1]
                obj_w = local[2]
                obj_h = local[3]
                obj_yaw = local[4]

                rect = utils.Rectangle(obj_x, obj_y, obj_w, obj_h, obj_yaw)
                obj_img = rect.contains_point_vec(x, y)
                local_img = obj_img 
                return local_img

            # def imagize_action(img_shape, local, actions):
            #     obj_x = local[:, 0]
            #     obj_y = local[:, 1]
            #     obj_w = local[:, 2]
            #     obj_h = local[:, 3]
            #     obj_yaw = local[:, 4]

            #     offset_x = actions[:, 0]
            #     offset_y = actions[:, 1]
            #     pos_x = obj_x + obj_w * offset_x * np.cos(obj_yaw) - \
            #             obj_h * offset_y * np.sin(obj_yaw)
            #     pos_y = obj_y + obj_w * offset_x * np.sin(obj_yaw) + \
            #             obj_h * offset_y * np.cos(obj_yaw)

            #     pos_x = np.clip(pos_x, 0, 20 - 1e-6)
            #     pos_y = np.clip(pos_y, 0, 20 - 1e-6)

            #     pos_x = (pos_x * img_shape[2] / 20).astype(int)
            #     pos_y = img_shape[1] - 1 - (pos_y * img_shape[1] / 20).astype(int)
            #     action_img = np.zeros(img_shape[:3])
            #     action_img[np.arange(img_shape[0]), pos_y, pos_x] = 1
            #     return action_img

            img_arr = np.concatenate((img_arr, 
                                      imagize_local(img_arr.shape, local_arr)[...,None]), axis=2)
            # img_arr = (img_arr - self._ebm._HID_mean[:-1]) / self._ebm._HID_std[:-1]
            img_arr = (img_arr - self._ebm._HID_mean) / self._ebm._HID_std
            
            x = img_arr.reshape(-1)
            params = np.array(self._ebm.predict_sample(x, local_arr, rng),
                              dtype=self._param_option.params_space.dtype).squeeze()
            params = np.random.randn(*params.shape).astype(params.dtype)

            # pos_x, pos_y = params
            # obj_x = local_arr[0]
            # obj_y = local_arr[1]
            # obj_w = local_arr[2]
            # obj_h = local_arr[3]
            # obj_yaw = local_arr[4]

            # offset_x = ((pos_x - obj_x) * np.cos(obj_yaw) + (pos_y - obj_y) * np.sin(obj_yaw)) / obj_w
            # offset_y = ((pos_y - obj_y) * np.cos(obj_yaw) - (pos_x - obj_x) * np.sin(obj_yaw)) / obj_h

            # params = np.array((offset_x, offset_y), dtype=self._param_option.params_space.dtype)
        # print(params)
        # exit()
        print(params)
        return params

