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
from predicators.teacher import Teacher, TeacherInteractionMonitor
from predicators.approaches.sampler_learning_approach import _featurize_state

class SamplerLearningApproachMix(BilevelPlanningApproach):
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
        return "sampler_learning_mix"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._nsrts

    def _create_network(self):
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
                    timesteps=CFG.num_diffusion_steps,#100,
                    learning_rate=CFG.learning_rate
                )
            else:
                raise ValueError('Can only use ebm class ebm or diff')
        else:
            raise ValueError('This sampler approach only uses EBM/diffusion')
    
        return classifier

    def _initialize_ebm_samplers(self) -> None:
        new_nsrts = []
        if CFG.use_ebm:
            self._ebms = []
        else:
            self._gaussians = []
            self._classifiers = []

        self._option_needs_generic_sampler = {}
        self._generic_option_samplers = {}
        for nsrt in self._nsrts:
            if nsrt.option not in self._option_needs_generic_sampler:
                self._option_needs_generic_sampler[nsrt.option] = False
            else:
                self._option_needs_generic_sampler[nsrt.option] = True
                self._generic_option_samplers[nsrt.option] = self._create_network()


        for nsrt in self._nsrts:
            classifier = self._create_network()
            if self._option_needs_generic_sampler[nsrt.option]:
                generic_classifier = self._generic_option_samplers[nsrt.option]
            else:
                generic_classifier = None

            if CFG.mix_samplers_uniform:    # Uniformly choose between samplers
                choice_probabilities = None
                random_points_distance = None
            elif CFG.ebm_aux_training:#CFG.ebm_train_reconstruction:  # Use error signal from reconstruction to choose between samplers
                choice_probabilities = None
                if CFG.ebm_aux_training == 'reconstruct':
                    random_points_distance = torch.load(f'data_{CFG.env}/data/{nsrt.name}_distance_random_samples.pt')['dist']
                elif CFG.ebm_aux_training == 'geometry':
                    random_points_distance = torch.load(f'data_{CFG.env}/data/{nsrt.name}_distance_random_samples_geometry.pt')['dist']
                elif CFG.ebm_aux_training == 'geometry+':
                    random_points_distance = torch.load(f'data_{CFG.env}/data/{nsrt.name}_distance_random_samples_geometry+.pt')['norm_vec']
            elif CFG.mix_samplers_predictor:    # Use a trained predictor (on test data) to choose between samplers
                sampler_choice_data = torch.load(f'data_{CFG.env}/sampler_choices/{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks_sampler_choices_seed{CFG.seed}.pt')
                print({k: sampler_choice_data[k].shape for k in sampler_choice_data})
                X = sampler_choice_data['states']
                y = sampler_choice_data['sampler_idxs']
                mean = X.mean(axis=0)
                std = X.std(axis=0)
                std = np.where(std > 0, std, 1)
                X = (X - mean) / std

                tensor_X = torch.from_numpy(X).float().to('cuda' if CFG.use_cuda else 'cpu')
                tensor_y = torch.from_numpy(y).long().to('cuda' if CFG.use_cuda else 'cpu')
                net = torch.nn.Sequential(
                    torch.nn.Linear(X.shape[1], 256),
                    torch.nn.Dropout(0.5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.Dropout(0.5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, len(np.unique(y)))
                ).to('cuda' if CFG.use_cuda else 'cpu')
                net.train()
                data = torch.utils.data.TensorDataset(tensor_X, tensor_y)
                dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
                optimizer = torch.optim.Adam(net.parameters())
                for epoch in range(100):#range(CFG.sampler_mlp_classifier_max_itr):#
                    cum_loss = 0
                    for batch_X, batch_y in dataloader:
                        yhat = net(batch_X)
                        loss = torch.nn.functional.cross_entropy(yhat, batch_y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        cum_loss += loss.item() * batch_X.shape[0]
                    if epoch % 100 == 0:
                        logging.info(f'Sampler choice classifier, epoch: {epoch}, loss: {cum_loss / tensor_X.shape[0]}')
                net.to('cpu')
                net.eval()
                choice_probabilities = {
                    'mean': mean,
                    'std': std,
                    'net': net
                }
                random_points_distance = None
            else:   # Use frequency of each sampler (on test data) to choose between samplers
                sampler_choice_data = torch.load(f'data_{CFG.env}/sampler_choices/{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks_sampler_choices_seed{CFG.seed}.pt')
                idxs = sampler_choice_data['sampler_idxs']
                _, cnt = np.unique(idxs, return_counts=True)
                choice_probabilities = cnt / cnt.sum()
                random_points_distance = None



            new_sampler = _LearnedSampler(nsrt.name, classifier, generic_classifier, nsrt.parameters, nsrt.option, self._nsrts, self._horizon, nsrt.sampler, choice_probabilities, random_points_distance).sampler
            self._ebms.append(classifier)
            

            new_nsrts.append(NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions,
                                  nsrt.add_effects, nsrt.delete_effects,
                                  nsrt.ignore_effects, nsrt.option, 
                                  nsrt.option_vars, new_sampler))


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
            "partial_planning",
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
            query_policy = self._create_none_query_policy()
            requests.append(InteractionRequest(train_task_idx, act_policy, query_policy, termination_fn, skeleton))
            self._request_task_idx.append(train_task_idx)
            logging.info('Using Mix sampler learning, which collects a single sample for compatibility, but shouldnt be used for data collection in general')
            break
        # assert len(requests) == CFG.interactive_num_requests_per_cycle
        return requests
    
    def _create_none_query_policy(self) -> Callable[[State], Optional[Query]]:
        def _query_policy(s: State) -> Optional[GroundAtomsHoldQuery]:
            return None
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

        ##### Changing for data load from file #####
        for ebm, nsrt in zip(self._ebms, self._nsrts):
            ##### Changing for model load from file #####
            ebm.eval()
            logging.info(f'Loading {nsrt.name} model from file; this approach assumes pretrained nets')
            if CFG.num_diffusion_steps == 100:
                model_state = torch.load(f'models_{CFG.env}/{nsrt.name}{self._ebm_class}{f"_{CFG.ebm_aux_training}" if CFG.ebm_aux_training else ""}_obs_{CFG.use_full_state}_myopic_True_dropout_False_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt', map_location='cpu' if not CFG.use_cuda else None)
            else:
                # model_state = torch.load(f'models_{CFG.env}/{nsrt.name}{self._ebm_class}_{CFG.num_diffusion_steps}steps{f"_{CFG.ebm_aux_training}" if CFG.ebm_aux_training else ""}_obs_{CFG.use_full_state}_myopic_True_dropout_False_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt', map_location='cpu' if not CFG.use_cuda else None)
                model_state = torch.load(f'models_{CFG.env}/{nsrt.name}{self._ebm_class}_{CFG.num_diffusion_steps}steps{f"_distilled" if CFG.distill_steps else ""}{f"_{CFG.ebm_aux_training}" if CFG.ebm_aux_training else ""}_obs_{CFG.use_full_state}_myopic_True_dropout_False_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt', map_location='cpu' if not CFG.use_cuda else None)
            ebm._x_dim = model_state['x_dim']
            x_cond_dim = sum(len(p.type.feature_names) for p in nsrt.parameters)
            if CFG.use_full_state:
                x_cond_dim += 32
            ebm._t_dim = (x_cond_dim // 2) * 2
            ebm._y_dim = nsrt.option.params_space.shape[0] - 1 
            ebm._x_cond_dim = x_cond_dim
            # ebm._t_dim = model_state['t_dim']
            # ebm._y_dim = model_state['y_dim']
            ebm._input_scale = model_state['input_scale']
            ebm._input_shift = model_state['input_shift']
            ebm._output_scale = model_state['output_scale']
            ebm._output_shift = model_state['output_shift']
            ebm._y_aux_dim = model_state['y_aux_dim']
            ebm._output_aux_scale = model_state['output_aux_scale']
            ebm._output_aux_shift = model_state['output_aux_shift']
            ebm._y_out_dim = 1
            ebm.is_trained = True
            ebm._initialize_net()
            ebm.to(ebm._device)
            ebm.load_state_dict(model_state['state_dict'])
            #############################################

        for option in self._option_needs_generic_sampler:
            if self._option_needs_generic_sampler[option]:
                ebm = self._generic_option_samplers[option]
                ebm.eval()
                logging.info(f'Loading {option.name} model from file; this approach assumes pretrained nets')
                model_state = torch.load(f'models_{CFG.env}/{option.name}{self._ebm_class}{f"_{CFG.ebm_aux_training}" if CFG.ebm_aux_training else ""}_obs_{CFG.use_full_state}_myopic_True_dropout_False_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt', map_location='cpu' if not CFG.use_cuda else None)
                ebm._x_dim = model_state['x_dim']
                x_cond_dim = sum(len(t.feature_names) for t in option.types)
                if CFG.use_full_state:
                    x_cond_dim += 32
                ebm._t_dim = (x_cond_dim // 2) * 2
                ebm._y_dim = option.params_space.shape[0] - 1
                ebm._x_cond_dim = x_cond_dim
                # ebm._t_dim = model_state['t_dim']
                # ebm._y_dim = model_state['y_dim']
                ebm._input_scale = model_state['input_scale']
                ebm._input_shift = model_state['input_shift']
                ebm._output_scale = model_state['output_scale']
                ebm._output_shift = model_state['output_shift']
                ebm._y_aux_dim = model_state['y_aux_dim']
                ebm._output_aux_scale = model_state['output_aux_scale']
                ebm._output_aux_shift = model_state['output_aux_shift']
                ebm.is_trained = True
                ebm._initialize_net()
                ebm.to(ebm._device)
                ebm.load_state_dict(model_state['state_dict'])

        if CFG.mix_samplers_uniform:
            env = get_or_create_env(CFG.env)
            test_interaction_requests = self._get_test_interaction_requests(env)
            test_interaction_results, _ = self._generate_test_interaction_results(
                env, Teacher(env.get_test_tasks()), test_interaction_requests)

            traj_list = [LowLevelTrajectory(result.states, result.actions) for result in test_interaction_results]
            skeleton_list = [result.skeleton for result in test_interaction_results]
            for nsrt in self._nsrts:
                states = []
                sampler_idxs = []
                for traj, skeleton in zip(traj_list, skeleton_list):
                    for t, (state, action, ground_nsrt) in enumerate(zip(traj.states[:-1], traj.actions, skeleton)):
                        option = action.get_option()
                        if nsrt.option.name == option.name:
                            x = _featurize_state(([nsrt.name for nsrt in self._nsrts], [nsrt.parameters for nsrt in self._nsrts], self._horizon, state, ground_nsrt.objects, [s.name for s in skeleton[t + 1:]], [s.objects for s in skeleton[t + 1:]]))
                            idx = option.params[0]
                            states.append(x)
                            sampler_idxs.append(idx)
                states_arr = np.array(states)
                sampler_idxs_arr = np.array(sampler_idxs)
                torch.save({
                    'states': states_arr,
                    'sampler_idxs': sampler_idxs_arr
                    }, f'data_{CFG.env}/sampler_choices/{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks_sampler_choices_seed{CFG.seed}.pt')



    def _get_test_interaction_requests(self, env):
        requests = []
        # Create the explorer.
        explorer = create_explorer(
            "partial_planning",
            self._initial_predicates,
            self._initial_options,
            self._types,
            self._action_space,
            env.get_test_tasks(),
            self._get_current_nsrts(),
            self._option_model)
        # total_requests = CFG.num_test_tasks
        total_requests = len(env.get_test_tasks())
        logging.info(f"Collecting sampler data for {total_requests} tasks")
        for idx in range(total_requests):
            logging.info(f"\t{idx} / {total_requests}")
            # Determine the action policy and termination function.
            act_policy, termination_fn, skeleton = explorer.get_exploration_strategy(
                idx, CFG.timeout)
            # Determine the query policy.
            query_policy = self._create_none_query_policy()
            requests.append(InteractionRequest(idx, act_policy, query_policy, termination_fn, skeleton))
        return requests
    
    def _generate_test_interaction_results(self, env, teacher, requests):
        """Given a sequence of InteractionRequest objects, handle the requests and
        return a list of InteractionResult objects."""
        logging.info("Generating test interaction results...")
        results = []
        query_cost = 0.0
        total_requests = len(requests)
        for curr_request, request in enumerate(requests):
            logging.info(f"\t{curr_request} / {total_requests}")
            monitor = TeacherInteractionMonitor(request, teacher)
            traj, _ = utils.run_policy(
                request.act_policy,
                env,
                "test",
                request.train_task_idx,
                request.termination_function,
                max_num_steps=CFG.max_num_steps_interaction_request,
                exceptions_to_break_on={
                    utils.EnvironmentFailure, utils.OptionExecutionFailure,
                    utils.RequestActPolicyFailure
                },
                monitor=monitor)
            request_responses = monitor.get_responses()
            query_cost += monitor.get_query_cost()
            result = InteractionResult(traj.states, traj.actions,
                                       request_responses, request.skeleton)
            results.append(result)
        return results, query_cost


@dataclass(frozen=True, eq=False, repr=False)
class _LearnedSampler:
    """A convenience class for holding the models underlying a learned
    sampler."""
    _name: str
    _ebm: BinaryEBM
    _generic_ebm: BinaryEBM
    _variables: Sequence[Variable]
    _param_option: ParameterizedOption
    _nsrts: List[NSRT]
    _horizon: int
    _original_sampler: NSRTSampler
    _choice_probabilities: Array
    _random_aux_error: float

    def sampler(self, state: State, goal: Set[GroundAtom],
                rng: np.random.Generator, objects: Sequence[Object],
                skeleton: List[Any]) -> Array:
        """The sampler corresponding to the given models.

        May be used as the _sampler field in an NSRT.
        """
        assert self._ebm.is_trained and (self._generic_ebm is None or self._generic_ebm.is_trained)
        # if not self._ebm.is_trained:
        #     return self._original_sampler(state, goal, rng, objects, skeleton)
        x_lst: List[Any] = []  
        sub = dict(zip(self._variables, objects))
        for var in self._variables:
            x_lst.extend(state[sub[var]])
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
        
        if self._generic_ebm is None:
            num_samplers = 2
        else:
            num_samplers = 3
        # chosen_sampler_idx = rng.integers(num_samplers)

        if CFG.ebm_aux_training:#CFG.ebm_train_reconstruction:
            if CFG.ebm_aux_training == 'reconstruct':
                def _aux_error_fn(model, x):
                    a_multi = []
                    for _ in range(CFG.ebm_aux_n_samples):
                        a = np.array(model.predict_sample(x, rng),
                                     dtype=self._param_option.params_space.dtype)
                        a_multi.append(a)
                    a_multi = np.stack(a_multi, axis=0)
                    x_rep = np.repeat(x.reshape(1, -1), CFG.ebm_aux_n_samples, axis=0)
                    x_hat = model.predict_aux(x_rep, a_multi)
                    err_multi = np.linalg.norm(x_rep - x_hat, axis=1) ** 2
                    best_idx = err_multi.argmin()
                    a = a_multi[best_idx]
                    err = err_multi[best_idx, None]
                    return a, err
            elif CFG.ebm_aux_training == 'geometry':
                if self._name.startswith('NavigateTo'):
                    def _aux_error_fn(model, x):
                        obj_x = x[0]
                        obj_y = x[1]
                        obj_w = x[2]
                        obj_h = x[3]
                        obj_yaw = x[4]
                        rect = utils.Rectangle(obj_x, obj_y, obj_w, obj_h, obj_yaw)
                        a_multi = []
                        for _ in range(CFG.ebm_aux_n_samples):
                            a = np.array(model.predict_sample(x, rng),
                                         dtype=self._param_option.params_space.dtype)
                            a_multi.append(a)
                        a_multi = np.stack(a_multi, axis=0)
                        offset_x = a_multi[:, -2]
                        offset_y = a_multi[:, -1]
                        pos_x = obj_x + obj_w * offset_x * np.cos(obj_yaw) - \
                                obj_h * offset_y * np.sin(obj_yaw)
                        pos_y = obj_y + obj_w * offset_x * np.sin(obj_yaw) + \
                                obj_h * offset_y * np.cos(obj_yaw)
                        pos_x = np.clip(pos_x, 0, 20 - 1e-6)
                        pos_y = np.clip(pos_y, 0, 20 - 1e-6)
                        aux_multi = []
                        if self._name == 'NavigateToCup':
                            for i in range(CFG.ebm_aux_n_samples):
                                aux_multi.append(rect.line_segments[0].distance_nearest_point(pos_x[i], pos_y[i]))
                        elif self._name == 'NavigateToTray':
                            x1 = obj_x
                            y1 = obj_y
                            x2 = x1 + (obj_w - obj_h) * np.cos(obj_yaw)
                            y2 = y1 + (obj_w - obj_h) * np.sin(obj_yaw)
                            rect1 = utils.Rectangle(x1, y1, obj_h, obj_h, obj_yaw)
                            rect2 = utils.Rectangle(x2, y2, obj_h, obj_h, obj_yaw)
                            for i in range(CFG.ebm_aux_n_samples):
                                aux_multi.append(min(rect1.distance_nearest_point(pos_x[i], pos_y[i]),
                                                    rect2.distance_nearest_point(pos_x[i], pos_y[i])))
                        else:
                            for i in range(CFG.ebm_aux_n_samples):
                                aux_multi.append(rect.distance_nearest_point(pos_x[i], pos_y[i]))
                        aux_multi = np.array(aux_multi)[:, None]
                        x_rep = np.repeat(x.reshape(1, -1), CFG.ebm_aux_n_samples, axis=0)
                        aux_hat = model.predict_aux(x_rep, a_multi)
                        err_multi = (aux_hat - aux_multi) ** 2
                        best_idx = err_multi.argmin()
                        a = a_multi[best_idx]
                        err = err_multi[best_idx, None]
                        return a, err
                elif self._name.startswith('Pick'):
                    def _aux_error_fn(model, x):
                        obj_x = x[0]
                        obj_y = x[1]
                        obj_w = x[2]
                        obj_h = x[3]
                        obj_yaw = x[4]
                        rect = utils.Rectangle(obj_x, obj_y, obj_w, obj_h, obj_yaw)
                        a_multi = []
                        for _ in range(CFG.ebm_aux_n_samples):
                            a = np.array(model.predict_sample(x, rng),
                                         dtype=self._param_option.params_space.dtype)
                            a_multi.append(a)
                        a_multi = np.stack(a_multi, axis=0)
                        robby_x = x[6]
                        robby_y = x[7]
                        robby_yaw = x[8]
                        offset_gripper = a_multi[:, -2]
                        tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
                        tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)
                        aux_multi = []
                        for i in range(CFG.ebm_aux_n_samples):
                            aux_multi.append(rect.distance_nearest_point(tip_x[i], tip_y[i]))
                        aux_multi = np.array(aux_multi)[:, None]
                        x_rep = np.repeat(x.reshape(1, -1), CFG.ebm_aux_n_samples, axis=0)
                        aux_hat = model.predict_aux(x_rep, a_multi)
                        err_multi = (aux_hat - aux_multi) ** 2
                        best_idx = err_multi.argmin()
                        a = a_multi[best_idx]
                        err = err_multi[best_idx, None]
                        return a, err
                elif self._name.startswith('Place'):
                    def _aux_error_fn(model, x):
                        obj_x = x[6]
                        obj_y = x[7]
                        obj_w = x[8]
                        obj_h = x[9]
                        obj_yaw = x[10]
                        rect = utils.Rectangle(obj_x, obj_y, obj_w, obj_h, obj_yaw)
                        a_multi = []
                        for _ in range(CFG.ebm_aux_n_samples):
                            a = np.array(model.predict_sample(x, rng),
                                         dtype=self._param_option.params_space.dtype)
                            a_multi.append(a)
                        a_multi = np.stack(a_multi, axis=0)
                        robby_x = x[12]
                        robby_y = x[13]
                        robby_yaw = x[14]
                        offset_gripper = a_multi[:, -1]
                        tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
                        tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)
                        aux_multi = []
                        for i in range(CFG.ebm_aux_n_samples):
                            aux_multi.append(rect.distance_nearest_point(tip_x[i], tip_y[i]))
                        aux_multi = np.array(aux_multi)[:, None]
                        x_rep = np.repeat(x.reshape(1, -1), CFG.ebm_aux_n_samples, axis=0)
                        aux_hat = model.predict_aux(x_rep, a_multi)
                        err_multi = (aux_hat - aux_multi) ** 2
                        best_idx = err_multi.argmin()
                        a = a_multi[best_idx]
                        err = err_multi[best_idx, None]
                        return a, err
            elif CFG.ebm_aux_training == 'geometry+':
                if self._name.startswith('NavigateTo'):
                    def _aux_error_fn(model, x):
                        obj_x = x[0]
                        obj_y = x[1]
                        obj_w = x[2]
                        obj_h = x[3]
                        obj_yaw = x[4]
                        rect = utils.Rectangle(obj_x, obj_y, obj_w, obj_h, obj_yaw)
                        a_multi = []
                        for _ in range(CFG.ebm_aux_n_samples):
                            a = np.array(model.predict_sample(x, rng),
                                         dtype=self._param_option.params_space.dtype)
                            a_multi.append(a)
                        a_multi = np.stack(a_multi, axis=0)
                        offset_x = a_multi[:, -2]
                        offset_y = a_multi[:, -1]
                        pos_x = obj_x + obj_w * offset_x * np.cos(obj_yaw) - \
                                obj_h * offset_y * np.sin(obj_yaw)
                        pos_y = obj_y + obj_w * offset_x * np.sin(obj_yaw) + \
                                obj_h * offset_y * np.cos(obj_yaw)
                        pos_x = np.clip(pos_x, 0, 20 - 1e-6)
                        pos_y = np.clip(pos_y, 0, 20 - 1e-6)
                        if self._name == 'NavigateToTray':
                            x1 = obj_x
                            y1 = obj_y
                            x2 = x1 + (obj_w - obj_h) * np.cos(obj_yaw)
                            y2 = y1 + (obj_w - obj_h) * np.sin(obj_yaw)
                            rect1 = utils.Rectangle(x1, y1, obj_h, obj_h, obj_yaw)
                            rect2 = utils.Rectangle(x2, y2, obj_h, obj_h, obj_yaw)
                        aux_multi = np.empty((CFG.ebm_aux_n_samples, 7))
                        for i in range(CFG.ebm_aux_n_samples):
                            if self._name == 'NavigateToCup':
                                aux_multi[i, 0] = rect.line_segments[0].distance_nearest_point(pos_x[i], pos_y[i])
                            elif self._name == 'NavigateToTray':
                                aux_multi[i, 0] = min(rect1.distance_nearest_point(pos_x[i], pos_y[i]),
                                                        rect2.distance_nearest_point(pos_x[i], pos_y[i]))
                            else:
                                aux_multi[i, 0] = rect.distance_nearest_point(pos_x[i], pos_y[i])
                            aux_multi[i, 1], aux_multi[i, 2] = rect.nearest_point(pos_x[i], pos_y[i])
                            aux_multi[i, 3], aux_multi[i, 4] = pos_x[i], pos_y[i]
                            aux_multi[i, 5], aux_multi[i, 6] = rect.relative_reoriented_coordinates(pos_x[i], pos_y[i])
                        x_rep = np.repeat(x.reshape(1, -1), CFG.ebm_aux_n_samples, axis=0)
                        err_multi = model.aux_square_error(x_rep, a_multi, aux_multi)
                        best_idx = err_multi.sum(axis=1).argmin()
                        a = a_multi[best_idx]
                        err = err_multi[best_idx]
                        return a, err
                elif self._name.startswith('Pick'):
                    def _aux_error_fn(model, x):
                        book_x = x[0]
                        book_y = x[1]
                        book_w = x[2]
                        book_h = x[3]
                        book_yaw = x[4]
                        rect = utils.Rectangle(book_x, book_y, book_w, book_h, book_yaw)
                        a_multi = []
                        for _ in range(CFG.ebm_aux_n_samples):
                            a = np.array(model.predict_sample(x, rng),
                                         dtype=self._param_option.params_space.dtype)
                            a_multi.append(a)
                        a_multi = np.stack(a_multi, axis=0)
                        robby_x = x[6]
                        robby_y = x[7]
                        robby_yaw = x[8]
                        offset_gripper = a_multi[:, -2]
                        tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
                        tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)
                        aux_multi = np.empty((CFG.ebm_aux_n_samples, 4))
                        for i in range(CFG.ebm_aux_n_samples):
                            aux_multi[i, 0], aux_multi[i, 1] = tip_x[i], tip_y[i]
                            aux_multi[i, 2], aux_multi[i, 3] = rect.relative_reoriented_coordinates(tip_x[i], tip_y[i])
                        x_rep = np.repeat(x.reshape(1, -1), CFG.ebm_aux_n_samples, axis=0)
                        err_multi = model.aux_square_error(x_rep, a_multi, aux_multi)
                        best_idx = err_multi.sum(axis=1).argmin()
                        a = a_multi[best_idx]
                        err = err_multi[best_idx]
                        return a, err
                elif self._name.startswith('Place'):
                    def _aux_error_fn(model, x):
                        book_relative_x = x[0]
                        book_relative_y = x[1]
                        book_w = x[2]
                        book_h = x[3]
                        book_relative_yaw = x[4]

                        shelf_x = x[6]
                        shelf_y = x[7]
                        shelf_w = x[8]
                        shelf_h = x[9]
                        shelf_yaw = x[10]
                        shelf_rect = utils.Rectangle(shelf_x, shelf_y, shelf_w, shelf_h, shelf_yaw)
                        a_multi = []
                        for _ in range(CFG.ebm_aux_n_samples):
                            a = np.array(model.predict_sample(x, rng),
                                         dtype=self._param_option.params_space.dtype)
                            a_multi.append(a)
                        a_multi = np.stack(a_multi, axis=0)
                        robby_x = x[12]
                        robby_y = x[13]
                        robby_yaw = x[14]
                        offset_gripper = a_multi[:, -1]
                        tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
                        tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)
                        place_x = tip_x + book_relative_x * np.sin(
                            robby_yaw) + book_relative_y * np.cos(robby_yaw)
                        place_y = tip_y + book_relative_y * np.sin(
                            robby_yaw) - book_relative_x * np.cos(robby_yaw)
                        place_yaw = book_relative_yaw + robby_yaw
                        while place_yaw > np.pi:
                            place_yaw -= 2 * np.pi
                        while place_yaw < -np.pi:
                            place_yaw += 2 * np.pi
                        for i in range(CFG.ebm_aux_n_samples):
                            book_rect = utils.Rectangle(place_x[i], place_y[i], book_w, book_h, place_yaw)
                            com_x, com_y = book_rect.center
                            aux_multi = np.empty((CFG.ebm_aux_n_samples, 4))
                            aux_multi[i, 0], aux_multi[i, 1] = com_x, com_y
                            aux_multi[i, 2], aux_multi[i, 3] = shelf_rect.relative_reoriented_coordinates(com_x, com_y)
                        x_rep = np.repeat(x.reshape(1, -1), CFG.ebm_aux_n_samples, axis=0)
                        err_multi = model.aux_square_error(x_rep, a_multi, aux_multi)
                        best_idx = err_multi.sum(axis=1).argmin()
                        a = a_multi[best_idx]
                        err = err_multi[best_idx]
                        return a, err

            ebm_a, ebm_square_err = _aux_error_fn(self._ebm, x)
            original_err = np.sqrt(ebm_square_err.shape[0])    # This is because we'll be normalizing by the _random_aux_error vector
            ebm_err = np.sqrt(np.sum(ebm_square_err / (self._random_aux_error ** 2)))
    
            choice_probabilities = 1 / np.array([ebm_err + 1e-6, original_err])
            if num_samplers == 3:
                generic_ebm_a, generic_ebm_square_err = _aux_error_fn(self._generic_ebm, x)
                generic_ebm_err = np.sqrt(np.sum(generic_ebm_square_err / (self._random_aux_error ** 2)))
                choice_probabilities = np.r_[choice_probabilities, 1 / (generic_ebm_err + 1e-6)]
            choice_probabilities /= choice_probabilities.sum()
        elif CFG.mix_samplers_predictor:
            mean = self._choice_probabilities['mean']
            std = self._choice_probabilities['std']
            net = self._choice_probabilities['net']
            tensor_x = torch.from_numpy(((x - mean) / std ).reshape(1, -1)).float()
            choice_probabilities = net(tensor_x).softmax(dim=1).detach().numpy().squeeze()
        else:
            choice_probabilities = self._choice_probabilities

        chosen_sampler_idx = rng.choice(num_samplers, replace=False, p=choice_probabilities)
        if chosen_sampler_idx == 0:
            if CFG.ebm_aux_training is not None:# and CFG.ebm_aux_training.startswith('geometry'):
                params = ebm_a
            else:
                params = np.array(self._ebm.predict_sample(x, rng),
                                  dtype=self._param_option.params_space.dtype)
        elif chosen_sampler_idx == 1:
            params = self._original_sampler(state, goal, rng, objects, skeleton)[1:]
        else:
            if CFG.ebm_aux_training is not None:# and CFG.ebm_aux_training.startswith('geometry'):
                params = generic_ebm_a
            else:
                params = np.array(self._generic_ebm.predict_sample(x, rng),
                                  dtype=self._param_option.params_space.dtype)
        # print(params)
        # exit()
        return np.r_[chosen_sampler_idx, params]
