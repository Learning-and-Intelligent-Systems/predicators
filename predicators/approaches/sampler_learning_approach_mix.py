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
                    timesteps=100,
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
            new_sampler = _LearnedSampler(classifier, generic_classifier, nsrt.parameters, nsrt.option, self._nsrts, self._horizon, nsrt.sampler).sampler
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
            model_state = torch.load(f'models/{nsrt.name}{self._ebm_class}_obs_{CFG.use_full_state}_myopic_True_dropout_False_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt')
            ebm._x_dim = model_state['x_dim']
            x_cond_dim = sum(len(p.type.feature_names) for p in nsrt.parameters)
            if CFG.use_full_state:
                x_cond_dim += 32
            ebm._t_dim = (x_cond_dim // 2) * 2
            ebm._y_dim = nsrt.option.params_space.shape[0] - 1 
            # ebm._t_dim = model_state['t_dim']
            # ebm._y_dim = model_state['y_dim']
            ebm._input_scale = model_state['input_scale']
            ebm._input_shift = model_state['input_shift']
            ebm._output_scale = model_state['output_scale']
            ebm._output_shift = model_state['output_shift']
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
                model_state = torch.load(f'models/{option.name}{self._ebm_class}_obs_{CFG.use_full_state}_myopic_True_dropout_False_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks.pt')
                ebm._x_dim = model_state['x_dim']
                x_cond_dim = sum(len(t.feature_names) for t in option.types)
                if CFG.use_full_state:
                    x_cond_dim += 32
                ebm._t_dim = (x_cond_dim // 2) * 2
                ebm._y_dim = option.params_space.shape[0] - 1
                # ebm._t_dim = model_state['t_dim']
                # ebm._y_dim = model_state['y_dim']
                ebm._input_scale = model_state['input_scale']
                ebm._input_shift = model_state['input_shift']
                ebm._output_scale = model_state['output_scale']
                ebm._output_shift = model_state['output_shift']
                ebm.is_trained = True
                ebm._initialize_net()
                ebm.to(ebm._device)
                ebm.load_state_dict(model_state['state_dict'])

        env = get_or_create_env(CFG.env)
        test_interaction_requests = self._get_test_interaction_requests(env)
        test_interaction_results = self._generate_test_interaction_results(
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
                }, f'data/{nsrt.name}_full_obs_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks_sampler_choices.pt')

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
        total_requests = 1#CFG.num_test_tasks
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
                })
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
    _ebm: BinaryEBM
    _generic_ebm: BinaryEBM
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
        chosen_sampler_idx = rng.integers(num_samplers)
        if chosen_sampler_idx == 0:
            params = np.array(self._ebm.predict_sample(x, rng),
                              dtype=self._param_option.params_space.dtype)
        elif chosen_sampler_idx == 1:
            params = self._original_sampler(state, goal, rng, objects, skeleton)[1:]
        else:
            params = np.array(self._generic_ebm.predict_sample(x, rng),
                              dtype=self._param_option.params_space.dtype)
        # print(params)
        # exit()
        return np.r_[chosen_sampler_idx, params]
