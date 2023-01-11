import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np
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
from predicators.ml_models import BinaryCNNEBM, BinaryEBM, MLPBinaryClassifier, NeuralGaussianRegressor
from predicators import utils
from predicators.envs import get_or_create_env

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
        self._X_replay = []
        self._Y_replay = []
        for nsrt in self._nsrts:
            if CFG.use_ebm:
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

            self._X_replay.append([])
            self._Y_replay.append([])

        self._nsrts = new_nsrts

    def load(self, online_learning_cycle: Optional[int]) -> None:
        raise NotImplementedError

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
        for train_task_idx in self._select_interaction_train_task_idxs():
            # Determine the action policy and termination function.
            act_policy, termination_fn, skeleton = explorer.get_exploration_strategy(
                train_task_idx, CFG.timeout)
            # Determine the query policy.
            # query_policy = self._create_goal_query_policy(train_task_idx)
            query_policy = self._create_all_query_policy()
            request = InteractionRequest(train_task_idx, act_policy,
                                         query_policy, termination_fn, skeleton)
            requests.append(request)
        assert len(requests) == CFG.interactive_num_requests_per_cycle
        return requests
    
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

        for result in results:
            # final_response = result.responses[-1]
            # satisficing_trajectory = all(final_response.holds.values())
            # state_annotations = [satisficing_trajectory for _ in range(len(result.states))]

            state_annotations = []
            atoms_prev_state = set(atom for atom, value in result.responses[0].holds.items() if value)
            for response_state, action in zip(result.responses[1:], result.skeleton):
                atoms_state = set(atom for atom, value in response_state.holds.items() if value)
                expected_atoms = (atoms_prev_state | action.add_effects) - action.delete_effects
                # state_annotations.append(expected_atoms == atoms_state)
                # TODO: the line below is closer to what we should do for 
                # efficiency according to l561 in planning.py, but that
                # requires looking at the state instead of these responses,
                # and checking whether we need to get rid of the queries 
                # altogether
                # state_annotations.append(all(a.holds(response_state) for a in expected_atoms))
                state_annotations.append(expected_atoms.issubset(atoms_state))
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

    def _update_samplers(self, trajectories: List[LowLevelTrajectory], annotations_list: List[Any], skeletons: List[Any]) -> None:
        """Learns the sampler in a self-supervised fashion."""
        logging.info("\nUpdating the samplers...")
        if self._online_learning_cycle == 0:
            self._initialize_ebm_samplers()


        env = get_or_create_env(CFG.env)
        nsrt_names = [nsrt.name for nsrt in self._nsrts]
        num_nsrts = len(self._nsrts)

        if CFG.use_ebm:
            loop_generator = zip(self._ebms, self._nsrts, self._X_replay, self._Y_replay)
        else:
            loop_generator = zip(self._gaussians, self._classifiers, self._nsrts, self._X_replay, self._Y_replay)

        for loop_variables in loop_generator:
            if CFG.use_ebm:
                ebm, nsrt, X_replay, Y_replay = loop_variables
            else:
                gaussian, classifier, nsrt, X_replay, Y_replay = loop_variables

            X = []
            Y = []
            X_replay_new = []
            Y_replay_new = []
            for traj, annotations, skeleton in zip(trajectories, annotations_list, skeletons):
                for t, (state, action, annotation, ground_nsrt) in enumerate(zip(traj.states, traj.actions, annotations, skeleton)):
                    # Assume there's a single sampler per option
                    option = action.get_option()
                    if nsrt.option.name == option.name:
                        x = state.vec(ground_nsrt.objects)
                        a = option.params
                        if CFG.use_full_state:
                            # The full state is represented as the image observation of the env
                            img = env.render_state(state, None)[0][:,:,:3].mean(axis=2).reshape(-1)
                            x = np.r_[img, x]
                        if CFG.use_skeleton_state:
                            # The skeleton representation is a series of self._horizon one-hot vectors
                            # indicating which action is executed, plus a series of self._horizon * num_actions
                            # vectors, where the chosen action per step contains the object features of the 
                            # operator objects, while the other actions contain all-zeros
                            skeleton_rep = np.zeros(0)
                            for t_prime in range(t+1, t+self._horizon):
                                one_hot = np.zeros(num_nsrts)
                                if t_prime < len(skeleton):
                                    one_hot[nsrt_names.index(skeleton[t_prime].name)] = 1
                                nsrt_object_rep = np.zeros(0)
                                for nsrt_tmp in self._nsrts:
                                    if t_prime < len(skeleton) and nsrt_tmp.name == skeleton[t_prime].name:
                                        rep = state.vec(skeleton[t_prime].objects)
                                        assert state.vec(skeleton[t_prime].objects).shape[0] == sum(obj.type.dim for obj in nsrt_tmp.parameters), f'{state.vec(skeleton[t_prime].objects).shape[0]}, {sum(obj.type.dim for obj in nsrt_tmp.parameters)}, {nsrt_tmp.name}, {skeleton[t_prime].objects}, {nsrt_tmp.parameters}'
                                    else:
                                        rep = np.zeros(sum(obj.type.dim for obj in nsrt_tmp.parameters))
                                    nsrt_object_rep = np.r_[nsrt_object_rep, rep]
                                skeleton_rep = np.r_[skeleton_rep, one_hot, nsrt_object_rep]
                            x = np.r_[x, skeleton_rep]
                        
                        assert len(X) == 0 or np.r_[x, a].shape == X[-1].shape, f'{np.r_[x,a].shape}, {X[-1].shape}'
                        X.append(np.r_[x, a])
                        r = annotations[t + min(len(traj.actions) - t - 1, self._horizon - 1)]
                        Y.append(r)

                        if not annotation:
                            # Store only single-step failed actions, because those are always on-policy
                            X_replay_new.append(np.r_[x, a])
                            Y_replay_new.append(annotation)
            if len(X) > 0 and 0 < sum(Y + Y_replay) < len(Y + Y_replay):
                X_arr = np.array(X + X_replay)
                Y_arr = np.array(Y + Y_replay)
                print(X_arr.shape, Y_arr.shape)
                print('Training data success for', nsrt.name, ':', Y_arr.mean())
                if CFG.use_ebm:
                    ebm.fit(X_arr, Y_arr)
                else:
                    gaussian.fit(X_arr[Y_arr == 1, :x.shape[0]], X_arr[Y_arr == 1, x.shape[0]:])
                    classifier.fit(X_arr, Y_arr)

                # norm_X = (X_arr - ebm._input_shift) / ebm._input_scale

                # import torch
                # Y_hat = ebm(torch.tensor(norm_X.astype(np.float32)).to(ebm._device)).detach().cpu().numpy()
                # print('accuracy: ', (Y_arr == (Y_hat >= 0)).mean())
                # import matplotlib
                # matplotlib.use('TkAgg')
                # import matplotlib.pyplot as plt
                # plt.scatter(X_arr[Y_arr == 0, -2], X_arr[Y_arr == 0, -1], c='r')
                # plt.scatter(X_arr[Y_arr == 1, -2], X_arr[Y_arr == 1, -1], c='b')
                # plt.figure()
                # plt.scatter(X_arr[Y_hat < 0, -2], X_arr[Y_hat < 0, -1], c='r')
                # plt.scatter(X_arr[Y_hat >= 0, -2], X_arr[Y_hat >= 0, -1], c='b')

                # samples = []
                # cnt = 0
                # for x in X_arr[::10]:
                #     samples.append(ebm.predict_sample(x[:-2], self._rng))
                #     cnt += 1
                #     # if cnt == 100:
                #     #     break
                # samples = np.array(samples)
                # plt.scatter(samples[:, 0], samples[:, 1], c='green', alpha=0.3)

                # norm_samples = (samples - ebm._input_shift[-2:]) / ebm._input_scale[-2:]
                # new_X = np.c_[norm_X[::10, :-2], norm_samples]
                # new_Yhat = ebm(torch.tensor(new_X.astype(np.float32)).to(ebm._device)).detach().cpu().numpy()
                # print('sample positive %:', (new_Yhat >= 0).mean())

                # plt.show()
                # exit()
            else:
                print('No data or success for', nsrt.name, ':', len(X), sum(Y))
            replay_capacity = 1000
            if len(X_replay) + len(X_replay_new) < replay_capacity:
                X_replay += X_replay_new
                Y_replay += Y_replay_new
            elif len(X_replay_new) < replay_capacity:
                num_keep_old = replay_capacity - len(X_replay_new)
                X_replay += X_replay_new
                Y_replay += Y_replay_new
                del X_replay[:-num_keep_old]
                del Y_replay[:-num_keep_old]
            else:
                X_replay += X_replay_new
                Y_replay += Y_replay_new
                del X_replay[:-len(X_replay_new)]
                del Y_replay[:-len(Y_replay_new)]

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
            img = env.render_state(state, None)[0][:,:,:3].mean(axis=2).reshape(-1)
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
            img = env.render_state(state, None)[0][:,:,:3].mean(axis=2).reshape(-1)
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