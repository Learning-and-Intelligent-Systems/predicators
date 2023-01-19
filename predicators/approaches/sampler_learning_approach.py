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
        self._gamma = 0.999
        self._reward_scale = CFG.sql_reward_scale

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

            self._replay.append(([], [], [], [], [], []))   # states_replay, actions_replay, rewards_replay, next_states_replay, next_ground_nsrts_replay, terminals_replay

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

    def _featurize_state(self, state, ground_nsrt, skeleton):
        x = state.vec(ground_nsrt.objects)
        if CFG.use_full_state:
            # The full state is represented as the image observation of the env
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
            for t in range(self._horizon):
                one_hot = np.zeros(num_nsrts)
                if t < len(skeleton):
                    one_hot[nsrt_names.index(skeleton[t].name)] = 1
                nsrt_object_rep = np.zeros(0)
                for nsrt_tmp in self._nsrts:
                    if t < len(skeleton) and nsrt_tmp.name == skeleton[t].name:
                        rep = state.vec(skeleton[t].objects)
                        assert state.vec(skeleton[t].objects).shape[0] == sum(obj.type.dim for obj in nsrt_tmp.parameters), f'{state.vec(skeleton[t_prime].objects).shape[0]}, {sum(obj.type.dim for obj in nsrt_tmp.parameters)}, {nsrt_tmp.name}, {skeleton[t_prime].objects}, {nsrt_tmp.parameters}'
                    else:
                        rep = np.zeros(sum(obj.type.dim for obj in nsrt_tmp.parameters))
                    nsrt_object_rep = np.r_[nsrt_object_rep, rep]
                skeleton_rep = np.r_[skeleton_rep, one_hot, nsrt_object_rep]
            x = np.r_[x, skeleton_rep]
        return x

    # TD-learning (SQL)
    def _update_samplers(self, trajectories: List[LowLevelTrajectory], annotations_list: List[Any], skeletons: List[Any]) -> None:
        """Learns the sampler in a self-supervised fashion."""
        logging.info("\nUpdating the samplers...")
        if self._online_learning_cycle == 0:
            self._initialize_ebm_samplers()

        env = get_or_create_env(CFG.env)

        for ebm, nsrt, replay in zip(self._ebms, self._nsrts, self._replay):
            states = []
            actions = []
            rewards = []
            next_states = []
            next_ground_nsrts = []
            terminals = []

            for traj, annotations, skeleton in zip(trajectories, annotations_list, skeletons):
                for t, (state, action, annotation, ground_nsrt, next_state, next_ground_nsrt) in enumerate(zip(traj.states[:-1], traj.actions, annotations, skeleton, traj.states[1:], (skeleton[1:] + [None]))):
                    # Assume there's a single sampler per option
                    option = action.get_option()
                    if nsrt.option.name == option.name:
                        x = self._featurize_state(state, ground_nsrt, skeleton[t + 1:])
                        a = option.params
                        if next_ground_nsrt is not None:
                            next_x = self._featurize_state(next_state, next_ground_nsrt, skeleton[t + 2:])
                        else:
                            next_x = np.empty(0)
                        
                        states.append(x)
                        actions.append(a)
                        rewards.append(annotations[t] * self._reward_scale)
                        next_states.append(next_x)
                        next_ground_nsrts.append(next_ground_nsrt)
                        terminals.append(next_ground_nsrt is None or not annotations[t])

            states_replay, actions_replay, rewards_replay, next_states_replay, next_ground_nsrts_replay, terminals_replay = replay
            states_replay += states
            actions_replay += actions
            rewards_replay += rewards
            next_states_replay += next_states
            next_ground_nsrts_replay += next_ground_nsrts
            terminals_replay += terminals

            if len(states_replay) > 0:
                ebm_target = ebm    # TODO: this should be a copy and done iteratively, but for now it'll do
                states_arr = np.array(states_replay)
                actions_arr = np.array(actions_replay)
                rewards_arr = np.array(rewards_replay, dtype=np.float32)
                next_states_arr = np.array(next_states_replay)
                terminals_arr = np.array(terminals_replay)

                if len(rewards) > 0:
                    print(nsrt.name, 'success rate:', sum(rewards)/len(rewards)/self._reward_scale)
                assert terminals_arr[rewards_arr == 0].all(), 'all r=0 should be terminal {}'.format(terminals_arr[rewards_arr == 0])

                next_v = np.zeros(states_arr.shape[0])
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
                x = np.c_[states_arr, actions_arr]
                print('max r:', rewards_arr.max())
                print('max v:', next_v.max())
                print('max target:', target.max())
                if (target > 0).any():
                    with torch.autograd.set_detect_anomaly(True):
                        ebm.fit(x, target)

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