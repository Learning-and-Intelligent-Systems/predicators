import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, cast

import numpy as np
from scipy.special import logsumexp
import torch
from gym.spaces import Box

from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.explorers import create_explorer
from predicators.ground_truth_nsrts import get_gt_nsrts, _get_options_by_names
from predicators.settings import CFG
from predicators.structs import Array, Dataset, GroundAtom, GroundAtomsHoldQuery, \
    GroundAtomsHoldResponse, InteractionRequest, InteractionResult, \
    LowLevelTrajectory, NSRT, NSRTSampler, Object, ParameterizedOption, PNAD, \
    Predicate, Query, State, Task, Type, Variable, VarToObjSub
from predicators.ml_models import BinaryCNNEBM, BinaryEBM, MLPBinaryClassifier, \
    NeuralGaussianRegressor, DiffusionRegressor, CNNDiffusionRegressor
from predicators.nsrt_learning.nsrt_learning_main import _learn_pnad_samplers
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.sampler_learning import _LearnedSampler
from predicators import utils
from predicators.envs import get_or_create_env
from multiprocessing import Pool

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

class NSRTSamplerLearningApproach(BilevelPlanningApproach):
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
        self._ebm_class = CFG.ebm_class

    @classmethod
    def get_name(cls) -> str:
        return "nsrt_learning_sampler_only"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._nsrts

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
        
        ####################### Store data
        # for t in trajectories:
        #     for a in t.actions:
        #         a.option_name = a.get_option().name
        #         a.option_objs = a.get_option().objects
        #         a.option_params = a.get_option().params
        #         del a._option

        # torch.save({'trajectories': trajectories, 'train_task_idx': self._request_task_idx}, f'data_{CFG.env}/data/50ksplit/{CFG.data_collection_process_idx:02}_50ktasks_trajectories.pt')
        # exit()
        ######################

        if CFG.learn_or_eval == 'learn':
            ###################### Train models
            # Load data form file
            data = torch.load(f'data_{CFG.env}/data/50ktasks_trajectories.pt')
            trajectories = data['trajectories']
            train_task_idx = data['train_task_idx']

            # Trim trajectories
            if CFG.num_train_tasks < 50000:
                mask = train_task_idx < 2 * CFG.num_train_tasks
                train_task_idx_tmp = np.array(train_task_idx)[mask]
                max_task_idx = np.unique(train_task_idx_tmp)[CFG.num_train_tasks - 1]
            else:
                max_task_idx = 50000
            mask = train_task_idx < max_task_idx
            trajectories = [trajectories[i] for i in range(len(trajectories)) if mask[i]]
            for traj in trajectories:
                for s, a in zip(traj.states, traj.actions):
                    param_opt = _get_options_by_names(CFG.env, [a.option_name])[0]
                    a._option = param_opt.ground(a.option_objs, a.option_params)
                    a._option.initiable(s)

            # Apply predicates to data, producing a dataset of abstract states.
            ground_atom_dataset = utils.create_ground_atom_dataset(
                trajectories, self._get_current_predicates())

            assert CFG.segmenter == 'option_changes'
            assert CFG.return_learned_sampler
            segmented_trajs = [segment_trajectory(traj) for traj in ground_atom_dataset]
            segments = [seg for segs in segmented_trajs for seg in segs]
            pnads = [PNAD(nsrt.op, [], (nsrt.option, nsrt.option_vars)) for nsrt in self._nsrts]

            for seg in segments:
                seg_opt = seg.get_option()
                for pnad in pnads:
                    pnad_opt, pnad_opt_vars = pnad.option_spec
                    if seg_opt.name == pnad_opt.name and all(o.is_instance(v.type) for o, v in zip(seg_opt.objects, pnad_opt_vars)):
                        # print(seg.add_effects)
                        # print(pnad.op.add_effects)
                        # print()
                        # print(seg.delete_effects)
                        # print(pnad.op.delete_effects)
                        # print()
                        # print(seg_opt.parent)
                        # print(pnad_opt)
                        # print()
                        # print(seg_opt.objects)
                        # print(pnad_opt_vars)
                        # print()

                        # suc, ent_to_ent_sub = utils.unify_preconds_effects_options(
                        #     frozenset(),
                        #     frozenset(),
                        #     frozenset(seg.add_effects),
                        #     frozenset(pnad.op.add_effects),
                        #     frozenset(seg.delete_effects),
                        #     frozenset(pnad.op.delete_effects),
                        #     seg_opt.parent,
                        #     pnad_opt,
                        #     tuple(seg_opt.objects),
                        #     tuple(pnad_opt_vars))
                        # assert suc
                        # sub = cast(VarToObjSub,
                        #            {v: o
                        #             for o, v in ent_to_ent_sub.items()})

                        sub: VarToObjSub = {}
                        for o, v in zip(seg_opt.objects, pnad_opt_vars):
                            assert o.type == v.type, f'{v}, {o}'
                            assert v not in sub or sub[v] == o, f'{v}, {o}'
                            sub[v] = o

                        pnad.add_to_datastore((seg, sub), check_effect_equality=False)
            sampler_objs = _learn_pnad_samplers(pnads, CFG.sampler_learner)
            for pnad in pnads:
                for pnad, sampler in zip(pnads, sampler_objs):
                    torch.save({
                        'regressor_state_dict': sampler._regressor.state_dict(),
                        'regressor_x_dim': sampler._regressor._x_dim,
                        'regressor_y_dim': sampler._regressor._y_dim,
                        'regressor_input_scale': sampler._regressor._input_scale,
                        'regressor_input_shift': sampler._regressor._input_shift,
                        'regressor_output_scale': sampler._regressor._output_scale,
                        'regressor_output_shift': sampler._regressor._output_shift,
                        #
                        'classifier_state_dict': sampler._classifier.state_dict(),
                        'classifier_x_dim': sampler._classifier._x_dim,
                        'classifier_input_scale': sampler._classifier._input_scale,
                        'classifier_input_shift': sampler._classifier._input_shift
                        },
                        f'models_{CFG.env}/{pnad.op.name}_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks_gaussian.pt')
            exit()
            #######################
        elif CFG.learn_or_eval == 'eval':
            ###################### Load models
            assert CFG.sampler_learning_regressor_model == 'neural_gaussian'
            new_nsrts = []
            for nsrt in self._nsrts:
                models = torch.load(f'models_{CFG.env}/{nsrt.op.name}_{CFG.num_train_tasks if CFG.num_train_tasks < 1000 else (str(CFG.num_train_tasks//1000) +"k")}tasks_gaussian.pt')
                
                regressor: DistributionRegressor = NeuralGaussianRegressor(
                    seed=CFG.seed,
                    hid_sizes=CFG.neural_gaus_regressor_hid_sizes,
                    max_train_iters=CFG.neural_gaus_regressor_max_itr,
                    clip_gradients=CFG.mlp_regressor_clip_gradients,
                    clip_value=CFG.mlp_regressor_gradient_clip_value,
                    learning_rate=CFG.learning_rate)
                classifier = MLPBinaryClassifier(
                    seed=CFG.seed,
                    balance_data=CFG.mlp_classifier_balance_data,
                    max_train_iters=CFG.sampler_mlp_classifier_max_itr,
                    learning_rate=CFG.learning_rate,
                    n_iter_no_change=CFG.mlp_classifier_n_iter_no_change,
                    hid_sizes=CFG.mlp_classifier_hid_sizes,
                    n_reinitialize_tries=CFG.sampler_mlp_classifier_n_reinitialize_tries,
                    weight_init="default")
                regressor._x_dim = models['regressor_x_dim']
                regressor._y_dim = models['regressor_y_dim']
                regressor._input_scale = models['regressor_input_scale']
                regressor._input_shift = models['regressor_input_shift']
                regressor._output_scale = models['regressor_output_scale']
                regressor._output_shift = models['regressor_output_shift']
                regressor._initialize_net()
                regressor.load_state_dict(models['regressor_state_dict'])

                classifier._x_dim = models['classifier_x_dim']
                classifier._input_scale = models['classifier_input_scale']
                classifier._input_shift = models['classifier_input_shift']
                classifier._initialize_net()
                classifier.load_state_dict(models['classifier_state_dict'])

                sampler = _LearnedSampler(classifier, regressor, nsrt.parameters, nsrt.option).sampler
                new_nsrts.append(NSRT(
                                nsrt.name,
                                nsrt.parameters,
                                nsrt.preconditions,
                                nsrt.add_effects,
                                nsrt.delete_effects,
                                nsrt.ignore_effects,
                                nsrt.option,
                                nsrt.option_vars,
                                sampler))



            self._nsrts = new_nsrts
            logging.info("\nLearned NSRTs:")
            for nsrt in self._nsrts:
                logging.info(nsrt)
            logging.info("")
