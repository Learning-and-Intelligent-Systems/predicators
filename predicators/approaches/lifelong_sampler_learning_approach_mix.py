import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set
import copy
from collections import defaultdict
import time
import pickle as pkl

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

def _featurize_state(state, ground_nsrt_objects):
    assert not CFG.use_full_state
    assert not CFG.use_skeleton_state
    return state.vec(ground_nsrt_objects)

def _aux_labels(nsrt_name, x, a):
    if nsrt_name.startswith('NavigateTo'):
        obj_x = x[0]
        obj_y = x[1]
        obj_w = x[2]
        obj_h = x[3]
        obj_yaw = x[4]

        offset_x = a[-2]
        offset_y = a[-1]
        pos_x = obj_x + obj_w * offset_x * np.cos(obj_yaw) - \
                obj_h * offset_y * np.sin(obj_yaw)
        pos_y = obj_y + obj_w * offset_x * np.sin(obj_yaw) + \
                obj_h * offset_y * np.cos(obj_yaw)

        pos_x = np.clip(pos_x, 0, 20 - 1e-6)
        pos_y = np.clip(pos_y, 0, 20 - 1e-6)

        aux_labels = np.empty(7)
        rect = utils.Rectangle(obj_x, obj_y, obj_w, obj_h, obj_yaw)
        if nsrt_name == 'NavigateToCup':
            aux_labels[0] = rect.line_segments[0].distance_nearest_point(pos_x, pos_y)
        elif nsrt_name == 'NavigateToTray':
            x1 = obj_x
            y1 = obj_y
            x2 = x1 + (obj_w - obj_h) * np.cos(obj_yaw)
            y2 = y1 + (obj_w - obj_h) * np.sin(obj_yaw)
            rect1 = utils.Rectangle(x1, y1, obj_h, obj_h, obj_yaw)
            rect2 = utils.Rectangle(x2, y2, obj_h, obj_h, obj_yaw)
            aux_labels[0] = min(rect1.distance_nearest_point(pos_x, pos_y),
                                    rect2.distance_nearest_point(pos_x, pos_y))
        else:
            aux_labels[0] = rect.distance_nearest_point(pos_x, pos_y)
        aux_labels[1], aux_labels[2] = rect.nearest_point(pos_x, pos_y)
        aux_labels[3], aux_labels[4] = pos_x, pos_y
        aux_labels[5], aux_labels[6] = rect.relative_reoriented_coordinates(pos_x, pos_y)
    elif nsrt_name.startswith('Pick'):
        obj_x = x[0]
        obj_y = x[1]
        obj_w = x[2]
        obj_h = x[3]
        obj_yaw = x[4]

        robby_x = x[6]
        robby_y = x[7]
        robby_yaw = x[8]

        offset_gripper = a[-2]

        tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
        tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)

        aux_labels = np.empty(4)
        rect = utils.Rectangle(obj_x, obj_y, obj_w, obj_h, obj_yaw)
        aux_labels[0], aux_labels[1] = tip_x, tip_y
        aux_labels[2], aux_labels[3] = rect.relative_reoriented_coordinates(tip_x, tip_y)
    elif nsrt_name.startswith('Place'):
        obj_relative_x = x[0]
        obj_relative_y = x[1]
        obj_w = x[2]
        obj_h = x[3]
        obj_relative_yaw = x[4]

        target_x = x[6]
        target_y = x[7]
        target_w = x[8]
        target_h = x[9]
        target_yaw = x[10]

        robby_x = x[12]
        robby_y = x[13]
        robby_yaw = x[14]

        offset_gripper = a[-1]

        tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
        tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)

        place_x = tip_x + obj_relative_x * np.sin(
            robby_yaw) + obj_relative_y * np.cos(robby_yaw)
        place_y = tip_y + obj_relative_y * np.sin(
            robby_yaw) - obj_relative_x * np.cos(robby_yaw)
        place_yaw = obj_relative_yaw + robby_yaw

        aux_labels = np.empty(4)
        target_rect = utils.Rectangle(target_x, target_y, target_w, target_h, target_yaw)
        
        obj_yaw = place_yaw
        while obj_yaw > np.pi:
            obj_yaw -= (2 * np.pi)
        while obj_yaw < -np.pi:
            obj_yaw += (2 * np.pi)
        obj_rect = utils.Rectangle(place_x, place_y, obj_w, obj_h, obj_yaw)
        com_x, com_y = obj_rect.center

        aux_labels[0], aux_labels[1] = com_x, com_y
        aux_labels[2], aux_labels[3] = target_rect.relative_reoriented_coordinates(com_x, com_y)

    return aux_labels


class LifelongSamplerLearningApproachMix(BilevelPlanningApproach):
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
        self._next_train_task = 0
        assert CFG.timeout == float('inf'), "We don't want to let methods time out in these experiments"
        assert CFG.bookshelf_add_sampler_idx_to_params, "Code assumes the env expects one extra dummy param"
        self._save_dict = {}

    @classmethod
    def get_name(cls) -> str:
        return "lifelong_sampler_learning_mix"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._nsrts

    def _create_network(self) -> DiffusionRegressor:
        return DiffusionRegressor(
                    seed=CFG.seed,
                    hid_sizes=CFG.mlp_classifier_hid_sizes,
                    max_train_iters=CFG.sampler_mlp_classifier_max_itr,
                    timesteps=100,
                    learning_rate=CFG.learning_rate
               )

    def _initialize_ebm_samplers(self) -> None:
        new_nsrts = []
        self._ebms = []
        self._replay = []
        self._option_needs_generic_sampler = {}
        self._generic_option_samplers = {}
        for nsrt in self._nsrts:
            if nsrt.option not in self._option_needs_generic_sampler:
                self._option_needs_generic_sampler[nsrt.option] = False
            else:
                self._option_needs_generic_sampler[nsrt.option] = True
                self._generic_option_samplers[nsrt.option] = self._create_network()

        for nsrt in self._nsrts:
            states_replay = []
            actions_replay = []
            aux_labels_replay = []
            self._replay.append((states_replay, actions_replay, aux_labels_replay))   

            ebm = self._create_network()
            if self._option_needs_generic_sampler[nsrt.option]:
                generic_ebm = self._generic_option_samplers[nsrt.option]
            else:
                generic_ebm = None

            assert CFG.ebm_aux_training == 'geometry+'
            choice_probabilities = None
            random_points_distance = torch.load(f'data_{CFG.env}/data/{nsrt.name}_distance_random_samples_geometry+.pt')['norm_vec']
            new_sampler = _LearnedSampler(nsrt.name, ebm, generic_ebm, nsrt.parameters, nsrt.option, self._nsrts, nsrt.sampler, choice_probabilities, random_points_distance).sampler
            self._ebms.append(ebm)
            
            new_nsrts.append(NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions,
                                  nsrt.add_effects, nsrt.delete_effects,
                                  nsrt.ignore_effects, nsrt.option, 
                                  nsrt.option_vars, new_sampler))
        self._nsrts = new_nsrts

    def load(self, online_learning_cycle: Optional[int]) -> None:
        raise 'NotImplementedError'
        # TODO: I should probably implement checkpointing here

    def load_checkpoint(self) -> int:
        self._save_dict = torch.load(f"{CFG.results_dir}/{utils.get_config_path_str()}__checkpoint.pt")
        self._initialize_ebm_samplers()
        self._online_learning_cycle = 1 + max(nsrt_dict["online_learning_cycle"] for nsrt_dict in self._save_dict.values())
        
        def load_ebm(state_dict, ebm):
            ebm._input_scale = state_dict["input_scale"]
            ebm._input_shift = state_dict["input_shift"]
            ebm._output_scale = state_dict["output_scale"]
            ebm._output_shift = state_dict["output_shift"]
            ebm._output_aux_scale = state_dict["output_aux_scale"]
            ebm._output_aux_shift = state_dict["output_aux_shift"]
            ebm.is_trained = state_dict["is_trained"]
            ebm._x_cond_dim = state_dict["x_cond_dim"]
            ebm._t_dim = state_dict["t_dim"]
            ebm._y_dim = state_dict["y_dim"]
            ebm._x_dim = state_dict["x_dim"]
            ebm._y_aux_dim = state_dict["y_aux_dim"]
            ebm._initialize_net()
            ebm.to(ebm._device)
            ebm.load_state_dict(state_dict["model_state"])
            ebm._create_optimizer()
            ebm._optimizer.load_state_dict(state_dict["optimizer_state"])

        for ebm, nsrt, replay in zip(self._ebms, self._nsrts, self._replay):
            if nsrt.name in self._save_dict:
                nsrt_dict = self._save_dict[nsrt.name]
                assert len(replay[0]) == len(replay[1]) == len(replay[2]) == 0
                replay[0].extend(nsrt_dict["replay"][0])
                replay[1].extend(nsrt_dict["replay"][1])
                replay[2].extend(nsrt_dict["replay"][2])
                load_ebm(nsrt_dict, ebm)
                logging.info(f"Successfully loaded model for {nsrt.name}")

        for option in self._option_needs_generic_sampler:
            if self._option_needs_generic_sampler[option] and option.name in self._save_dict:
                option_dict = self._save_dict[option.name]
                ebm = self._generic_option_samplers[option]
                load_ebm(option_dict, ebm)
                logging.info(f"Successfully loaded model for {option.name}")

        self._next_train_task = (CFG.lifelong_burnin_period or CFG.interactive_num_requests_per_cycle) + (self._online_learning_cycle - 1) * CFG.interactive_num_requests_per_cycle

        return self._online_learning_cycle 


    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # TODO: I'm not sure whether it makes more sense to treat the initial
        # data collection like all the others or have it be "demos"
        assert len(dataset.trajectories) == 0

    def get_interaction_requests(self):
        requests = []
        explorer = create_explorer(
            "partial_planning",
            self._initial_predicates,
            self._initial_options,
            self._types,
            self._action_space,
            self._train_tasks,
            self._get_current_nsrts(),
            self._option_model)

        metrics: Metrics = defaultdict(float)

        if CFG.lifelong_burnin_period is None or self._online_learning_cycle > 0:
            num_tasks = CFG.interactive_num_requests_per_cycle
        else:
            num_tasks = CFG.lifelong_burnin_period
        first_train_task = self._next_train_task
        self._next_train_task += num_tasks
        # Get the next tasks in the sequence
        total_time = 0
        for train_task_idx in range(first_train_task, self._next_train_task):
            query_policy = self._create_none_query_policy()

            explore_start = time.perf_counter()
            act_policy, termination_fn, skeleton = explorer.get_exploration_strategy(
                train_task_idx, CFG.timeout)
            explore_time = time.perf_counter() - explore_start
            requests.append(InteractionRequest(train_task_idx, act_policy, query_policy, termination_fn, skeleton))
            total_time += explore_time
        num_unsolved = explorer.metrics["num_unsolved"]
        num_solved = explorer.metrics["num_solved"]
        num_total = num_unsolved + num_solved
        assert num_total == num_tasks
        avg_time = total_time / num_tasks
        metrics["num_solved"] = num_solved
        metrics["num_unsolved"] = num_unsolved
        metrics["num_total"] = num_tasks
        metrics["avg_time"] = avg_time
        metrics["min_num_samples"] = explorer.metrics["min_num_samples"]
        metrics["max_num_samples"] = explorer.metrics["max_num_samples"]
        metrics["min_num_skeletons_optimized"] = explorer.metrics["min_num_skeletons_optimized"]
        metrics["max_num_skeletons_optimized"] = explorer.metrics["max_num_skeletons_optimized"]
        metrics["num_solve_failures"] = num_unsolved

        for metric_name in [
                "num_samples", "num_skeletons_optimized", "num_nodes_expanded",
                "num_nodes_created", "num_nsrts", "num_preds", "plan_length",
                "num_failures_discovered"
        ]:
            total = explorer.metrics[f"total_{metric_name}"]
            metrics[f"avg_{metric_name}"] = (
                total / num_solved if num_solved > 0 else float("inf"))
        total = explorer.metrics["total_num_samples_failed"]
        metrics["avg_num_samples_failed"] = total / num_unsolved if num_unsolved > 0 else float("inf")

        if CFG.env == "planar_behavior":
            env = get_or_create_env(CFG.env)
            for subenv_indicator, subenv in env._indicator_to_env_map.items():
                subenv_name = subenv.get_name()
                num_unsolved_env = explorer.metrics[f"env_{subenv_name}_num_unsolved"]
                num_solved_env = explorer.metrics[f"env_{subenv_name}_num_solved"]
                total_tasks_env = num_solved_env + num_unsolved_env


                metrics[f"{subenv_name}_num_solved"] = num_solved_env
                metrics[f"{subenv_name}_num_total"] = total_tasks_env
                metrics[f"{subenv_name}_min_num_samples"] = explorer.metrics[
                    f"env_{subenv_indicator}_min_num_samples"] if explorer.metrics[f"env_{subenv_indicator}_min_num_samples"] < float(
                        "inf") else 0
                metrics[f"{subenv_name}_max_num_samples"] = explorer.metrics[f"env_{subenv_indicator}_max_num_samples"]
                metrics[f"{subenv_name}_min_skeletons_optimized"] = explorer.metrics[
                    f"env_{subenv_indicator}_min_num_skeletons_optimized"] if explorer.metrics[
                        f"env_{subenv_indicator}_min_num_skeletons_optimized"] < float("inf") else 0
                metrics[f"{subenv_name}_max_skeletons_optimized"] = explorer.metrics[
                    f"env_{subenv_indicator}_max_num_skeletons_optimized"]
                metrics[f"{subenv_name}_num_solve_failures"] = num_unsolved_env
                # Handle computing averages of total approach metrics wrt the
                # number of found policies. Note: this is different from computing
                # an average wrt the number of solved tasks, which might be more
                # appropriate for some metrics, e.g. avg_suc_time above.
                for metric_name in [
                        "num_samples", "num_skeletons_optimized", "num_nodes_expanded",
                        "num_nodes_created", "num_nsrts", "num_preds", "plan_length",
                        "num_failures_discovered"
                ]:
                    total = explorer.metrics[f"env_{subenv_indicator}_total_{metric_name}"]
                    metrics[f"{subenv_name}_avg_{metric_name}"] = (
                        total / num_solved_env if num_solved_env > 0 else float("inf"))
                total = explorer.metrics[f"env_{subenv_indicator}_total_num_samples_failed"]
                metrics[f"{subenv_name}_avg_num_samples_failed"] = total / num_unsolved_env if num_unsolved_env > 0 else float("inf")

        logging.info(f"Tasks solved: {int(num_solved)} / {num_tasks}")
        outfile = (f"{CFG.results_dir}/{utils.get_config_path_str()}__"
               f"{self._online_learning_cycle}.pkl")
        
        cfg_copy = copy.copy(CFG)
        cfg_copy.pybullet_robot_ee_orns = None
        cfg_copy.get_arg_specific_settings = None
        outdata = {
            "config": cfg_copy,
            "results": metrics.copy(),
            "git_commit_hash": utils.get_git_commit_hash()
        }
        # Dump the CFG, results, and git commit hash to a pickle file.
        with open(outfile, "wb") as f:
            pkl.dump(outdata, f)

        logging.info(f"Exploration results: {metrics}")
        logging.info(f"Average time per task: {avg_time:.5f} seconds")
        logging.info(f"Wrote out test results to {outfile}")

        return requests

    def _create_none_query_policy(self) -> Callable[[State], Optional[Query]]:
        def _query_policy(s: State) -> Optional[GroundAtomsHoldQuery]:
            return None
        return _query_policy
    
    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        if CFG.lifelong_eval_no_learning: #CFG.oracle_samplers: NOTE: I replaced oracle samplers with this so it also "works" for going from checkpoint w/o training
            self._online_learning_cycle += 1
            return

        traj_list: List[LowLevelTrajectory] = []
        annotations_list: List[Any] = []
        skeleton_list: List[Any] = []

        for result in results:
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

    def _update_samplers(self, trajectories: List[LowLevelTrajectory], annotations_list: List[Any], skeletons: List[Any]) -> None:
        logging.info("\nUpdating the samplers...")
        if self._online_learning_cycle == 0:
            self._initialize_ebm_samplers()
        logging.info("Featurizing the samples...")

        option_generic_states = {opt: [] for opt, needs in self._option_needs_generic_sampler.items() if needs}
        option_generic_actions = {opt: [] for opt, needs in self._option_needs_generic_sampler.items() if needs}
        option_generic_aux_labels = {opt: [] for opt, needs in self._option_needs_generic_sampler.items() if needs}
        option_generic_replay_states = {opt: [] for opt, needs in self._option_needs_generic_sampler.items() if needs}
        option_generic_replay_actions = {opt: [] for opt, needs in self._option_needs_generic_sampler.items() if needs}
        option_generic_replay_aux_labels = {opt: [] for opt, needs in self._option_needs_generic_sampler.items() if needs}

        for ebm, nsrt, replay in zip(self._ebms, self._nsrts, self._replay):
            # Build replay buffer for generic option sampler training in the other loop
            # Note: it's important to do this at the top and not the bottom of this loop bc we later add the _new_ data into the replay buffer
            if self._option_needs_generic_sampler[nsrt.option]:
                option_generic_replay_states[nsrt.option].extend(replay[0])
                option_generic_replay_actions[nsrt.option].extend(replay[1])
                option_generic_replay_aux_labels[nsrt.option].extend(replay[2])

            # Get the data corresponding to the current NSRT
            states = []
            actions = []
            aux_labels = []
            for traj, annotations, skeleton in zip(trajectories, annotations_list, skeletons):
                for state, action, annotation, ground_nsrt in zip(traj.states[:-1], traj.actions, annotations, skeleton):
                    option = action.get_option()
                    # Get this NSRT's positive (successful) data only
                    if annotation > 0:
                        if nsrt.name == ground_nsrt.name: 
                            x = _featurize_state(state, ground_nsrt.objects)
                            a = option.params[1:]
                            aux = _aux_labels(nsrt.name, x, a)
                            states.append(x)
                            actions.append(a)
                            aux_labels.append(aux)
                            if self._option_needs_generic_sampler[nsrt.option]:
                                # Get the generic option's new data
                                option_generic_states[nsrt.option].append(x)
                                option_generic_actions[nsrt.option].append(a)
                                option_generic_aux_labels[nsrt.option].append(aux)

            states_arr = np.array(states)
            actions_arr = np.array(actions)
            aux_labels_arr = np.array(aux_labels)

            logging.info(f"{nsrt.name}: {states_arr.shape[0]} samples")
            if states_arr.shape[0] > 0:
                start_time = time.perf_counter()
                if not ebm.is_trained:
                    ebm.fit(states_arr, actions_arr, aux_labels_arr)
                else:
                    states_replay = np.array(replay[0])
                    actions_replay = np.array(replay[1])
                    aux_labels_replay = np.array(replay[2])

                    if CFG.lifelong_method == "distill" or CFG.lifelong_method == "2-distill":
                        # First copy: train model just on new data
                        ebm_new = copy.deepcopy(ebm)
                        ebm_new.fit(states_arr, actions_arr, aux_labels_arr)

                        # Second copy: previous version to distill into updated model
                        ebm_old = copy.deepcopy(ebm)

                        # Distill new and old models into updated model
                        ebm_new_data = (ebm_new, (states_arr, actions_arr, aux_labels_arr))
                        ebm_old_data = (ebm_old, (states_replay, actions_replay, aux_labels_replay))
                        ebm.distill(ebm_old_data, ebm_new_data)
                    elif CFG.lifelong_method == "retrain":
                        # Instead, try re-training the model as a performance upper bound
                        states_full = np.r_[states_arr, states_replay]
                        actions_full = np.r_[actions_arr, actions_replay]
                        aux_labels_full = np.r_[aux_labels_arr, aux_labels_replay]
                        ebm.fit(states_full, actions_full, aux_labels_full)
                    elif CFG.lifelong_method == "retrain-scratch":
                        ebm._linears = torch.nn.ModuleList()
                        ebm._optimizer = None
                        ebm.is_trained = False
                        states_full = np.r_[states_arr, states_replay]
                        actions_full = np.r_[actions_arr, actions_replay]
                        aux_labels_full = np.r_[aux_labels_arr, aux_labels_replay]
                        ebm.fit(states_full, actions_full, aux_labels_full)
                    elif CFG.lifelong_method == 'finetune':
                        ebm.fit(states_arr, actions_arr, aux_labels_arr)
                    elif CFG.lifelong_method == "retrain-balanced":
                        new_data = (states_arr, actions_arr, aux_labels_arr)
                        old_data = (states_replay, actions_replay, aux_labels_replay)
                        ebm.fit_balanced(old_data, new_data)
                    else:
                        raise NotImplementedError(f"Unknown lifelong method {CFG.lifelong_method}")            
                end_time = time.perf_counter()
                logging.info(f"Training time: {(end_time - start_time):.5f} seconds")
                replay[0].extend(states)
                replay[1].extend(actions)
                replay[2].extend(aux_labels)
                self._save_dict[nsrt.name] = {
                    "optimizer_state": ebm._optimizer.state_dict(),
                    "model_state": ebm.state_dict(),
                    "input_scale": ebm._input_scale,
                    "input_shift": ebm._input_shift,
                    "output_scale": ebm._output_scale,
                    "output_shift": ebm._output_shift,
                    "output_aux_scale": ebm._output_aux_scale,
                    "output_aux_shift": ebm._output_aux_shift,
                    "is_trained": ebm.is_trained,
                    "x_cond_dim": ebm._x_cond_dim,
                    "t_dim": ebm._t_dim,
                    "y_dim": ebm._y_dim,
                    "x_dim": ebm._x_dim,
                    "y_aux_dim": ebm._y_aux_dim,
                    "replay": replay,
                    "online_learning_cycle": self._online_learning_cycle,
                }
        for option in self._option_needs_generic_sampler:
            if self._option_needs_generic_sampler[option]:
                ebm = self._generic_option_samplers[option]
                states_arr = np.array(option_generic_states[option])
                actions_arr = np.array(option_generic_actions[option])
                aux_labels_arr = np.array(option_generic_aux_labels[option])

                logging.info(f"{option.name}: {states_arr.shape[0]} samples")
                if states_arr.shape[0] > 0:
                    start_time = time.perf_counter()
                    if not ebm.is_trained:
                        ebm.fit(states_arr, actions_arr, aux_labels_arr)
                    else:
                        states_replay = np.array(option_generic_replay_states[option])
                        actions_replay = np.array(option_generic_replay_actions[option])
                        aux_labels_replay = np.array(option_generic_replay_aux_labels[option])
                        if CFG.lifelong_method == "distill" or CFG.lifelong_method == "2-distill":
                            # First copy: train model on just new data
                            ebm_new = copy.deepcopy(ebm)
                            ebm_new.fit(states_arr, actions_arr, aux_labels_arr)

                            # Second copy: previous version to distill into updated model
                            ebm_old = copy.deepcopy(ebm)

                            # Distill new and old models into updated model
                            ebm_new_data = (ebm_new, (states_arr, actions_arr, aux_labels_arr))
                            ebm_old_data = (ebm_old, (states_replay, actions_replay, aux_labels_replay))
                            ebm.distill(ebm_old_data, ebm_new_data)
                        elif CFG.lifelong_method == "retrain":
                            # Instead, try re-training the model as a performance upper bound
                            states_full = np.r_[states_arr, states_replay]
                            actions_full = np.r_[actions_arr, actions_replay]
                            aux_labels_full = np.r_[aux_labels_arr, aux_labels_replay]
                            ebm.fit(states_full, actions_full, aux_labels_full)
                        elif CFG.lifelong_method == "retrain-scratch":
                            ebm._linears = torch.nn.ModuleList()
                            ebm._optimizer = None
                            ebm.is_trained = False
                            print(states_arr.shape)
                            print(states_replay.shape)
                            states_full = np.r_[states_arr, states_replay]
                            actions_full = np.r_[actions_arr, actions_replay]
                            aux_labels_full = np.r_[aux_labels_arr, aux_labels_replay]
                            ebm.fit(states_full, actions_full, aux_labels_full)
                        elif CFG.lifelong_method == "finetune":
                            ebm.fit(states_arr, actions_arr, aux_labels_arr)
                        elif CFG.lifelong_method == "retrain-balanced":
                            new_data = (states_arr, actions_arr, aux_labels_arr)
                            old_data = (states_replay, actions_replay, aux_labels_replay)
                            ebm.fit_balanced(old_data, new_data)
                        else:
                            raise NotImplementedError(f"Unknown lifelong method {CFG.lifelong_method}")
                    end_time = time.perf_counter()
                    logging.info(f"Training time: {(end_time - start_time):.5f} seconds")
                    self._save_dict[option.name] = {
                        "optimizer_state": ebm._optimizer.state_dict(),
                        "model_state": ebm.state_dict(),
                        "input_scale": ebm._input_scale,
                        "input_shift": ebm._input_shift,
                        "output_scale": ebm._output_scale,
                        "output_shift": ebm._output_shift,
                        "output_aux_scale": ebm._output_aux_scale,
                        "output_aux_shift": ebm._output_aux_shift,
                        "is_trained": ebm.is_trained,
                        "x_cond_dim": ebm._x_cond_dim,
                        "t_dim": ebm._t_dim,
                        "y_dim": ebm._y_dim,
                        "x_dim": ebm._x_dim,
                        "y_aux_dim": ebm._y_aux_dim,
                        "online_learning_cycle": self._online_learning_cycle,
                    }
        torch.save(self._save_dict, f"{CFG.results_dir}/{utils.get_config_path_str()}__checkpoint.pt")

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
    _original_sampler: NSRTSampler
    _choice_probabilities: Array
    _random_aux_error: float

    def sampler(self, state: State, goal: Set[GroundAtom],
                rng: np.random.Generator, objects: Sequence[Object],
                skeleton: List[Any]) -> Array:
        """The sampler corresponding to the given models.

        May be used as the _sampler field in an NSRT.
        """
        x_lst: List[Any] = []  
        sub = dict(zip(self._variables, objects))
        for var in self._variables:
            x_lst.extend(state[sub[var]])
        x = np.array(x_lst)
        assert (x == state.vec(objects)).all()
        if not self._ebm.is_trained:
            # If I haven't trained the specialized model, uniformly choose between original and generic
            if self._generic_ebm is not None and self._generic_ebm.is_trained:
                chosen_sampler_idx = rng.choice([1, 2])
                if chosen_sampler_idx == 2:
                    params = np.array(self._generic_ebm.predict_sample(x, rng),
                                     dtype=self._param_option.params_space.dtype)
                    return np.r_[chosen_sampler_idx, params]
            return self._original_sampler(state, goal, rng, objects, skeleton)
        
        if self._generic_ebm is None:
            num_samplers = 2
        else:
            num_samplers = 3
        # chosen_sampler_idx = rng.integers(num_samplers)

        ebm_a = np.array(self._ebm.predict_sample(x, rng),
                         dtype=self._param_option.params_space.dtype)
        aux = _aux_labels(self._name, x, ebm_a)
        ebm_square_err = self._ebm.aux_square_error(x[None], ebm_a[None], aux[None])
        original_err = np.sqrt(ebm_square_err.shape[0])    # This is because we'll be normalizing by the _random_aux_error vector
        ebm_err = np.sqrt(np.sum(ebm_square_err / (self._random_aux_error ** 2)))
        choice_probabilities = 1 / np.array([ebm_err + 1e-6, original_err])
        if num_samplers == 3:
            generic_ebm_a = np.array(self._generic_ebm.predict_sample(x, rng),
                                     dtype=self._param_option.params_space.dtype)
            aux = _aux_labels(self._name, x, generic_ebm_a)
            generic_ebm_square_err = self._generic_ebm.aux_square_error(x[None], generic_ebm_a[None], aux[None])
            generic_ebm_err = np.sqrt(np.sum(generic_ebm_square_err / (self._random_aux_error ** 2)))
            choice_probabilities = np.r_[choice_probabilities, 1 / (generic_ebm_err + 1e-6)]
        choice_probabilities /= choice_probabilities.sum()

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
        # print(self._param_option.params_space.shape)
        # exit()
        return np.r_[chosen_sampler_idx, params]
