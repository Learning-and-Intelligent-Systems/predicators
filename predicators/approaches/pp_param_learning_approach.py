"""pp_param_learning_approach module."""
import logging
import os
import random
import time
from collections import defaultdict
from pprint import pformat
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from gym.spaces import Box
from torch import Tensor
from torch.optim import LBFGS, Adam
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from predicators import utils
from predicators.approaches.process_planning_approach import \
    BilevelProcessPlanningApproach
from predicators.ground_truth_models import get_gt_processes
from predicators.option_model import _OptionModelBase
from predicators.planning_with_processes import process_task_plan_grounding
from predicators.settings import CFG
from predicators.structs import NSRT, AtomOptionTrajectory, CausalProcess, \
    Dataset, EndogenousProcess, ExogenousProcess, GroundAtom, \
    LowLevelTrajectory, ParameterizedOption, Predicate, Task, Type, \
    _GroundCausalProcess


class ParamLearningBilevelProcessPlanningApproach(
        BilevelProcessPlanningApproach):
    """A bilevel planning approach that uses hand-specified processes."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1,
                 bilevel_plan_without_sim: Optional[bool] = None,
                 processes: Optional[Set[CausalProcess]] = None,
                 option_model: Optional[_OptionModelBase] = None):
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         task_planning_heuristic,
                         max_skeletons_optimized,
                         bilevel_plan_without_sim,
                         option_model=option_model)
        if processes is None:
            processes = get_gt_processes(CFG.env, self._initial_predicates,
                                         self._initial_options)
        self._processes: Set[CausalProcess] = processes
        self._offline_dataset = Dataset([])

    @classmethod
    def get_name(cls) -> str:
        return "param_learning_process_planning"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_processes(self) -> Set[CausalProcess]:
        return self._processes

    def _get_current_exogenous_processes(self) -> Set[ExogenousProcess]:
        """Get the current set of exogenous processes."""
        return {p for p in self._processes if isinstance(p, ExogenousProcess)}

    def _get_current_endogenous_processes(self) -> Set[EndogenousProcess]:
        """Get the current set of endogenous processes."""
        return {p for p in self._processes if isinstance(p, EndogenousProcess)}

    def _get_current_nsrts(self) -> Set[NSRT]:
        """Get the current set of NSRTs."""
        return set()

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """Learn parameters of processes from the offline datasets.

        This is currently achieved by optimizing the marginal data
        likelihood.
        """
        self._learn_process_parameters(dataset.trajectories)

    def _learn_process_parameters(
        self,
        trajectories: List[LowLevelTrajectory],
        use_lbfgs: bool = False,
    ) -> None:
        """Stochastic (mini-batch) optimisation of process parameters."""
        processes = sorted(self._get_current_processes())
        _, scores = learn_process_parameters(
            trajectories[:1],
            self._get_current_predicates(),
            processes,
            use_lbfgs=use_lbfgs,
            lbfgs_max_iter=CFG.process_param_learning_num_steps,
            adam_num_steps=CFG.process_param_learning_num_steps,
            early_stopping_patience=20,
            use_empirical=CFG.process_param_learning_use_empirical,
        )
        logging.debug(f"ELBO: {scores[0]}, exp_state: {scores[1]}, "
                      f"exp_delay: {scores[2]}, entropy: {scores[3]}")
        logging.debug("Learned processes:")
        for p in processes:
            logging.debug(pformat(p))
        logging.debug(f"Log frame strength: {scores[4]}")


def learn_process_parameters(
    trajectories: List[LowLevelTrajectory],
    predicates: Set[Predicate],
    processes: Sequence[CausalProcess],
    use_lbfgs: bool = False,
    plot_training_curve: bool = True,
    lbfgs_max_iter: int = 200,
    seed: int = 0,
    display_progress: bool = True,
    adam_num_steps: int = 200,
    std_regularization: Optional[int] = None,
    early_stopping_patience: Optional[int] = None,
    early_stopping_tolerance: float = 1e-4,
    check_condition_overall: bool = True,
    batch_size: int = 16,
    debug_log: bool = False,
    use_empirical: bool = False,
) -> Tuple[Sequence[CausalProcess], Tuple[float, float, float, float, float]]:
    """Learn process parameters using stochastic optimization or empirical
    estimation.

    If use_empirical=True, bypasses variational inference and directly
    estimates delay parameters from observed data.
    """

    # If using empirical estimation, bypass all the variational inference
    if use_empirical:
        processes, _stats = learn_process_parameters_empirical(
            trajectories, predicates, processes, use_empirical=True)

        # Even when using empirical estimation, we need to
        # prepare data and evaluate properly
        max_traj_len = max(
            len(traj.states)
            for traj in trajectories) if len(trajectories) > 0 else 0

        per_traj_data, proc_and_guide_params_full, num_proc_params = \
            _prepare_training_data_and_model_params(
                predicates,
                processes,
                trajectories,
                check_condition_overall,
            )

        # Initialize guide parameters randomly since we
        # don't learn them empirically
        guide_params = proc_and_guide_params_full[num_proc_params:]
        final_frame_param = torch.tensor(1.0)  # Default frame strength

        # Evaluate the empirically set model on the dataset
        (mean_elbo, mean_exp_state,
         mean_exp_delay, mean_entropy) = \
            evaluate_model_on_dataset(
                per_traj_data=per_traj_data,
                frame_param=final_frame_param,
                guide_params=guide_params,
                debug_log=debug_log)

        return processes, (mean_elbo, mean_exp_state, mean_exp_delay,
                           mean_entropy, 1.0)

    if use_lbfgs:
        num_steps = 1
        batch_size = 100
        inner_lbfgs_max_iter = lbfgs_max_iter
    else:
        num_steps = adam_num_steps

    torch.manual_seed(seed)
    random.seed(seed)

    # -------------------------------------------------------------- #
    # 0.  Cache per-trajectory data & build a global param layout     #
    # -------------------------------------------------------------- #
    max_traj_len = max(len(traj.states) for traj in trajectories)\
        if len(trajectories) > 0 else 0

    per_traj_data, proc_and_guide_params_full, num_proc_params = \
        _prepare_training_data_and_model_params(
            predicates,
            processes,
            trajectories,
            check_condition_overall,
        )

    # --- Optionally initialize process parameters with empirical estimates ---
    if CFG.use_empirical_init_for_vi_params:
        _initialize_params_with_empirical_estimates(
            trajectories, predicates, processes, proc_and_guide_params_full,
            num_proc_params)

    # --- Separate parameter tensor into logical, learnable components ---

    # All process parameters (strength + delay) from the initial tensor
    proc_params_full = proc_and_guide_params_full[:num_proc_params]

    learnable_params_for_optim = []

    guide_params = torch.nn.Parameter(
        proc_and_guide_params_full[num_proc_params:])
    learnable_params_for_optim.append(guide_params)

    learnable_proc_params = torch.nn.Parameter(proc_params_full)
    learnable_params_for_optim.append(learnable_proc_params)

    frame_param = torch.nn.Parameter(torch.randn(1) * 0.01)
    learnable_params_for_optim.append(frame_param)

    init_proc_param = proc_params_full.detach()
    _set_process_parameters(processes, init_proc_param,
                            **{'max_k': max_traj_len})

    # ------------------- progress bar -------------------------- #
    if use_lbfgs:
        pbar_total = num_steps * inner_lbfgs_max_iter
        desc = "Training (mini‑batch LBFGS)"
    else:
        pbar_total = num_steps
        desc = "Training (Adam)"
    if display_progress:
        pbar = tqdm(total=pbar_total, desc=desc)
    else:
        pbar = None

    best_elbo = -float("inf")
    curve: Dict = {
        "iterations": [],
        "elbos": [],
        "best_elbos": [],
        "wall_time": []
    }
    training_start_time = time.time()

    # --- Early stopping setup ---
    patience_counter = early_stopping_patience
    optim: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    if use_lbfgs:
        # LBFGS is re-initialized per outer step or initialized once here.
        # optim = LBFGS([params], max_iter=inner_lbfgs_max_iter,
        # line_search_fn="strong_wolfe")
        pass  # Will be initialized in the loop
    else:
        # 1e-1; 500 steps: 983;
        #       5k steps (no scheduler): 987.0547/987.0568
        #       5k steps (scheduler):    987.0547/987.0537
        #       10k      (no schedule) : 987.0574
        # 5e-2; 500 steps: 980/981; 5k steps: 986/987
        # 1e-2; 500 steps: 953/962; 5k steps: 984/984
        lr = 1e-1
        if debug_log:
            logging.debug(f"lr={lr}")
        optim = Adam(learnable_params_for_optim, lr=lr)
        # scheduler = ReduceLROnPlateau(optim,
        #                               mode='min',
        #                               factor=0.5,
        #                               patience=20,
        #                               verbose=True,)
        if debug_log:
            if scheduler:
                logging.debug(f"Scheduler initialized: {scheduler}")

    # ------------------- training loop ----------------------------- #
    iteration = 0
    for _ in range(num_steps):
        current_optim: Optional[LBFGS] = None
        if use_lbfgs:
            current_optim = LBFGS(learnable_params_for_optim,
                                  max_iter=inner_lbfgs_max_iter,
                                  line_search_fn="strong_wolfe")
        else:
            current_optim = optim  # type: ignore[assignment]

        assert current_optim is not None, "Optimizer not initialized"

        # remaining_ids = list(range(1, len(per_traj_data)))
        # additional_samples = min(batch_size - 1, len(remaining_ids))
        # batch_ids = [0] + random.sample(remaining_ids, k=additional_samples)
        num_trajs = len(per_traj_data)
        batch_ids = random.sample(range(num_trajs),
                                  k=min(batch_size, num_trajs))

        # pylint: disable=cell-var-from-loop
        def closure() -> float:
            nonlocal best_elbo, iteration
            nonlocal patience_counter  #, best_params_state

            if current_optim:
                current_optim.zero_grad(set_to_none=True)

            proc_param = learnable_proc_params

            _set_process_parameters(processes, proc_param)

            guide_flat = guide_params
            frame = frame_param

            elbo = torch.tensor(0.0, dtype=frame.dtype, device=frame.device)

            for tidx in batch_ids:
                td = per_traj_data[tidx]
                guide_dict = _create_guide_dict_for_trajectory(
                    td, guide_flat, td["traj_len"])

                data_elbo, _, _, _ = elbo_torch(
                    td["trajectory"],
                    td["sparse_trajectory"],
                    td["ground_causal_processes"],
                    td["start_times_per_gp"],
                    guide_dict,
                    frame,
                    set(td["all_atoms"]),
                    td["atom_to_val_to_gps"],
                    td["condition_cache"],
                )
                elbo = elbo + data_elbo

            loss = -(elbo / len(batch_ids))
            if std_regularization and learnable_params_for_optim:
                loss = loss + std_regularization * (proc_param[2::3].sum())

            if learnable_params_for_optim:
                loss.backward()  # type: ignore

            detached_elbo_item = elbo.detach().item()

            # --- Early stopping check ---
            if early_stopping_patience is not None:
                if detached_elbo_item > best_elbo + early_stopping_tolerance:
                    best_elbo = detached_elbo_item
                    # best_params_state = [
                    #     p.clone().detach() for p in learnable_params_for_optim
                    # ]
                    patience_counter = early_stopping_patience
                else:
                    if patience_counter is not None:
                        patience_counter -= 1
            elif detached_elbo_item > best_elbo:
                best_elbo = detached_elbo_item

            curve["iterations"].append(iteration)
            curve["elbos"].append(detached_elbo_item)
            curve["best_elbos"].append(best_elbo)
            curve["wall_time"].append(time.time() - training_start_time)
            if pbar:
                pbar.set_postfix(ELBO=detached_elbo_item, best=best_elbo)
                pbar.update(1)

            iteration += 1
            return loss.item()

        # pylint: enable=cell-var-from-loop

        if use_lbfgs:
            current_optim.step(closure)  # type: ignore[misc,no-untyped-call]
        else:
            loss = closure()
            # pylint: disable-next=no-value-for-parameter
            current_optim.step()  # type: ignore[call-arg,no-untyped-call]
            if scheduler:
                if debug_log:
                    prev_lr = scheduler.get_last_lr()
                scheduler.step(loss)  # type: ignore[arg-type]
                if debug_log:
                    curr_lr = scheduler.get_last_lr()
                    if curr_lr != prev_lr:
                        logging.debug(
                            f"decreasing lr from {prev_lr} to {curr_lr}")

        # --- Trigger early stop if patience has run out ---
        if (early_stopping_patience is not None
                and patience_counter is not None and patience_counter <= 0):
            break

    if pbar:
        pbar.close()

    # --- Persist Final Parameters and Evaluate ---
    final_guide_params = guide_params.detach()
    final_proc_params = learnable_proc_params.detach()

    _set_process_parameters(processes, final_proc_params)
    final_frame_param = frame_param.detach()

    # Call the new independent evaluation function
    (mean_elbo, mean_exp_state,
     mean_exp_delay, mean_entropy) = \
        evaluate_model_on_dataset(
            per_traj_data=per_traj_data,
            frame_param=final_frame_param,
            guide_params=final_guide_params,
            debug_log=debug_log)

    if plot_training_curve:
        _plot_training_curve(curve)

    return processes, (mean_elbo, mean_exp_state, mean_exp_delay, mean_entropy,
                       final_frame_param.item())


def elbo_torch(
    atom_option_trajectory: AtomOptionTrajectory,
    sparse_trajectory: List[Tuple[Set[GroundAtom], int, int]],
    ground_processes: List[
        _GroundCausalProcess],  # All potential ground causal processes
    start_times_per_gp: List[
        List[int]],  # start_times_per_gp[gp_idx] is list of
    # s_i for ground_processes[gp_idx]
    guide: Dict[_GroundCausalProcess,
                Dict[int, Tensor]],  # Variational params q(z_t ; gp, s_i)
    log_frame_strength: Tensor,
    all_possible_atoms: Set[GroundAtom],
    atom_to_val_to_gps: Dict[GroundAtom, Dict[bool,
                                              Set[_GroundCausalProcess]]],
    condition_cache: Dict[_GroundCausalProcess, Dict[int, Dict[int, bool]]],
    use_sparse_trajectory: bool = True,
    debug_log: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Differentiable ELBO with cached condition checks."""
    trajectory = atom_option_trajectory
    num_time_steps = len(trajectory.states)

    ll = torch.tensor(0.0, dtype=log_frame_strength.dtype)
    yt_prev = trajectory.states[0]

    # -----------------------------------------------------------------
    # 1.  Expected log state probabilities
    # -----------------------------------------------------------------
    exp_state_prob = torch.tensor(0.0, dtype=log_frame_strength.dtype)
    if use_sparse_trajectory:
        if debug_log:
            logging.debug(f"Compute exp_state_prob from "
                          f"{len(sparse_trajectory)-1} segments")
        for i, (yt, start_t, _) in enumerate(sparse_trajectory[1:]):
            state_prob_t = torch.tensor(0.0, dtype=log_frame_strength.dtype)
            E_log_Zt = torch.tensor(0.0, dtype=log_frame_strength.dtype)
            for atom, val_to_gps in atom_to_val_to_gps.items():
                sum_ytj = torch.tensor(0.0, dtype=log_frame_strength.dtype)
                for val in (True, False):
                    # Ground processes which have this atom in the add or delete
                    # effects.
                    gps = val_to_gps.get(val, set())

                    prod = torch.tensor(1.0, dtype=log_frame_strength.dtype)
                    for gp in gps:
                        for st, q in guide.get(gp, {}).items():
                            if st < start_t:
                                # --- Efficient Cache Lookup ---
                                # Default to True if not in
                                # cache (e.g. no overall cond.)
                                condition_overall_holds = condition_cache.get(
                                    gp, {}).get(st, {}).get(start_t - 1, True)
                                prev_val = atom in yt_prev
                                # --- Numerator Part ---
                                if val == (atom
                                           in yt) and condition_overall_holds:
                                    exp_state_prob = exp_state_prob + \
                                        q[start_t] * \
                                        gp.factored_effect_factor(val, atom,
                                                prev_val)
                                    state_prob_t = state_prob_t + \
                                        q[start_t] * \
                                        gp.factored_effect_factor(val, atom,
                                                prev_val)

                                # --- Denominator Part ---
                                if condition_overall_holds:
                                    prod = prod * (q[start_t] * torch.exp(
                                        gp.factored_effect_factor(
                                            val, atom, prev_val)) +
                                                   (1 - q[start_t]))
                    sum_ytj = sum_ytj + prod * torch.exp(log_frame_strength *
                                                         (val ==
                                                          (atom in yt_prev)))
                E_log_Zt = E_log_Zt + torch.log(sum_ytj + 1e-12)

            # Atoms not referenced in any process law
            add_atoms = yt - yt_prev
            del_atoms = yt_prev - yt
            atoms_unchanged = all_possible_atoms - del_atoms - add_atoms
            exp_state_prob = exp_state_prob + log_frame_strength * len(
                atoms_unchanged)
            state_prob_t = state_prob_t + log_frame_strength * len(
                atoms_unchanged)

            # Normalization contribution from atoms not
            # described by the processes
            atoms_in_law_effects = set(atom_to_val_to_gps)
            atoms_not_in_law_effects = all_possible_atoms - atoms_in_law_effects
            E_log_Zt = E_log_Zt + len(atoms_not_in_law_effects) * torch.log(
                1 + torch.exp(log_frame_strength))

            exp_state_prob = exp_state_prob - E_log_Zt
            state_prob_t = state_prob_t - E_log_Zt
            yt_prev = yt
            if debug_log:
                logging.debug(
                    f"seg {i}: start_t={start_t}, "
                    f"exp_state_prob_t={state_prob_t.detach().item():.4f}, "
                    f"add atoms: {add_atoms}, del atoms: {del_atoms}")
    else:
        for t in range(1, num_time_steps):
            yt = trajectory.states[t]

            E_log_Zt = torch.tensor(0.0, dtype=log_frame_strength.dtype)
            for atom, val_to_gps in atom_to_val_to_gps.items():
                sum_ytj = torch.tensor(0.0, dtype=log_frame_strength.dtype)
                for val in (True, False):
                    gps = val_to_gps.get(val, set())

                    prod = torch.tensor(1.0, dtype=log_frame_strength.dtype)
                    for gp in gps:
                        for st, q in guide.get(gp, {}).items():
                            if st < t:
                                # --- Efficient Cache Lookup ---
                                # Default to True if not in
                                # cache (e.g. no overall cond.)
                                condition_overall_holds = condition_cache.get(
                                    gp, {}).get(st, {}).get(t - 1, True)

                                prev_val = atom in yt_prev
                                # --- Numerator Part ---
                                if val == (atom
                                           in yt) and condition_overall_holds:
                                    exp_state_prob = exp_state_prob + q[t] * \
                                        gp.factored_effect_factor(val, atom,
                                        prev_val)
                                # --- Denominator Part ---
                                if condition_overall_holds:
                                    prod = prod * (q[t] * torch.exp(
                                        gp.factored_effect_factor(
                                            val, atom, prev_val)) + (1 - q[t]))

                    sum_ytj = sum_ytj + prod * torch.exp(log_frame_strength *
                                                         (val ==
                                                          (atom in yt_prev)))
                E_log_Zt = E_log_Zt + torch.log(sum_ytj + 1e-12)

            # Atoms not referenced in any process law
            add_atoms = yt - yt_prev
            del_atoms = yt_prev - yt
            atoms_unchanged = all_possible_atoms - del_atoms - add_atoms
            exp_state_prob = exp_state_prob + log_frame_strength * len(
                atoms_unchanged)

            # Normalization contribution from atoms not
            # described by the processes
            atoms_in_law_effects = set(atom_to_val_to_gps)
            atoms_not_in_law_effects = all_possible_atoms - atoms_in_law_effects
            E_log_Zt = E_log_Zt + len(atoms_not_in_law_effects) * torch.log(
                1 + torch.exp(log_frame_strength))

            exp_state_prob = exp_state_prob - E_log_Zt
            yt_prev = yt
    ll = ll + exp_state_prob

    # -----------------------------------------------------------------
    # 2.  Expected Delay probabilities
    # -----------------------------------------------------------------
    exp_delay_prob = torch.tensor(0.0, dtype=log_frame_strength.dtype)
    for gp_idx, gp_obj in enumerate(ground_processes):
        for s_i in start_times_per_gp[gp_idx]:
            if s_i + 1 < num_time_steps:
                delay_values = torch.arange(1,
                                            num_time_steps - s_i,
                                            dtype=torch.long,
                                            device=log_frame_strength.device)
                if delay_values.numel() == 0:
                    continue
                t_indices_for_guide = s_i + delay_values
                all_delay_log_probs = (
                    gp_obj.delay_distribution  # type: ignore[attr-defined]
                    .log_prob(delay_values))
                q_dist_for_instance = guide.get(gp_obj, {}).get(s_i, None)
                if q_dist_for_instance is None:
                    raise Exception("Guide distribution not found"
                                    f" for {gp_obj} at s_i={s_i}")
                guide_slice_for_delays = q_dist_for_instance[
                    t_indices_for_guide]
                valid_mask = ~torch.isneginf(all_delay_log_probs) & (
                    guide_slice_for_delays > 1e-9)
                if valid_mask.any():
                    single_exp_delay_prob = torch.sum(
                        guide_slice_for_delays[valid_mask] *
                        all_delay_log_probs[valid_mask])
                    exp_delay_prob = exp_delay_prob + single_exp_delay_prob
                    if debug_log:
                        logging.debug(
                            "exp_delay_prob="
                            f"{single_exp_delay_prob.detach().item():.4f} "
                            f"start_t={s_i}, "
                            f"max_guide_values: at "
                            f"t={torch.argmax(q_dist_for_instance)}"
                            f": {torch.max(q_dist_for_instance)}")
                        logging.debug("guide_prob at arrival_t (94): "
                                      f"{q_dist_for_instance[94]}")

    ll = ll + exp_delay_prob

    # -----------------------------------------------------------------
    # 3.  Entropy of the variational distributions
    # -----------------------------------------------------------------
    num_started_delays = 0
    entropy = torch.tensor(0.0, dtype=log_frame_strength.dtype)
    for start_time_q_map in guide.values():
        for q_dist_for_instance in start_time_q_map.values():
            mask = q_dist_for_instance > 1e-9
            if mask.any():
                entropy -= torch.sum(q_dist_for_instance[mask] *
                                     torch.log(q_dist_for_instance[mask]))
            num_started_delays += 1
    # Add entropy for guide for delay variables who were not activated
    num_gp = len(ground_processes)
    num_unstarted_delays = num_gp * num_time_steps - num_started_delays
    unstarted_delay_entropy = num_unstarted_delays * torch.log(
        torch.tensor(1 / num_time_steps, dtype=log_frame_strength.dtype))
    entropy -= unstarted_delay_entropy

    elbo = ll + entropy
    return elbo, exp_state_prob, exp_delay_prob, entropy


def compute_empirical_delays(
    trajectories: List[LowLevelTrajectory],
    predicates: Set[Predicate],
    processes: Sequence[CausalProcess],
) -> Dict[str, List[int]]:
    """Compute empirical delays for each process type from trajectory data.

    Returns a dictionary mapping process names to lists of observed
    delays.
    """
    atom_option_dataset = utils.create_ground_atom_option_dataset(
        trajectories, predicates)

    # Dictionary to store delays for each process type
    process_delays: Dict[str, List[int]] = defaultdict(list)

    for traj in atom_option_dataset:
        traj_len = len(traj.states)
        ll_states = traj._low_level_states  # pylint: disable=protected-access
        objs = set(ll_states[0])

        # Ground the processes for this trajectory
        _ground_processes, _ = process_task_plan_grounding(
            init_atoms=set(),
            objects=objs,
            cps=processes,
            allow_waits=True,
            compute_reachable_atoms=False,
        )
        ground_processes = [
            gp for gp in _ground_processes
            if isinstance(gp, _GroundCausalProcess)
        ]

        # For each ground process, find when it was triggered
        # and when effects appeared
        for gp in ground_processes:
            # Find all times when this process was triggered
            trigger_times = []
            for t in range(traj_len):
                if gp.cause_triggered(traj.states[:t + 1],
                                      traj.actions[:t + 1]):
                    trigger_times.append(t)

            # For each trigger time, find when the effects appeared
            for trigger_t in trigger_times:
                # Check when the add effects appear
                for effect_t in range(trigger_t + 1, traj_len):
                    # Check if all add effects are present
                    # and all delete effects are gone
                    add_satisfied = gp.add_effects.issubset(
                        traj.states[effect_t])
                    delete_satisfied = not any(atom in traj.states[effect_t]
                                               for atom in gp.delete_effects)

                    if add_satisfied and delete_satisfied:
                        # Found the effect time - compute delay
                        delay = effect_t - trigger_t
                        process_delays[gp.parent.name].append(delay)
                        break

    return process_delays


def learn_process_parameters_empirical(
    trajectories: List[LowLevelTrajectory],
    predicates: Set[Predicate],
    processes: Sequence[CausalProcess],
    use_empirical: bool = False,
) -> Tuple[Sequence[CausalProcess], Dict[str, Tuple[Optional[float],
                                                    Optional[float]]]]:
    """Learn process parameters using empirical estimation of delays.

    When use_empirical=True, directly computes mean and std from
    observed delays. Returns the processes with updated parameters and a
    dict of statistics.
    """
    if not use_empirical:
        raise ValueError("This function is only for empirical estimation")

    # Compute empirical delays for each process type
    process_delays = compute_empirical_delays(trajectories, predicates,
                                              processes)

    # Statistics dictionary to return
    stats: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    # Update each process with empirical parameters
    for process in processes:
        if process.name in process_delays and len(
                process_delays[process.name]) > 0:
            delays = torch.tensor(process_delays[process.name],
                                  dtype=torch.float32)

            # Compute mean and std
            empirical_mean = delays.mean()
            empirical_std = delays.std() if len(delays) > 1 else torch.tensor(
                0.1)

            # Ensure std is not too small
            empirical_std = torch.clamp(empirical_std, min=0.1)

            # Create parameter tensor [log_strength, log_mu, log_sigma]
            # We'll keep strength at 1.0 (log(1) = 0) since we're ignoring it
            params = torch.tensor([
                0.0,  # log_strength = 0 (strength = 1)
                torch.log(empirical_mean),  # log_mu
                torch.log(empirical_std)  # log_sigma
            ])

            # Update the process parameters
            process._set_parameters(  # pylint: disable=protected-access
                params.tolist())

            # Store statistics
            stats[process.name] = (empirical_mean.item(), empirical_std.item())

            print(f"Process {process.name}:")
            print(f"  Observed delays: {process_delays[process.name]}")
            print(f"  Empirical mean: {empirical_mean:.2f}")
            print(f"  Empirical std: {empirical_std:.2f}")
        else:
            # No observations for this process - use defaults
            print(f"Process {process.name}: No observations "
                  "found, keeping defaults")
            stats[process.name] = (None, None)

    return processes, stats


@torch.no_grad()
def evaluate_model_on_dataset(
        per_traj_data: List[Dict[str, Any]],
        frame_param: torch.Tensor,
        guide_params: torch.Tensor,
        ignore_entropy: bool = False,
        debug_log: bool = False) -> Tuple[float, float, float, float]:
    """Evaluates a trained model on the full dataset.

    TODO: maybe normalize by number of segments? or total number of steps?
    """
    total_elbo, total_exp_state = 0.0, 0.0
    total_exp_delay, total_entropy = 0.0, 0.0

    for td in per_traj_data:
        guide_dict = _create_guide_dict_for_trajectory(td, guide_params,
                                                       td["traj_len"])

        data_elbo, data_exp_state, data_exp_delay, data_entropy = elbo_torch(
            td["trajectory"],
            td["sparse_trajectory"],
            td["ground_causal_processes"],
            td["start_times_per_gp"],
            guide_dict,
            frame_param,
            set(td["all_atoms"]),
            td["atom_to_val_to_gps"],
            td["condition_cache"],
            debug_log=debug_log)
        total_elbo += data_elbo.item()
        if ignore_entropy:
            total_elbo -= data_entropy.item()
        total_exp_state += data_exp_state.item()
        total_exp_delay += data_exp_delay.item()
        total_entropy += data_entropy.item()

    num_trajectories = len(per_traj_data)
    mean_elbo = total_elbo / num_trajectories
    mean_exp_state = total_exp_state / num_trajectories
    mean_exp_delay = total_exp_delay / num_trajectories
    mean_entropy = total_entropy / num_trajectories

    return mean_elbo, mean_exp_state, mean_exp_delay, mean_entropy


def _set_process_parameters(processes: Sequence[CausalProcess],
                            parameters: Tensor, **kwargs: Any) -> None:
    # Parameters are for the CausalProcess types, not ground instances.
    # Assumes 3 parameters per CausalProcess type
    # (e.g., for its delay distribution)
    num_causal_process_types = len(processes)
    expected_len = 3 * num_causal_process_types
    assert len(parameters) == expected_len, \
        f"Expected {expected_len} params, got {len(parameters)}"

    # Loop through the CausalProcess types
    for i in range(num_causal_process_types):
        param_slice = parameters[i * 3:(i + 1) * 3]
        processes[i]._set_parameters(  # pylint: disable=protected-access
            param_slice.tolist(), **kwargs)


def _compute_condition_cache_for_traj(
    ground_processes: List[_GroundCausalProcess],
    start_times_per_gp: List[List[int]], history: List[Set[GroundAtom]],
    num_time_steps: int
) -> Dict[_GroundCausalProcess, Dict[int, Dict[int, bool]]]:
    """Pre-computes which `condition_overall` holds at each time step for a
    single trajectory."""
    condition_cache: Dict[_GroundCausalProcess, Dict[int, Dict[int,
                                                               bool]]] = {}
    for gp_idx, gp in enumerate(ground_processes):
        # Only need to cache for processes that have an overall condition
        if not gp.condition_overall:
            continue
        condition_cache[gp] = {}
        for st in start_times_per_gp[gp_idx]:
            condition_cache[gp][st] = {}
            # Use dynamic programming: the result at `t`
            # depends on the result at `t-1`
            is_still_holding = True
            for t_interval in range(st + 1, num_time_steps):
                # Check only the new state at the end of the interval
                if not gp.condition_overall.issubset(history[t_interval]):
                    is_still_holding = False
                # The result for the interval [st+1, t_interval+1] is stored
                condition_cache[gp][st][t_interval] = is_still_holding
    return condition_cache


def _prepare_training_data_and_model_params(
    predicates: Set[Predicate], processes: Sequence[CausalProcess],
    trajectories: List[LowLevelTrajectory], check_condition_overall: bool
) -> Tuple[List[Dict[str, Any]], torch.nn.Parameter, int]:
    """Cache per-trajectory data, build global param layout for process and
    guide parameters, and initialize them."""
    atom_option_dataset = utils.create_ground_atom_option_dataset(
        trajectories, predicates)

    per_traj_data: List[Dict[str, Any]] = []
    # num_proc_params is now just the number of process parameters
    num_proc_params = 3 * len(processes)
    q_offset = 0

    for traj in atom_option_dataset:
        traj_len = len(traj.states)
        ll_st = traj._low_level_states  # pylint: disable=protected-access
        objs = set(ll_st[0])

        _ground_processes, _ = process_task_plan_grounding(
            init_atoms=set(),
            objects=objs,
            cps=processes,
            allow_waits=True,
            compute_reachable_atoms=False,
        )
        ground_processes = [
            gp for gp in _ground_processes
            if isinstance(gp, _GroundCausalProcess)
        ]

        atom_to_val_to_gps: Dict[GroundAtom, Dict[
            bool,
            Set[_GroundCausalProcess]]] = defaultdict(lambda: defaultdict(set))
        for gp in ground_processes:
            for a in gp.add_effects:
                atom_to_val_to_gps[a][True].add(gp)
            for a in gp.delete_effects:
                atom_to_val_to_gps[a][False].add(gp)

        start_times = [[
            t for t in range(traj_len)
            if gp.cause_triggered(traj.states[:t + 1], traj.actions[:t + 1])
        ] for gp in ground_processes]

        # Pre-compute the condition cache for this trajectory
        condition_cache: Dict[_GroundCausalProcess,
                              Dict[int, Dict[int, bool]]] = {}
        if check_condition_overall:
            condition_cache = _compute_condition_cache_for_traj(
                ground_processes, start_times, traj.states, traj_len)

        gp_qparam_id_map: Dict[Tuple[_GroundCausalProcess, int],
                               Tuple[int, int]] = {}
        for gp_idx, gp in enumerate(ground_processes):
            for s_i in start_times[gp_idx]:
                lo, hi = q_offset, q_offset + traj_len
                gp_qparam_id_map[(gp, s_i)] = (lo, hi)
                q_offset = hi

        # 1. Create sparse representation: [(state, start_time, end_time)]
        sparse_trajectory = []
        if len(traj.states) > 1:
            yt_prev = traj.states[0]
            start_t = 0
            for t in range(1, len(traj.states)):
                if traj.states[t] != yt_prev:
                    sparse_trajectory.append((yt_prev, start_t, t - 1))
                    yt_prev = traj.states[t]
                    start_t = t
            sparse_trajectory.append((yt_prev, start_t, len(traj.states) - 1))

        per_traj_data.append({
            "trajectory":
            traj,
            "sparse_trajectory":
            sparse_trajectory,
            "traj_len":
            traj_len,
            "ground_causal_processes":
            ground_processes,
            "start_times_per_gp":
            start_times,
            "atom_to_val_to_gps":
            atom_to_val_to_gps,
            "all_atoms":
            utils.all_possible_ground_atoms(
                traj._low_level_states[0],  # pylint: disable=protected-access
                predicates),
            "gp_qparam_id_map":
            gp_qparam_id_map,
            "condition_cache":
            condition_cache
        })

    # Total parameters for processes and the guide ONLY
    total_params_len = num_proc_params + q_offset
    model_params = torch.nn.Parameter(torch.randn(total_params_len) * 0.01)

    return per_traj_data, model_params, num_proc_params


def _initialize_params_with_empirical_estimates(
    trajectories: List[LowLevelTrajectory],
    predicates: Set[Predicate],
    processes: Sequence[CausalProcess],
    model_params: torch.nn.Parameter,
    _num_proc_params: int,
) -> None:
    """Initialize process parameters using empirical estimates from trajectory
    data.

    This function computes empirical delays from trajectory
    data and uses them to initialize the process parameters in
    the model_params tensor. Only the process parameters
    (first num_proc_params elements) are modified - guide parameters
    remain randomly initialized.
    """
    # Compute empirical delays for each process type
    process_delays = compute_empirical_delays(trajectories, predicates,
                                              processes)

    # Initialize process parameters with empirical estimates
    with torch.no_grad():
        for i, process in enumerate(processes):
            param_start_idx = i * 3  # 3 parameters per process

            if process.name in process_delays and len(
                    process_delays[process.name]) > 0:
                delays = torch.tensor(process_delays[process.name],
                                      dtype=torch.float32)

                # Compute mean and std
                empirical_mean = delays.mean()
                empirical_std = delays.std(
                ) if len(delays) > 1 else torch.tensor(0.1)
                empirical_std = torch.clamp(empirical_std, min=0.1)

                # Set parameters: [log_strength, log_mu, log_sigma]
                model_params.data[
                    param_start_idx] = 0.0  # log_strength = 0 (strength = 1)
                model_params.data[param_start_idx + 1] = torch.log(
                    empirical_mean)  # log_mu
                model_params.data[param_start_idx + 2] = torch.log(
                    empirical_std)  # log_sigma

                print(f"Empirically initialized process {process.name}:")
                print(f"  Mean delay: {empirical_mean:.2f}")
                print(f"  Std delay: {empirical_std:.2f}")
            else:
                # No observations - keep random initialization but log it
                print(f"Process {process.name}: No empirical "
                      "data, keeping random initialization")


def _create_guide_dict_for_trajectory(
    td: Dict[str, Any],
    guide_flat: Tensor,
    traj_len: int,
) -> Dict[_GroundCausalProcess, Dict[int, Tensor]]:
    """Helper to create the guide distribution dictionary for a single
    trajectory."""
    guide_dict: Dict[_GroundCausalProcess, Dict[int,
                                                Tensor]] = defaultdict(dict)
    for (gp, s_i), (lo, hi) in td["gp_qparam_id_map"].items():
        # Create the causality mask to prevent effects
        # from occurring at or before the cause
        mask = torch.ones(traj_len,
                          dtype=torch.float32,
                          device=guide_flat.device)
        mask[:s_i + 1] = 0

        # Current behavior: softmax over learnable logits
        raw = guide_flat[lo:hi]
        probs = torch.softmax(raw + torch.log(mask + 1e-20), dim=0)

        guide_dict[gp][s_i] = probs
    return guide_dict


def _plot_training_curve(training_curve: Dict,
                         image_dir: str = "images") -> None:
    """Plot the training curve showing ELBO over iterations."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    iterations = training_curve['iterations']
    elbos = training_curve['elbos']
    best_elbos = training_curve['best_elbos']
    wall_time = training_curve['wall_time']

    plt.figure(figsize=(18, 6))  # Adjusted figure size for three plots

    # Plot current ELBO vs Iteration
    plt.subplot(1, 2, 1)
    plt.plot(iterations, elbos, 'b-', alpha=0.7, label='Current ELBO')
    plt.plot(iterations, best_elbos, 'r-', linewidth=2, label='Best ELBO')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.title('ELBO vs Iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot ELBO vs Wall Time
    plt.subplot(1, 2, 2)
    plt.plot(wall_time, elbos, 'b-', alpha=0.7, label='Current ELBO')
    plt.plot(wall_time, best_elbos, 'r-', linewidth=2, label='Best ELBO')
    plt.xlabel('Wall Time (s)')
    plt.ylabel('ELBO')
    plt.title('ELBO vs Wall Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    filename = "training_curve.png"
    plt.savefig(os.path.join(image_dir, filename))
    logging.info(f"Training curve saved to {filename}")
    plt.close()
