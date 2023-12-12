import copy
import os
import sys
import time
from collections import defaultdict
from itertools import combinations
from typing import List, Optional, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.ground_truth_models import get_gt_nsrts
from predicators.nsrt_learning.nsrt_learning_main import learn_new_nsrts_from_data
from predicators.option_model import _OptionModelBase
from predicators.planning import generate_sas_file_for_fd, fd_plan_from_sas_file, PlanningFailure, run_low_level_search
from predicators.settings import CFG
from predicators.structs import NSRT, Dataset, GroundAtom, LiftedAtom, Metrics, Object, \
    ParameterizedOption, Predicate, Task, Type, _GroundNSRT

class InteractiveNSRTLearningApproach(NSRTLearningApproach):
    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        gt_nsrts = get_gt_nsrts(CFG.env, self._initial_predicates,
                                   self._initial_options)
        self._fixed_nsrts = {nsrt for nsrt in gt_nsrts if nsrt.name != CFG.excluded_nsrts}
        self._learned_nsrts: Set[NSRT] = set()
        self._online_learning_cycle = 0

    @classmethod
    def get_name(cls) -> str:
        return "interactive_nsrt_learning"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._fixed_nsrts | self._learned_nsrts

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Apply predicates to data, producing a dataset of abstract states.
        ground_atom_dataset = utils.create_ground_atom_dataset(
            dataset.trajectories, self._get_current_predicates())
        self._learned_nsrts, self._info = \
            learn_new_nsrts_from_data(dataset.trajectories,
                                      self._fixed_nsrts,
                                      self._train_tasks,
                                      self._get_current_predicates(),
                                      self._initial_options,
                                      self._action_space,
                                      ground_atom_dataset,
                                      sampler_learner=CFG.sampler_learner,
                                      annotations=dataset.annotations if dataset.has_annotations else None)

        ### Compute initial probabilities for each precondition
        self._predicate_probabilities = {}
        local_vars = set(self._info['local_vars_effects'] + self._info['local_vars_setup'])
        obj_to_var = self._info['obj_to_var']
        for atom in self._info['pre_image']:
            # Only add predicates whose objects are all "local"
            if all(o in obj_to_var and obj_to_var[o] in local_vars for o in atom.objects):
                objects_changed_by_action = all(obj_to_var[o] in self._info['local_vars_effects'] for o in atom.objects)
                objects_changed_by_setup = all(obj_to_var[o] in self._info['local_vars_effects'] + self._info['local_vars_setup'] for o in atom.objects)
                objects_changed_by_setup_action_mixed = all(obj_to_var[o] in self._info['local_vars_setup'] for o in atom.objects)
                predicate_changed_unintended = atom in (self._info['added_effects'] - self._info['intended_added_effects'])
                predicate_changed_explained = atom in (self._info['explained_atoms'] & self._info['intended_added_effects'])
                predicate_changed_by_setup = atom in self._info['added_effects']

                # I'm missing here adding the delete effects, which is another good candidate for preconditions
                # Also, this is not the most efficient way to write this if-else, but it is somewhat more readable
                if predicate_changed_by_setup and not predicate_changed_unintended and not predicate_changed_explained:
                    if objects_changed_by_action:
                        p = 0.75
                    else:
                        p = 0.5
                elif predicate_changed_by_setup and not predicate_changed_unintended and predicate_changed_explained:
                    if objects_changed_by_action:
                        p = 0.5
                    else:
                        p = 0.25
                elif predicate_changed_by_setup and predicate_changed_unintended:
                    if objects_changed_by_action:
                        p = 0.25
                    else:
                        p = 0.125
                # The predicate was not changed by setup
                elif objects_changed_by_action and objects_changed_by_setup:
                    p = 0.5
                elif objects_changed_by_action:
                    p = 0.25
                elif objects_changed_by_setup or objects_changed_by_setup_action_mixed:
                    p = 0.125
                else:
                    # print('objects_changed_by_action:', objects_changed_by_action)
                    # print('objects_changed_by_setup:', objects_changed_by_setup)
                    # print('predicate_changed_unintended:', predicate_changed_unintended)
                    # print('predicate_changed_explained:', predicate_changed_explained)
                    # print('predicate_changed_by_setup:', predicate_changed_by_setup)
                    raise ValueError("Seems like I didn't consider all cases :-(")
                self._predicate_probabilities[atom] = p

        ### This was wrong! It would be a bad idea to delete unattainable predicates, which may still encode
        # necessary properties (see my notes from 2023-09-13)
        # preds_to_delete = []
        # for atom in self._predicate_probabilities:
        #     pred_not_attainable = True
        #     for op in self._get_current_nsrts():
        #         for atom_tmp in op.add_effects:
        #             if atom.predicate.name == atom_tmp.predicate.name and atom.predicate.types == atom_tmp.predicate.types:
        #                 pred_not_attainable = False
        #                 break
        #         if not pred_not_attainable:
        #             break
        #     if pred_not_attainable:
        #         preds_to_delete.append(atom)
        # for pred in preds_to_delete:
        #     del self._predicate_probabilities[pred]
        self._predicate_probabilities = {atom.lift(obj_to_var): prob for atom, prob in sorted(self._predicate_probabilities.items(), key=lambda x: x[1], reverse=True)}
        print('(',len(self._predicate_probabilities), '):', self._predicate_probabilities)

        ### Construct sets of preconditions (initial, fixed, ...)
        self._fixed_preconditions: Set[LiftedAtom] =  set()

    def interact_and_learn(self) -> None:
        ### Pick the task that will be used for interaction
        train_task_idx = self._online_learning_cycle + 1

        ### TODO: eventually loop over orderings (hill-climbing)
        ### For the current ordering, loop over possible places to try new action
        curr_preconditions = copy.copy(self._fixed_preconditions)
        goal = self._train_tasks[train_task_idx].goal
        init_atoms = utils.abstract(self._train_tasks[train_task_idx].init, self._get_current_predicates())
        objects = list(self._train_tasks[train_task_idx].init)

        len_success_list = []
        len_fail_list = []
        p_success_list = []
        skeleton_list = []
        p_success = 0
        learned_nsrt = next(iter(self._learned_nsrts))  # It should be a single learned NSRT, so use this hack to get it
        next_predicate = 0
        atoms = init_atoms
        while True:
            curr_parameters = {o for atom in curr_preconditions | learned_nsrt.add_effects | learned_nsrt.delete_effects for o in atom.variables}
            curr_parameters |= set(learned_nsrt.option_vars)
            curr_nsrt = NSRT(learned_nsrt.name, curr_parameters, curr_preconditions, learned_nsrt.add_effects,
                             learned_nsrt.delete_effects, learned_nsrt.ignore_effects, learned_nsrt.option,
                             learned_nsrt.option_vars, learned_nsrt.sampler)
            curr_nsrt.fancy_ignore_effects = learned_nsrt.fancy_ignore_effects

            try:
                skeleton, atoms_sequence = self._generate_skeleton_via_fast_downward(
                    self._train_tasks[train_task_idx],
                    atoms,
                    objects,
                    self._option_model,
                    self._fixed_nsrts | {curr_nsrt},
                    self._get_current_predicates(),
                    self._types,
                    CFG.timeout,
                    CFG.seed,
                    max_horizon=CFG.horizon,
                    optimal=CFG.sesame_task_planner == "fdopt")
            except PlanningFailure:
                print('removing:', new_precondition)
                curr_preconditions.remove(new_precondition)
                self._predicate_probabilities[new_precondition] = 0
                if next_predicate == len(self._predicate_probabilities):
                    break
                new_precondition = list(self._predicate_probabilities.keys())[next_predicate]
                curr_preconditions.add(new_precondition)
                print()
                print('adding:', new_precondition)
                next_predicate += 1
                continue
            skeleton_list.append(skeleton)
            len_success_list.append(len(skeleton))
            for i in range(len(skeleton)):
                if skeleton[i].name == learned_nsrt.name:
                    break
                atoms = utils.apply_operator(skeleton[i], atoms)
            else:   # never had to try new NSRT, so skip learning or something
                return
            len_fail_list.append(i + 1)
            # TODO: the line below assumes that the probability of success only depends on the atoms that were explicitly in the
            # preconditions, but doesn't check if those atoms are true anyway -- this is because we don't have a good way to
            # find a matching from objects to variables (see my own notes from [2023-08-29])
            p_success = np.prod([1 - prob for atom, prob in self._predicate_probabilities.items() if atom not in curr_preconditions])
            print('p succes:', p_success)
            p_success_list.append(p_success)

            print('skeleton:')
            for nsrt in skeleton:
                print('\t', nsrt.name, nsrt.objects)
            print()
            if next_predicate == len(self._predicate_probabilities):
                break
            new_precondition = list(self._predicate_probabilities.keys())[next_predicate]
            curr_preconditions.add(new_precondition)
            print()
            print('adding:', new_precondition)
            next_predicate += 1

        # print(len(len_success_list), len_success_list)
        # print(len(len_fail_list), len_fail_list)
        # print(len(p_success_list), p_success_list)
        # print(self._predicate_probabilities)
        # for i, pred in enumerate(self._predicate_probabilities):
        #     print(pred, self._predicate_probabilities[pred], len_success_list[i], len_fail_list[i], p_success_list[i])

        costs = [len_success_list[-1]]  # assume including all predicates leads to 100% p_success
        decisions = []


        curr_predicate_probabilities = {k: v for k, v in self._predicate_probabilities.items() if v > 0}

        # for i, (pred, len_success, len_fail, p_success) in reversed(list(enumerate(zip(curr_predicate_probabilities, len_success_list, len_fail_list, p_success_list)))):
        for i in range(len(curr_predicate_probabilities) - 1, -1, -1):
            pred = list(curr_predicate_probabilities.keys())[i]
            len_success = len_success_list[i + 1]
            len_fail = len_fail_list[i + 1]
            p_success = p_success_list[i + 1]
            skeleton = skeleton_list[i + 1]

            # Costs to try/not try without including the i-th predicate in the precondition set
            cost_try_succeed = len_success
            cost_try_fail = len_fail + costs[-1]
            cost_try = p_success * cost_try_succeed + (1 - p_success) * cost_try_fail
            cost_not_try = len_fail - 1 + costs[-1]

            if self._online_learning_cycle == 0:
                # In the first loop, try everything
                if skeleton[0].name != learned_nsrt.name:
                    costs.append(cost_try)
                    decisions.append(True)
                else:
                    costs.append(cost_not_try)
                    decisions.append(False)
            elif cost_try < cost_not_try:
                costs.append(cost_try)
                decisions.append(True)
            else:
                costs.append(cost_not_try)
                decisions.append(False)
            print(pred, self._predicate_probabilities[pred], len_success, len_fail, p_success, cost_try, cost_not_try, decisions[-1])
        decisions = decisions[::-1]

        ### TODO Run low-level search on "longest" plan? Or on "current" plan in for loop?
        # For now, I'll just assume that we can always low-level search and there's no dependency between before and after the new action
        curr_preconditions = set()
        failed_once = False
        for pred, try_without in zip(curr_predicate_probabilities, decisions):
            print(pred, try_without)
            if try_without:
                curr_parameters = {o for atom in curr_preconditions | learned_nsrt.add_effects | learned_nsrt.delete_effects for o in atom.variables}
                curr_parameters |= set(learned_nsrt.option_vars)
                curr_nsrt = NSRT(learned_nsrt.name, curr_parameters, curr_preconditions, learned_nsrt.add_effects,
                                 learned_nsrt.delete_effects, learned_nsrt.ignore_effects, learned_nsrt.option,
                                 learned_nsrt.option_vars, learned_nsrt.sampler)
                curr_nsrt.fancy_ignore_effects = learned_nsrt.fancy_ignore_effects
                skeleton, atoms_sequence = self._generate_skeleton_via_fast_downward(
                    self._train_tasks[train_task_idx],
                    init_atoms,
                    objects,
                    self._option_model,
                    self._fixed_nsrts | {curr_nsrt},
                    self._get_current_predicates(),
                    self._types,
                    CFG.timeout,
                    CFG.seed,
                    max_horizon=CFG.horizon,
                    optimal=CFG.sesame_task_planner == "fdopt")
                partial_skeleton = []
                partial_atoms_sequence = [atoms_sequence[0]]
                for i in range(len(skeleton)):
                    partial_skeleton.append(skeleton[i])
                    partial_atoms_sequence.append(atoms_sequence[i+1])
                    if skeleton[i].name == learned_nsrt.name:
                        break
                
                metrics: Metrics = defaultdict(float)
                plan, suc, traj = run_low_level_search(self._train_tasks[train_task_idx],
                                                 self._option_model, partial_skeleton,
                                                 partial_atoms_sequence, CFG.seed,
                                                 CFG.timeout, metrics, CFG.horizon,
                                                 return_traj=True)
                init_atoms = utils.abstract(traj[-1], self._get_current_predicates())   # for next round

                if suc:
                    print('Succeeded! Using:', curr_preconditions)
                    break

                # Otherwise, this is our first failure. Mark the state prior to failure and move on
                failed_once = True
                failure_atoms = utils.abstract(traj[-2], self._get_current_predicates())    # to compute delta that leads to success
            # Grow the precondition set
            curr_preconditions.add(pred)
        else:   # didn't succeed yet, repeat
            curr_parameters = {o for atom in curr_preconditions | learned_nsrt.add_effects | learned_nsrt.delete_effects for o in atom.variables}
            curr_parameters |= set(learned_nsrt.option_vars)
            curr_nsrt = NSRT(learned_nsrt.name, curr_parameters, curr_preconditions, learned_nsrt.add_effects,
                             learned_nsrt.delete_effects, learned_nsrt.ignore_effects, learned_nsrt.option,
                             learned_nsrt.option_vars, learned_nsrt.sampler)
            curr_nsrt.fancy_ignore_effects = learned_nsrt.fancy_ignore_effects
            skeleton, atoms_sequence = self._generate_skeleton_via_fast_downward(
                self._train_tasks[train_task_idx],
                init_atoms,
                objects,
                self._option_model,
                self._fixed_nsrts | {curr_nsrt},
                self._get_current_predicates(),
                self._types,
                CFG.timeout,
                CFG.seed,
                max_horizon=CFG.horizon,
                optimal=CFG.sesame_task_planner == "fdopt")
            partial_skeleton = []
            partial_atoms_sequence = [atoms_sequence[0]]
            for i in range(len(skeleton)):
                partial_skeleton.append(skeleton[i])
                partial_atoms_sequence.append(atoms_sequence[i+1])
                if skeleton[i].name == learned_nsrt.name:
                    break
            
            metrics: Metrics = defaultdict(float)
            plan, suc, traj = run_low_level_search(self._train_tasks[train_task_idx],
                                             self._option_model, partial_skeleton,
                                             partial_atoms_sequence, CFG.seed,
                                             CFG.timeout, metrics, CFG.horizon,
                                             return_traj=True)

            if suc:
                print("Succeeded! Using:", curr_preconditions)

        ### Keep only atoms in the intersection
        new_ground_nsrt = partial_skeleton[-1]
        local_objs = new_ground_nsrt.objects
        local_atoms = {atom for atom in partial_atoms_sequence[-2] if all(o in local_objs for o in atom.objects)}

        # Find the matching between local_atoms and _predicate_probabilities
        obj_to_var = {}
        # drop_variables = []
        for var, obj in new_ground_nsrt.var_to_obj.items():
            if obj in obj_to_var:
                if var in self._info['local_vars_effects']:
                    vart_tmp = obj_to_var[obj]
                    # drop_variables.append(vart_tmp)
                    obj_to_var[obj] = var
            else:
                obj_to_var[obj] = var

        local_atoms_lifted = {atom.lift(obj_to_var) for atom in local_atoms}
        if failed_once:
            failure_atoms_lifted = {atom.lift(obj_to_var) for atom in failure_atoms if all(o in obj_to_var for o in atom.objects)}

        # sort in increasing order of probs so the renaming keeps the highest-prob atoms
        self._predicate_probabilities = {pred: prob for pred, prob in sorted(self._predicate_probabilities.items(), key=lambda x: x[1])}

        for pred in self._predicate_probabilities:
            if pred not in local_atoms_lifted:
                local_vars = set(self._info['local_vars_effects'] + self._info['local_vars_setup'])
                if all(o in self._info['local_vars_effects'] for o in pred.variables):
                    # In this case, we know the correct binding, and the atom didn't hold, so we can drop it
                    self._predicate_probabilities[pred] = 0
                else:
                    # We are only "guessing" that the atom didn't hold because it is over unaffected objects 
                    # and we didn't deliberately set it in the preconditions, so we just decrease its p
                    self._predicate_probabilities[pred] /= 2

        if failed_once:
            set_that_leads_to_success = (local_atoms_lifted - failure_atoms_lifted).intersection(self._predicate_probabilities.keys())

            def inclusion_exclusion_probabilities(event_probabilities, max_depth=5):
                n = len(event_probabilities)
                total_probability = sum(event_probabilities)

                inclusion_terms = [(-1) ** i * sum( [np.prod(comb) for comb in combinations(event_probabilities, i + 1)] ) for i in range(min(max_depth, n))]

                probability_union = sum(inclusion_terms)
                return probability_union

            normalizer = inclusion_exclusion_probabilities([self._predicate_probabilities[pred] for pred in set_that_leads_to_success])
            for pred in set_that_leads_to_success:
                self._predicate_probabilities[pred] /= normalizer
        self._predicate_probabilities = {k: v for k, v in self._predicate_probabilities.items() if v > 0}
        self._predicate_probabilities = {pred: prob for pred, prob in sorted(self._predicate_probabilities.items(), key=lambda x: x[1], reverse=True)}
        print('(',len(self._predicate_probabilities), '):', self._predicate_probabilities)
        self._online_learning_cycle += 1
        return

    def _generate_skeleton_via_fast_downward(self,
        task: Task, init_atoms: List[Set[GroundAtom]], objects: List[Object],
        option_model: _OptionModelBase, nsrts: Set[NSRT],
        predicates: Set[Predicate], types: Set[Type], timeout: float, seed: int,
        max_horizon: int, optimal: bool
    ) -> Tuple[List[_GroundNSRT], List[Set[GroundAtom]]] :  # pragma: no cover 
        timeout_cmd = "gtimeout" if sys.platform == "darwin" else "timeout"
        if optimal:
            alias_flag = "--alias seq-opt-lmcut"
        else:  # satisficing
            alias_flag = "--alias lama-first"
            # alias_flag = "--alias seq-sat-lama-2011"
        # Run Fast Downward followed by cleanup. Capture the output.
        assert "FD_EXEC_PATH" in os.environ, \
            "Please follow the instructions in the docstring of this method!"
        fd_exec_path = os.environ["FD_EXEC_PATH"]
        exec_str = os.path.join(fd_exec_path, "fast-downward.py")
        start_time = time.perf_counter()
        sas_file = generate_sas_file_for_fd(task, nsrts, predicates, types,
                                            timeout, timeout_cmd, alias_flag,
                                            exec_str, objects, init_atoms)

        skeleton, atoms_sequence, metrics = fd_plan_from_sas_file(
            sas_file, timeout_cmd, timeout, exec_str, alias_flag, start_time,
            objects, init_atoms, nsrts, max_horizon)
        necessary_atoms_seq = utils.compute_necessary_atoms_seq(
                skeleton, atoms_sequence, task.goal)
        skeleton, necessary_atoms_seq = utils.trim_skeleton_to_necessary_atoms(
            skeleton, necessary_atoms_seq)
        return skeleton, necessary_atoms_seq