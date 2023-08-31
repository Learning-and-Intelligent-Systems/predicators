import copy
import os
import sys
import time
from typing import List, Optional, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.ground_truth_models import get_gt_nsrts
from predicators.nsrt_learning.nsrt_learning_main import learn_new_nsrts_from_data
from predicators.option_model import _OptionModelBase
from predicators.planning import generate_sas_file_for_fd, fd_plan_from_sas_file, PlanningFailure
from predicators.settings import CFG
from predicators.structs import NSRT, Dataset, GroundAtom, Object, \
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
        objects = {o for atom in self._info['pre_image'] for o in atom.objects}
        objects_lst = sorted(objects)
        params = utils.create_new_variables([o.type.parent or o.type for o in objects_lst])
        obj_to_var = dict(zip(objects_lst, params))
        for atom in self._info['pre_image']:
            if len(atom.objects) == 0:
                p_local = 1
            else:
                p_local = sum(o in self._info['local_objects'] for o in atom.objects) / len(atom.objects)
            if atom in self._info['intended_added_effects']:
                if atom in self._info['explained_atoms']:
                    p_setup = 0.5
                else:
                    p_setup = 1
            elif atom in self._info['added_effects']:
                p_setup = 0.25
            else:
                p_setup = 0
            p = 0.5 * p_local + 0.5 * p_setup
            self._predicate_probabilities[atom.lift(obj_to_var)] = p
        self._predicate_probabilities = {atom: prob for atom, prob in sorted(self._predicate_probabilities.items(), key=lambda x: x[1], reverse=True)}
        # print('{')
        # for atom, prob in self._predicate_probabilities.items():
        #     print(f'\t{atom.predicate.name} ({atom.variables}): {prob}')
        # print('}')
        
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

        ### Find var_to_obj sub from precondition dict to task init state

        len_success_list = []
        len_fail_list = []
        p_success_list = []
        p_success = 0
        learned_nsrt = next(iter(self._learned_nsrts))  # It should be a single learned NSRT, so use this hack to get it
        next_predicate = 0
        while p_success < 0.99:
            curr_parameters = {o for atom in curr_preconditions | learned_nsrt.add_effects | learned_nsrt.delete_effects for o in atom.variables}
            curr_parameters |= set(learned_nsrt.option_vars)
            # curr_parameters = {o for atom in self._predicate_probabilities for o in atom.variables}
            curr_nsrt = NSRT(learned_nsrt.name, curr_parameters, curr_preconditions, learned_nsrt.add_effects,
                             learned_nsrt.delete_effects, learned_nsrt.ignore_effects, learned_nsrt.option,
                             learned_nsrt.option_vars, learned_nsrt.sampler)
            curr_nsrt.fancy_ignore_effects = learned_nsrt.fancy_ignore_effects

            try:
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
            except PlanningFailure:
                print('removing:', new_precondition)
                curr_preconditions.remove(new_precondition)
                self._predicate_probabilities[new_precondition] = 0
                new_precondition = list(self._predicate_probabilities.keys())[next_predicate]
                curr_preconditions.add(new_precondition)
                print('adding:', new_precondition)
                next_predicate += 1
            len_success_list.append(len(skeleton))
            atoms = init_atoms
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
            init_atoms = atoms

            print('skeleton:')
            for nsrt in skeleton:
                print('\t', nsrt.name, nsrt.objects)
            print()
            new_precondition = list(self._predicate_probabilities.keys())[next_predicate]
            curr_preconditions.add(new_precondition)
            print('adding:', new_precondition)
            next_predicate += 1

        print(len_success_list)
        print(len_fail_list)
        print(p_success_list)


        costs = [len_success_list[-1]]  # assume including all predicates leads to 100% p_success
        decisions = []
        for i, (len_success, len_fail, p_success) in reversed(list(enumerate(zip(len_success_list, len_fail_list, p_success_list)))):
            # Costs to try/not try without including the i-th predicate in the precondition set
            cost_try_succeed = len_success
            cost_try_fail = len_fail + costs[-1]
            cost_try = p_success * cost_try_succeed + (1 - p_success) * cost_try_fail
            cost_not_try = len_fail - 1 + costs[-1]
            print(cost_try, cost_not_try)
            if cost_try < cost_not_try:
                costs.append(cost_try)
                decisions.append(True)
            else:
                costs.append(cost_not_try)
                decisions.append(False)

        print(decisions)
        exit()




        ### Get a skeleton for solving the new task
        skeleton = []
        while not skeleton:
            try:
                skeleton, atoms_sequence = self._generate_skeleton_via_fast_downward(
                    self._train_tasks[train_task_idx],
                    self._option_model,
                    self._get_current_nsrts(),
                    self._get_current_predicates(),
                    self._types,
                    max_horizon=CFG.horizon,
                    optimal=CFG.sesame_task_planner == "fdopt")
            except PlanningFailure:
                # TODO: do stuff to make planning feasible and re-try
                # One possibility is to trim preconditions based on likelihood
                # Another possibility is to figure out which things couldn't be made true during FD, but I don't think we know how to do that
                # Another possibility is to figure out which objects aren't present (is that even possible in lifted land? probs not)
                raise

        ### TODO Run low-level search on "longest" plan? Or on "current" plan in for loop?
            # For now, I'll just assume that we can always low-level search and there's no dependency between before and after the new action
    

        ### Update precondition probabilities

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