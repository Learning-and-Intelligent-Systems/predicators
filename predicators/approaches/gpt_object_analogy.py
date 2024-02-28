"""Use GPT for cross-domain policy learning in PG3.

Command for testing gripper / ferry:
    python predicators/main.py --approach gpt_obj --seed 0  \
        --env pddl_ferry_procedural_tasks --strips_learner oracle \
        --num_train_tasks 20 --pg3_init_policy gripper_ldl_policy.txt \
        --pg3_init_base_env pddl_gripper_procedural_tasks \
        --pg3_add_condition_allow_new_vars False

    python predicators/main.py --approach gpt_obj --seed 0  \
        --env pddl_gripper_procedural_tasks --strips_learner oracle \
        --num_train_tasks 20 --pg3_init_policy ferry_ldl_policy.txt \
        --pg3_init_base_env pddl_ferry_procedural_tasks \
        --pg3_add_condition_allow_new_vars False

    python predicators/main.py --approach gpt_obj --seed 0  \
        --env pddl_miconic_procedural_tasks --strips_learner oracle \
        --num_train_tasks 20 --pg3_init_policy gripper_ldl_policy.txt \
        --pg3_init_base_env pddl_gripper_procedural_tasks \
        --pg3_add_condition_allow_new_vars False

"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Iterator, List, Set, Tuple

import smepy
import numpy as np
import itertools
import math
from collections import Counter

from predicators.approaches.pg3_analogy_approach import PG3AnalogyApproach
from predicators.envs.base_env import BaseEnv
from predicators.settings import CFG
from predicators.structs import NSRT, LDLRule, LiftedAtom, \
    LiftedDecisionList, Predicate, Type, Variable, _GroundNSRT
from predicators.structs import Action, Box, Dataset, GroundAtom, \
    LiftedDecisionList, Object, ParameterizedOption, Predicate, State, Task, \
    Type, _GroundNSRT
from predicators.envs.pddl_env import _action_to_ground_strips_op
from predicators.envs import create_new_env
from predicators.datasets import create_dataset
from predicators.ground_truth_models import get_gt_options

from predicators import utils

DEBUG = True

class GPTObjectApproach(PG3AnalogyApproach):
    """Use GPT for cross-domain policy learning in PG3."""

    @classmethod
    def get_name(cls) -> str:
        return "gpt_obj"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Get a task and trajectory for correctness of generated conditions
        self._base_env = create_new_env(CFG.pg3_init_base_env)
        self._base_task = [t.task for t in self._base_env.get_train_tasks()]

        self._target_tasks = self._train_tasks
        self._target_states = [dataset.trajectories[i]._states for i in range(len(dataset.trajectories))]
        ordered_objs = [list(dataset.trajectories[i]._states[0]) for i in range(len(dataset.trajectories))]
        self._target_env = create_new_env(CFG.env)

        self._target_actions = []
        for i in range(len(dataset.trajectories)):
            self._target_actions.append([_action_to_ground_strips_op(a, ordered_objs[i], sorted(self._target_env._strips_operators)) for a in dataset.trajectories[i]._actions])
        super().learn_from_offline_dataset(dataset)

    def _induce_policies_by_analogy(
            self, base_policy: LiftedDecisionList, base_env: BaseEnv,
            target_env: BaseEnv, base_nsrts: Set[NSRT],
            target_nsrts: Set[NSRT]) -> List[LiftedDecisionList]:
        # Determine analogical mappings between the current env and the
        # base env that the initialized policy originates from.
        self._base_nsrts = base_nsrts
        self._target_nsrts = target_nsrts

        final_rules = []
        for rule in base_policy.rules:
            target_rule = self._generate_rules(rule)
            final_rules.append(target_rule)

        return [LiftedDecisionList(final_rules)]


    def _generate_rules(self, rule: LDLRule) -> List[LDLRule]:
        # Generates "best" rules in target environment given rule in base environment
        target_nsrts = self._get_analagous_nsrts(rule) 
        for target_nsrt in target_nsrts:
            new_rule = self._generate_best_rule(rule, target_nsrt)
    
    def _generate_best_rule(self, rule: LDLRule, target_nsrt: NSRT) -> LDLRule:
        # Generates best rule in target environment given rule in base environment and a target nsrt

        mapping = {} # Mapping from target param to rule variable
        for target_nsrt_param in target_nsrt.parameters:
            # Ask GPT what is the best variable in rule for target_nsrt_param
            best_rule_variable = self._get_analagous_variable(target_nsrt_param, rule, target_nsrt)
            if best_rule_variable != None:
                mapping[target_nsrt_param] = best_rule_variable
        
        # Search for a state that we can ground on the target environment
        best_score = -1.0
        best_object_mapping = None
        best_index = None
        best_task_index = None
        for task_index in range(len(self._target_tasks)):
            for i in range(len(self._target_states[task_index])-1):
                target_action = self._target_actions[task_index][i]
                if target_action.name != target_nsrt.name:
                    continue

                object_mapping, gini_index = self._get_object_distribution_and_score(rule, target_nsrt, mapping, task_index, i)
                if gini_index > best_score:
                    best_task_index = task_index
                    best_object_mapping = object_mapping
                    best_score = gini_index
                    best_index = i
            
        # Get useful objects
        one_to_one_object_mapping = self._filter_object_mapping(best_object_mapping)
        useful_objects = set(one_to_one_object_mapping.values())
        useful_objects.update(self._target_actions[best_task_index][best_index].objects)

        # Getting useful analagous predicates
        useful_pos_predicates = set()
        useful_goal_predicates = set()
        for pos_condition in rule.pos_state_preconditions:
            pos_predicate = pos_condition.predicate
            possible_pred_names = self._get_analagous_predicate_names(pos_predicate)
            if possible_pred_names is not None:
                pos_analagous_predicate_names = set(possible_pred_names)
                useful_pos_predicates.update(pos_analagous_predicate_names)
        for goal_condition in rule.goal_preconditions:
            goal_predicate = goal_condition.predicate
            goal_analagous_predicate_names = set(self._get_analagous_predicate_names(goal_predicate))
            useful_goal_predicates.update(goal_analagous_predicate_names)

        # Getting final ranking of atoms in ground state
        ranked_atoms = self._score_conds(useful_objects, useful_pos_predicates, useful_goal_predicates, best_task_index, best_index)
        if DEBUG:
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("RULE\n", rule)
            print("GROUND ACTION\n", self._target_actions[best_task_index][best_index])
            print("GROUND ACTIONS\n", [self._target_actions[best_task_index][i].name for i in range(len(self._target_actions[best_task_index]))])
            print("BEST TASK INDEX", best_task_index)
            print("BEST INDEX: ", best_index)
            print("GROUND STATE\n", sorted(self._target_states[best_task_index][best_index].simulator_state))
            print("GOAL\n", self._target_tasks[best_task_index].goal)
            print("USEFUL OBJECTS", useful_objects)
            print("USEFUL POS PREDICATES", useful_pos_predicates)
            print("USEFUL GOAL PREDICATES", useful_goal_predicates)
            print("PRECONDS", ranked_atoms)
        
    def _score_conds(self, useful_objects, useful_pos_preds, useful_goal_preds, task_index, index):
        ground_state = self._target_states[task_index][index] 
        useful_pos_atoms = {}
        final_atoms = set()
        for atom in ground_state.simulator_state:
            score = 0.0
            pred = atom.predicate
            if pred.name in useful_pos_preds:
                score += 0.5 
            for obj in atom.objects:
                if obj in useful_objects:
                    score += 1.5
            if score > 0.0:
                useful_pos_atoms[atom] = score
                final_atoms.add(atom)
        sorted_atoms, sorted_scores = sort_atoms_by_score(useful_pos_atoms)
        return sorted_atoms

    def _filter_object_mapping(self, object_mapping):
        final_mapping = {}
        rankings = []
        for var in object_mapping:
            for obj, obj_score in object_mapping[var].items():
                rankings.append((obj_score, obj, var))
        rankings.sort(reverse = True)
        for ranking in rankings:
            obj = ranking[1]
            var = ranking[2]
            if var not in final_mapping.keys() and obj not in final_mapping.values():
                final_mapping[var] = obj
        return final_mapping

    def _get_object_distribution_and_score(self, rule: LDLRule, target_nsrt: NSRT, existing_mapping: Dict[Variable, Variable], task_index: int, state_index: int):
        all_rule_variables = set([var for var in rule.parameters])
        used_rule_variables = set(existing_mapping.values())

        ground_target_nsrt = self._target_actions[task_index][state_index]
        mapping_to_objects = {}
        for target_var, base_var in existing_mapping.items():
            index_of_object = target_nsrt.parameters.index(target_var)
            target_object = ground_target_nsrt.objects[index_of_object]
            mapping_to_objects[base_var] = target_object

        ground_state = self._target_states[task_index][state_index]
        goal_state = self._target_tasks[task_index].goal
        if DEBUG:
            print("=================================")
            print(f"Rule:\n{rule}")
            print(f"Target NSRT:\n{target_nsrt}")
            print(f"Target Ground NSRT:\n{ground_target_nsrt}")
            print(f"Index: ", state_index)
            print(f"Ground state:\n{sorted(ground_state.simulator_state)}")
            print(f"Goal state:\n{goal_state}")
        state_entropy = 0.0

        var_to_obj_distributions = {}
        for unused_rule_var in sorted(all_rule_variables - used_rule_variables):

            pos_conditions = set()
            for condition in rule.pos_state_preconditions:
                if unused_rule_var in condition.variables:
                    pos_conditions.add(condition)
            
            goal_conditions = set()
            for condition in rule.goal_preconditions:
                if unused_rule_var in condition.variables:
                    goal_conditions.add(condition)
            
            object_distribution = {}
            for pos_cond in pos_conditions:
                analagous_preds = self._get_analagous_predicate_names(pos_cond.predicate)
                if analagous_preds == None:
                    continue

                needed_objects = set()
                for var in pos_cond.variables:
                    if var in mapping_to_objects:
                        needed_objects.add(mapping_to_objects[var])

                for pos_atom in ground_state.simulator_state:
                    # if analagous predicate
                    if pos_atom.predicate.name in analagous_preds and needed_objects.issubset(set(pos_atom.objects)):
                        for obj in pos_atom.objects:
                            
                            if obj in needed_objects:
                                continue

                            if obj not in object_distribution:
                                #print(f"OBJ {obj} HIT! with {pos_atom}")
                                object_distribution[obj] = 1
                            else:
                                #print(f"OBJ {obj} HIT! with {pos_atom}")
                                object_distribution[obj] += 1
            for goal_cond in goal_conditions:
                analagous_preds = self._get_analagous_predicate_names(goal_cond.predicate)
                if analagous_preds == None:
                    continue

                needed_objects = set()
                for var in pos_cond.variables:
                    if var in mapping_to_objects:
                        needed_objects.add(mapping_to_objects[var])

                for goal_atom in goal_state:
                    if goal_atom.predicate.name in analagous_preds and needed_objects.issubset(set(goal_atom.objects)):
                        for obj in goal_atom.objects:

                            if obj in needed_objects:
                                continue

                            if obj not in object_distribution:
                                object_distribution[obj] = 1
                            else:
                                object_distribution[obj] += 1

            object_entropy = gini_index_from_dict(object_distribution)
            state_entropy += object_entropy
            if object_entropy > 0.0: # If some useful information, add the distribution
                var_to_obj_distributions[unused_rule_var] = object_distribution
            else:
                var_to_obj_distributions[unused_rule_var] = {}
            if DEBUG:
                print('-----------------')
                print(f"Unused Rule Var {unused_rule_var}")
                print(f"Pos Conditions {pos_conditions}")
                print(f"Goal Conditions {goal_conditions}")
                print(f"Candidates: {object_distribution}")
                print(f"Object Gini: {object_entropy}")
                print(f"State Gini: {state_entropy}")
        return var_to_obj_distributions, state_entropy

    def _get_analagous_predicate_names(self, pred):
        # Gripper -> Ferry

        if 'gripper' in self._base_env.get_name() and 'ferry' in self._target_env.get_name():
            predicate_input = {
                "ball": ["car"],
                "free": ["empty-ferry"],
                "room": ["location"],
                "at-robby": ["at-ferry"],
                "at": ["at"],
                "carry": ["on"],
            }

        # Ferry -> Gripper
        if 'ferry' in self._base_env.get_name() and 'gripper' in self._target_env.get_name():
            predicate_input = {
                "car": ["ball"],
                "empty-ferry": ["free"],
                "location": ["room"],
                "at-ferry": ["at-robby"],
                "at": ["at"],
                "on": ["carry"],
            }

        # Gripper -> Detyped Miconic
        if 'gripper' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            predicate_input = {
                "ball": ["passenger"],
                "room": ["floor"],
                "at-robby": ["lift-at"],
                "at": ["origin", "destin"],
                "carry": ["boarded"],
            }

        # Ferry -> Detyped Miconic
        if 'ferry' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            predicate_input = {
                "car": ["ball"],
                "empty-ferry": ["free"],
                "location": ["room"],
                "at-ferry": ["at-robby"],
                "at": ["at"],
                "on": ["carry"],
            }

        target_env_pred_names_to_pred = {str(pred): pred for pred in self._target_env.predicates}
        ans = []
        if pred.name in predicate_input:
            for name in predicate_input[pred.name]:
                ans.append(target_env_pred_names_to_pred[name].name)
            return ans
        else:
            return None
    
    def _find_objects_and_conditions(self, rule: LDLRule, target_nsrt: NSRT, existing_mapping: Dict[Variable, Variable], state_index: int) -> Tuple([GroundLDLRule, float]):
        # Note: state_index is the index of self._target_states that we try grounding on
        all_rule_variables = set([var for var in rule.parameters])
        used_rule_variables = set(existing_mapping.values())
        
        # Getting mapping from base variable to object using the plan's ground action
        ground_target_nsrt = self._target_actions[state_index]
        mapping_to_objects = {}
        for target_var, base_var in existing_mapping.items():
            index_of_object = target_nsrt.parameters.index(target_var)
            target_object = ground_target_nsrt.objects[index_of_object]
            mapping_to_objects[base_var] = target_object
        
        ground_state = self._target_states[state_index]
        available_objects = sorted(set(ground_state.data.keys()) - set(mapping_to_objects.values()))

        final_ground_objects = set(mapping_to_objects.values())

        # Finding best object in ground state TODO: CAN CHANGE THIS: RIGHT NOW THIS IS A DETERMINISTIC PROCESS
        for unused_rule_var in sorted(all_rule_variables - used_rule_variables):
            relevant_variables = set([])
            for condition in rule.pos_state_preconditions:
                if unused_rule_var in condition.variables:
                    relevant_variables.update(condition.variables)
            # TODO: add neg_state_preconditions? (to find relevant variables)
            for condition in rule.goal_preconditions:
                if unused_rule_var in condition.variables:
                    relevant_variables.update(condition.variables)
            relevant_variables.remove(unused_rule_var)
            relevant_objects = set([mapping_to_objects[var] for var in relevant_variables])

            relevant_atoms = set()
            for atom in ground_state.simulator_state:
                if len(relevant_objects & set(atom.objects)) > 0 and len(set(atom.objects) - relevant_objects) > 0 : # If some relevant object is in atom, but not the only atom, include it
                    relevant_atoms.add(atom)
            
            best_objects = set()
            for relevant_atom in relevant_atoms:
                for obj in relevant_atom.objects:
                    if obj not in relevant_objects:
                        best_objects.add(obj)
            
            best_object = None
            if len(best_objects) > 0:
                best_object = sorted(best_objects)[0] # TODO: Probably find a better way to do this
                final_ground_objects.add(best_object)

        # Getting ground conditions for rule with desired objects 
        final_preconditions = set()
        final_goalconditions = set()

        for ground_atom in ground_state.simulator_state:
            # if all objects in ground_atom are in final_ground_objects
            ground_atom_objects = set(ground_atom.objects)
            if ground_atom_objects.issubset(final_ground_objects):
                final_preconditions.add(ground_atom)

        for ground_atom in self._target_task.goal:
            # if all objects in ground_atom are in final_ground_objects
            ground_atom_objects = set(ground_atom.objects)
            if ground_atom_objects.issubset(final_ground_objects):
                final_goalconditions.add(ground_atom)
        
        return final_ground_objects, final_preconditions, final_goalconditions

    def _design_lifted_rule_from_conditions(self, objects, target_nsrt, preconditions, goalconditions):
        # 1. Find asssignment of target_nsrt to preconditions (assumes only one)
        param_to_pos_preds: Dict[Variable, Set[Predicate]] = {
            p: set()
            for p in target_nsrt.parameters
        }
        object_to_loc = {
            o: set()
            for o in objects
        }
        for atom in target_nsrt.preconditions:
            for i in range(len(atom.variables)):
                var = atom.variables[i]
                param_to_pos_preds[var].add((atom.predicate.name, i))

        for atom in preconditions:
            for i in range(len(atom.objects)):
                obj = atom.objects[i]
                object_to_loc[obj].add((atom.predicate.name, i))
        
        mapping = {} # From var in target_nsrt to object
        param_choices = []  # list of lists of possible objects for each param
        for var in target_nsrt.parameters:
            var_locs = param_to_pos_preds[var]
            param_choices.append([])
            for obj, obj_locs in object_to_loc.items():
                if var_locs.issubset(obj_locs):
                    param_choices[-1].append(obj)
            # No satisfying assignment/matching
            if len(param_choices[-1]) == 0:
                return None, 0.0

        min_param = None
        min_unique_count = float('inf')
        for choice in itertools.product(*param_choices):
            unique_count = len(set(choice))
            if unique_count <= min_unique_count:
                min_param = choice
                min_unique_count = unique_count
        
        ground_nsrt = target_nsrt.ground(min_param)

        # 2. Create a lifted rule from this

        lifted_variables = target_nsrt.parameters.copy()
        mapping = {} # Mapping from object to variable
        for i in range(len(min_param)):
            mapping[min_param[i]] = target_nsrt.parameters[i]
        
        for obj in objects:
            if obj not in mapping.keys():
                new_var = _convert_object_to_variable(obj)
                mapping[obj] = new_var
                lifted_variables.append(new_var)

        lifted_preconditions = set()
        for precondition in preconditions:
            predicate = precondition.predicate
            params = [mapping[precondition.objects[i]] for i in range(len(precondition.objects))]
            lifted_preconditions.add(predicate(params))
        
        lifted_goalconditions = set()
        for precondition in goalconditions:
            predicate = precondition.predicate
            params = [mapping[precondition.objects[i]] for i in range(len(precondition.objects))]
            lifted_goalconditions.add(predicate(params))
        lifted_rule = LDLRule("gen-rule", lifted_variables, lifted_preconditions, set(), lifted_goalconditions, target_nsrt)
        return lifted_rule, 0.0

    
    def _get_analagous_nsrts(self, rule: LDLRule) -> List[NSRT]:
        # Returns NSRT(s) in target environment that is analagous to the NSRT in rule
        nsrt_input = None
        # Gripper -> Ferry
        if 'gripper' in self._base_env.get_name() and 'ferry' in self._target_env.get_name():
            nsrt_input = { "move": ["sail"], "pick": ["board"], "drop": ["debark"], }

        # Ferry -> Gripper
        if 'ferry' in self._base_env.get_name() and 'gripper' in self._target_env.get_name():
            nsrt_input = { "sail": ["move"], "board": ["pick"], "debark": ["drop"], }

        # Gripper -> Detyped Miconic
        if 'gripper' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            nsrt_input = { "move": ["up"], "pick": ["board"], "drop": ["depart"], }

        # Ferry -> Detyped Miconic
        if 'ferry' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            nsrt_input = { "sail": ["up"], "board": ["board"], "debark": ["depart"], }

        target_env_nsrt_name_to_nsrt = {nsrt.name: nsrt for nsrt in self._target_nsrts}
        analagous_target_nsrts = [target_env_nsrt_name_to_nsrt[nsrt_name] for nsrt_name in nsrt_input[rule.nsrt.name]]
        return analagous_target_nsrts

    def _get_analagous_variable(self, nsrt_param: Variable, rule: LDLRule, target_nsrt: NSRT) -> LDLRule:
        # Ask GPT what the best variable in rule for nsrt_param

        variable_input = None
        # Gripper -> Ferry
        if 'gripper' in self._base_env.get_name() and 'ferry' in self._target_env.get_name():
            variable_input = {
                ("sail", "move") : {"?from": "?from", "?to": "?to"},
                ("board", "pick") : {"?car": "?obj", "?loc": "?room"},
                ("debark", "drop") : {"?car": "?obj", "?loc": "?room"},
            }
        # Ferry -> Gripper
        if 'ferry' in self._base_env.get_name() and 'gripper' in self._target_env.get_name():
            variable_input = {
                ("move", "sail") : {"?from": "?from", "?to": "?to"},
                ("pick", "board") : {"?obj": "?car", "?room": "?loc"},
                ("drop", "debark") : {"?obj": "?car", "?room": "?loc"},
            }
        # Gripper -> Detyped Miconic
        if 'gripper' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            variable_input = {
                ("up", "move") : {"?f1": "?from", "?f2": "?to"},
                ("board", "pick") : {"?p": "?obj", "?f": "?room"},
                ("depart", "drop") : {"?p": "?obj", "?f": "?room"},
            }
        # Ferry -> Detyped Miconic
        if 'ferry' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            variable_input = {
                ("up", "sail") : {"?f1": "?from", "?f2": "?to"},
                ("board", "board") : {"?p": "?car", "?f": "?loc"},
                ("depart", "debark") : {"?p": "?car", "?f": "?loc"},
            }

        if nsrt_param.name in variable_input[(target_nsrt.name, rule.nsrt.name)]:
            var_name = variable_input[(target_nsrt.name, rule.nsrt.name)][nsrt_param.name]
            base_var_names_to_var = {var.name: var for var in rule.parameters}
            return base_var_names_to_var[var_name]
        else:
            return None


def _convert_object_to_variable(obj: Object) -> Variable:
    return obj.type("?" + obj.name)

def gini_index_from_dict(distribution_dict):
    """
    Calculate the Gini index for a given discrete distribution represented as a dictionary.

    :param distribution_dict: A dictionary where keys are elements and values are their frequencies
    :return: The Gini index as a float
    """
    values = list(distribution_dict.values())
    n = len(values)

    # If there are no elements in the distribution, return 0
    if n == 0:
        return 0

    sum_of_values = sum(values)
    sorted_values = sorted(values)

    # If the sum of values is 0 (all values are 0), return 0 to avoid division by zero
    if sum_of_values == 0:
        return 0

    cumulative_sum = 0
    for i, value in enumerate(sorted_values, 1):
        cumulative_sum += value * i

    # Gini index calculation
    gini = (2 * cumulative_sum) / (n * sum_of_values) - (n + 1) / n
    return gini

def sort_atoms_by_score(atoms_dict):
    # Group atoms by their scores
    grouped_atoms = {}
    for atom, score in atoms_dict.items():
        if score in grouped_atoms:
            grouped_atoms[score].append(atom)
        else:
            grouped_atoms[score] = [atom]

    # Sort the scores in descending order
    sorted_scores = sorted(grouped_atoms, reverse=True)

    # Arrange the atoms according to the sorted scores
    sorted_atoms = [grouped_atoms[score] for score in sorted_scores]
    
    return sorted_atoms, sorted_scores