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


class GPTObjectApproach(PG3AnalogyApproach):
    """Use GPT for cross-domain policy learning in PG3."""

    @classmethod
    def get_name(cls) -> str:
        return "gpt_obj"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Get a task and trajectory for correctness of generated conditions
        self._base_env = create_new_env(CFG.pg3_init_base_env)
        self._base_task = [t.task for t in self._base_env.get_train_tasks()]

        for i in range(len(self._train_tasks)):
            if len(dataset.trajectories[i]._states) == 5:
                final_i = i
                break

        self._target_task = self._train_tasks[final_i]
        self._target_states = dataset.trajectories[final_i]._states
        ordered_objs = list(self._target_states[final_i])
        self._target_env = create_new_env(CFG.env)
        self._target_actions = [_action_to_ground_strips_op(a, ordered_objs, sorted(self._target_env._strips_operators)) for a in dataset.trajectories[final_i]._actions]
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
            import ipdb; ipdb.set_trace();
    
    def _generate_best_rule(self, rule: LDLRule, target_nsrt: NSRT) -> LDLRule:
        # Generates best rule in target environment given rule in base environment and a target nsrt
        mapping = {} # Mapping from target param to rule variable
        all_rule_variables = set([var for var in rule.parameters])
        used_rule_variables = set()
        for target_nsrt_param in target_nsrt.parameters:
            # Ask GPT what is the best variable in rule for target_nsrt_param
            best_rule_variable = self._get_analagous_variable(target_nsrt_param, rule, target_nsrt)
            if best_rule_variable != None:
                mapping[target_nsrt_param] = best_rule_variable
        
        # Search for a state that we can ground on the target environment
        best_score = -1.0
        best_rule = None
        for i in range(len(self._target_states)-1):
            target_action = self._target_actions[i]
            if target_action.name != target_nsrt.name:
                continue

            self._experiment(rule, target_nsrt, mapping, i)
            import ipdb; ipdb.set_trace();
        """
            ground_objects, preconditions, goalconditions = self._find_objects_and_conditions(rule, target_nsrt, mapping, i)
            candidate_rule, score = self._design_lifted_rule_from_conditions(ground_objects, target_nsrt, preconditions, goalconditions)
            if score > best_score:
                best_rule = candidate_rule
                best_score = score
        
        return candidate_rule
        """

    def _experiment(self, rule: LDLRule, target_nsrt: NSRT, existing_mapping: Dict[Variable, Variable], state_index: int) -> Tuple([GroundLDLRule, float]):
        all_rule_variables = set([var for var in rule.parameters])
        used_rule_variables = set(existing_mapping.values())

        ground_target_nsrt = self._target_actions[state_index]
        mapping_to_objects = {}
        for target_var, base_var in existing_mapping.items():
            index_of_object = target_nsrt.parameters.index(target_var)
            target_object = ground_target_nsrt.objects[index_of_object]
            mapping_to_objects[base_var] = target_object

        ground_state = self._target_states[state_index]
        available_objects = sorted(set(ground_state.data.keys()) - set(mapping_to_objects.values()))

        final_ground_objects = set(mapping_to_objects.values())

        print("=================================")
        print(f"Rule:\n{rule}")
        print(f"Target NSRT:\n{target_nsrt}")
        print(f"Target Ground NSRT:\n{ground_target_nsrt}")
        print(f"Ground state:\n{sorted(ground_state.simulator_state)}")
        print(f"Goal state:\n{self._target_task.goal}")
        for unused_rule_var in sorted(all_rule_variables - used_rule_variables):
            print('-----------------')
            print(f"Unused Rule Var {unused_rule_var}")

            pos_conditions = set()
            for condition in rule.pos_state_preconditions:
                if unused_rule_var in condition.variables:
                    pos_conditions.add(condition)
            
            goal_conditions = set()
            for condition in rule.goal_preconditions:
                if unused_rule_var in condition.variables:
                    goal_conditions.add(condition)
            
            print(f"Pos Conditions {pos_conditions}")
            print(f"Goal Conditions {goal_conditions}")


            candidates = {}
            for pos_cond in pos_conditions:
                analagous_preds = self._get_analogy_pred_name(pos_cond.predicate)
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

                            if obj not in candidates:
                                #print(f"OBJ {obj} HIT! with {pos_atom}")
                                candidates[obj] = 1
                            else:
                                #print(f"OBJ {obj} HIT! with {pos_atom}")
                                candidates[obj] += 1
            for goal_cond in goal_conditions:
                analagous_preds = self._get_analogy_pred_name(goal_cond.predicate)
                if analagous_preds == None:
                    continue

                needed_objects = set()
                for var in pos_cond.variables:
                    if var in mapping_to_objects:
                        needed_objects.add(mapping_to_objects[var])

                for goal_atom in self._target_task.goal:
                    if goal_atom.predicate.name in analagous_preds and needed_objects.issubset(set(goal_atom.objects)):
                        for obj in goal_atom.objects:

                            if obj in needed_objects:
                                continue

                            if obj not in candidates:
                                #print(f"OBJ {obj} HIT! with {goal_atom}")
                                candidates[obj] = 1
                            else:
                                #print(f"OBJ {obj} HIT! with {goal_atom}")
                                candidates[obj] += 1

            """
            for already_used_obj in ground_target_nsrt.objects:
                if already_used_obj in candidates:
                    del candidates[already_used_obj]
            """
            print(f"Candidates: {candidates}")
            
    def _get_analogy_pred_name(self, pred):
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
        if nsrt_param.name in variable_input[(target_nsrt.name, rule.nsrt.name)]:
            var_name = variable_input[(target_nsrt.name, rule.nsrt.name)][nsrt_param.name]
            base_var_names_to_var = {var.name: var for var in rule.parameters}
            return base_var_names_to_var[var_name]
        else:
            return None


def _convert_object_to_variable(obj: Object) -> Variable:
    return obj.type("?" + obj.name)
