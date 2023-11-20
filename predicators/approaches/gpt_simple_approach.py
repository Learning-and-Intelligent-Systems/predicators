"""Use GPT for cross-domain policy learning in PG3.

Command for testing gripper / ferry:
    python predicators/main.py --approach gpts_pg3 --seed 0  \
        --env pddl_ferry_procedural_tasks --strips_learner oracle \
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

from predicators import utils


class GPTAnalogyApproach(PG3AnalogyApproach):
    """Use GPT for cross-domain policy learning in PG3."""

    @classmethod
    def get_name(cls) -> str:
        return "gpts_pg3"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Get a task and trajectory for correctness of generated conditions
        self._correctness_task = self._train_tasks[0]
        self._correctness_traj = dataset.trajectories[0]._states
        super().learn_from_offline_dataset(dataset)

    def _induce_policies_by_analogy(
            self, base_policy: LiftedDecisionList, base_env: BaseEnv,
            target_env: BaseEnv, base_nsrts: Set[NSRT],
            target_nsrts: Set[NSRT]) -> List[LiftedDecisionList]:
        # Determine analogical mappings between the current env and the
        # base env that the initialized policy originates from.
        analogies = _find_env_analogies(base_env, target_env, base_nsrts,
                                        target_nsrts)
        chem_analogies = _find_chem_analogies(base_env, target_env, analogies)
        target_policies: List[LiftedDecisionList] = []
        for chem_analogy in chem_analogies:
            # Use the analogy to create an initial policy for the target env.
            # target_policy = _apply_analogy_to_ldl(analogy, base_policy, self._correctness_traj, self._correctness_task, self._initial_predicates)
            """
            print(chem_analogy.predicates)
            print()
            print(chem_analogy.variables)
            print()
            print(chem_analogy.special_analogies)
            import ipdb; ipdb.set_trace()
            """
            target_policy = _apply_chem_analogy_to_ldl(chem_analogy, base_policy, self._correctness_traj, self._correctness_task, self._initial_predicates)
            target_policies.append(target_policy)
        return target_policies


@dataclass(frozen=True)
class _Analogy:
    # All maps are base -> target.
    predicates: Dict[Predicate, Predicate]
    nsrts: Dict[NSRT, NSRT]
    variables: Dict[Tuple[String, String], Dict[Variable, Variable]] #TODO: Fix typing

@dataclass(frozen=True)
class _ChemAnalogy:
    # All maps are base -> target.
    predicates: Dict[Predicate, Predicate]
    nsrts: Dict[NSRT, NSRT]
    variables: Dict[Tuple[String, String], Dict[Variable, Variable]] #TODO: Fix typing
    special_analogies: Dict[Tuple[String, String], Dict[frozenset[LiftedAtom]: frozenset[LiftedAtom]]] #TODO: Fix typing (set can't be indexed)

def _find_env_analogies(base_env: BaseEnv, target_env: BaseEnv,
                        base_nsrts: Set[NSRT],
                        target_nsrts: Set[NSRT]) -> List[_Analogy]:
    predicate_dict = {}
    nsrt_dict = {}
    """
    predicate_input = {
        "room": ["floor"],
        "ball": ["passenger"],
        "at-robby": ["lift-at"],
        "at": ["origin", "destin"],
        "carry": ["boarded"],
    }
    nsrt_input = {
        "move": ["up", "down"],
        "pick": ["board"],
        "drop": ["depart"],
    }

    variable_input = {
        ("move", "up") : {"?from": "?f1", "?to": "?f2"},
        ("move", "down") : {"?from": "?f1", "?to": "?f2"},
        ("pick", "board") : {"?obj": "?p", "?room": "?f", "?gripper": None},
        ("drop", "depart") : {"?obj": "?p", "?room": "?f", "?gripper": None},
    }
    """

    """
    predicate_input = {
        "ball": ["car"],
        "free": ["empty-ferry"],
        "room": ["location"],
        "at-robby": ["at-ferry"],
        "at": ["at"],
        "carry": ["on"],
    }
    nsrt_input = {
        "move": ["sail"],
        "pick": ["board"],
        "drop": ["debark"],
    }

    variable_input = {
        ("move", "sail") : {"?from": "?from", "?to": "?to"},
        ("pick", "board") : {"?obj": "?car", "?room": "?loc", "?gripper": None},
        ("drop", "debark") : {"?obj": "?car", "?room": "?loc", "?gripper": None},
    }
    """

    predicate_input = {
        "car": ["ball"],
        "empty-ferry": ["free"],
        "location": ["room"],
        "at-ferry": ["at-robby"],
        "at": ["at"],
        "on": ["carry"],
    }
    nsrt_input = {
        "sail": ["move"],
        "board": ["pick"],
        "debark": ["drop"],
    }

    variable_input = {
        ("sail", "move") : {"?from": "?from", "?to": "?to"},
        ("board", "pick") : {"?car": "?obj", "?loc": "?room"},
        ("debark", "drop") : {"?car": "?obj", "?loc": "?room"},
    }

    base_env_pred_names_to_pred = {str(pred): pred for pred in base_env.predicates}
    target_env_pred_names_to_pred = {str(pred): pred for pred in target_env.predicates}
    for base_pred, target_preds in predicate_input.items():
        predicate_dict[base_env_pred_names_to_pred[base_pred]] = [target_env_pred_names_to_pred[target_pred] for target_pred in target_preds]

    base_env_nsrt_names_to_nsrt = {nsrt.name: nsrt for nsrt in base_nsrts}
    target_env_nsrt_names_to_nsrt = {nsrt.name: nsrt for nsrt in target_nsrts}
    for base_nsrt, target_nsrts in nsrt_input.items():
        nsrt_dict[base_env_nsrt_names_to_nsrt[base_nsrt]] = [target_env_nsrt_names_to_nsrt[target_nsrt] for target_nsrt in target_nsrts]
    
    variable_dict = {}
    for operator_pair, operator_variables in variable_input.items():
        variable_dict[operator_pair] = {}
        base_nsrt = base_env_nsrt_names_to_nsrt[operator_pair[0]]
        target_nsrt = target_env_nsrt_names_to_nsrt[operator_pair[1]]
        base_var_names_to_var = {var.name: var for var in base_nsrt.parameters}
        target_var_names_to_var = {var.name: var for var in target_nsrt.parameters}
        for base_var_name, target_var_name in operator_variables.items():
            base_var = base_var_names_to_var[base_var_name]
            if target_var_name is None:
                variable_dict[operator_pair][base_var] = None
            else:
                target_var = target_var_names_to_var[target_var_name]
                variable_dict[operator_pair][base_var] = target_var
    
    return [_Analogy(predicate_dict, nsrt_dict, variable_dict)]

def _find_chem_analogies(base_env: BaseEnv, target_env: BaseEnv, analogies: List[_Analogy]):
    assert len(analogies) == 1
    analogy = analogies[0]
    special_analogies = {}
    
    # Find atoms with any leftover entity (predicate or variable)
    for base_nsrt, target_nsrts in analogy.nsrts.items():
        for target_nsrt in target_nsrts:
            nsrt_pair = (base_nsrt.name, target_nsrt.name)
            unmatched_base_preconditions = set()
            for base_precondition in base_nsrt.preconditions:
                if (not base_precondition.predicate in analogy.predicates) or (not all([var in analogy.variables[(base_nsrt.name, target_nsrt.name)] for var in base_precondition.variables])):
                    unmatched_base_preconditions.add(base_precondition)

            unmatched_target_preconditions = set()
            all_target_matched_predicates = {target_predicate for target_predicates in analogy.predicates.values() for target_predicate in target_predicates}
            for target_precondition in target_nsrt.preconditions:
                if (not target_precondition.predicate in all_target_matched_predicates) or (not all([var in analogy.variables[(base_nsrt.name, target_nsrt.name)].values() for var in target_precondition.variables])):
                    unmatched_target_preconditions.add(target_precondition)
        
            assert len(unmatched_base_preconditions) == 0 or len(unmatched_target_preconditions) == 0

            if len(unmatched_base_preconditions) == 0:

                target_to_base_predicate_analogies = {}
                for base_predicate, target_predicates in analogy.predicates.items():
                    for target_predicate in target_predicates:
                        target_to_base_predicate_analogies[target_predicate] = base_predicate
                kinda_matched_base_preconditions = set()
                for unmatched_target_precondition in unmatched_target_preconditions:
                    if unmatched_target_precondition.predicate in target_to_base_predicate_analogies:
                        kinda_matched_base_predicate = target_to_base_predicate_analogies[unmatched_target_precondition.predicate]
                        for base_precondition in base_nsrt.preconditions:
                            if base_precondition.predicate == kinda_matched_base_predicate:
                                kinda_matched_base_preconditions.add(base_precondition)
                
                if len(kinda_matched_base_preconditions) == 0:
                    special_analogies[None] = unmatched_target_preconditions
                else:
                    special_analogies[frozenset(kinda_matched_base_preconditions)] = frozenset(unmatched_target_preconditions)

            elif len(unmatched_target_preconditions) == 0:
                kinda_matched_target_preconditions = set()
                for unmatched_base_precondition in unmatched_base_preconditions:
                    if unmatched_base_precondition.predicate in analogy.predicates:
                        kinda_matched_target_predicates = analogy.predicates[unmatched_base_precondition.predicate]
                        for base_precondition in base_nsrt.preconditions:
                            if base_precondition.predicate in kinda_matched_target_predicates:
                                kinda_matched_target_preconditions.add(base_precondition)
                
                if len(kinda_matched_target_preconditions) == 0:
                    # special_analogies[frozenset(unmatched_base_preconditions)] = None # TODO: Maybe we want to keep this
                    pass
                else:
                    special_analogies[frozenset(unmatched_base_preconditions)] = frozenset(kinda_matched_target_preconditions)

            else:
                raise Exception("One set of unmatched should be empty")

    chem_analogy = _ChemAnalogy(analogy.predicates, analogy.nsrts, analogy.variables, special_analogies)
    return [chem_analogy]

def _apply_chem_analogy_to_ldl(chem_analogy: _Analogy,
                          ldl: LiftedDecisionList, 
                          correctness_traj, correctness_task, initial_predicates) -> LiftedDecisionList:
    new_rules = []

    for rule in ldl.rules:
        # Can't create a rule if there is no NSRT match.
        if rule.nsrt not in chem_analogy.nsrts:
            continue
        new_rule_nsrts = chem_analogy.nsrts[rule.nsrt]
        for new_rule_nsrt in new_rule_nsrts:
            new_rule_name = rule.name
            nsrt_name_pair = (rule.nsrt.name, new_rule_nsrt.name)
            import ipdb; ipdb.set_trace();
            # TODO
            # We apply the analogy in thse following way:
            # 1. Figure out all combinations of preconditions that fall into any special analogy
            # 2. Create their corresponding analagous set of predicates for positive and negative
            # 3. Test each new created predicate for validity -> Add to rule if valid
            # 4. For goal conditions, apply the fine-grained conditions
            # 5. Add new_rule_nsrt conditions

            created_pos_state_preconditions, created_neg_state_preconditions, created_goal_conditions = _apply_special_analogies(chem_analogy, rule, new_rule_nsrt)
            # ocreated_pos_state_preconditions, ocreated_neg_state_preconditions, ocreated_goal_conditions = _apply_regular_analogies(chem_analogy, rule, new_rule_nsrt)

            # Ensure pos conditions are disjoint from negative conditions (assume new_rule_nsrt has not negative preconditions)
            pre_pos_state_preconditions = created_pos_state_preconditions | new_rule_nsrt.preconditions
            pre_neg_state_preconditions = created_neg_state_preconditions - new_rule_nsrt.preconditions
            pre_goal_conditions = created_goal_conditions.copy()

            valid_positive_preconditions = set()
            for potential_precondition in pre_pos_state_preconditions:
                temporary_params = sorted({var for atom in pre_pos_state_preconditions for var in atom.variables} | {var for atom in pre_neg_state_preconditions for var in atom.variables} | set(new_rule_nsrt.parameters))
                temporary_pos_preconditions = new_rule_nsrt.preconditions.copy() | {potential_precondition}
                temporary_neg_preconditions =  set()
                temporary_goalconditions = set()
                temporary_ldl_rules = []


                new_rule = LDLRule("temporary-rule", temporary_params, temporary_pos_preconditions, temporary_neg_preconditions, temporary_goalconditions, new_rule_nsrt)
                temporary_ldl_rules.append(new_rule)
                temporary_ldl = LiftedDecisionList(temporary_ldl_rules)
                temp_init_atoms = utils.abstract(correctness_task.init, initial_predicates)
                found_valid_action = False
                for i in range(len(correctness_traj)-1):
                    ans = utils.query_ldl(temporary_ldl, correctness_traj[i].simulator_state, correctness_traj[i].data, correctness_task.goal, static_predicates=initial_predicates, init_atoms= correctness_traj[i].simulator_state) #TODO: CHECK INIT ATOMS (DOESN'T WORK OTHERWISE?)
                    if isinstance(ans, _GroundNSRT):
                        found_valid_action = True
                        break
                if found_valid_action:
                    valid_positive_preconditions.add(potential_precondition)
            
            valid_negative_preconditions = set()
            for potential_precondition in pre_neg_state_preconditions:
                temporary_params = sorted({var for atom in pre_pos_state_preconditions for var in atom.variables} | {var for atom in pre_neg_state_preconditions for var in atom.variables} | set(new_rule_nsrt.parameters))
                temporary_pos_preconditions = new_rule_nsrt.preconditions.copy() | valid_positive_preconditions
                temporary_neg_preconditions =  {potential_precondition}
                temporary_goalconditions = set()
                temporary_ldl_rules = []

                new_rule = LDLRule("temporary-rule", temporary_params, temporary_pos_preconditions, temporary_neg_preconditions, temporary_goalconditions, new_rule_nsrt)
                temporary_ldl_rules.append(new_rule)
                temporary_ldl = LiftedDecisionList(temporary_ldl_rules)
                temp_init_atoms = utils.abstract(correctness_task.init, initial_predicates)
                found_valid_action = False
                for i in range(len(correctness_traj)-1):
                    ans = utils.query_ldl(temporary_ldl, correctness_traj[i].simulator_state, correctness_traj[i].data, correctness_task.goal, static_predicates=initial_predicates, init_atoms= correctness_traj[i].simulator_state) #TODO: CHECK INIT ATOMS (DOESN'T WORK OTHERWISE?)
                    if isinstance(ans, _GroundNSRT):
                        found_valid_action = True
                        break
                if found_valid_action:
                    valid_negative_preconditions.add(potential_precondition)
            
            # If special analogies created no goals, then use classical analogies
            valid_goalconditions = pre_goal_conditions.copy()
            if len(valid_goalconditions) == 0:
                valid_goalconditions = _configure_goal(chem_analogy, rule, new_rule_nsrt)

                
            print(valid_positive_preconditions)
            print(valid_negative_preconditions)
            print(valid_goalconditions)

            import ipdb; ipdb.set_trace()
            # for potential_goalcondition in created_goal_conditions:

            



    return LiftedDecisionList(new_rules)


"""
TODO: MAYBE NEEDS MORE FINE-GRAINED CONVERSION OF "REGULAR" ANALOGIES
def _apply_regular_analogies(chem_analogy, rule, new_rule_nsrt):
    created_pos_state_preconditions = _apply_regular_analogies_against_rule_conditions(chem_analogy, rule.pos_state_preconditions, new_rule_nsrt)
    created_neg_state_preconditions = _apply_regular_analogies_against_rule_conditions(chem_analogy, rule.neg_state_preconditions, new_rule_nsrt)
    created_goal_conditions = _apply_regular_analogies_against_rule_conditions(chem_analogy, rule.goal_preconditions, new_rule_nsrt)
    return created_pos_state_preconditions, created_neg_state_preconditions, created_goal_conditions

def _apply_regular_analogies_against_rule_conditions(chem_analogy, rule_conditions, new_rule_nsrt):
    import ipdb; ipdb.set_trace()
    predicates_in_special_analogies = {atom.predicate for special_analogy in chem_analogy.special_analogies for atom in special_analogy}
    for rule_condition in rule_conditions:
        if rule_condition.predicate not in chem_analogy.predicates or rule_condition.predicate in predicates_in_special_analogies:
            continue
        new_predicates = chem_analogy.predicates[rule_condition.predicate]

        for new_predicate in new_predicates:
            print(rule_condition, new_predicate)
            import ipdb; ipdb.set_trace()
"""
def _configure_goal(chem_analogy, rule, new_rule_nsrt):
    """Returns set of goal conditions from rule after applying chem_analogy"""
    created_goal_conditions = set()
    for goal_condition in rule.goal_preconditions:
        if goal_condition.predicate not in chem_analogy.predicates:
            continue
        new_predicates = chem_analogy.predicates[goal_condition.predicate]

        # Check that all variables in goal_condition have an analogy as a base variable in new_rule_nsrt
        for new_predicate in new_predicates:
            print(goal_condition, new_predicate, new_predicate.arity, chem_analogy.variables[(rule.nsrt.name, new_rule_nsrt.name)])
            analogous_variables = []
            for goal_condition_variable in goal_condition.variables:
                if goal_condition_variable not in chem_analogy.variables[(rule.nsrt.name, new_rule_nsrt.name)]:
                    import ipdb; ipdb.set_trace(); 
                    raise Exception("Variable in goal condition does not have analogy")
                analogous_variables.append(chem_analogy.variables[(rule.nsrt.name, new_rule_nsrt.name)])
        
            created_goal_conditions.add(new_predicate(analogous_variables))
        
    return created_goal_conditions

def _apply_special_analogies(chem_analogy, rule, new_rule_nsrt):
    nsrt_name_pair = (rule.nsrt.name, new_rule_nsrt.name)

    variable_analogies = chem_analogy.variables
    special_analogies = chem_analogy.special_analogies

    created_pos_state_preconditions = set()
    created_neg_state_preconditions = set()
    created_goal_conditions = set()

    for base_special, target_special in special_analogies.items():
        new_pos_preconditions, new_neg_preconditions, new_goal_conditions = _find_possible_matches_for_special_analogy_against_rule(base_special, target_special, rule, new_rule_nsrt, variable_analogies)
        created_pos_state_preconditions.update(new_pos_preconditions)
        created_neg_state_preconditions.update(new_neg_preconditions)
        created_goal_conditions.update(new_goal_conditions)

    all_atoms = created_pos_state_preconditions | created_neg_state_preconditions | created_goal_conditions 
    new_rule_params_set = {v for a in all_atoms for v in a.variables}
    new_rule_params_set.update(new_rule_nsrt.parameters)
    new_rule_params = sorted(new_rule_params_set)
    assert set(new_rule_params).issuperset(set(new_rule_nsrt.parameters))

    return created_pos_state_preconditions, created_neg_state_preconditions, created_goal_conditions

def _find_possible_matches_for_special_analogy_against_rule(special_base, special_target, rule, new_rule_nsrt, variable_analogies):
    created_pos_state_preconditions = _find_possible_matches_for_special_analogy(special_base, special_target, rule.pos_state_preconditions, new_rule_nsrt, variable_analogies)
    created_neg_state_preconditions = _find_possible_matches_for_special_analogy(special_base, special_target, rule.neg_state_preconditions, new_rule_nsrt, variable_analogies)
    created_goal_conditions = _find_possible_matches_for_special_analogy(special_base, special_target, rule.goal_preconditions, new_rule_nsrt, variable_analogies)
    return created_pos_state_preconditions, created_neg_state_preconditions, created_goal_conditions
    
def _find_possible_matches_for_special_analogy(special_base, special_target, rule_conditions, new_rule_nsrt, variable_analogies):
    predicates_used_in_special_base = {atom.predicate: [] for atom in special_base}
    grounded_conditions = _get_grounded_atoms(rule_conditions)

    special_base_variables = sorted({var for atom in special_base for var in atom.variables})
    condition_objects = sorted({obj for atom in grounded_conditions for obj in atom.objects})

    matches = {}

    special_base_combinations = []

    for perm in list(itertools.permutations(condition_objects, len(special_base_variables))):
        var_to_obj = {}
        for i in range(len(special_base_variables)):
            var_to_obj[special_base_variables[i]] = perm[i]
        
        works = True
        special_base_passing_ground_atoms = set()
        for special_base_lifted_atom in special_base:
            obj_assignment = [var_to_obj[var] for var in special_base_lifted_atom.variables]
            special_base_ground_atom = GroundAtom(special_base_lifted_atom.predicate, obj_assignment)
            if special_base_ground_atom in grounded_conditions:
                special_base_passing_ground_atoms.add(special_base_ground_atom)
            else:
                works = False
                break
        
        if works:
            special_base_combinations.append(special_base_passing_ground_atoms)
    
    # Now we need to convert the lifted atoms
    # TODO: FIND A BETTER WAY TO CONVERT VARIABLES
    target_extra_conditions = {}
    if len(special_base_combinations) > 0:
        target_extra_conditions = set(special_target)
    
    return target_extra_conditions

def _get_grounded_atoms(lifted_atoms: Set[LiftedAtom]):
    grounded_atoms = set()
    for lifted_atom in lifted_atoms:
        objects = [_convert_variable_to_object(var) for var in lifted_atom.variables]
        if len(objects) > 0:
            grounded_atoms.add(lifted_atom.predicate(objects))
        else:
            grounded_atoms.add(GroundAtom(lifted_atom.predicate, []))
    return grounded_atoms

def _get_lifted_atoms(ground_atoms: Set[GroundAtom]):
    lifted_atoms = set()
    for ground_atom in ground_atoms:
        variables = [_convert_variable_to_object(obj) for obj in lifted_atom.objects]
        if len(variables) > 0:
            lifted_atoms.add(lifted_atom.predicate(variables))
        else:
            lifted_atoms.add(LiftedAtom(lifted_atom.predicate, []))
    return lifted_atoms

def _convert_variable_to_object(var: Variable):
    return var.type(var.name[1:]) 

def _convert_object_to_variable(obj: Object):
    return obj.type(obj.name[1:]) 