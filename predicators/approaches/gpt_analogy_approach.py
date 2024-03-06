"""Use GPT for cross-domain policy learning in PG3.

Command for testing gripper / ferry:
    python predicators/main.py --approach gpt_pg3 --seed 0  \
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
from predicators import utils


class GPTAnalogyApproach(PG3AnalogyApproach):
    """Use GPT for cross-domain policy learning in PG3."""

    @classmethod
    def get_name(cls) -> str:
        return "gpt_pg3"

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
        target_policies: List[LiftedDecisionList] = []
        for analogy in analogies:
            # Use the analogy to create an initial policy for the target env.
            target_policy = _apply_analogy_to_ldl(analogy, base_policy, self._correctness_traj, self._correctness_task, self._initial_predicates)
            target_policies.append(target_policy)
        return target_policies


@dataclass(frozen=True)
class _Analogy:
    # All maps are base -> target.
    predicates: Dict[Predicate, Predicate]
    nsrts: Dict[NSRT, NSRT]
    variables: Dict[Tuple, Dict[Variable, Variable]] #TODO: Fix typing


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



def _apply_analogy_to_ldl(analogy: _Analogy,
                          ldl: LiftedDecisionList, 
                          correctness_traj, correctness_task, initial_predicates) -> LiftedDecisionList:
    new_rules = []
    for rule in ldl.rules:
        # Can't create a rule if there is no NSRT match.
        if rule.nsrt not in analogy.nsrts:
            continue
        new_rule_nsrts = analogy.nsrts[rule.nsrt]
        for new_rule_nsrt in new_rule_nsrts:
            new_rule_name = rule.name 
            nsrt_name_pair = (rule.nsrt.name, new_rule_nsrt.name)
            new_rule_pos_preconditions = _apply_analogy_to_atoms(analogy, nsrt_name_pair, rule.pos_state_preconditions, new_rule_nsrt, "pos_state_preconditions", correctness_traj, correctness_task, initial_predicates)
            new_rule_pos_preconditions.update(new_rule_nsrt.preconditions)
            new_rule_neg_preconditions = _apply_analogy_to_atoms(analogy, nsrt_name_pair, rule.neg_state_preconditions, new_rule_nsrt, "neg_state_preconditions", correctness_traj, correctness_task, initial_predicates)
            new_rule_goal_preconditions = _apply_analogy_to_atoms(analogy, nsrt_name_pair, rule.goal_preconditions, new_rule_nsrt, "goal_preconditions", correctness_traj, correctness_task, initial_predicates)
            print(rule.goal_preconditions)
            all_atoms = new_rule_pos_preconditions | new_rule_neg_preconditions | \
                        new_rule_goal_preconditions
            new_rule_params_set = {v for a in all_atoms for v in a.variables}
            new_rule_params_set.update(new_rule_nsrt.parameters)
            new_rule_params = sorted(new_rule_params_set)
            assert set(new_rule_params).issuperset(set(new_rule_nsrt.parameters))
            new_rule = LDLRule(new_rule_name, new_rule_params,
                            new_rule_pos_preconditions,
                            new_rule_neg_preconditions,
                            new_rule_goal_preconditions, new_rule_nsrt)
            new_rules.append(new_rule)
    return LiftedDecisionList(new_rules)


def _apply_analogy_to_atoms(analogy: _Analogy, nsrt_pair: Tuple[str, str], atoms: Set[LiftedAtom], new_rule_nsrt: NSRT, atom_location: str, correctness_traj, correctness_task, initial_predicates) -> Set[LiftedAtom]:
    new_atoms: Set[LiftedAtom] = set()
    for atom in atoms:
        # Can't create an atom if there is no predicate match.
        if atom.predicate not in analogy.predicates:
            continue
        new_predicates = analogy.predicates[atom.predicate]

        for new_predicate in new_predicates:
            new_variables = _generate_variables(new_predicate, atom, analogy, nsrt_pair, new_rule_nsrt, atom_location, correctness_traj, correctness_task, initial_predicates)

    return new_atoms

def _generate_variables(predicate, atom: LiftedAtom, analogy: _Analogy, nsrt_pair: Tuple[str, str], new_rule_nsrt: NSRT, atom_location: str, correctness_traj, correctness_task, initial_predicates):
    # 3 cases: 
    # new predicate arity > old predicate arity, 
    # new predicate arity = old predicate arity,
    # new predicate arity < old predicate arity
    # what if a variable isn't one of the default ones?

    # Generating must use variables
    must_use_variables = []
    for var in atom.variables:
        if var in analogy.variables[nsrt_pair] and analogy.variables[nsrt_pair][var] != None:
            must_use_variables.append(analogy.variables[nsrt_pair][var])
    
    # Only use given variables
    if len(must_use_variables) >= predicate.arity:
        candidate_param = {} # find_correct_params(predicate, must_use_variables, new_rule_nsrt, atom_location, correctness_traj, correctness_task, initial_predicates)
        if candidate_param != None:
            return candidate_param
        else:
            raise Exception("SHOULD HAVE FOUND PARAMETERS WITHOUT FILLER")
    # Need to generate new variables
    else:
        #TODO: IMPLEMENT THIS FOLLOWING ALGORITHM WRITTEN IN PSEUDOCODE
        """
        1. Base - Base Variables
            a. Loop through each base variable found in atom, find it's projection from analogy. Add each to the set of new variables
        2. Create temporary "filler" variables
        3. Try different combinations of the base variable - ensuring correctness with the example, until a configuration is found
        4. The other ones are actually just new variables
        TODO: WHEN SHOULD WE TRY TO MATCH IT TO AN EXISTING VARIABLE?
        TODO: HOW TO DEAL WITH DUPLICATE VARIABLES CREATED SO FAR - FIX THIS
        
        """
        num_filler_variables_needed = predicate.arity - len(must_use_variables)
        temporary_variables = must_use_variables.copy()
        object_type = new_rule_nsrt.parameters[0].type
        for i in range(num_filler_variables_needed):
            temporary_variables.append(object_type(f"?tv{i}"))
        candidate_param = find_correct_params(predicate, temporary_variables, new_rule_nsrt, atom_location, correctness_traj, correctness_task, initial_predicates)
        if candidate_param != None:
            return candidate_param
        else:
            raise Exception("SHOULD HAVE FOUND PARAMETERS WITH FILLER")

def find_correct_params(predicate, all_variables, nsrt, atom_location, correctness_traj, correctness_task, initial_predicates):
    # Note: the number of all_variables can be more than predicate arity
    all_possible_params = list(itertools.permutations(all_variables, predicate.arity))
    for candidate_param in all_possible_params:
        temporary_condition = LiftedAtom(predicate, candidate_param)
        temporary_ldl_rules = []
        # TODO: ONE HACK IS WE ONLY USE ONE TRAJECTORY (SO IF IT DOESN"T USE EVERY NSRT, THAT"S A PROBLEM) - USE ALL PROBLEMS?
        temporary_params = sorted(set(candidate_param) | set(nsrt.parameters))
        temporary_preconditions = nsrt.preconditions.copy()
        temporary_negative_preconditions = set()
        temporary_goalconditions = set()
        
        if atom_location == "pos_state_preconditions":
            temporary_preconditions.add(temporary_condition)
        elif atom_location == "neg_state_preconditions":
            temporary_negative_preconditions.add(temporary_condition)
        elif atom_location == "goal_preconditions":
            temporary_goalconditions.add(temporary_condition)
        else:
            raise Exception("No valid atom location!")

        new_rule = LDLRule("temporary-rule", temporary_params, temporary_preconditions, temporary_negative_preconditions, temporary_goalconditions, nsrt)
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
            return candidate_param
    return None


