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

from predicators.approaches.pg3_analogy_approach import PG3AnalogyApproach
from predicators.envs.base_env import BaseEnv
from predicators.settings import CFG
from predicators.structs import NSRT, LDLRule, LiftedAtom, \
    LiftedDecisionList, Predicate, Type, Variable


class GPTAnalogyApproach(PG3AnalogyApproach):
    """Use GPT for cross-domain policy learning in PG3."""

    @classmethod
    def get_name(cls) -> str:
        return "gpt_pg3"

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
            target_policy = _apply_analogy_to_ldl(analogy, base_policy)
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
    """

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
                          ldl: LiftedDecisionList) -> LiftedDecisionList:
    new_rules = []
    for rule in ldl.rules:
        # Can't create a rule if there is no NSRT match.
        if rule.nsrt not in analogy.nsrts:
            continue
        new_rule_nsrts = analogy.nsrts[rule.nsrt]
        for new_rule_nsrt in new_rule_nsrts:
            new_rule_name = rule.name 
            nsrt_name_pair = (rule.nsrt.name, new_rule_nsrt.name)
            new_rule_pos_preconditions = _apply_analogy_to_atoms(analogy, nsrt_name_pair, rule.pos_state_preconditions)
            new_rule_pos_preconditions.update(new_rule_nsrt.preconditions)
            new_rule_neg_preconditions = _apply_analogy_to_atoms(analogy, nsrt_name_pair, rule.neg_state_preconditions)
            new_rule_goal_preconditions = _apply_analogy_to_atoms(analogy, nsrt_name_pair, rule.goal_preconditions)
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


def _create_variable_mapping_for_rule(
        analogy: _Analogy, rule: LDLRule) -> Dict[Variable, Variable]:
    old_to_new_var = analogy.base_nsrt_to_variable_analogy(rule.nsrt).copy()
    for old_var in rule.parameters:
        if old_var in old_to_new_var:
            continue
        new_var_type = analogy.types[old_var.type]
        new_var = Variable(old_var.name, new_var_type)
        old_to_new_var[old_var] = new_var
    return old_to_new_var


def _apply_analogy_to_atoms(analogy: _Analogy, nsrt_pair: Tuple[str, str], atoms: Set[LiftedAtom]) -> Set[LiftedAtom]:
    new_atoms: Set[LiftedAtom] = set()
    for atom in atoms:
        # Can't create an atom if there is no predicate match.
        if atom.predicate not in analogy.predicates:
            continue
        new_predicates = analogy.predicates[atom.predicate]

        for new_predicate in new_predicates:
            new_variables = _get_variables(new_predicate, analogy, nsrt_pair, atoms)

        """
        new_variables = []
        for var in atom.variables:
            new_var = None
            if var in analogy.variables[nsrt_pair]:
                new_var = analogy.variables[nsrt_pair][var]
            else:
                new_var = var
            
            if new_var != None:
                new_variables.append(new_var)

        
        for new_predicate in new_predicates:

            if new_predicate.arity < len(new_variables):
                new_variables = new_variables[:new_predicate.arity] # TODO: FIX THIS IS VERY HACKY
            else:
                possible_variables = []
                for var in analogy.variables[nsrt_pair].values():
                    if var is not None:
                        possible_variables.append(var)

                new_variables = np.random.choice(possible_variables, new_predicate.arity)
            # print(LiftedAtom(new_predicate, new_variables))
            # import ipdb; ipdb.set_trace();
            new_atoms.add(LiftedAtom(new_predicate, new_variables)) 
        """
    return new_atoms

