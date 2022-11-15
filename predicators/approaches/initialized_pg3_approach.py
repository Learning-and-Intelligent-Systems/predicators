"""Policy-guided planning for generalized policy generation (PG3) with an
initialized policy.

PG3 requires known STRIPS operators. The command below uses oracle operators,
but it is also possible to use this approach with operators learned from
demonstrations.

Example. First run:
    python predicators/main.py --approach pg3 --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --strips_learner oracle --num_train_tasks 10

Then run:
    python predicators/main.py --approach initialized_pg3 --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --strips_learner oracle --num_train_tasks 10 \
        --pg3_init_base_env pddl_easy_delivery_procedural_tasks

Alternatively, define an initial LDL in plain text and save it to
<path to file>.txt Then run:
    python predicators/main.py --approach initialized_pg3 --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --strips_learner oracle --num_train_tasks 10 \
        --pg3_init_policy <path to file>.txt \
        --pg3_init_base_env pddl_easy_delivery_procedural_tasks
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Set

import dill as pkl
import smepy

from predicators import utils
from predicators.approaches.pg3_approach import PG3Approach
from predicators.envs import get_or_create_env
from predicators.envs.base_env import BaseEnv
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.settings import CFG
from predicators.structs import NSRT, LDLRule, LiftedAtom, \
    LiftedDecisionList, Predicate, Type, Variable, _TypedEntity


class InitializedPG3Approach(PG3Approach):
    """Policy-guided planning for generalized policy generation (PG3) with
    initialized policy."""

    @classmethod
    def get_name(cls) -> str:
        return "initialized_pg3"

    @staticmethod
    def _get_policy_search_initial_ldl() -> LiftedDecisionList:
        # Create base and target envs.
        base_env_name = CFG.pg3_init_base_env
        target_env_name = CFG.env
        base_env = get_or_create_env(base_env_name)
        target_env = get_or_create_env(target_env_name)
        base_nsrts = get_gt_nsrts(base_env.get_name(), base_env.predicates,
                                  base_env.options)
        target_nsrts = get_gt_nsrts(target_env.get_name(),
                                    target_env.predicates, target_env.options)
        # Initialize with initialized policy from file.
        if CFG.pg3_init_policy is None:  # pragma: no cover
            # By default, use policy from base domain.
            save_path = utils.get_approach_save_path_str()
            pg3_init_policy_file = f"{save_path}_None.ldl"
        else:
            pg3_init_policy_file = CFG.pg3_init_policy
        # Can load from a pickled LDL or a plain text LDL.
        _, file_extension = os.path.splitext(pg3_init_policy_file)
        assert file_extension in (".ldl", ".txt")
        if file_extension == ".ldl":
            with open(pg3_init_policy_file, "rb") as fb:
                base_policy = pkl.load(fb)
        else:
            with open(pg3_init_policy_file, "r", encoding="utf-8") as f:
                base_policy_str = f.read()
            base_policy = utils.parse_ldl_from_str(base_policy_str,
                                                   base_env.types,
                                                   base_env.predicates,
                                                   base_nsrts)
        # Determine an analogical mapping between the current env and the
        # base env that the initialized policy originates from.
        
        analogy = _find_env_analogy(base_env, target_env, base_nsrts,
                                    target_nsrts)
        # Use the analogy to create an initial policy for the target env.
        target_policy = _apply_analogy_to_ldl(analogy, base_policy)
        # Initialize PG3 search with this new target policy.
        return target_policy


@dataclass(frozen=True)
class _Analogy:
    # All maps are base -> target.
    predicates: Dict[Predicate, Predicate]
    nsrts: Dict[NSRT, NSRT]
    #types: Dict[Type, Type]
    vars: Dict[Variable,Variable]

def to_s_exp(expr, op_name):
    if isinstance(expr, LiftedAtom):
        return [f"term{len(expr.entities)}", expr.predicate.name] + [to_s_exp(s, op_name) for s in expr.entities]
    elif isinstance(expr, _TypedEntity):
        return f"{op_name}-{expr.name.replace('?', '')}"

def to_sme_struct(pddl, name):
    """
        Parses a PDDL domain into StructCase for SME
        Assumes a BaseEnv as an input
    """

    actions = []
    vars = {}
    # get actions -- the only thing that we need
    for operator in pddl._strips_operators:
        new_action = []
        new_action.append('action')
        new_action.append(operator.name)
        new_action.append(['precon', ['and'] + [to_s_exp(s, operator.name) for s in operator.preconditions]])
        new_action.append(['effects', ['and'] + [to_s_exp(s, operator.name) for s in operator.add_effects] + [['not', to_s_exp(s, operator.name)] for s in operator.delete_effects]])

        actions.append(new_action)

        # extract variable types
        for var in operator.parameters:
            var_name = f"{operator.name}-{var.name.replace('?', '')}"
            vars[var_name] = var

    return smepy.StructCase(actions, name), vars

def run_magic_analogy_machine(base_pddl, target_pddl, max_mappings=1):
    smepy.declare_nary("and") # necessary to interpret the files properly
    base_struct, base_vars = to_sme_struct(base_pddl, "base")
    target_struct, target_vars = to_sme_struct(target_pddl, "target")

    sme = smepy.SME(base_struct, target_struct, max_mappings=max_mappings)
    gms = sme.match()

    return {'mappings': gms[0], 'base_vars': base_vars, 'target_vars': target_vars}

def find_among_predicates(name, env):
    """
        Finds the predicate with the given name in the env
        Return the pred if one is found, otherwise None
    """
    for pred in env._predicates:
        if pred.name == name:
            return pred

    return None

def find_among_nsrts(name, nsrts):
    """
        finds the NSRT with the matching name
                Returns one if found, otherwise None
    """
    for n in nsrts:
        if n.name == name:
            return n

    return None


def parse_analogy_result(matching_result, base_env, base_nsrts, target_env, target_nsrts, base_vars, target_vars):
    predicate_map = {}
    nsrt_map = {}
    var_map = {}

    for match in matching_result.entity_matches():
        # check if in preds
        pred_match = find_among_predicates(match.base.name, base_env)
        if pred_match is not None:
            predicate_map[pred_match] = find_among_predicates(match.target.name, target_env)
            continue

        # check if in nsrts
        nsrt_match = find_among_nsrts(match.base.name, base_nsrts)
        if nsrt_match is not None:
            nsrt_map[nsrt_match] = find_among_nsrts(match.target.name, target_nsrts)
            continue

        # check if in types
        if match.base.name in base_vars:
            # type_key = base_vars[match.base.name]
            # type_val = target_vars[match.target.name]
            
            # if type_key not in var_map:
            #     var_map[type_key] = type_val

            var_map[base_vars[match.base.name]] = target_vars[match.target.name]

    return predicate_map, nsrt_map, var_map


def _find_env_analogy(base_env: BaseEnv, target_env: BaseEnv,
                      base_nsrts: Set[NSRT],
                      target_nsrts: Set[NSRT]) -> _Analogy:
    # Create PDDL domains from environments.
    # base_pddl = utils.create_pddl_domain(base_nsrts, base_env.predicates,
    #                                      base_env.types, base_env.get_name())
    # target_pddl = utils.create_pddl_domain(target_nsrts, target_env.predicates,
    #                                        target_env.types,
    #                                        target_env.get_name())
    # Call external module to find analogy.
    # TODO (not sure exactly what form this will take)
    analogy_result = run_magic_analogy_machine(base_env, target_env)
    # Parse the results of the external module back into our data structures.
    # TODO
    predicate_map, nsrt_map, var_map = parse_analogy_result(analogy_result['mappings'], base_env, base_nsrts, target_env,  \
                                            target_nsrts, analogy_result['base_vars'], analogy_result['target_vars'])
    
    return _Analogy(predicate_map, nsrt_map, var_map)


def _apply_analogy_to_ldl(analogy: _Analogy,
                          ldl: LiftedDecisionList) -> LiftedDecisionList:
    new_rules = []
    for rule in ldl.rules:
        new_rule_name = rule.name
        new_rule_parameters = [
            _apply_analogy_to_variable(analogy, p) for p in rule.parameters
        ]
        new_rule_pos_preconditions = _apply_analogy_to_atoms(
            analogy, rule.pos_state_preconditions)
        new_rule_neg_preconditions = _apply_analogy_to_atoms(
            analogy, rule.neg_state_preconditions)
        new_rule_goal_preconditions = _apply_analogy_to_atoms(
            analogy, rule.goal_preconditions)
        new_rule_nsrt = analogy.nsrts.get(rule.nsrt, None)

        if new_rule_nsrt is None:
            # If we couldn't match the operator, skip this rule
            continue
        

        new_rule = LDLRule(new_rule_name, new_rule_parameters,
                           new_rule_pos_preconditions.union({x for x in new_rule_nsrt.preconditions if x not in new_rule_pos_preconditions}),  # we would never want to suggest an action that’s actually not applicable in the current state, but it can happen that not everything is mapped between domains
                           new_rule_neg_preconditions,
                           new_rule_goal_preconditions, new_rule_nsrt)
        new_rules.append(new_rule)
    return LiftedDecisionList(new_rules)


def _apply_analogy_to_atoms(analogy: _Analogy,
                            atoms: Set[LiftedAtom]) -> Set[LiftedAtom]:
    new_atoms: Set[LiftedAtom] = set()
    for atom in atoms:
        new_variables = [
            _apply_analogy_to_variable(analogy, v) for v in atom.variables
        ]
        new_predicate = analogy.predicates.get(atom.predicate, None)
        if new_predicate is None or None in new_variables:
            continue
        else:
            new_atoms.add(LiftedAtom(new_predicate, new_variables))
    return new_atoms


def _apply_analogy_to_variable(analogy: _Analogy,
                               variable: Variable) -> Variable:
    if variable in analogy.vars:
        return analogy.vars[variable]
    else:
        return variable
