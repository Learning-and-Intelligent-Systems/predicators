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

Command for testing gripper / ferry:
    python predicators/main.py --approach initialized_pg3 --seed 0  \
        --env pddl_ferry_procedural_tasks --strips_learner oracle \
        --num_train_tasks 20 --pg3_init_policy gripper_ldl_policy.txt \
        --pg3_init_base_env pddl_gripper_procedural_tasks \
        --pg3_add_condition_allow_new_vars False
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import dill as pkl
import smepy

from predicators import utils
from predicators.approaches.pg3_approach import PG3Approach
from predicators.envs import get_or_create_env
from predicators.envs.base_env import BaseEnv
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.settings import CFG
from predicators.structs import NSRT, LDLRule, LiftedAtom, \
    LiftedDecisionList, Predicate, Type, Variable


class InitializedPG3Approach(PG3Approach):
    """Policy-guided planning for generalized policy generation (PG3) with
    initialized policy."""

    @classmethod
    def get_name(cls) -> str:
        return "initialized_pg3"

    @staticmethod
    def _get_policy_search_initial_ldls() -> List[LiftedDecisionList]:
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
    types: Dict[Type, Type]
    # TODO: see if we can remove (I think we can)
    variables: Dict[Variable, Variable]


def _find_env_analogies(base_env: BaseEnv, target_env: BaseEnv,
                        base_nsrts: Set[NSRT],
                        target_nsrts: Set[NSRT]) -> List[_Analogy]:
    # Use external SME module to find analogies.
    smepy.declare_nary("and")
    base_sme_struct, base_sme_vars = _create_sme_inputs(base_env, base_nsrts)
    target_sme_struct, target_sme_vars = _create_sme_inputs(
        target_env, target_nsrts)
    sme = smepy.SME(base_sme_struct,
                    target_sme_struct,
                    max_mappings=CFG.pg3_max_analogies)
    analogies: List[_Analogy] = []
    for sme_mapping in sme.match():
        analogy = _sme_mapping_to_analogy(sme_mapping, base_env, target_env,
                                          base_nsrts, target_nsrts,
                                          base_sme_vars, target_sme_vars)
        analogies.append(analogy)
    return analogies


def _apply_analogy_to_ldl(analogy: _Analogy,
                          ldl: LiftedDecisionList) -> LiftedDecisionList:
    new_rules = []
    for rule in ldl.rules:
        # Can't create a rule if there is no NSRT match.
        if rule.nsrt not in analogy.nsrts:
            continue
        # Can't create a rule if there is no type match.
        if not all(p.type in analogy.types for p in rule.parameters):
            continue
        new_rule_name = rule.name
        new_rule_nsrt = analogy.nsrts[rule.nsrt]
        new_rule_pos_preconditions = _apply_analogy_to_atoms(
            analogy, rule.pos_state_preconditions)
        # Always add the NSRT preconditions because we would never want to
        # take an action that's inapplicable in a current state.
        new_rule_pos_preconditions.update(new_rule_nsrt.preconditions)
        new_rule_neg_preconditions = _apply_analogy_to_atoms(
            analogy, rule.neg_state_preconditions)
        new_rule_goal_preconditions = _apply_analogy_to_atoms(
            analogy, rule.goal_preconditions)
        # Reconstruct parameters from the other components of the LDL.
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


def _apply_analogy_to_atoms(analogy: _Analogy,
                            atoms: Set[LiftedAtom]) -> Set[LiftedAtom]:
    new_atoms: Set[LiftedAtom] = set()
    for atom in atoms:
        new_variables = [
            _apply_analogy_to_variable(analogy, v) for v in atom.variables
        ]
        # Can't create an atom if there is no predicate match.
        if atom.predicate not in analogy.predicates:
            continue
        new_predicate = analogy.predicates[atom.predicate]
        new_atoms.add(LiftedAtom(new_predicate, new_variables))
    return new_atoms


def _apply_analogy_to_variable(analogy: _Analogy,
                               variable: Variable) -> Variable:
    return Variable(variable.name, analogy.types[variable.type])


def _create_sme_inputs(
        env: BaseEnv,
        nsrts: Set[NSRT]) -> Tuple[smepy.StructCase, Dict[str, Variable]]:
    action_terms: List[Any] = []
    var_name_to_var = {}
    # Sort to ensure determinism.
    for nsrt in sorted(nsrts):
        new_action_terms: List[Any] = []
        new_action_terms.append('action')
        new_action_terms.append(nsrt.name)
        new_action_terms.append([
            'precon', ['and'] +
            [_atom_to_s_exp(a, nsrt.name) for a in nsrt.preconditions]
        ])
        add_effect_terms = [
            _atom_to_s_exp(a, nsrt.name) for a in nsrt.add_effects
        ]
        delete_effect_terms = [['not', _atom_to_s_exp(a, nsrt.name)]
                               for a in nsrt.delete_effects]
        # Need to ignore types because of mypy issues with recursive types.
        effect_terms = add_effect_terms + delete_effect_terms  # type: ignore
        new_action_terms.append(['effects', ['and'] + effect_terms])
        action_terms.append(new_action_terms)
        # Save mapping from variable names to variables.
        for variable in nsrt.parameters:
            var_name = _variable_to_s_exp(variable, nsrt.name)
            var_name_to_var[var_name] = variable
    struct_case = smepy.StructCase(action_terms, env.get_name())
    return struct_case, var_name_to_var


def _sme_mapping_to_analogy(
        sme_mapping: smepy.Mapping, base_env: BaseEnv, target_env: BaseEnv,
        base_nsrts: Set[NSRT], target_nsrts: Set[NSRT],
        base_var_name_to_var: Dict[str, Variable],
        target_var_name_to_var: Dict[str, Variable]) -> _Analogy:
    # Used to construct the analogy.
    analogy_maps: Dict[str, Dict[str, Any]] = {
        "predicates": {},
        "nsrts": {},
        "types": {},
        "variables": {}
    }

    base_names_to_instances = _create_name_to_instances(
        base_env, base_nsrts, base_var_name_to_var)
    target_names_to_instances = _create_name_to_instances(
        target_env, target_nsrts, target_var_name_to_var)
    assert set(base_names_to_instances) == set(target_names_to_instances)
    assert set(base_names_to_instances) == set(analogy_maps)

    match_found = False
    for match in sme_mapping.entity_matches():
        base_name = match.base.name
        target_name = match.target.name
        for group in sorted(base_names_to_instances):
            base_map = base_names_to_instances[group]
            target_map = target_names_to_instances[group]
            if base_name in base_map and target_name in target_map:
                base_instance = base_map[base_name]
                target_instance = target_map[target_name]
                analogy_maps[group][base_instance] = target_instance
                match_found = True
        if match_found:
            break

    # Ignore types in favor of a more concise class instantiation.
    return _Analogy(**analogy_maps)  # type: ignore


def _atom_to_s_exp(atom: LiftedAtom, nsrt_name: str) -> str:
    # The NSRT name is used to rename the variables.
    pred = atom.predicate
    var_exps = [_variable_to_s_exp(v, nsrt_name) for v in atom.variables]
    return [f"term{pred.arity}", pred.name] + var_exps  # type: ignore


def _variable_to_s_exp(variable: Variable, nsrt_name: str) -> str:
    # The NSRT name is used to rename the variables.
    # Replace question marks because smepy doesn't like them.
    return f"{nsrt_name}-{variable.name.replace('?', '')}"


def _create_name_to_instances(
        env: BaseEnv, nsrts: Set[NSRT],
        var_name_to_var: Dict[str, Variable]) -> Dict[str, Dict[str, Any]]:
    # Helper for _sme_mapping_to_analogy().
    pred_name_to_pred = {p.name: p for p in env.predicates}
    type_name_to_type = {t.name: t for t in env.types}
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    names_to_instances: Dict[str, Dict[str, Any]] = {
        "predicates": pred_name_to_pred,
        "types": type_name_to_type,
        "nsrts": nsrt_name_to_nsrt,
        "variables": var_name_to_var,
    }
    return names_to_instances
