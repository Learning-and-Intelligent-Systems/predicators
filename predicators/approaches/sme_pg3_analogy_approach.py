"""Use SME for cross-domain policy learning in PG3.

Command for testing gripper / ferry:
    python predicators/main.py --approach sme_pg3 --seed 0  \
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

from predicators.approaches.pg3_analogy_approach import PG3AnalogyApproach
from predicators.envs.base_env import BaseEnv
from predicators.settings import CFG
from predicators.structs import NSRT, LDLRule, LiftedAtom, \
    LiftedDecisionList, Predicate, Type, Variable


class SMEPG3AnalogyApproach(PG3AnalogyApproach):
    """Use SME for cross-domain policy learning in PG3."""

    @classmethod
    def get_name(cls) -> str:
        return "sme_pg3"

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
    # Keep track of variables on a per-NSRT basis because some variable names
    # might be the same across multiple NSRTs, but we don't want to match them.
    nsrt_variables: Dict[Tuple[NSRT, Variable], Tuple[NSRT, Variable]]

    @cached_property
    def types(self) -> Dict[Type, Type]:
        """Infer type analogy from variables."""
        type_map: Dict[Type, Type] = {}
        for nsrt in self.nsrts:
            var_map = self.base_nsrt_to_variable_analogy(nsrt)
            for old_var, new_var in var_map.items():
                type_map[old_var.type] = new_var.type
        return type_map

    def base_nsrt_to_variable_analogy(
            self, base_nsrt: NSRT) -> Dict[Variable, Variable]:
        """Create a map of base to target variables for a given base NSRT."""
        old_to_new_var: Dict[Variable, Variable] = {}
        for (old_n, old_v), (new_n, new_v) in self.nsrt_variables.items():
            if old_n != base_nsrt:
                continue
            assert old_v not in old_to_new_var
            # Don't match variables between different NSRTs.
            if new_n != self.nsrts[old_n]:
                continue
            old_to_new_var[old_v] = new_v
        return old_to_new_var


def _find_env_analogies(base_env: BaseEnv, target_env: BaseEnv,
                        base_nsrts: Set[NSRT],
                        target_nsrts: Set[NSRT]) -> List[_Analogy]:
    # Use external SME module to find analogies.
    smepy.declare_nary("and")
    base_sme_struct = _create_sme_inputs(base_env, base_nsrts)
    target_sme_struct = _create_sme_inputs(target_env, target_nsrts)
    analogies: List[_Analogy] = []
    for sme_mapping in _query_sme(base_sme_struct, target_sme_struct):
        analogy = _sme_mapping_to_analogy(sme_mapping, base_env, target_env,
                                          base_nsrts, target_nsrts)
        analogies.append(analogy)
    return analogies


def _query_sme(
    base_sme_struct: smepy.StructCase, target_sme_struct: smepy.StructCase
) -> Iterator[smepy.Mapping]:  # pragma: no cover
    # Not unit-tested because slow.
    sme = smepy.SME(base_sme_struct,
                    target_sme_struct,
                    max_mappings=CFG.pg3_max_analogies)
    return sme.match()


def _apply_analogy_to_ldl(analogy: _Analogy,
                          ldl: LiftedDecisionList) -> LiftedDecisionList:
    new_rules = []
    for rule in ldl.rules:
        # Can't create a rule if there is no NSRT match.
        if rule.nsrt not in analogy.nsrts:
            continue
        # Create a rule-specific mapping for variables that is a superset of
        # the general analogy between NSRT variables.
        old_to_new_vars = _create_variable_mapping_for_rule(analogy, rule)
        new_rule_name = rule.name
        new_rule_nsrt = analogy.nsrts[rule.nsrt]
        new_rule_pos_preconditions = _apply_analogy_to_atoms(
            analogy, old_to_new_vars, rule.pos_state_preconditions)
        # Always add the NSRT preconditions because we would never want to
        # take an action that's inapplicable in a current state.
        new_rule_pos_preconditions.update(new_rule_nsrt.preconditions)
        new_rule_neg_preconditions = _apply_analogy_to_atoms(
            analogy, old_to_new_vars, rule.neg_state_preconditions)
        new_rule_goal_preconditions = _apply_analogy_to_atoms(
            analogy, old_to_new_vars, rule.goal_preconditions)
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


def _apply_analogy_to_atoms(analogy: _Analogy, old_to_new_vars: Dict[Variable,
                                                                     Variable],
                            atoms: Set[LiftedAtom]) -> Set[LiftedAtom]:
    new_atoms: Set[LiftedAtom] = set()
    for atom in atoms:
        new_variables = [old_to_new_vars[v] for v in atom.variables]
        # Can't create an atom if there is no predicate match.
        if atom.predicate not in analogy.predicates:
            continue
        new_predicate = analogy.predicates[atom.predicate]
        new_atoms.add(LiftedAtom(new_predicate, new_variables))
    return new_atoms


def _create_sme_inputs(env: BaseEnv, nsrts: Set[NSRT]) -> smepy.StructCase:
    action_terms: List[Any] = []
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
    struct_case = smepy.StructCase(action_terms, env.get_name())
    return struct_case


def _sme_mapping_to_analogy(sme_mapping: smepy.Mapping, base_env: BaseEnv,
                            target_env: BaseEnv, base_nsrts: Set[NSRT],
                            target_nsrts: Set[NSRT]) -> _Analogy:
    # Used to construct the analogy.
    analogy_maps: Dict[str, Dict[str, Any]] = {
        "predicates": {},
        "nsrts": {},
        "nsrt_variables": {}
    }

    base_names_to_instances = _create_name_to_instances(base_env, base_nsrts)
    target_names_to_instances = _create_name_to_instances(
        target_env, target_nsrts)
    assert set(base_names_to_instances) == set(target_names_to_instances)
    assert set(base_names_to_instances) == set(analogy_maps)

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


def _create_name_to_instances(env: BaseEnv,
                              nsrts: Set[NSRT]) -> Dict[str, Dict[str, Any]]:
    # Helper for _sme_mapping_to_analogy().
    pred_name_to_pred = {p.name: p for p in env.predicates}
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    var_name_to_nsrt_variables = {
        _variable_to_s_exp(v, n.name): (n, v)
        for n in nsrts
        for v in n.parameters
    }
    names_to_instances: Dict[str, Dict[str, Any]] = {
        "predicates": pred_name_to_pred,
        "nsrts": nsrt_name_to_nsrt,
        "nsrt_variables": var_name_to_nsrt_variables,
    }
    return names_to_instances
