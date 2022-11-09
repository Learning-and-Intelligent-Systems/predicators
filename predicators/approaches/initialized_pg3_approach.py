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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

import dill as pkl

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
    def _get_policy_search_initial_ldl() -> LiftedDecisionList:
        # Initialize with initialized policy from file.
        if CFG.pg3_init_policy is None:  # pragma: no cover
            # By default, use policy from base domain.
            save_path = utils.get_approach_save_path_str()
            pg3_init_policy_file = f"{save_path}_None.ldl"
        else:
            pg3_init_policy_file = CFG.pg3_init_policy
        with open(pg3_init_policy_file, "rb") as f:
            base_policy = pkl.load(f)
        # Determine an analogical mapping between the current env and the
        # base env that the initialized policy originates from.
        base_env_name = CFG.pg3_init_base_env
        target_env_name = CFG.env
        base_env = get_or_create_env(base_env_name)
        target_env = get_or_create_env(target_env_name)
        analogy = _find_env_analogy(base_env, target_env)
        # Use the analogy to create an initial policy for the target env.
        target_policy = _apply_analogy_to_ldl(analogy, base_policy)
        # Initialize PG3 search with this new target policy.
        return target_policy


@dataclass(frozen=True)
class _Analogy:
    # All maps are base -> target.
    predicates: Dict[Predicate, Predicate]
    nsrts: Dict[NSRT, NSRT]
    types: Dict[Type, Type]


def _find_env_analogy(base_env: BaseEnv, target_env: BaseEnv) -> _Analogy:
    assert base_env.get_name() == target_env.get_name(), \
        "Only trivial env mappings are implemented so far"
    env = base_env
    predicate_map = {p: p for p in env.predicates}
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, env.options)
    nsrt_map = {n: n for n in nsrts}
    type_map = {t: t for t in env.types}
    return _Analogy(predicate_map, nsrt_map, type_map)


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
        new_rule_nsrt = analogy.nsrts[rule.nsrt]
        new_rule = LDLRule(new_rule_name, new_rule_parameters,
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
        new_predicate = analogy.predicates[atom.predicate]
        new_atoms.add(LiftedAtom(new_predicate, new_variables))
    return new_atoms


def _apply_analogy_to_variable(analogy: _Analogy,
                               variable: Variable) -> Variable:
    return Variable(variable.name, analogy.types[variable.type])
