"""Policy-guided planning for generalized policy generation (PG3) with an
initialized policy.

PG3 requires known STRIPS operators. The command below uses oracle operators,
but it is also possible to use this approach with operators learned from
demonstrations.

Example command line:
    python predicators/main.py --approach initialized_pg3 --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --strips_learner oracle --num_train_tasks 10
"""
from __future__ import annotations

import dill as pkl
import os
import re
from typing import List, Set, Dict
import numpy as np

from predicators.approaches.pg3_approach import PG3Approach
from predicators.settings import CFG
from predicators.structs import LiftedDecisionList, GroundAtom, NSRT, LDLRule, Predicate
from predicators.ground_truth_nsrts import get_gt_nsrts


class InitializedPG3Approach(PG3Approach):
    """Policy-guided planning for generalized policy generation (PG3) with
    initialized policy."""

    @classmethod
    def get_name(cls) -> str:
        return "initialized_pg3"

    def _get_policy_search_initial_ldl(self) -> LiftedDecisionList:
        # Initialize with initialized policy from file.
        assert CFG.pg3_init_policy

        substitutions = {}
        for initial_predicate in self._initial_predicates:
            new_predicate_name = "pre" + initial_predicate.name
            substitutions[initial_predicate] = Predicate(new_predicate_name, initial_predicate.types, initial_predicate._classifier)

        filename, file_extension = os.path.splitext(CFG.pg3_init_policy)
        assert file_extension == ".ldl" or file_extension == ".txt"
        if file_extension == ".ldl":
            with open(CFG.pg3_init_policy, "rb") as f:
                first_policy = pkl.load(f)
                new_policy = InitializedPG3Approach.apply_hardcoded_substitutions(first_policy, substitutions)
                return new_policy

        elif file_extension == ".txt":
            with open(CFG.pg3_init_policy, "r") as f:
                policy_str = f.read()
                policy_parser = PolicyParser(self._types, self._initial_predicates, self._initial_options)
                first_policy = policy_parser.parse_policy(policy_str)
                new_policy = InitializedPG3Approach.apply_hardcoded_substitutions(first_policy, substitutions)
                return new_policy

    @staticmethod
    def apply_hardcoded_substitutions(
            initial_ldl: LiftedDecisionList,
            substitutions: Dict[Predicate, Predicate]) -> LiftedDecisionList:
        # Return new LiftedDecisionList replacing all predicates using hardcoded substitutions
        policy_rules = []
        
        for rule in initial_ldl.rules:
            
            new_rule_name = rule.name
            new_rule_parameters = rule.parameters.copy()

            new_rule_pos_preconditions = set()
            for pos_precond in rule.pos_state_preconditions:
                variables = pos_precond.variables
                new_predicate = substitutions[pos_precond.predicate]
                new_rule_pos_preconditions.add(new_predicate(variables))

            new_rule_neg_preconditions = set()
            for neg_precond in rule.neg_state_preconditions:
                variables = neg_precond.variables
                new_predicate = substitutions[neg_precond.predicate]
                new_rule_goal_preconditions.add(new_predicate(variables))

            new_rule_goal_preconditions = set()
            for goal_precond in rule.pos_state_preconditions:
                variables = goal_precond.variables
                new_predicate = substitutions[goal_precond.predicate]
                new_rule_goal_preconditions.add(new_predicate(variables))

            new_rule_nsrt = rule.nsrt

            policy_rules.append(LDLRule(new_rule_name, new_rule_parameters, new_rule_pos_preconditions, new_rule_neg_preconditions, new_rule_goal_preconditions, new_rule_nsrt))

        return LiftedDecisionList(policy_rules)

class PolicyParser():
    """
    Parser for ordered-decision lists composed of rules with name, paramaters, 
    preconditions, goal conditions, and action
    """
    def __init__(self, types, predicates, options):
        nsrts = get_gt_nsrts(predicates, options)
        self.nsrt_name_to_nsrt = {nsrt.name: nsrt for nsrt in nsrts}
        self.type_name_to_type = {t.name: t for t in types}
        self.predicate_name_to_predicate = {p.name: p for p in predicates}

    def parse_policy(self, policy_str: str) -> LiftedDecisionList:
        policy_rules = []

        rule_matches = re.finditer(r"\(:rule", policy_str)
        rule_pattern = r"\(:rule(.*):parameters(.*):preconditions(.*):goals(.*):action(.*)\)"
        for rule_start in rule_matches:

            rule_str = self._find_balanced_expression(policy_str, rule_start.start())
            rule_name_str, params_str, preconds_str, goals_str, action_str = re.match(rule_pattern, rule_str, re.DOTALL).groups()
        
            rule_name_str = rule_name_str.strip()
            params_str = params_str.strip()
            preconds_str = preconds_str.strip()
            goals_str = goals_str.strip()
            action_str = action_str.strip()

            params_str = params_str[1:-1].split("?")
            params_mapping = [(param.strip().split("-", 1)[0].strip(), param.strip().split("-", 1)[1].strip()) for param in params_str[1:]]

            params = set()
            variable_name_to_variable = {}
            for par_name, par_type in params_mapping:
                lifted_par_name = "?" + par_name
                par = self.type_name_to_type[par_type](lifted_par_name)
                params.add(par)
                variable_name_to_variable[lifted_par_name] = par

            preconds = self._parse_into_literal(preconds_str, variable_name_to_variable, self.predicate_name_to_predicate)
            pos_preconds = set()
            neg_preconds = set()
            for precond in preconds:
                if precond.predicate.name.startswith("NOT-"):
                    neg_preconds.add(precond)
                else:
                    pos_preconds.add(precond)

            goals = self._parse_into_literal(goals_str, variable_name_to_variable, self.predicate_name_to_predicate)

            action = self._parse_into_nsrt(action_str, self.nsrt_name_to_nsrt)

            policy_rules.append(LDLRule(rule_name_str, params, pos_preconds, neg_preconds, goals, action))
        
        return LiftedDecisionList(policy_rules)
    
    def _parse_into_nsrt(self, string, nsrt_name_to_nsrt) -> NSRT:
        """Parse the given string (representing either preconditions or effects)
        into a NSRT. 
        """
        assert string[0] == "("
        assert string[-1] == ")"
        string = string[1:-1].split()
        return nsrt_name_to_nsrt[string[0]]

    def _parse_into_literal(self, string, variable_name_to_variable, predicate_name_to_predicate) -> set[GroundAtom]:
        """Parse the given string (representing either preconditions or effects)
        into a literal. Check against params to make sure typing is correct.
        Code taken from pddlgym
        """
        assert string[0] == "("
        assert string[-1] == ")"

        if string.startswith("(and") and string[4] in (" ", "\n", "("):
            clauses = self._find_all_balanced_expressions(string[4:-1].strip())
            literals = set()
            for clause in clauses:
                literals = literals | self._parse_into_literal(clause, variable_name_to_variable, predicate_name_to_predicate)
            return literals

        elif string.startswith("(not") and string[4] in (" ", "\n", "("): 
            #Only contains a single literal inside not
            new_string = string[4:-1].strip()[1: -1].strip().split()
            pred = predicate_name_to_predicate[new_string[0]].get_negation()
            args = [variable_name_to_variable[arg] for arg in new_string[1:]]
            return {pred(args)}

        string = string[1:-1].split()
        pred = predicate_name_to_predicate[string[0]]
        args = [variable_name_to_variable[arg] for arg in string[1:]]

        return {pred(args)}

    def _find_balanced_expression(self, string: str, index: int) -> str:
        """Find balanced expression in string starting from given index.
        Code taken from pddlgym.
        """
        assert string[index] == "("
        start_index = index
        balance = 1
        while balance != 0:
            index += 1
            symbol = string[index]
            if symbol == "(":
                balance += 1
            elif symbol == ")":
                balance -= 1
        return string[start_index:index+1]

    def _find_all_balanced_expressions(self, string: str) -> str:
        """Return a list of all balanced expressions in a string,
        starting from the beginning.
        Code taken from pddlgym
        """
        assert string[0] == "("
        assert string[-1] == ")"
        exprs = []
        index = 0
        start_index = index
        balance = 1
        while index < len(string)-1:
            index += 1
            if balance == 0:
                exprs.append(string[start_index:index])
                # Jump to next "(".
                while True:
                    if string[index] == "(":
                        break
                    index += 1
                start_index = index
                balance = 1
                continue
            symbol = string[index]
            if symbol == "(":
                balance += 1
            elif symbol == ")":
                balance -= 1
        assert balance == 0
        exprs.append(string[start_index:index+1])
        return exprs

