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
from predicators.envs.pddl_env import _action_to_ground_strips_op
from predicators.envs import create_new_env

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
        ordered_objs = list(self._correctness_traj[0])
        self._env = create_new_env(CFG.env)
        self._correctness_actions = [_action_to_ground_strips_op(a, ordered_objs, sorted(self._env._strips_operators)) for a in dataset.trajectories[0]._actions]
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
            target_policy = _apply_chem_analogy_to_ldl(chem_analogy, base_policy, self._correctness_traj, self._correctness_actions, self._correctness_task, self._initial_predicates)
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
    special_analogies: Dict[Tuple[String, String], Dict[frozenset[LiftedAtom]: frozenset[LiftedAtom]]]

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
                          correctness_traj, correctness_actions, correctness_task, initial_predicates) -> LiftedDecisionList:
    new_rules = []

    for rule in ldl.rules:
        # Can't create a rule if there is no NSRT match.
        if rule.nsrt not in chem_analogy.nsrts:
            continue
        new_rule_nsrts = chem_analogy.nsrts[rule.nsrt]
        for new_rule_nsrt in new_rule_nsrts:
            new_rule_name = rule.name
            nsrt_name_pair = (rule.nsrt.name, new_rule_nsrt.name)

            new_rule = _apply_special_analogies(chem_analogy, rule, new_rule_nsrt, correctness_traj, correctness_actions, correctness_task, initial_predicates)

    return LiftedDecisionList(new_rules)


def _apply_special_analogies(chem_analogy, rule, new_rule_nsrt, correctness_traj, correctness_actions, correctness_task, initial_predicates):
    """Returns rule that applies chem_analogy from rule to new_rule_nsrt"""
    nsrt_name_pair = (rule.nsrt.name, new_rule_nsrt.name)

    superfluous_pos_conditions, superfluous_neg_conditions, superfluous_goal_preconditions = _create_superfluous_conditions(chem_analogy, rule, new_rule_nsrt)

def _create_superfluous_conditions(chem_analogy, rule, new_rule_nsrt):
    """Returns tuple of superfluous pos state preconditions, neg state preconditions, goal conditions"""
    rule_ground_objects = tuple([_convert_variable_to_object(param) for param in rule.parameters])
    ground_rule = rule.ground(rule_ground_objects)

    variable_count = 0

    all_superfluous_pos_state_preconditions = set()
    all_superfluous_neg_state_preconditions = set()
    all_superfluous_goal_preconditions = set()
     
    for base_template, target_template in chem_analogy.special_analogies.items():
        base_template = set(base_template)
        target_template = set(target_template)
        superfluous_pos_state_preconditions, variable_count = _apply_single_special_analogy(rule.pos_state_preconditions, base_template, target_template, variable_count)
        superfluous_neg_state_preconditions, variable_count = _apply_single_special_analogy(rule.neg_state_preconditions, base_template, target_template, variable_count)
        superfluous_goal_preconditions, variable_count = _apply_single_special_analogy(rule.goal_preconditions, base_template, target_template, variable_count)

        all_superfluous_pos_state_preconditions.update(superfluous_pos_state_preconditions)
        all_superfluous_neg_state_preconditions.update(superfluous_neg_state_preconditions)
        all_superfluous_goal_preconditions.update(superfluous_goal_preconditions)
    
    return all_superfluous_pos_state_preconditions, all_superfluous_neg_state_preconditions, all_superfluous_goal_preconditions
 
def _apply_single_special_analogy(conditions: Set[LiftedAtom], base_template: Set[LiftedAtom], target_template: Set[LiftedAtom], starting_variable_count: int) -> Tuple(Set[LiftedAtom], int):
    """Applies a special analogy (consisting of a base_template, target_template) pair to a set/list of conditions
    Returns set of superfluous conditions after applying special analogy and updated variable counter"""

    satisfying_base_atoms = _get_all_matches(base_template, conditions)
    num_satisfying_base_combs = len(satisfying_base_atoms)
    superfluous_conditions = set()

    variable_count = starting_variable_count
    for i in range(num_satisfying_base_combs):
        for target_atom in target_template:
            target_var_to_superfluous_var = {}
            for target_atom_var in target_atom.variables:
                target_var_to_superfluous_var[target_atom_var] = target_atom_var.type(f"?var-{variable_count}")
                variable_count += 1
            superfluous_atom = target_atom.substitute(target_var_to_superfluous_var)
            superfluous_conditions.add(superfluous_atom)

    return superfluous_conditions, variable_count

def _get_all_matches(template: Set[LiftedAtom], state: Set[LiftedAtom]) -> List[List[LiftedAtom]]:
    """Returns list of all sets matching template in state
    TODO: THIS IS FINNICKY"""
    ground_state = _get_grounded_atoms(state)
    template_parameters = sorted(set([var for atom in template for var in atom.variables]))
    static_predicates = set([atom.predicate for atom in template])
    objects = sorted(set([obj for atom in ground_state for obj in atom.objects]))

    param_to_preds: Dict[Variable, Set[Predicate]] = {
        p: set()
        for p in template_parameters
    }

    zero_arity_predicates = set()
    for atom in template:
        pred = atom.predicate
        if pred.arity == 0:
            zero_arity_predicates.add(pred)
        if pred in static_predicates and pred.arity == 1:
            param = atom.variables[0]
            param_to_preds[param].add(pred)
    
    # Checking zero arity conditions are in
    for zero_arity_pred in zero_arity_predicates:
        ground_zero_arity_condition = GroundAtom(zero_arity_pred, [])
        if ground_zero_arity_condition not in state:
            return []

    param_choices = []  # list of lists of possible objects for each param
    init_atom_tups = {(a.predicate, tuple(a.objects)) for a in ground_state}
    for param in template_parameters:
        choices = []
        for obj in objects:
            # Types must match, as usual.
            if obj.type != param.type:
                continue
            # Check the static conditions.
            binding_valid = True
            for pred in param_to_preds[param]:
                if (pred, (obj, )) not in init_atom_tups:
                    binding_valid = False
                    break
            if binding_valid:
                choices.append(obj)
        # Must be sorted for consistency with other grounding code.
        param_choices.append(sorted(choices))
    satisfying_ground_atoms = []
    for choice in itertools.product(*param_choices):
        param_to_variable = {template_parameters[i]: _convert_object_to_variable(choice[i]) for i in range(len(choice))}
        satisfying_ground_atoms_for_choice = [atom.substitute(param_to_variable) for atom in template]
        satisfying_ground_atoms.append(satisfying_ground_atoms_for_choice)
    return satisfying_ground_atoms

def _get_grounded_atoms(lifted_atoms: Set[LiftedAtom]) -> Set[GroundAtom]:
    grounded_atoms = set()
    for lifted_atom in lifted_atoms:
        objects = [_convert_variable_to_object(var) for var in lifted_atom.variables]
        if len(objects) > 0:
            grounded_atoms.add(lifted_atom.predicate(objects))
        else:
            grounded_atoms.add(GroundAtom(lifted_atom.predicate, []))
    return grounded_atoms

def _get_lifted_atoms(ground_atoms: Set[GroundAtom]) -> Set[LiftedAtom]:
    lifted_atoms = set()
    for ground_atom in ground_atoms:
        variables = [_convert_variable_to_object(obj) for obj in lifted_atom.objects]
        if len(variables) > 0:
            lifted_atoms.add(lifted_atom.predicate(variables))
        else:
            lifted_atoms.add(LiftedAtom(lifted_atom.predicate, []))
    return lifted_atoms

def _convert_variable_to_object(var: Variable) -> Object:
    return var.type(var.name[1:]) 

def _convert_object_to_variable(obj: Object) -> Variable:
    return obj.type("?" + obj.name) 