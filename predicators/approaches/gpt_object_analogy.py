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

from typing import Any, Dict, List, Set, Tuple, Union
from pg3.policy_search import score_policy

from predicators.approaches.pg3_analogy_approach import PG3AnalogyApproach
from predicators.envs.base_env import BaseEnv
from predicators.settings import CFG
from predicators.structs import NSRT, LDLRule, LiftedAtom, \
    LiftedDecisionList, Predicate, Variable, _GroundNSRT
from predicators.structs import Dataset, GroundAtom, \
    LiftedDecisionList, Object, Predicate, _GroundNSRT
from predicators.envs.pddl_env import _action_to_ground_strips_op
from predicators.envs import create_new_env

from predicators import utils
from predicators.llm_interface import OpenAILLM
from openai import OpenAI
from predicators.approaches.prompt_gen import get_prompt
import os

DEBUG = False
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class GPTObjectApproach(PG3AnalogyApproach):
    """Use GPT for cross-domain policy learning in PG3."""

    @classmethod
    def get_name(cls) -> str:
        return "gpt_obj"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Get a task and trajectory for correctness of generated conditions
        self._base_env = create_new_env(CFG.pg3_init_base_env)
        self._base_task = [t.task for t in self._base_env.get_train_tasks()]

        self._target_tasks = self._train_tasks
        self._target_states = [dataset.trajectories[i]._states for i in range(len(dataset.trajectories))]
        ordered_objs = [list(dataset.trajectories[i]._states[0]) for i in range(len(dataset.trajectories))]
        self._target_env = create_new_env(CFG.env)

        self._predicate_analogies = {} # Base Predicate Name -> Target Predicate
        self._nsrt_analogies = {} # Base NSRT -> Target NSRT
        self._variable_analogies = {} # Base Variable -> Target Variable
        self.setup_basic_predicate_analogies()
        self._target_actions = []
        for i in range(len(dataset.trajectories)):
            self._target_actions.append([_action_to_ground_strips_op(a, ordered_objs[i], sorted(self._target_env._strips_operators)) for a in dataset.trajectories[i]._actions])
        super().learn_from_offline_dataset(dataset)

    def _induce_policies_by_analogy(
            self, base_policy: LiftedDecisionList, base_env: BaseEnv,
            target_env: BaseEnv, base_nsrts: Set[NSRT],
            target_nsrts: Set[NSRT]) -> List[LiftedDecisionList]:
        # Generates new policy in target environment using base policy and analogies

        self._base_nsrts = base_nsrts
        self._target_nsrts = target_nsrts

        self._generate_expanded_predicates(base_policy)
        new_policy_rules = []
        for base_rule in base_policy.rules:
            target_rules = self._generate_rules(base_rule)
            for target_rule in target_rules:
                new_policy_rules.append(target_rule)

        pre_prune_policy = LiftedDecisionList(new_policy_rules)
        post_prune_policy = self.pg3_prune_policy(pre_prune_policy)
        return [post_prune_policy]

    def _generate_rules(self, rule: LDLRule) -> List[LDLRule]:
        # Generates "best" rules in target environment given rule in base environment
        target_nsrts = self._get_analagous_nsrts(rule) 
        translated_rules = []
        for target_nsrt in target_nsrts:
            new_rule = self._generate_best_rule(rule, target_nsrt)
            translated_rules.append(new_rule)
        return translated_rules
    
    def _generate_best_rule(self, rule: LDLRule, target_nsrt: NSRT) -> LDLRule:
        # Generates best rule in target environment given rule in base environment and a target nsrt

        # Search across all tasks and state for mapping
        best_object_mapping, best_task_index, best_state_index = self.search_for_best_state_and_mapping(rule, target_nsrt) 

        # Get useful objects
        useful_objects = set(best_object_mapping.values())
        useful_objects.update(self._target_actions[best_task_index][best_state_index].objects)

        # Getting useful analagous predicates
        useful_predicates = set()
        for pos_condition in rule.pos_state_preconditions:
            pos_predicate = pos_condition.predicate
            possible_pred_names = self.get_analagous_predicates(pos_predicate, True)
            if possible_pred_names is not None:
                pos_analagous_predicate_names = set(possible_pred_names)
                useful_predicates.update(pos_analagous_predicate_names)
        for neg_condition in rule.neg_state_preconditions:
            neg_predicate = add_not_prefix(neg_condition.predicate)
            neg_analagous_predicate_names = self.get_analagous_predicates(neg_predicate, True) 
            if neg_analagous_predicate_names is not None:
                set_neg_analagous_predicate_names = set(self.get_analagous_predicates(neg_predicate, True))
                useful_predicates.update(set_neg_analagous_predicate_names)
        for goal_condition in rule.goal_preconditions:
            goal_predicate = add_wanted_prefix(goal_condition.predicate)
            goal_analagous_predicate_names = self.get_analagous_predicates(goal_predicate, True)
            if goal_analagous_predicate_names is not None:
                set_goal_analagous_predicate_names = set(self.get_analagous_predicates(goal_predicate, True))
                useful_predicates.update(set_goal_analagous_predicate_names)

        # Getting final ranking of atoms in ground state
        ranked_atoms = self._score_conds(useful_objects, useful_predicates, best_task_index, best_state_index)
        useful_atoms = [atom for sublist in ranked_atoms for atom in sublist] 
        if DEBUG:
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("RULE\n", rule)
            print("GROUND ACTION\n", self._target_actions[best_task_index][best_state_index])
            print("GROUND ACTIONS\n", [self._target_actions[best_task_index][i].name for i in range(len(self._target_actions[best_task_index]))])
            print("BEST TASK INDEX", best_task_index)
            print("BEST INDEX: ", best_state_index)
            print("GROUND STATE\n", sorted(self._target_states[best_task_index][best_state_index].simulator_state))
            print("GOAL\n", self._target_tasks[best_task_index].goal)
            print("USEFUL OBJECTS", useful_objects)
            print("USEFUL POS PREDICATES", useful_predicates)
            print("PRECONDS", ranked_atoms)

        # Creates the lifted rule from useful atoms (preconditions) and ground action
        new_rule = self.create_lifted_rule(useful_atoms, best_object_mapping, self._target_actions[best_task_index][best_state_index])
        return new_rule

    # =========================== LIFTED RULE FINALIZATION ===========================

    def create_lifted_rule(self, useful_atoms: List[LiftedAtom], obj_to_var_mapping: Dict[Object, Variable], ground_action: _GroundNSRT) -> LDLRule:
        # Preprocessing
        lifted_action = ground_action.parent
        obj_to_var_mapping = {}
        all_final_objects = {saycan_atom_object for useful_atom in useful_atoms for saycan_atom_object in useful_atom.objects}
        for i in range(len(ground_action.objects)):
            obj_to_var_mapping[ground_action.objects[i]] = lifted_action.parameters[i]
        for leftover_object in sorted(all_final_objects - set(obj_to_var_mapping.keys())):
            new_var = leftover_object.type("?" + leftover_object.name)
            obj_to_var_mapping[leftover_object] = new_var

        # Creating the rule
        rule_pos_state_preconditions = set()
        rule_neg_state_preconditions = set()
        rule_goal_preconditions = set()
        for useful_atom in useful_atoms:
            if "WANT" in useful_atom.predicate.name:
                new_predicate = remove_wanted_prefix(useful_atom.predicate)
                new_variables = [obj_to_var_mapping[o] for o in useful_atom.objects]
                rule_goal_preconditions.add(new_predicate(new_variables))
                rule_neg_state_preconditions.add(new_predicate(new_variables))
            elif "NOT" in useful_atom.predicate.name:
                new_predicate = remove_not_prefix(useful_atom.predicate)
                new_variables = [obj_to_var_mapping[o] for o in useful_atom.objects]
                rule_neg_state_preconditions.add(new_predicate(new_variables))
            else:
                new_variables = [obj_to_var_mapping[o] for o in useful_atom.objects]
                rule_pos_state_preconditions.add(useful_atom.predicate(new_variables))
        
        for needed_condition in lifted_action.preconditions:
            rule_pos_state_preconditions.add(needed_condition)
        final_rule = LDLRule("generated-rule", sorted(obj_to_var_mapping.values()), rule_pos_state_preconditions, rule_neg_state_preconditions, rule_goal_preconditions, lifted_action)
        return final_rule

    def _score_conds(self, useful_objects, useful_predicates, task_index, index):
        # Returns a list of atoms sorted by score

        # Creates the ground state consisting of positive atoms, goal atoms, and negated atoms over the plan
        ground_state = self._target_states[task_index][index].simulator_state.copy()
        for atom in self._target_tasks[task_index].goal:
            wanted_atom = add_wanted_prefix(atom)
            ground_state.add(wanted_atom)
        for action in self._target_actions[task_index]: 
            deleted_atoms = action.delete_effects.copy() 
            for deleted_atom in deleted_atoms:
                if deleted_atom not in ground_state:
                    ground_state.add(add_not_prefix(deleted_atom))
        useful_pos_atoms = {}
        final_atoms = set()
        for atom in ground_state:
            score = 0.0
            pred = atom.predicate
            if pred.name in useful_predicates:
                score += 0.5
            for obj in atom.objects:
                if obj in useful_objects:
                    score += 1.0/len(atom.objects)
            if score >= 1.0:
                useful_pos_atoms[atom] = score
                final_atoms.add(atom)
        sorted_atoms, sorted_scores = sort_atoms_by_score(useful_pos_atoms)
        return sorted_atoms

    # =========================== PRUNING ===========================
    def pg3_prune_policy(self, policy: LiftedDecisionList):
        # Prunes the policy to remove extraneous conditions using PG3 score function
        # Generating all deletions
        list_of_rules = policy.rules.copy()

        things_to_remove = []
        for i in range(len(list_of_rules)):
            rule = list_of_rules[i]
            necessary_conditions = rule.nsrt.preconditions
            for index, group in [(0, rule.pos_state_preconditions), (1, rule.neg_state_preconditions), (2, rule.goal_preconditions), (3, [None])]:
                for condition in group:
                    if index == 0 and condition in necessary_conditions:
                        continue
                    things_to_remove.append((i, index, condition))
        
        current_score = self.get_pg3_scores([str(policy)])[0]
        current_list_of_rules = policy.rules.copy()
        num_deleted_rules = 0
        print(f"DONE {current_score}")
        print(policy)
        for rule_index, inner_rule_index, condition in things_to_remove:
            rule_index -= num_deleted_rules
            rule = current_list_of_rules[rule_index]
            new_pos_state_preconditions = rule.pos_state_preconditions.copy()
            new_neg_state_preconditions = rule.neg_state_preconditions.copy()
            new_goal_preconditions = rule.goal_preconditions.copy()
            if inner_rule_index == 0:
                new_pos_state_preconditions.remove(condition)
            elif inner_rule_index == 1:
                new_neg_state_preconditions.remove(condition)
            elif inner_rule_index == 2:
                new_goal_preconditions.remove(condition)
            new_rule = LDLRule("temp-rule", rule.parameters, new_pos_state_preconditions, new_neg_state_preconditions, new_goal_preconditions, rule.nsrt)

            # Creating new policy
            new_rules = []
            for j in range(len(current_list_of_rules)):
                if j != rule_index:
                    new_rules.append(current_list_of_rules[j])
                else:
                    if inner_rule_index != 3:
                        new_rules.append(new_rule)
                
            input = [str(LiftedDecisionList(new_rules))]
            pg3_score = self.get_pg3_scores(input)[0]
            if pg3_score < current_score:
                current_list_of_rules = new_rules
                current_score = pg3_score
                print("===========================")
                print("DELETED", rule_index, inner_rule_index, condition)
                print(pg3_score)
                print(LiftedDecisionList(new_rules))
            elif pg3_score <= current_score and inner_rule_index == 3:
                current_list_of_rules = new_rules
                num_deleted_rules += 1
                print("===========================")
                print("DELETED", rule_index, inner_rule_index, condition)
                print(pg3_score)
                print(LiftedDecisionList(new_rules))
            else:
                print("===========================")
                print("SKIPPED", rule_index, inner_rule_index, condition)
                
        print("LOWEST SCORE POLICY")
        lowest_score_policy = LiftedDecisionList(current_list_of_rules)
        print(lowest_score_policy)
        return lowest_score_policy
    
    def get_pg3_scores(self, policies: List[str]) -> List[float]:
        # Get the score of policies using PG3
        nsrts = self._get_current_nsrts()
        predicates = self._get_current_predicates()
        types = self._types
        domain_name = CFG.env
        domain_str = utils.create_pddl_domain(nsrts, predicates, types,
                                              domain_name)

        # Create the problem strs.
        problem_strs = []

        for i, train_task in enumerate(self._train_tasks):
            problem_name = f"problem{i}"
            goal = train_task.goal
            objects = set(train_task.init)
            init_atoms = utils.abstract(train_task.init, predicates)
            problem_str = utils.create_pddl_problem(objects, init_atoms, goal,
                                                    domain_name, problem_name)
            problem_strs.append(problem_str)
        
        scores =score_policy(
            domain_str,
            problem_strs,
            horizon=CFG.horizon,
            heuristic_name=CFG.pg3_heuristic,
            search_method=CFG.pg3_search_method,
            max_policy_guided_rollout=CFG.pg3_max_policy_guided_rollout,
            gbfs_max_expansions=CFG.pg3_gbfs_max_expansions,
            hc_enforced_depth=CFG.pg3_hc_enforced_depth,
            allow_new_vars=CFG.pg3_add_condition_allow_new_vars,
            initial_policy_strs=policies) 
        return scores
    
    # =========================== SEARCH FOR BEST STATE AND MAPPING ===========================
    def search_for_best_state_and_mapping(self, rule: LDLRule, target_nsrt: NSRT) -> Tuple[Dict[Variable, Object], int, int]:
        mapping = {} # Mapping from target param to rule variable
        for target_nsrt_param in target_nsrt.parameters:
            # Ask GPT what is the best variable in rule for target_nsrt_param
            best_rule_variable = self._get_analagous_variable(target_nsrt_param, rule, target_nsrt)
            if best_rule_variable != None:
                mapping[target_nsrt_param] = best_rule_variable
        
        # Search for a state that we can ground on the target environment
        best_score = -1.0
        best_object_mapping = None
        best_state_index = None
        best_task_index = None
        for task_index in range(len(self._target_tasks)):
            # In case planning for a task fails
            if task_index >= len(self._target_states) or len(self._target_states[task_index]) == 0:
                continue
            for i in range(len(self._target_states[task_index])-1):
                target_action = self._target_actions[task_index][i]
                if target_action.name != target_nsrt.name:
                    continue

                object_mapping, gini_index = self._get_object_distribution_and_score(rule, target_nsrt, mapping, task_index, i)
                if gini_index > best_score:
                    best_task_index = task_index
                    best_object_mapping = object_mapping
                    best_score = gini_index
                    best_state_index = i
        
        return best_object_mapping, best_task_index, best_state_index
 
    def _get_object_distribution_and_score(self, rule: LDLRule, target_nsrt: NSRT, existing_mapping: Dict[Variable, Variable], task_index: int, state_index: int):
        # Performs best-first search filling variables with objects, if possible
        initial_variables = set([var for var in rule.parameters]) # - set(existing_mapping.values())

        ground_target_nsrt = self._target_actions[task_index][state_index]
        constraints = {} # Base domain var to object
        """
        for target_var, base_var in existing_mapping.items():
            index_of_object = target_nsrt.parameters.index(target_var)
            target_object = ground_target_nsrt.objects[index_of_object]
            constraints[base_var] = target_object
        """
        
        distributions = self.find_distribution(rule, initial_variables, constraints, task_index, state_index)

        gini_state_score = 0.0

        while True:
            best_var = None
            best_gini_score = 0.0
            for var, var_distribution in distributions.items():
                gini_score = gini_index_from_dict(var_distribution)
                if gini_score > best_gini_score:
                    best_var = var
                    best_gini_score = gini_score
            
            if best_var is None:
                break

            best_object = max(distributions[best_var], key=distributions[best_var].get)
            constraints[best_var] = best_object
            initial_variables.remove(best_var)
            gini_state_score += best_gini_score
            distributions = self.find_distribution(rule, initial_variables, constraints, task_index, state_index)
        
        return constraints, gini_state_score

    def find_distribution(self, rule: LDLRule, available_variables: List[Variable], constraints: Dict[Variable, Object], task_index: int, state_index: int):
        # Gets the distribution of variables to objects in a current state under constraints
        ground_state = self._target_states[task_index][state_index].simulator_state.copy()
        goal_state = self._target_tasks[task_index].goal
        for atom in goal_state:
            wanted_atom = add_wanted_prefix(atom)
            ground_state.add(wanted_atom)
        
        all_distributions = {}
        
        for available_variable in available_variables:
            pos_conditions = set()
            for condition in rule.pos_state_preconditions:
                if available_variable in condition.variables:
                    pos_conditions.add(condition)
            
            goal_conditions = set()
            for condition in rule.goal_preconditions:
                if available_variable in condition.variables:
                    goal_conditions.add(condition)

            predicate_frequencies = {predicate: 0 for predicate in self._target_env.predicates}
            matches_array = []
            for list_index, conditions in enumerate([pos_conditions, goal_conditions]):
                for cond in conditions:
                    if list_index == 0: # If pos conditions, don't add WANT
                        analagous_preds = self.get_analagous_predicates(cond.predicate, True)
                    else: # If goal condition, add WANT
                        analagous_preds = self.get_analagous_predicates(add_wanted_prefix(cond.predicate), True)

                    if analagous_preds == None:
                        continue
                    needed_objects = set()
                    for var in cond.variables:
                        if var in constraints:
                            needed_objects.add(constraints[var])
                    for pos_atom in ground_state:
                        # if analagous predicate
                        if pos_atom.predicate.name in analagous_preds and needed_objects.issubset(set(pos_atom.objects)):
                            for obj in pos_atom.objects:
                                if obj in needed_objects or obj in constraints.values():
                                    continue
                                matches_array.append((pos_atom.predicate.name, obj))
                                if pos_atom.predicate.name in predicate_frequencies:
                                    predicate_frequencies[pos_atom.predicate.name] += 1
                                else:
                                    predicate_frequencies[pos_atom.predicate.name] = 1

            object_distribution = {obj: 1 for obj in self._target_states[task_index][state_index].data}
            for predicate_name, obj in matches_array:
                if obj not in object_distribution:
                    import ipdb; ipdb.set_trace();
                object_distribution[obj] += 1.0/predicate_frequencies[predicate_name]
            for tobj, object_score in object_distribution.items():
                object_distribution[tobj] = object_score * object_score * object_score

            all_distributions[available_variable] = object_distribution
        return all_distributions


    # =========================== UTILS ===========================
    def get_analagous_predicates(self, pred: Union[Predicate, str], want_name: bool = False):
        predicate_name = None
        if isinstance(pred, Predicate):
            predicate_name = pred.name
        else:
            predicate_name =  pred
        
        if predicate_name not in self._predicate_analogies:
            return None

        analagous_predicates = self._predicate_analogies[predicate_name]
        if want_name:
            return [analagous_predicate.name for analagous_predicate in analagous_predicates]
        else:
            return analagous_predicates.copy()

    def _generate_expanded_predicates(self, base_policy: LiftedDecisionList):
        temp_classifier = lambda s, o: False

        # Collecting all base environment predicates
        base_goal_predicates = set([goal_condition.predicate for rule in base_policy.rules for goal_condition in rule.goal_preconditions])
        base_neg_predicates = set([neg_condition.predicate for rule in base_policy.rules for neg_condition in rule.neg_state_preconditions])
        base_env_name_to_predicate = {}
        for base_goal_predicate in sorted(base_goal_predicates):
            new_base_goal_predicate_name = f"WANT-{base_goal_predicate.name}"
            new_base_goal_predicate = Predicate(new_base_goal_predicate_name, base_goal_predicate.types.copy(), temp_classifier)
            base_env_name_to_predicate[new_base_goal_predicate_name] = new_base_goal_predicate
        for base_neg_predicate in sorted(base_neg_predicates):
            new_base_neg_predicate_name = f"NOT-{base_neg_predicate.name}"
            new_base_neg_predicate = Predicate(new_base_neg_predicate_name, base_neg_predicate.types.copy(), temp_classifier)
            base_env_name_to_predicate[new_base_neg_predicate_name] = new_base_neg_predicate
        for base_pos_predicate in self._base_env.predicates:
            base_env_name_to_predicate[base_pos_predicate.name] = base_pos_predicate

        # Collecting all target environment predicates
        target_env_name_to_predicate = {}
        target_goal_predicates = set([goal_condition.predicate for task in self._target_tasks for goal_condition in task.goal])
        for target_goal_predicate in target_goal_predicates:
            new_target_goal_predicate_name = f"WANT-{target_goal_predicate.name}"
            new_target_goal_predicate = Predicate(new_target_goal_predicate_name, target_goal_predicate.types.copy(), temp_classifier)
            target_env_name_to_predicate[new_target_goal_predicate_name] = new_target_goal_predicate
        for target_predicate in self._target_env.predicates:
            target_env_name_to_predicate[target_predicate.name] = target_predicate
            # Adding all negative predicates
            new_target_neg_predicate_name = f"NOT-{target_predicate.name}"
            new_target_neg_predicate = Predicate(new_target_neg_predicate_name, target_predicate.types.copy(), temp_classifier)
            target_env_name_to_predicate[new_target_neg_predicate_name] = new_target_neg_predicate
        
        name_analogies = {}
        # Gripper -> Ferry
        if 'gripper' in self._base_env.get_name() and 'ferry' in self._target_env.get_name():
            name_analogies = {'WANT-at': ['WANT-at']}
            self._predicate_analogies['WANT-at'] = [target_env_name_to_predicate['WANT-at']]

        # Ferry -> Gripper
        if 'ferry' in self._base_env.get_name() and 'gripper' in self._target_env.get_name():
            name_analogies = {'WANT-at': ['WANT-at']}
            self._predicate_analogies['WANT-at'] = [target_env_name_to_predicate['WANT-at']]

        # Gripper -> Detyped Miconic
        if 'gripper' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            name_analogies = {'WANT-at': ['destin', 'WANT-served'], 'free': ['NOT-boarded']}

        # Ferry -> Detyped Miconic
        if 'ferry' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            name_analogies = {'WANT-at': ['destin', 'WANT-served'], 'empty-ferry': ['NOT-boarded']}

        # Gripper -> Detyped Delivery
        if 'gripper' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            # name_analogies = {'WANT-at': ['destin', 'WANT-served'], 'free': ['NOT-boarded']}
            pass

        # Ferry -> Detyped Delivery
        if 'ferry' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            # name_analogies = {'WANT-at': ['destin', 'WANT-served'], 'empty-ferry': ['NOT-boarded']}
            pass

        # Gripper -> Detyped Forest
        if 'gripper' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            pass

        # Ferry -> Detyped Forest
        if 'ferry' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            pass

        # Gripper -> Logistics
        if 'gripper' in self._base_env.get_name() and 'logistics' in self._target_env.get_name():
            name_analogies = {'WANT-at': ['WANT-at']}
        
        for base_name, target_names in name_analogies.items():
            self._predicate_analogies[base_name] = [target_env_name_to_predicate[target_name] for target_name in target_names]

    def setup_basic_predicate_analogies(self):
        predicate_input = {}
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

        # Gripper -> Detyped Delivery
        if 'gripper' in self._base_env.get_name() and 'detypeddelivery' in self._target_env.get_name():
            predicate_input = {
                "ball": ["paper"],
                "room": ["loc"],
                "at-robby": ["at"],
                "carry": ["carrying"],
            }

        # Ferry -> Detyped Delivery
        if 'ferry' in self._base_env.get_name() and 'detypeddelivery' in self._target_env.get_name():
            predicate_input = {
                "car": ["paper"],
                "location": ["loc"],
                "at-ferry": ["at"],
                "on": ["carrying"],
            }

        # Gripper -> Detyped Miconic
        if 'gripper' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            predicate_input = {
                "ball": ["passenger"],
                "room": ["floor"],
                "at-robby": ["lift-at"],
                "at": ["origin", "destin"],
                "carry": ["boarded"],
            }

        # Ferry -> Detyped Miconic
        if 'ferry' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            predicate_input = {
                "car": ["passenger"],
                "location": ["floor"],
                "at-ferry": ["lift-at"],
                "at": ["origin", "destin"],
                "on": ["boarded"],
            }

        # Gripper -> Detyped Forest
        if 'gripper' in self._base_env.get_name() and 'detypedforest' in self._target_env.get_name():
            predicate_input = {
                "room": ["loc"],
                "at-robby": ["at"],
                "at": ["at"],
            }

        # Ferry -> Detyped Miconic
        if 'ferry' in self._base_env.get_name() and 'detypedforest' in self._target_env.get_name():
            predicate_input = {
                "location": ["loc"],
                "at-ferry": ["at"],
                "at": ["at"],
            }

        # Gripper -> Logistics
        if 'gripper' in self._base_env.get_name() and 'logistics' in self._target_env.get_name():
            predicate_input = {
                "room": ["location", "city", "airport"],
                "ball": ["obj"],
                "gripper": ["truck", "airplane"],
                "at-robby": ["at"],
                "at": ["at"],
                "carry": ["in"],
            }

        # Gripper -> Detyped Spanner
        if 'gripper' in self._base_env.get_name() and 'detypedspanner' in self._target_env.get_name():
            predicate_input = {
                "room": ["location"],
                "ball": ["spanner"],
                "gripper": ["man"],
                "at-robby": ["at"],
                "at": ["at"],
                "carry": ["carrying"],
            }

        target_env_name_to_predicate = {}
        for predicate in self._target_env.predicates:
            target_env_name_to_predicate[predicate.name] = predicate
        
        for base_name, target_names in predicate_input.items():
            target_predicates = [target_env_name_to_predicate[target_name] for target_name in target_names]
            self._predicate_analogies[base_name] = target_predicates
    
    def _get_analagous_nsrts(self, rule: LDLRule) -> List[NSRT]:
        # Returns NSRT(s) in target environment that is analagous to the NSRT in rule
        nsrt_input = None
        # Gripper -> Ferry
        if 'gripper' in self._base_env.get_name() and 'ferry' in self._target_env.get_name():
            nsrt_input = { "move": ["sail"], "pick": ["board"], "drop": ["debark"], }

        # Ferry -> Gripper
        if 'ferry' in self._base_env.get_name() and 'gripper' in self._target_env.get_name():
            nsrt_input = { "sail": ["move"], "board": ["pick"], "debark": ["drop"], }

        # Gripper -> Detyped Miconic
        if 'gripper' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            nsrt_input = { "move": ["up", "down"], "pick": ["board"], "drop": ["depart"], }

        # Ferry -> Detyped Miconic
        if 'ferry' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            nsrt_input = { "sail": ["up", "down"], "board": ["board"], "debark": ["depart"], }

        # Gripper -> Detyped Delivery
        if 'gripper' in self._base_env.get_name() and 'detypeddelivery' in self._target_env.get_name():
            nsrt_input = { "move": ["move"], "pick": ["pick-up"], "drop": ["deliver"], }

        # Ferry -> Detyped Delivery
        if 'ferry' in self._base_env.get_name() and 'detypeddelivery' in self._target_env.get_name():
            nsrt_input = { "sail": ["move"], "board": ["pick-up"], "debark": ["deliver"], }

        # Gripper -> Detyped Forest
        if 'gripper' in self._base_env.get_name() and 'detypedforest' in self._target_env.get_name():
            nsrt_input = { "move": ["walk", "climb"]}

        # Ferry -> Detyped Forest
        if 'ferry' in self._base_env.get_name() and 'detypedforest' in self._target_env.get_name():
            nsrt_input = { "sail": ["walk", "climb"]}

        # Gripper -> Logistics
        if 'gripper' in self._base_env.get_name() and 'logistics' in self._target_env.get_name():
            nsrt_input = { "move": ["drive-truck", "fly-airplane"],
                            "pick": ["load-truck", "load-airplane"],
                            "drop": ["unload-truck", "unload-airplane"]}

        # Gripper -> Detyped Spanner
        if 'gripper' in self._base_env.get_name() and 'detypedspanner' in self._target_env.get_name():
            nsrt_input = { "move": ["walk"],
                            "pick": ["pickupspanner"]}

        if rule.nsrt.name not in nsrt_input:
            return []
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
        # Gripper -> Detyped Miconic
        if 'gripper' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            variable_input = {
                ("up", "move") : {"?f1": "?from", "?f2": "?to"},
                ("down", "move") : {"?f1": "?from", "?f2": "?to"},
                ("board", "pick") : {"?p": "?obj", "?f": "?room"},
                ("depart", "drop") : {"?p": "?obj", "?f": "?room"},
            }

        # Ferry -> Detyped Miconic
        if 'ferry' in self._base_env.get_name() and 'detypedmiconic' in self._target_env.get_name():
            variable_input = {
                ("up", "sail") : {"?f1": "?from", "?f2": "?to"},
                ("down", "sail") : {"?f1": "?from", "?f2": "?to"},
                ("board", "board") : {"?p": "?car", "?f": "?loc"},
                ("depart", "debark") : {"?p": "?car", "?f": "?loc"},
            }
        
        # Gripper -> Detyped Delivery
        if 'gripper' in self._base_env.get_name() and 'detypeddelivery' in self._target_env.get_name():
            variable_input = {
                ("move", "move") : {"?from": "?from", "?to": "?to"},
                ("pick-up", "pick") : {"?paper": "?obj", "?loc": "?room"},
                ("deliver", "drop") : {"?paper": "?obj", "?loc": "?room"},
            }

        # Ferry -> Detyped Delivery
        if 'ferry' in self._base_env.get_name() and 'detypeddelivery' in self._target_env.get_name():
            variable_input = {
                ("move", "sail") : {"?from": "?from", "?to": "?to"},
                ("pick-up", "board") : {"?paper": "?car", "?loc": "?loc"},
                ("deliver", "debark") : {"?paper": "?car", "?loc": "?loc"},
            }

         # Gripper -> Detyped Forest
        if 'gripper' in self._base_env.get_name() and 'detypedforest' in self._target_env.get_name():
            variable_input = {
                ("walk", "move") : {"?from": "?from", "?to": "?to"},
                ("climb", "move") : {"?from": "?from", "?to": "?to"},
            }

        # Ferry -> Detyped Forest
        if 'ferry' in self._base_env.get_name() and 'detypedforest' in self._target_env.get_name():
            variable_input = {
                ("walk", "sail") : {"?from": "?from", "?to": "?to"},
                ("climb", "sail") : {"?from": "?from", "?to": "?to"},
            }

         # Gripper -> Logistics
        if 'gripper' in self._base_env.get_name() and 'logistics' in self._target_env.get_name():
            variable_input = {
                ("drive-truck", "move") : {"?loc_from": "?from", "?loc_to": "?to"},
                ("fly-airplane", "move") : {"?loc_from": "?from", "?loc_to": "?to"},
                ("unload-truck", "drop") : {"?obj": "?obj", "?loc": "?room", "?truck": "?gripper"},
                ("unload-airplane", "drop") : {"?obj": "?obj", "?loc": "?room", "?airplane": "?gripper"},
                ("load-truck", "pick") : {"?obj": "?obj", "?loc": "?room", "?truck": "?gripper"},
                ("load-airplane", "pick") : {"?obj": "?obj", "?loc": "?room", "?airplane": "?gripper"},
            }

        # Gripper -> Detyped Spanner
        if 'gripper' in self._base_env.get_name() and 'detypedspanner' in self._target_env.get_name():
            variable_input = {
                ("walk", "move") : {"?start": "?from", "?end": "?to"},
                ("pickupspanner", "pick") : {"?s": "?obj", "?l": "?room", "?m": "?gripper"},
            }

        if nsrt_param.name in variable_input[(target_nsrt.name, rule.nsrt.name)]:
            var_name = variable_input[(target_nsrt.name, rule.nsrt.name)][nsrt_param.name]
            base_var_names_to_var = {var.name: var for var in rule.parameters}
            return base_var_names_to_var[var_name]
        else:
            return None

def add_not_prefix(element: Union[Predicate, LiftedAtom, GroundAtom]):
    """Returns same type but with the Predicate becoming NOT-Predicate"""
    temp_classifier = lambda s, o: False
    if isinstance(element, Predicate):
        new_predicate_name = f"NOT-{element.name}"
        new_predicate = Predicate(new_predicate_name, element.types.copy(), temp_classifier)
        return new_predicate
    elif isinstance(element, LiftedAtom):
        new_predicate_name = f"NOT-{element.predicate.name}"
        new_predicate = Predicate(new_predicate_name, element.predicate.types.copy(), temp_classifier)
        new_atom = LiftedAtom(new_predicate, element.variables)
        return new_atom
    elif isinstance(element, GroundAtom):
        new_predicate_name = f"NOT-{element.predicate.name}"
        new_predicate = Predicate(new_predicate_name, element.predicate.types.copy(), temp_classifier)
        new_atom = GroundAtom(new_predicate, element.objects)
        return new_atom
    else:
        return None

def remove_not_prefix(element: Union[Predicate, LiftedAtom, GroundAtom]):
    """Returns same type but with the Predicate not having WANT"""
    temp_classifier = lambda s, o: False
    if isinstance(element, Predicate):
        new_predicate_name = element.name[4:]
        new_predicate = Predicate(new_predicate_name, element.types.copy(), temp_classifier)
        return new_predicate
    elif isinstance(element, LiftedAtom):
        new_predicate_name = element.predicate.name[4:]
        new_predicate = Predicate(new_predicate_name, element.predicate.types.copy(), temp_classifier)
        new_atom = LiftedAtom(new_predicate, element.variables)
        return new_atom
    elif isinstance(element, GroundAtom):
        new_predicate_name = element.predicate.name[4:]
        new_predicate = Predicate(new_predicate_name, element.predicate.types.copy(), temp_classifier)
        new_atom = GroundAtom(new_predicate, element.objects)
        return new_atom
    else:
        return None

def add_wanted_prefix(element: Union[Predicate, LiftedAtom, GroundAtom]):
    """Returns same type but with the Predicate becoming WANT-Predicate"""
    temp_classifier = lambda s, o: False
    if isinstance(element, Predicate):
        new_predicate_name = f"WANT-{element.name}"
        new_predicate = Predicate(new_predicate_name, element.types.copy(), temp_classifier)
        return new_predicate
    elif isinstance(element, LiftedAtom):
        new_predicate_name = f"WANT-{element.predicate.name}"
        new_predicate = Predicate(new_predicate_name, element.predicate.types.copy(), temp_classifier)
        new_atom = new_predicate(element.variables)
        return new_atom
    elif isinstance(element, GroundAtom):
        new_predicate_name = f"WANT-{element.predicate.name}"
        new_predicate = Predicate(new_predicate_name, element.predicate.types.copy(), temp_classifier)
        new_atom = new_predicate(element.objects)
        return new_atom
    else:
        return None

def remove_wanted_prefix(element: Union[Predicate, LiftedAtom, GroundAtom]):
    """Returns same type but with the Predicate not having WANT"""
    temp_classifier = lambda s, o: False
    if isinstance(element, Predicate):
        new_predicate_name = element.name[5:]
        new_predicate = Predicate(new_predicate_name, element.types.copy(), temp_classifier)
        return new_predicate
    elif isinstance(element, LiftedAtom):
        new_predicate_name = element.predicate.name[5:]
        new_predicate = Predicate(new_predicate_name, element.predicate.types.copy(), temp_classifier)
        new_atom = new_predicate(element.variables)
        return new_atom
    elif isinstance(element, GroundAtom):
        new_predicate_name = element.predicate.name[5:]
        new_predicate = Predicate(new_predicate_name, element.predicate.types.copy(), temp_classifier)
        new_atom = new_predicate(element.objects)
        return new_atom
    else:
        return None

def _convert_object_to_variable(obj: Object) -> Variable:
    return obj.type("?" + obj.name)

def gini_index_from_dict(distribution_dict):
    """
    Calculate the Gini index for a given discrete distribution represented as a dictionary.

    :param distribution_dict: A dictionary where keys are elements and values are their frequencies
    :return: The Gini index as a float
    """
    values = list(distribution_dict.values())
    n = len(values)

    # If there are no elements in the distribution, return 0
    if n == 0:
        return 0

    sum_of_values = sum(values)
    sorted_values = sorted(values)

    # If the sum of values is 0 (all values are 0), return 0 to avoid division by zero
    if sum_of_values == 0:
        return 0

    cumulative_sum = 0
    for i, value in enumerate(sorted_values, 1):
        cumulative_sum += value * i

    # Gini index calculation
    gini = (2 * cumulative_sum) / (n * sum_of_values) - (n + 1) / n
    return gini

def sort_atoms_by_score(atoms_dict):
    # Group atoms by their scores
    grouped_atoms = {}
    for atom, score in atoms_dict.items():
        if score in grouped_atoms:
            grouped_atoms[score].append(atom)
        else:
            grouped_atoms[score] = [atom]

    # Sort the scores in descending order
    sorted_scores = sorted(grouped_atoms, reverse=True)

    # Arrange the atoms according to the sorted scores
    sorted_atoms = [grouped_atoms[score] for score in sorted_scores]
    
    return sorted_atoms, sorted_scores

def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4-0125-preview",
    max_tokens=1,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
    top_logprobs=None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion