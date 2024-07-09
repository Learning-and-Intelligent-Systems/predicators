"""
Example command line:
    export OPENAI_API_KEY=<your API key>
"""
import ast
import base64
import importlib.util
import inspect
import json
import logging
import os
import re
import subprocess
import textwrap
import time
from collections import defaultdict, namedtuple
from copy import deepcopy
from inspect import getsource
from pprint import pformat
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Sequence, \
    Set, Tuple

import dill
import imageio
import numpy as np
from gym.spaces import Box
from tabulate import tabulate
from tqdm import tqdm

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.grammar_search_invention_approach import \
    _create_grammar, _GivenPredicateGrammar, create_score_function
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.envs import BaseEnv
from predicators.ground_truth_models import get_gt_nsrts
from predicators.llm_interface import OpenAILLM, OpenAILLMNEW
from predicators.predicate_search_score_functions import \
    _ClassificationErrorScoreFunction, _PredicateSearchScoreFunction, \
    create_score_function
from predicators.settings import CFG
from predicators.structs import Action, AnnotatedPredicate, Dataset, \
    GroundAtomTrajectory, GroundOptionRecord, LowLevelTrajectory, Object, \
    Optional, ParameterizedOption, Predicate, State, Task, Type, _Option, \
    _TypedEntity
from predicators.utils import EnvironmentFailure, OptionExecutionFailure, \
    option_plan_to_policy

import_str = """
import numpy as np
from typing import Sequence
from predicators.structs import State, Object, Predicate, Type
from predicators.utils import RawState, NSPredicate
"""

PlanningResult = namedtuple("PlanningResult", ['succeeded', 'info'])

def are_equal_by_obj(list1: List[_Option], list2: List[_Option]) -> bool:
    if len(list1) != len(list2):
        return False

    return all(
        option1.eq_by_obj(option2) for option1, option2 in zip(list1, list2))


def print_confusion_matrix(tp: float, tn: float, fp: float, fn: float) -> None:
    """Compate and print the confusion matrix."""
    precision = round(tp / (tp + fp), 2) if tp + fp > 0 else 0
    recall = round(tp / (tp + fn), 2) if tp + fn > 0 else 0
    specificity = round(tn / (tn + fp), 2) if tn + fp > 0 else 0
    accuracy = round(
        (tp + tn) / (tp + tn + fp + fn), 2) if tp + tn + fp + fn > 0 else 0
    f1_score = round(2 * (precision * recall) /
                     (precision + recall), 2) if precision + recall > 0 else 0

    table = [[
        "",
        "Positive",
        "Negative",
        "Precision",
        "Recall",
        "Specificity",
        "Accuracy",
        "F1 Score",
    ], ["True", tp, tn, "", "", "", "", ""],
             ["False", fp, fn, "", "", "", "", ""],
             ["", "", "", precision, recall, specificity, accuracy, f1_score]]
    logging.info(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


# Function to encode the image
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def add_python_quote(text: str) -> str:
    return f"```python\n{text}\n```"


def d2s(dict_with_arrays: Dict) -> str:
    # Convert State data with numpy arrays to lists, and to string
    return str({
        k: [round(i, 2) for i in v.tolist()]
        for k, v in dict_with_arrays.items()
    })


class VlmInventionApproach(NSRTLearningApproach):
    """Predicate Invention with VLMs."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Initial Predicates
        nsrts = get_gt_nsrts(CFG.env, self._initial_predicates,
                             self._initial_options)
        self._nsrts = nsrts

        self._learned_predicates: Set[Predicate] = set()
        # self._candidates: Set[Predicate] = set()
        self._num_inventions = 0
        # Set up the VLM
        self._vlm = OpenAILLMNEW(CFG.vlm_model_name)
        self._type_dict = {type.name: type for type in self._types}

    @classmethod
    def get_name(cls) -> str:
        return "vlm_online_invention"

    @property
    def is_offline_learning_based(self) -> bool:
        return False

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        pass

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates

    def load(self, online_learning_cycle: Optional[int]) -> None:
        super().load(online_learning_cycle)

        preds, _ = utils.extract_preds_and_types(self._nsrts)
        self._learned_predicates = (set(preds.values()) -
                                    self._initial_predicates)

    def _solve_tasks(self, env: BaseEnv, tasks: List[Task], ite: int) -> \
        List[PlanningResult]:
        """When return_trajctories is True, return the dataset of trajectories
        otherwise, return the results of solving the tasks (succeeded/failed
        plans)."""
        results = []
        trajectories = []
        for idx, task in enumerate(tasks):
            logging.info(f"Ite {ite}. Solving Task {idx}")
            # logging.debug(f"Init: {init_atoms} \nGoals: {task.goal}")
            # task.init.labeled_image.save(f"images/trn_{idx}_init.png")
            try:
                policy = self.solve(task, timeout=CFG.timeout)
            except (ApproachTimeout, ApproachFailure) as e:
                logging.info(f"Planning failed: {str(e)}")
                if "metrics" not in e.info:
                    # In the case of not dr-reachable
                    metrics, p_ref = None, []
                else:
                    metrics = e.info["metrics"],
                    p_ref = e.info["partial_refinements"]
                result = PlanningResult(succeeded=False,
                                        info={
                                            "metrics": metrics,
                                            "partial_refinements": p_ref,
                                            "error": str(e)
                                        })
            else:
                # logging.info(f"--> Succeeded")
                # This is essential, otherwise would cause errors
                # policy = utils.option_plan_to_policy(self._last_plan)

                result = PlanningResult(succeeded=True,
                                        info={
                                            "option_plan": self._last_plan,
                                            "nsrt_plan": self._last_nsrt_plan,
                                            "metrics": self._last_metrics,
                                            "partial_refinements":
                                            self._last_partial_refinements,
                                            "policy": policy
                                        })
                # Collect trajectory
                # try:
                traj, _ = utils.run_policy(
                    policy,
                    env,
                    "train",
                    idx,
                    termination_function=lambda s: False,
                    max_num_steps=CFG.horizon,
                    exceptions_to_break_on={
                        utils.OptionExecutionFailure, ApproachFailure
                    })
                # except:
                #     breakpoint()
                self.task_to_latest_traj[idx] = LowLevelTrajectory(
                    traj.states,
                    traj.actions,
                    _is_demo=True,
                    _train_task_idx=idx)
                # trajectories.append(traj)
            results.append(result)
        # dataset = Dataset(trajectories)
        return results

    def learn_from_tasks(self, env: BaseEnv, tasks: List[Task]) -> None:
        """Learn from interacting with the offline dataset."""
        for i, task in enumerate(tasks):
            task.init.state_image.save(f"images/init_state{i}.png")
            task.init.labeled_image.save(f"images/init_label{i}.png")
        # breakpoint()
        self.env_name = env.get_name()
        num_tasks = len(tasks)
        propose_ite = 0
        max_invent_ite = 20
        add_new_proposal_at_every_ite = False  # Invent at every iterations
        manual_prompt = True
        regenerate_response = False
        # solve_rate, prev_solve_rate = 0.0, np.inf  # init to inf
        best_solve_rate, best_ite, clf_acc = 0.0, 0.0, 0.0
        clf_acc_at_best_solve_rate = 0.0
        best_nsrt, best_preds = deepcopy(self._nsrts), set()
        self._learned_predicates = set()
        # Keep a copy in case it doesn't learn it from data.
        self._init_nsrts = deepcopy(self._nsrts)
        no_improvement = False
        self.state_cache: Dict[int, RawState] = {}
        self.base_candidates: Set[Predicate] = self._initial_predicates.copy()

        # Init data collection
        logging.debug(f"Initial predicates: {self._get_current_predicates()}")
        logging.debug(f"Initial operators: {pformat(self._init_nsrts)}")

        # For storing the results found at every iteration
        self.task_to_latest_traj: Dict[int, LowLevelTrajectory] = dict()
        # For help checking if a new plan is unique, to control data collection
        self.task_to_plans: Dict[int, List[_Option]] = defaultdict(list)
        # Organize a dataset for operator learning. This becomes the operator
        # learning dataset when the trajectories are put together.
        self.task_to_trajs: Dict[int, List[LowLevelTrajectory]] = \
            defaultdict(list)
        # Storing the prefix of partial trajectories
        self.task_to_partial_trajs: Dict[int, List[LowLevelTrajectory]] = \
            defaultdict(list)

        # Return the results and populate self.task_to_latest_traj
        num_init_nsrts = len(self._nsrts)
        self._nsrts = utils.reduce_nsrts(self._nsrts)
        num_reduced_nsrts = num_init_nsrts - len(self._nsrts)
        self._reduced_nsrts = deepcopy(self._nsrts)
        self._previous_nsrts = deepcopy(self._nsrts)
        logging.debug(f"Initial operators after pruning {num_reduced_nsrts}:\n"
                      f"{pformat(self._nsrts)}")
        results = self.collect_dataset(0, env, tasks)
        num_solved = sum([r.succeeded for r in results])
        num_failed_plans = prev_num_failed_plans = sum([len(
                r.info['partial_refinements']) for r in results])
        solve_rate = prev_solve_rate = num_solved / num_tasks
        logging.info(f"===ite 0; no invent solve rate {solve_rate}; "
                    f"num skeletons failed {num_failed_plans}\n")

        self.succ_optn_dict: Dict[str, GroundOptionRecord] =\
            defaultdict(GroundOptionRecord)
        self.fail_optn_dict: Dict[str, GroundOptionRecord] =\
            defaultdict(GroundOptionRecord)


        for ite in range(1, max_invent_ite + 1):
            logging.info(f"===Starting iteration {ite}...")
            # Reset at every iteration
            if CFG.reset_optn_state_dict_at_every_ite:
                self.succ_optn_dict = defaultdict(GroundOptionRecord)
                self.fail_optn_dict = defaultdict(GroundOptionRecord)
            # This will update self.task_to_tasjs
            self._process_interaction_result(env,
                                             results,
                                             tasks,
                                             ite,
                                             use_only_first_solution=False)
            #### End of data collection

            # Invent when no improvement in solve rate
            self._prev_learned_predicates: Set[Predicate] =\
                self._learned_predicates

            if ite == 1 or no_improvement:  #or add_new_proposal_at_every_ite:
                logging.info("Accquiring new predicates...")
                # Invent only when there is no improvement in solve rate
                # Or when add_new_proposal_at_every_ite is True
                #   Create prompt to inspect the execution
                # self.base_candidates: candidates to be unioned with the init set
                if CFG.vlm_predicator_oracle_base_grammar:
                    if CFG.neu_sym_predicate:
                        # If using the oracle predicates
                        # With NSP, we only want the GT NSPs besides the initial
                        # predicates
                        # Want to remove the predicates of the same name
                        # Currently assume this is correct
                        new_proposals = env.ns_predicates -\
                            self._initial_predicates
                    else:
                        new_proposals = env.oracle_proposed_predicates -\
                                            self._initial_predicates
                else:
                    # Use the results to prompt the llm
                    prompt = self._create_prompt(env, ite, 10, 2, 2, 
                                            categories_to_show=['tp', 'fp'])
                    breakpoint()
                    response_file =\
                        f'./prompts/invent_{self.env_name}_{ite}.response'
                    # f'./prompts/invent_{self.env_name}_{ite}.response'
                    new_proposals = self._get_llm_predictions(
                        prompt, response_file, manual_prompt,
                        regenerate_response)
                logging.info(
                    f"Done: created {len(new_proposals)} candidates:" +
                    f"{new_proposals}")
                propose_ite += 1

            # [Start moving out]
            # Apply the candidate predicates to the data.
            all_trajs = []            
            # needs to be improved
            if CFG.use_partial_plans_prefix_as_demo and num_solved == 0:
                logging.info(f"Learning from only failed plans")
                iterator = self.task_to_partial_trajs.items()
            else:
                logging.info(f"Learning from only full solution trajectories.")
                iterator = self.task_to_trajs.items()
            for _, trajs in iterator:
                for traj in trajs:
                    all_trajs.append(traj)
            logging.info(f"Learning from {len(all_trajs)} trajectories.")

            if CFG.llm_predicator_oracle_learned:
                self._learned_predicates = new_proposals
            else:
                # Select a subset candidates by score optimization
                self.base_candidates |= new_proposals

                ### Predicate Search
                # Optionally add grammar to the candidates
                all_candidates: Dict[Predicate, float] = {}
                if CFG.llm_predicator_use_grammar:
                    grammar = _create_grammar(dataset=Dataset(all_trajs),
                                              given_predicates=\
                                self.base_candidates|self._initial_predicates)
                else:
                    grammar = _GivenPredicateGrammar(
                        self.base_candidates | self._initial_predicates)
                all_candidates.update(grammar.generate(
                        max_num=CFG.grammar_search_max_predicates))
                # logging.debug(f"all candidates {pformat(all_candidates)}")
                # breakpoint()
                # Add a atomic states for succ_optn_dict and fail_optn_dict
                logging.info("[Start] Applying predicates to data...")
                if num_solved == 0:
                    score_func_name = "operator_classification_error"
                else:
                    score_func_name = "expected_nodes_created"
                    # score_function = CFG.grammar_search_score_function

                if score_func_name == "operator_classification_error":
                    # Abstract here because it's used in the score function
                    # Evaluate the newly proposed predicates; the values for
                    # previous proposed should have been cached by the previous
                    # abstract calls.
                    num_states = len(set(state for optn_dict in
                                    [self.succ_optn_dict, self.fail_optn_dict]
                                    for g_optn in optn_dict.keys() for state in
                                    optn_dict[g_optn].states))
                    logging.debug(f"There are {num_states} distinct states.")
                    for optn_dict in [self.succ_optn_dict, self.fail_optn_dict]:
                        for g_optn in optn_dict.keys():
                            atom_states = []
                            for state in optn_dict[g_optn].states:
                                atom_states.append(
                                    utils.abstract(state, set(all_candidates)))
                            optn_dict[g_optn].abstract_states = atom_states

                # This step should only make VLM calls on the end state
                # becuaes it would have labled all the other success states
                # from the previous step.
                atom_dataset: List[GroundAtomTrajectory] =\
                    utils.create_ground_atom_dataset(all_trajs, 
                                                     set(all_candidates))
                logging.info("[Finish] Applying predicates to data....")

                logging.info("[Start] Predicate search from " +
                             f"{self._initial_predicates}...")
                score_function = create_score_function(
                    score_func_name,
                    self._initial_predicates, atom_dataset, all_candidates,
                    self._train_tasks, self.succ_optn_dict,
                    self.fail_optn_dict)
                start_time = time.perf_counter()
                self._learned_predicates = \
                    self._select_predicates_by_score_hillclimbing(
                        all_candidates,
                        score_function,
                        initial_predicates = self._initial_predicates)
                logging.info("[Finish] Predicate search.")
                logging.info(
                    f"Total search time {time.perf_counter()-start_time:.2f} "
                    "seconds")
            # [End moving out]

            breakpoint()
            # Finally, learn NSRTs via superclass, using all the kept predicates.
            self._learn_nsrts(all_trajs,
                              online_learning_cycle=None,
                              annotations=None,
                              fail_optn_dict=self.fail_optn_dict)
            breakpoint()

            # Add init_nsrts whose option isn't in the current nsrts to
            # Is this sufficient? Or should I add back all the operators?
            # Because if it only learned move to one then can it use it to do
            # move to two?
            cur_options = [nsrt.option for nsrt in self._nsrts]
            # When starting to use complete trajectory to learn operators,
            # add the previous nsrts whose option is no longer in the current
            # nsrt, e.g. (twist when the tasks with non-twist jug is solved).
            for p_nsrt in self._previous_nsrts:
                if not p_nsrt.option in cur_options:
                    logging.debug(f"Adding back nsrt: {pformat(p_nsrt)}")
                    self._nsrts.add(p_nsrt)
            # Add the initial nsrts back to the nsrts
            # for p_nsrts in self._init_nsrts:
            #     if not p_nsrts.option in cur_options:
            #         self._nsrts.add(p_nsrts)
            # self._nsrts |= self._reduced_nsrts
            print("All NSRTS after learning", pformat(self._nsrts))

            # Collect Data again
            # Set up load/save filename for interaction dataset
            results = self.collect_dataset(ite, env, tasks)
            num_solved = sum([r.succeeded for r in results])
            num_failed_plans = sum([len(r.info['partial_refinements']) for r in 
                                results])
            solve_rate = num_solved / num_tasks
            no_improvement = not (solve_rate > prev_solve_rate)

            # Print the new classification results with the new operators
            tp, tn, fp, fn, _ = utils.count_classification_result_for_ops(
                self._nsrts,
                self.succ_optn_dict,
                self.fail_optn_dict,
                return_str=False,
                initial_ite=False,
                print_cm=True)
            clf_acc = (tp + tn) / (tp + tn + fp + fn)
            logging.info(f"\n===ite {ite} finished. "
                         f"Solve rate {num_solved / num_tasks} "
                         f"Prev solve rate {prev_solve_rate} "
                         f"Num skeletons failed {num_failed_plans} "
                         f"Clf accuracy: {clf_acc:.2f}\n")
            breakpoint()

            # Save the best model
            if solve_rate > best_solve_rate:
                best_solve_rate = solve_rate
                clf_acc_at_best_solve_rate = clf_acc
                best_ite = ite
                best_nsrt = self._nsrts
                best_preds = self._learned_predicates
            prev_solve_rate = solve_rate
            prev_num_failed_plans = num_failed_plans
            self._previous_nsrts = deepcopy(self._nsrts)
            if solve_rate == 1 and num_failed_plans == 0:
                break
            time.sleep(5)

        logging.info("Invention finished.")
        logging.info(
            f"\nBest solve rate {best_solve_rate} first achieved at ite "
            f"{best_ite}; clf accuracy {clf_acc_at_best_solve_rate}")
        logging.info(f"Predicates learned {best_preds}")
        logging.info(f"NSRTs learned {pformat(best_nsrt)}")
        breakpoint()
        self._nsrts = best_nsrt
        self._learned_predicates = best_preds
        return

    def _process_interaction_result(self, env: BaseEnv,
                                    results: List[PlanningResult],
                                    tasks: List[Task], ite: int,
                                    use_only_first_solution: bool) -> Dataset:
        """Process the data obtained in solving the tasks into ground truth
        positive and negative states for the ground options.

        Deprecated:
        When add_intermediate_details == True, detailed interaction
        trajectories are added to the return string
        """

        # num_solved = sum([isinstance(r, tuple) for r in results])
        # num_attempted = len(results)
        # logging.info(f"The agent solved {num_solved} out of " +
        #                 f"{num_attempted} tasks.\n")
        logging.info("===Processing the interaction results...\n")
        num_tasks = len(tasks)
        # suc_state_trajs = []
        if ite == 1:
            self.solve_log = [False] * num_tasks

        # Add a progress bar
        for i, _ in tqdm(enumerate(tasks),
                         total=num_tasks,
                         desc="Processing Interaction results"):
            result = results[i]

            if result.succeeded:
                # Found a successful plan
                # logging.info(f"Task {i}: planning succeeded.")
                nsrt_plan = result.info['nsrt_plan']
                option_plan = result.info['option_plan'].copy()
                logging.debug(
                    f"[ite {ite} task {i}] Processing succeeded " +
                    f"plan {[op.name + str(op.objects) for op in option_plan]}"
                )

                # Check before processing for some efficiency gain
                if use_only_first_solution:
                    if self.solve_log[i]:
                        # If the task has previously been solved
                        continue  # continue to logging the next task
                    else:
                        # Otherwise, update the log
                        self.solve_log[i] = True

                # Check if the current plan is novel; only log the
                # plan for predicate/operator learning if it's novel
                if i in self.task_to_plans:
                    # Has been solved before
                    if any(
                            are_equal_by_obj(option_plan, plan)
                            for plan in self.task_to_plans[i]):
                        continue
                    else:
                        logging.warning(f"Found a novel plan for task {i}")
                        self.task_to_plans[i].append(option_plan.copy())
                else:
                    # Has not been solved before.
                    self.task_to_plans[i].append(option_plan.copy())

                # If the code has got here: it's the 1st time solving task i OR
                #   we've found a new plan.
                states, actions = self._execute_succ_plan_and_track_state(
                    env.reset(train_or_test='train', task_idx=i), env,
                    nsrt_plan, option_plan)

                # Add it to the trajectory dictionary
                self.task_to_trajs[i].append(
                    LowLevelTrajectory(states,
                                       actions,
                                       _is_demo=True,
                                       _train_task_idx=i))

            # The failed refinements (negative samples)
            # This result is either a Result tuple or an exception
            for p_idx, p_ref in enumerate(result.info['partial_refinements']):
                # longest option refinement
                option_plan = p_ref[1].copy()
                nsrt_plan = p_ref[0]
                logging.debug(f"[ite {ite} task {i}] Processing failed plan "\
                f"{p_idx} of len {len(option_plan)}: "\
                f"{[op.name + str(op.objects) for op in option_plan]}")
                failed_opt_idx = len(option_plan) - 1

                # As above, check if the p-plan is novel
                if i in self.task_to_plans:
                    # Has been solved before
                    if any(
                            are_equal_by_obj(option_plan, plan)
                            for plan in self.task_to_plans[i]):
                        continue
                    else:
                        # logging.warning(f"Found a novel pplan for task {i}")
                        self.task_to_plans[i].append(option_plan.copy())
                else:
                    # Has not been solved before.
                    self.task_to_plans[i].append(option_plan.copy())

                state = env.reset(train_or_test='train', task_idx=i)
                # Successful part
                if failed_opt_idx > 0:
                    states, actions = self._execute_succ_plan_and_track_state(
                        state,
                        env,
                        nsrt_plan,
                        option_plan[:-1],
                        ite=ite,
                        task=i,
                        p_idx=p_idx,)
                    state = states[-1]
                    # Take the prefix of the pplan and use it to learn
                    #   operators.
                    if len(states) <= len(actions):
                        logging.warning("states is not 1 more than actions")
                        breakpoint()
                        # hacky fix, should really figure out why the option
                        # plan
                        # actions = actions[:len(states)-1]
                    if CFG.use_partial_plans_prefix_as_demo:
                        self.task_to_partial_trajs[i].append(
                            LowLevelTrajectory(states,
                                               actions,
                                               _is_demo=True,
                                               _train_task_idx=i))

                # Failed part
                ppp = [o.name for o in option_plan[:-1]]
                _, _ = self._execute_succ_plan_and_track_state(
                    state,
                    env,
                    nsrt_plan[failed_opt_idx:],
                    option_plan[-1:],
                    failed_opt=True,
                    partial_plan_prefix=ppp)
        logging.debug("Collected Positive states for "
                      f"{list(self.succ_optn_dict.keys())}")
        logging.debug("Collected Negative states for "
                      f"{list(self.fail_optn_dict.keys())}")

    def _execute_succ_plan_and_track_state(
            self,
            init_state: State,
            env: BaseEnv,
            nsrt_plan: List,
            option_plan: List,
            failed_opt: bool = False,
            ite: Optional[int] = None,
            task: Optional[int] = None,
            p_idx: Optional[int] = None,
            partial_plan_prefix: Optional[List[str]] = None,
            ) -> Tuple[List[State], List[Action]]:
        """Similar to _execute_plan_and_track_state but only run in successful
        policy because we only need the initial state for the failed option.

        Return:
        -------
        states: List[State]
            The states before executing each option in the option plan and last
            state before returning.
        actions: List[Action]
            The first action from each option in the option plan. The length of
            this will be 1 less than the number of states.
        partial_plan_prefix: Optional[str] = None,
            For failed options, the list of option successfully executed before
            them.
        """
        state = init_state

        def policy(_: State) -> Action:
            raise OptionExecutionFailure("placeholder policy")

        steps = 0
        nsrt_counter = 0
        env_step_counter = 0
        states, actions = [], []
        first_option_action = False
        if failed_opt:
            state_hash = state.__hash__()
            if state_hash in self.state_cache:
                option_start_state = self.state_cache[state_hash].copy()
            else:
                option_start_state = env.get_observation(
                    render=CFG.vlm_predicator_render_option_state)
                if CFG.neu_sym_predicate:
                    option_start_state.add_bbox_features()
                option_start_state.option_history = partial_plan_prefix
                self.state_cache[state_hash] = option_start_state.copy()
            g_nsrt = nsrt_plan[0]
            gop_str = g_nsrt.ground_option_str(
                use_object_id=CFG.neu_sym_predicate)
            self.fail_optn_dict[gop_str].append_state(
                option_start_state,
                utils.abstract(option_start_state,
                               self._get_current_predicates()),
                g_nsrt.option_objs, g_nsrt.parent.option_vars, g_nsrt.option)
        else:
            for steps in range(CFG.horizon):
                try:
                    act = policy(state)
                except OptionExecutionFailure as e:
                    # When the one-option policy reaches terminal state
                    # we're cetain the plan is successfully terminated
                    # because this is a successful plan.
                    if str(e) == "placeholder policy" or\
                    (str(e) == "Option plan exhausted!") or\
                    (str(e) == "Encountered repeated state."):
                        try:
                            option = option_plan.pop(0)
                        except IndexError:
                            # When the option_plan is exhausted
                            # Rendering the final state for success traj
                            state_hash = state.__hash__()
                            if state_hash in self.state_cache:
                                option_start_state = self.state_cache[
                                    state_hash].copy()
                            else:
                                option_start_state = env.get_observation(
                                    render=CFG.
                                    vlm_predicator_render_option_state)
                                if CFG.neu_sym_predicate:
                                    option_start_state.add_bbox_features()
                                # add plan prefix
                                option_start_state.option_history = [
                                    n.option.name for n in 
                                    nsrt_plan[:nsrt_counter]]
                                self.state_cache[
                                    state_hash] = option_start_state.copy()
                            # For debugging incomplete options
                            states.append(option_start_state)
                            break
                        else:
                            # raise_error_on_repeated_state is set to true in simple
                            # environments, but causes the option to not finish in
                            # the pybullet environment, hence are disabled in
                            # testing neu-sym-predicates.
                            # We are okay with this because the failure options
                            # have been handled above.
                            policy = utils.option_plan_to_policy(
                                [option], raise_error_on_repeated_state=False)
                            # [option], raise_error_on_repeated_state=True)
                            state_hash = state.__hash__()
                            if state_hash in self.state_cache:
                                option_start_state = self.state_cache[
                                    state_hash].copy()
                            else:
                                option_start_state = env.get_observation(
                                    render=CFG.
                                    vlm_predicator_render_option_state)
                                # add plan prefix
                                option_start_state.option_history = [
                                    n.option.name for n in 
                                    nsrt_plan[:nsrt_counter]]
                                if CFG.neu_sym_predicate:
                                    option_start_state.add_bbox_features()
                                self.state_cache[
                                    state_hash] = option_start_state.copy()
                            # option_start_state = env.get_observation(
                            #     render=CFG.vlm_predicator_render_option_state)
                            # logging.info("Start new option at step "+
                            #                 f"{env_step_counter}")
                            g_nsrt = nsrt_plan[nsrt_counter]
                            gop_str = g_nsrt.ground_option_str(
                                use_object_id=CFG.neu_sym_predicate)
                            states.append(option_start_state)
                            first_option_action = True
                            self.succ_optn_dict[gop_str].append_state(
                                option_start_state,
                                utils.abstract(option_start_state,
                                               self._get_current_predicates()),
                                g_nsrt.option_objs, g_nsrt.parent.option_vars,
                                g_nsrt.option)
                            nsrt_counter += 1
                    else:
                        break
                else:
                    state = env.step(act)
                    if first_option_action:
                        actions.append(act)
                        first_option_action = False
                    env_step_counter += 1
            if steps == CFG.horizon - 1:
                logging.warning("Processing stopped as steps reach the max.")

        # logging.debug(f"Finish executing after {steps} steps in the loop.")
        return states, actions

    def collect_dataset(self, ite: int, env: BaseEnv,
                        tasks: List[Task]) -> List[PlanningResult]:

        ds_fname = utils.llm_pred_dataset_save_name(ite)
        if CFG.load_llm_pred_invent_dataset and os.path.exists(ds_fname):
            with open(ds_fname, 'rb') as f:
                results = dill.load(f)
            logging.info(f"Loaded dataset from {ds_fname}\n")
        else:
            # Ask it to solve the tasks
            results = self._solve_tasks(env, tasks, ite)
            if CFG.save_llm_pred_invent_dataset:
                os.makedirs(os.path.dirname(ds_fname), exist_ok=True)
                with open(ds_fname, 'wb') as f:
                    dill.dump(results, f)
                logging.info(f"Saved dataset to {ds_fname}\n")
        return results

    def _get_llm_predictions(
            self,
            prompt: str,
            response_file: str,
            manual_prompt: bool = False,
            regenerate_response: bool = False) -> Set[Predicate]:
        if not os.path.exists(response_file) or regenerate_response:
            if manual_prompt:
                # create a empty file for pasting chatGPT response
                with open(response_file, 'w') as file:
                    pass
                logging.info(f"## Please paste the response from the LLM " +
                             f"to {response_file}")
                input("Press Enter when you have pasted the " + "response.")
            else:
                raise NotImplementedError("Automatic prompt generation not "+\
                                          "updated")
                self._vlm.sample_completions(prompt,
                                             temperature=CFG.llm_temperature,
                                             seed=CFG.seed,
                                             save_file=response_file)[0]
        new_candidates = self._parse_predicate_predictions(response_file)
        return new_candidates

    def _select_predicates_by_score_hillclimbing(
            self,
            candidates: Dict[Predicate, float],
            score_function: _PredicateSearchScoreFunction,
            initial_predicates: Set[Predicate] = set(),
            atom_dataset: List[GroundAtomTrajectory] = [],
            train_tasks: List[Task] = []) -> Set[Predicate]:
        """Perform a greedy search over predicate sets."""

        # There are no goal states for this search; run until exhausted.
        def _check_goal(s: FrozenSet[Predicate]) -> bool:
            del s  # unused
            return False

        # Successively consider larger predicate sets.
        def _get_successors(
            s: FrozenSet[Predicate]
        ) -> Iterator[Tuple[None, FrozenSet[Predicate], float]]:
            for predicate in sorted(set(candidates) - s):  # determinism
                # Actions not needed. Frozensets for hashing. The cost of
                # 1.0 is irrelevant because we're doing GBFS / hill
                # climbing and not A* (because we don't care about the
                # path).
                # pre_str = [p.name for p in (s | {predicate})]
                # if sorted(pre_str) == \
                #     sorted(["Clear", "Holding", "On", "OnTable"]):
                #     breakpoint()
                yield (None, frozenset(s | {predicate}), 1.0)
            # for predicate in sorted(s):  # determinism
            #     # Actions not needed. Frozensets for hashing. The cost of
            #     # 1.0 is irrelevant because we're doing GBFS / hill
            #     # climbing and not A* (because we don't care about the
            #     # path).
            #     yield (None, frozenset(set(s) - {predicate}), 1.0)

        # Start the search with no candidates.
        init: FrozenSet[Predicate] = frozenset(initial_predicates)
        # init: FrozenSet[Predicate] = frozenset(candidates.keys())

        # calculate the number of total combinations of all sizes
        num_combinations = 2**len(set(candidates))

        # Greedy local hill climbing search.
        if CFG.grammar_search_search_algorithm == "hill_climbing":
            path, _, heuristics = utils.run_hill_climbing(
                init,
                _check_goal,
                _get_successors,
                score_function.evaluate,
                enforced_depth=CFG.grammar_search_hill_climbing_depth,
                parallelize=CFG.grammar_search_parallelize_hill_climbing)
            logging.info("\nHill climbing summary:")
            for i in range(1, len(path)):
                new_additions = path[i] - path[i - 1]
                assert len(new_additions) == 1
                new_addition = next(iter(new_additions))
                h = heuristics[i]
                prev_h = heuristics[i - 1]
                logging.info(f"\tOn step {i}, added {new_addition}, with "
                             f"heuristic {h:.3f} (an improvement of "
                             f"{prev_h - h:.3f} over the previous step)")
        elif CFG.grammar_search_search_algorithm == "gbfs":
            path, _ = utils.run_gbfs(
                init,
                _check_goal,
                _get_successors,
                score_function.evaluate,
                max_evals=CFG.grammar_search_gbfs_num_evals,
                full_search_tree_size=num_combinations,
            )
        else:
            raise NotImplementedError(
                "Unrecognized grammar_search_search_algorithm: "
                f"{CFG.grammar_search_search_algorithm}.")
        kept_predicates = path[-1]
        # The total number of predicate sets evaluated is just the
        # ((number of candidates selected) + 1) * total number of candidates.
        # However, since 'path' always has length one more than the
        # number of selected candidates (since it evaluates the empty
        # predicate set first), we can just compute it as below.
        # assert self._metrics.get("total_num_predicate_evaluations") is None
        self._metrics["total_num_predicate_evaluations"] = len(path) * len(
            candidates)

        # # Filter out predicates that don't appear in some operator
        # # preconditions.
        # logging.info("\nFiltering out predicates that don't appear in "
        #              "preconditions...")
        # preds = kept_predicates | initial_predicates
        # pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset, preds)
        # segmented_trajs = [
        #     segment_trajectory(ll_traj, set(preds), atom_seq=atom_seq)
        #     for (ll_traj, atom_seq) in pruned_atom_data
        # ]
        # low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
        # preds_in_preconds = set()
        # for pnad in learn_strips_operators(low_level_trajs,
        #                                    train_tasks,
        #                                    set(kept_predicates
        #                                        | initial_predicates),
        #                                    segmented_trajs,
        #                                    verify_harmlessness=False,
        #                                    annotations=None,
        #                                    verbose=False):
        #     for atom in pnad.op.preconditions:
        #         preds_in_preconds.add(atom.predicate)
        # kept_predicates &= preds_in_preconds

        logging.info(f"\nSelected {len(kept_predicates)} predicates out of "
                     f"{len(candidates)} candidates:")
        for pred in kept_predicates:
            logging.info(f"\t{pred}")
        score_function.evaluate(kept_predicates)  # log useful numbers
        logging.info(f"\nSelected {len(kept_predicates)} predicates out of "
                     f"{len(candidates)} candidates:")
        for pred in kept_predicates:
            logging.info(f"\t{pred}")

        return set(kept_predicates)

    def _create_prompt(
        self,
        env: BaseEnv,
        ite: int,
        max_num_options: int = 10,  # Number of options to show
        max_num_groundings: int = 2,  # Number of ground options per option.3
        max_num_examples: int = 2,  # Number of examples per ground option.
        categories_to_show: List[str] = ['tp', 'fp'],
        seperate_prompt_per_option: bool = False,
    ) -> str:
        """Compose a prompt for VLM for predicate invention."""
        # Read the shared template
        with open(f'./prompts/invent_0_simple.outline', 'r') as file:
            template = file.read()
        # Get the different parts of the prompt
        if CFG.neu_sym_predicate:
            instr_fn = "raw"
        else:
            instr_fn = "oo"
        with open(f'./prompts/invent_0_{instr_fn}_state_simple.outline',
                  'r') as f:
            instruction = f.read()
        template += instruction

        ##### Meta Environment
        # Structure classes

        # with open('./prompts/class_definitions.py', 'r') as f:
        #     struct_str = f.read()
        if CFG.neu_sym_predicate:
            with open('./prompts/api_raw_state.py', 'r') as f:
                state_str = f.read()
            with open('./prompts/api_nesy_predicate.py', 'r') as f:
                pred_str = f.read()
        else:
            with open('./prompts/api_oo_state.py', 'r') as f:
                state_str = f.read()
            with open('./prompts/api_sym_predicate.py', 'r') as f:
                pred_str = f.read()

        template = template.replace(
            '[STRUCT_DEFINITION]',
            add_python_quote(state_str + '\n\n' + pred_str))

        ##### Environment
        self.env_source_code = getsource(env.__class__)
        # Type Instances
        if CFG.neu_sym_predicate:
            # New version: just read from a file
            with open(f"./prompts/types_{self.env_name}.py", 'r') as f:
                type_instan_str = f.read()
        else:
            # Old version: extract directly from the source code
            type_instan_str = self._env_type_str(self.env_source_code)
        type_instan_str = add_python_quote(type_instan_str)
        template = template.replace("[TYPES_IN_ENV]", type_instan_str)

        # Predicates
        # If NSP, provide the GT goal NSPs, although they are never used.
        pred_str_lst = []
        pred_str_lst.append(self._init_predicate_str(env,
                                                     self.env_source_code))
        # if ite > 1:
        #     pred_str_lst.append("The previously invented predicates are:")
        #     pred_str_lst.append(self._invented_predicate_str(ite))
        pred_str = '\n'.join(pred_str_lst)
        template = template.replace("[PREDICATES_IN_ENV]", pred_str)

        # Options
        '''Template: The set of options the robot has are:
        [OPTIONS_IN_ENV]'''
        options_str_set = set()
        for nsrt in self._nsrts:
            options_str_set.add(nsrt.option_str_annotated())
        options_str = '\n'.join(list(options_str_set))
        template = template.replace("[OPTIONS_IN_ENV]", options_str)

        # NSRTS
        nsrt_str = []
        for nsrt in self._nsrts:
            nsrt_str.append(str(nsrt).replace("NSRT-", "Operator-"))
        template = template.replace("[NSRTS_IN_ENV]", '\n'.join(nsrt_str))

        _, _, _, _, summary_str = utils.count_classification_result_for_ops(
            self._nsrts,
            self.succ_optn_dict,
            self.fail_optn_dict,
            return_str=True,
            initial_ite=(ite == 0),
            print_cm=True,
            max_num_options=max_num_options,
            max_num_groundings=max_num_groundings,
            max_num_examples=max_num_examples,
            categories_to_show=categories_to_show,
            )
        template = template.replace("[OPERATOR_PERFORMACE]", summary_str)

        # Save the text prompt
        with open(f'./prompts/invent_{self.env_name}_{ite}.prompt', 'w') as f:
            f.write(template)
        prompt = template

        return prompt

    def _parse_predicate_predictions(self,
                                     prediction_file: str) -> Set[Predicate]:
        # Read the prediction file
        with open(prediction_file, 'r') as file:
            response = file.read()

        # Regular expression to match Python code blocks
        pattern = re.compile(r'```python(.*?)```', re.DOTALL)
        python_blocks = []
        # Find all Python code blocks in the text
        for match in pattern.finditer(response):
            # Extract the Python code block and add it to the list
            python_blocks.append(match.group(1).strip())

        candidates = set()
        context: Dict = {}
        # add the classifiers of the existing predicates to the context for
        # potential reuse
        for p in self._initial_predicates:
            context[f"_{p.name}_NSP_holds"] = p._classifier

        type_init_str = self._env_type_str(self.env_source_code)
        constants_str = self._constants_str(self.env_source_code)
        for code_str in python_blocks:
            # Extract name from code block
            match = re.search(r'(\w+)\s*=\s*(NS)?Predicate', code_str)
            if match is None:
                raise ValueError("No predicate name found in the code block")
            pred_name = match.group(1)
            logging.info(f"Found definition for predicate {pred_name}")

            # # Type check the code
            # passed = False
            # while not passed:
            #     result, passed = self.type_check_proposed_predicates(pred_name,
            #                                                          code_str)
            #     if not passed:
            #         # Ask the LLM or the User to fix the code
            #         pass
            #     else:
            #         break

            # Instantiate the predicate
            exec(
                '\n'.join([import_str, type_init_str, constants_str,
                           code_str]), context)
            candidates.add(context[pred_name])

        return candidates

    def type_check_proposed_predicates(self, predicate_name: str,
                                       code_block: str) -> Tuple[str, bool]:
        # Write the definition to a python file
        predicate_fname = f'./prompts/oi1_predicate_{predicate_name}.py'
        with open(predicate_fname, 'w') as f:
            f.write(import_str + '\n' + code_block)

        # Type check
        logging.info(f"Start type checking the predicate " +
                     f"{predicate_name}...")
        result = subprocess.run([
            "mypy", "--strict-equality", "--disallow-untyped-calls",
            "--warn-unreachable", "--disallow-incomplete-defs",
            "--show-error-codes", "--show-column-numbers",
            "--show-error-context", predicate_fname
        ],
                                capture_output=True,
                                text=True)
        stdout = result.stdout
        passed = result.returncode == 0
        return stdout, passed

    def _env_type_str(self, source_code: str) -> str:
        type_pattern = r"(    # Types.*?)(?=\n\s*\n|$)"
        type_block = re.search(type_pattern, source_code, re.DOTALL)
        if type_block is not None:
            type_init_str = type_block.group()
            type_init_str = textwrap.dedent(type_init_str)
            # type_init_str = add_python_quote(type_init_str)
            return type_init_str
        else:
            raise Exception("No type definitions found in the environment.")

    def _constants_str(self, source_code: str) -> str:
        # Some constants, if any, defined in the environment are
        constants_str = ''
        pattern = r"(    # Constants present in goal predicates.*?)(?=\n\s*\n|$)"
        match = re.search(pattern, source_code, re.DOTALL)
        if match:
            constants_str = match.group(1)
            constants_str = textwrap.dedent(constants_str)
        return constants_str

    def _init_predicate_str(self, env: BaseEnv, source_code: str) -> str:
        """Extract the initial predicates from the environment source code If
        NSP, provide the GT goal NSPs, although they are never used."""
        init_pred_str = []
        init_pred_str.append(
            str({p.pretty_str_with_types()
                #  for p in self._initial_predicates}) + "\n")
                 for p in self.base_candidates}) + "\n")

        # Print the variable definitions
        constants_str = self._constants_str(source_code)
        if constants_str:
            init_pred_str.append(
                "The environment defines the following constants that can be "+\
                "used in defining predicates:")
            init_pred_str.append(add_python_quote(constants_str))

        # Get the entire predicate instantiation code block.
        predicate_pattern = r"(# Predicates.*?)(?=\n\s*\n|$)"
        predicate_block = re.search(predicate_pattern, source_code, re.DOTALL)
        if predicate_block is not None:
            pred_instantiation_str = predicate_block.group()

            if CFG.neu_sym_predicate:
                init_pred = [
                    p for p in env.ns_predicates
                    if p in self._initial_predicates
                ]
            else:
                init_pred = self._initial_predicates

            for p in init_pred:

                p_name = p.name
                # Get the instatiation code for p from the code block
                if CFG.neu_sym_predicate:
                    p_instan_pattern = r"(self\._" + re.escape(p_name) +\
                                    r"_NSP = NSPredicate\(.*?\n.*?\))"
                else:
                    p_instan_pattern = r"(self\._" + re.escape(p_name) +\
                                    r" = Predicate\(.*?\n.*?\))"
                block = re.search(p_instan_pattern, pred_instantiation_str,
                                  re.DOTALL)
                if block is not None:
                    p_instan_str = block.group()
                    pred_str = "Predicate " + p.pretty_str()[1] +\
                                " is defined by:\n" +\
                                add_python_quote(p.classifier_str() +\
                                p_instan_str)
                    init_pred_str.append(pred_str.replace("self.", ""))

        return '\n'.join(init_pred_str)

    def _invented_predicate_str(self, ite: int) -> str:
        """Get the predicate definitions from the previous response file."""
        new_predicate_str = []
        new_predicate_str.append(str(self._learned_predicates) + '\n')
        prediction_file = f'./prompts/invent_{self.env_name}_{ite-1}'+\
            ".response"
        with open(prediction_file, 'r') as file:
            response = file.read()

        # Regular expression to match Python code blocks
        code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
        for match in code_pattern.finditer(response):
            python_block = match.group(1).strip()
            pred_match = re.search(r'name\s*(:\s*str)?\s*= "([^"]*)"',
                                   python_block)
            if pred_match is not None:
                pred_name = pred_match.group(2)
                pred = next(
                    (p
                     for p in self._learned_predicates if p.name == pred_name),
                    None)
                if pred:
                    new_predicate_str.append("Predicate " +
                                             pred.pretty_str()[1] +
                                             " is defined by\n" +
                                             add_python_quote(python_block))
        has_not_or_forall = [
            p.name.startswith("NOT") or p.name.startswith("Forall")
            for p in self._learned_predicates
        ]
        if has_not_or_forall:
            new_predicate_str.append(
                "Predicates with names starting with " +
                "'NOT' or 'Forall' are defined by taking the negation or adding"
                + "universal quantifiers over other existing predicates.")
        return '\n'.join(new_predicate_str)