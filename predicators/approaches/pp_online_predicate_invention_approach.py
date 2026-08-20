"""pp_online_predicate_invention_approach module."""
import logging
import os
import re
import time
import traceback
from collections import defaultdict
from pprint import pformat
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Sequence, \
    Set, Tuple

import dill as pkl
import PIL
from gym.spaces import Box
from PIL import ImageDraw, ImageFont

from predicators import utils
from predicators.approaches.grammar_search_invention_approach import \
    _create_grammar, _GivenPredicateGrammar
from predicators.approaches.pp_online_process_learning_approach import \
    OnlineProcessLearningAndPlanningApproach
from predicators.approaches.pp_predicate_invention_approach import \
    PredicateInventionProcessPlanningApproach
from predicators.envs import create_new_env
from predicators.nsrt_learning.process_learning_main import \
    filter_explained_segment
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.option_model import _OptionModelBase
from predicators.planning_with_processes import process_task_plan_grounding
from predicators.predicate_search_score_functions import \
    _ExpectedNodesScoreFunction
from predicators.settings import CFG
from predicators.structs import CausalProcess, Dataset, DerivedPredicate, \
    EndogenousProcess, ExogenousProcess, GroundAtomTrajectory, Image, \
    InteractionResult, LowLevelTrajectory, ParameterizedOption, Predicate, \
    Segment, State, Task, Type, _GroundExogenousProcess


class OnlinePredicateInventionProcessPlanningApproach(
        PredicateInventionProcessPlanningApproach,
        OnlineProcessLearningAndPlanningApproach):
    """A bilevel planning approach that invent predicates."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1,
                 bilevel_plan_without_sim: Optional[bool] = None,
                 option_model: Optional[_OptionModelBase] = None):
        # just used for oracle predicate proposal or learned predicate
        self._oracle_predicates = create_new_env(
            CFG.env, use_gui=False).target_predicates
        self._candidate_predicates: Set[Predicate] = set()
        self._llm = utils.create_llm_by_name(CFG.llm_model_name)
        self._vlm = utils.create_vlm_by_name(
            CFG.llm_model_name)  # type: ignore[assignment]
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         task_planning_heuristic,
                         max_skeletons_optimized,
                         bilevel_plan_without_sim,
                         option_model=option_model)

    @classmethod
    def get_name(cls) -> str:
        return "online_predicate_invention_and_process_planning"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Just store the dataset, don't learn from it yet.
        self._offline_dataset = dataset
        # proposed_predicates = self._get_predicate_proposals(
        #     "transition_modelling",
        #     self._offline_dataset.trajectories)
        self.save()

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        # --- Process the interaction results ---
        assert self._requests_train_task_idxs is not None, \
            "Missing request->task index mapping."
        for i, result in enumerate(results):
            task_idx = self._requests_train_task_idxs[i]
            traj = LowLevelTrajectory(result.states,
                                      result.actions,
                                      _train_task_idx=task_idx)
            self._online_dataset.append(traj)

        all_trajs = self._offline_dataset.trajectories + \
            self._online_dataset.trajectories

        # Future: change to only propose when stop improving?
        # Test to only generate proposals at cycle 0.
        if self._online_learning_cycle == 0:
            proposed_predicates = self._get_predicate_proposals(
                "subgoals", all_trajs)
        else:
            proposed_predicates = set()
        logging.info(f"Done: created {len(proposed_predicates)} predicates")

        # --- Select the predicates to keep ---
        self._select_predicates_and_learn_processes(
            _ite=self._online_learning_cycle,
            all_trajs=all_trajs,
            proposed_predicates=proposed_predicates,
            train_tasks=self._train_tasks)
        logging.debug(f"Learned predicates: "
                      f"{self._learned_predicates-self._initial_predicates}")

        if CFG.learn_process_parameters:
            self._learn_process_parameters(all_trajs)
        self.save(self._online_learning_cycle)

        self._online_learning_cycle += 1

    def save(self, online_learning_cycle: Optional[int] = None) -> None:
        """Save."""
        # Saving the learned processes, dataset, candidate predicates
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.PROCes", "wb") as f:
            save_dict = {
                "processes": self._processes,
                "learned_predicates": self._learned_predicates,
                "candidate_predicates": self._candidate_predicates,
                "offline_dataset": self._offline_dataset,
                "online_dataset": self._online_dataset,
                "online_learning_cycle": self._online_learning_cycle
            }
            pkl.dump(save_dict, f)
            logging.info(f"Saved approach to {save_path}_"
                         f"{online_learning_cycle}.PROCes")

    def load(self, online_learning_cycle: Optional[int] = None) -> None:
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}_{online_learning_cycle}.PROCes", "rb") as f:
            save_dict = pkl.load(f)
        # check save_dict has "processes", "candidate_predicate" values
        assert "processes" in save_dict, "Processes not found in save_dict"
        assert "candidate_predicates" in save_dict, \
            "Candidate predicates not found in save_dict"
        assert "offline_dataset" in save_dict, \
            "Offline dataset not found in save_dict"
        assert "online_dataset" in save_dict, \
            "Online dataset not found in save_dict"
        self._processes = save_dict["processes"]
        self._learned_predicates = save_dict["learned_predicates"]
        self._candidate_predicates = save_dict["candidate_predicates"]
        self._offline_dataset = save_dict["offline_dataset"]
        self._online_dataset = save_dict["online_dataset"]
        self._online_learning_cycle = save_dict["online_learning_cycle"] + 1
        logging.info("\n\nLoaded Processes:")
        for process in sorted(self._processes):
            logging.info(process)
        logging.info(
            f"Loaded {len(self._learned_predicates)} learned predicates")
        logging.info(f"{sorted(self._learned_predicates)}")
        logging.info(
            f"Loaded {len(self._processes)} processes, "
            f"{len(self._candidate_predicates)} candidate predicates, "
            f"{len(self._offline_dataset.trajectories)} offline trajectories, "
            f"{len(self._online_dataset.trajectories)} online trajectories\n")

        for proc in self._processes:
            if isinstance(proc, EndogenousProcess):
                proc.option.params_space.seed(CFG.seed)

    def _get_predicate_proposals(
            self, proposal_method: str,
            trajectories: List[LowLevelTrajectory]) -> Set[Predicate]:
        if CFG.vlm_predicator_oracle_base_predicates:
            base_candidates = self._oracle_predicates - self._initial_predicates
        else:
            base_candidates: Set[Predicate] = set()  # type: ignore[no-redef]

            # noisy_but_complete_proposal = True
            # if noisy_but_complete_proposal:
            #     base_candidates |= set(p for p in self._oracle_predicates
            #                            if p.name in [
            #                                #    "NoWaterSpilled",
            #                                "NoJugAtFaucetOrAtFaucetAndFilled"
            #                            ])

            for i in range(CFG.vlm_predicator_num_proposal_batches):
                base_candidates |= self._get_predicate_proposals_from_fm(
                    proposal_method, trajectories, i,
                    invent_derived_predicates=\
                        CFG.predicate_invent_invent_derived_predicates)
            # Future: filter semantically equivalent predicates
        return base_candidates

    def _get_predicate_proposals_from_fm(
            self, proposal_method: str, trajectories: List[LowLevelTrajectory],
            proposal_batch_id: int,
            invent_derived_predicates: bool) -> Set[Predicate]:
        """Get predicate proposals from the FM."""
        ###### Invent predicates in NL based on the dataset ######
        b_id = proposal_batch_id
        seed = CFG.seed * 100 + self._online_learning_cycle * 10 + b_id

        assert proposal_method in [
            "transition_modeling", "discrimination", "unconditional",
            "subgoals"
        ]

        # transition modelling (2 fm calls): spec -> implementation
        # discrimination (3 fm calls): nl -> spec -> implementation
        # unconditional: (3 calls): spec -> primitive impl -> concept impl
        if proposal_method in ["transition_modeling", "subgoals"]:
            # 1. Get template
            successful_trajectory = traj_is_successful(trajectories[0],
                                                       self._train_tasks)
            if successful_trajectory:
                if invent_derived_predicates:
                    prompt_template_f = f"prompts/invent_{proposal_method}"\
                        "_solved_derived.outline"
                else:
                    prompt_template_f =\
                        f"prompts/invent_{proposal_method}_solved.outline"
            else:
                prompt_template_f = f"prompts/invent_{proposal_method}_failed"\
                                    f".outline"
            with open(prompt_template_f, "r", encoding='utf-8') as f:
                prompt_template = f.read()

            # 2. Fill and save the template
            pred_str = _get_predicates_str(self._get_current_predicates())
            types = set(o.type for o in set(trajectories[0].states[0]))
            logging.info("Inventing predicates from only the offline dataset.")
            experience_str, state_str = _get_transition_str(
                self._offline_dataset.trajectories,  # +\
                # self._online_dataset.trajectories,
                self._train_tasks,
                self._get_current_predicates(),
                ite=self._online_learning_cycle,
                use_abstract_state_str=invent_derived_predicates,
            )
            prompt = prompt_template.format(
                PREDICATES_IN_ENV=pred_str,
                TYPES_IN_ENV=_get_types_str(types),
                EXPERIENCE_IN_ENV=experience_str,
                GOAL_PREDICATE=self._train_tasks[0].goal)
            with open(
                    f"{CFG.log_file}/ite{self._online_learning_cycle}_b{b_id}"
                    f"_s1.prompt",
                    "w",
                    encoding='utf-8') as f:
                f.write(prompt)

            # 3. Get spec proposals
            temperature = 0.2
            if CFG.rgb_observation:
                images = load_images_from_directory(
                    CFG.log_file +
                    f"ite{self._online_learning_cycle}_b{b_id}_obs/")
                vlm = self._vlm
                assert vlm is not None
                spec_response = vlm.sample_completions(prompt,
                                                       images,
                                                       temperature=temperature,
                                                       num_completions=1,
                                                       seed=seed)[0]
            else:
                spec_response = self._llm.sample_completions(
                    prompt,
                    imgs=None,
                    temperature=temperature,
                    num_completions=1,
                    seed=seed)[0]
            with open(
                    f"{CFG.log_file}/ite{self._online_learning_cycle}_b{b_id}"
                    f"_s1.response",
                    "w",
                    encoding='utf-8') as f:
                f.write(spec_response)
        elif proposal_method == "discrimination":
            # Method 1: Find each state, if it satisfies the
            #   condition of an exogenous process, check later
            #   that its effect did take place, save it if not.
            #   Then for each exogenous process, compare the
            #   above negative state with positive states where
            #   the effect took place (e.g. in the demo).
            # Maybe this will mirror the planner.
            # Remember to reset at the end

            # Step 1: Find the false positive examples
            exogenous_processes = list(self._get_current_exogenous_processes())
            false_positive_process_state = get_false_positive_states(
                self._online_dataset.trajectories,
                self._get_current_predicates(), exogenous_processes)

            # Step 2: Find the true positive examples
            # For each expected effect that did not take place,
            # find in the demo the initial state where it did
            # take place, and save it as a positive
            #  example.
            get_true_positive_process_states(
                self._get_current_predicates(), exogenous_processes,
                list(false_positive_process_state.keys()),
                self._offline_dataset.trajectories)

            # Step 3: Prompt VLM to invent predicates
            # Pending: prepare the prompt
            # Pending: implement the prompt and parse logic
        else:
            raise NotImplementedError

        ###### Implement the predicates in python ######
        # Create the implementation prompt
        if CFG.predicate_invent_neural_symbolic_predicates:
            raise NotImplementedError
        template_f = \
            "prompts/invent_sym_pred_implementation.outline"
        state_api_f = "prompts/api_oo_state.py"
        pred_api_f = "prompts/api_sym_predicate.py"

        with open(f"./{template_f}", "r", encoding='utf-8') as f:
            template = f.read()
        with open(f"./{state_api_f}", "r", encoding='utf-8') as f:
            state_cls_str = f.read()
        with open(f"./{pred_api_f}", "r", encoding='utf-8') as f:
            pred_cls_str = f.read()

        prompt = template.format(
            STRUCT_DEFINITION=add_python_quote(state_cls_str + "\n\n" +
                                               pred_cls_str),
            TYPES_IN_ENV=add_python_quote(
                _get_types_str(types, use_python_def_str=True)),
            PREDICATES_IN_ENV=pred_str,
            LISTED_STATES=state_str,
            PREDICATE_SPECS=spec_response,
        )
        with open(
                f"{CFG.log_file}/ite{self._online_learning_cycle}_b{b_id}"
                f"_s2_impl.prompt",
                "w",
                encoding='utf-8') as f:
            f.write(prompt)

        impl_response = self._llm.sample_completions(prompt,
                                                     imgs=None,
                                                     temperature=0,
                                                     num_completions=1,
                                                     seed=seed)[0]
        with open(
                f"{CFG.log_file}/ite{self._online_learning_cycle}_b{b_id}"
                f"_s2_impl.response",
                "w",
                encoding='utf-8') as f:
            f.write(impl_response)

        prim_predicates, deri_predicates =\
                _parse_predicates_predictions(impl_response,
                                            self._initial_predicates,
                                            self._candidate_predicates,
                                            types,
                                            self._train_tasks[0].init
                                            )
        base_candidates = set(prim_predicates) | set(deri_predicates)
        return base_candidates

    def _select_predicates_and_learn_processes(
        self,
        _ite: int,
        all_trajs: List[LowLevelTrajectory],
        proposed_predicates: Set[Predicate],
        train_tasks: Optional[List[Task]] = None,
        enumerate_processes: bool = False,
    ) -> None:
        if train_tasks is None:
            train_tasks = []
        if CFG.vlm_predicator_oracle_learned_predicates:
            if CFG.boil_goal_simple_human_happy:
                selected_preds = {
                    p
                    for p in proposed_predicates if p.name in {"JugFilled"}
                }
            else:
                selected_preds = proposed_predicates
            self._learned_predicates |= selected_preds
            # --- Learn processes & parameters ---
            self._learn_processes(
                all_trajs, online_learning_cycle=self._online_learning_cycle)
        else:
            self._candidate_predicates |= proposed_predicates

            all_candidates: Dict[Predicate, float] = {
                p: p.arity
                for p in self._initial_predicates
            }
            if CFG.vlm_predicator_use_grammar:
                grammar = _create_grammar(dataset=Dataset(all_trajs),
                                          given_predicates=\
                                            self._candidate_predicates)
            else:
                grammar = _GivenPredicateGrammar(self._candidate_predicates)
            all_candidates.update(
                grammar.generate(  # type: ignore[arg-type]
                    max_num=CFG.grammar_search_max_predicates))

            atom_dataset: List[GroundAtomTrajectory] =\
                        utils.create_ground_atom_dataset(all_trajs,
                                                        set(all_candidates))

            new_preds = set(all_candidates) - self._initial_predicates
            logging.info(f"Candidate predicates:\n{pformat(new_preds)}")
            self._learned_predicates = set(all_candidates)  # temp
            # Future: save the top ranking conditions here
            # so they can be used later in predicate selection.
            self._learn_processes(
                all_trajs, online_learning_cycle=self._online_learning_cycle)

            if CFG.learn_process_parameters:
                self._learn_process_parameters(all_trajs)
            # Whether to do predicate selection by scoring different predicate
            # set or by scoring different process set.
            start_time = time.perf_counter()
            if enumerate_processes:
                # Learn processes based on all the candidates.

                # Search by scoring different set of processes.
                # When commented out: keeping all candidates.
                selected_processes =\
                    self._select_processes_by_score_optimization(train_tasks,
                                                self._processes, atom_dataset)
                self._processes = selected_processes
                # Future: remove duplicate predicates
                self._learned_predicates = self._get_predicates_in_processes(
                    self._processes, set(all_candidates))
            else:
                # select predicates
                logging.info("[Start] Predicate search.")
                self._learned_predicates =\
                    self._select_predicates_by_score_optimization(
                        train_tasks,
                        all_candidates,  # type: ignore[arg-type]
                        self._processes,
                        all_trajs, atom_dataset)
            logging.info("[Finished] Predicate search.")
            logging.info("Total search time "
                         f"{time.perf_counter() - start_time:.2f}s")

    def _get_predicates_in_processes(
            self, processes: Set[CausalProcess],
            all_candidates: Set[Predicate]) -> Set[Predicate]:
        """Get the predicates in the processes."""
        all_process_predicates = set()
        for process in processes:
            all_process_predicates |= {
                atom.predicate
                for atom in process.condition_at_start
            }
            all_process_predicates |= {
                atom.predicate
                for atom in process.add_effects
            }
            all_process_predicates |= {
                atom.predicate
                for atom in process.delete_effects
            }
        selected_predicates = set()
        for pred in all_candidates:
            if pred in all_process_predicates:
                selected_predicates.add(pred)
        return selected_predicates

    def _select_processes_by_score_optimization(
        self,
        train_tasks: List[Task],
        all_processes: Set[CausalProcess],
        atom_dataset: List[GroundAtomTrajectory],
    ) -> Set[CausalProcess]:
        """Perform a greedy search over process sets."""
        endogenous_processes = {
            p
            for p in all_processes if isinstance(p, EndogenousProcess)
        }
        exogenous_processes = {
            p
            for p in all_processes if isinstance(p, ExogenousProcess)
        }

        # Precompute stuff for scoring.
        segmented_trajs = [
            segment_trajectory(ll_traj, self._get_current_predicates(),
                               atom_seq)
            for (ll_traj, atom_seq) in atom_dataset
        ]
        score_func = _ExpectedNodesScoreFunction(
            _initial_predicates=set(),
            _atom_dataset=[],
            _candidates={},
            _train_tasks=train_tasks,
            _current_processes=set(),
            _use_processes=True,
            metric_name="num_nodes_expanded")

        # Define the score function for a set of processes.
        def _score_processes(
            candidate_exogenous_processes: FrozenSet[ExogenousProcess]
        ) -> float:
            process_score = score_func.evaluate_with_operators(
                candidate_predicates=self._get_current_predicates(
                ),  # type: ignore[arg-type]
                low_level_trajs=self._offline_dataset.trajectories +
                self._online_dataset.trajectories,
                segmented_trajs=segmented_trajs,
                strips_ops=
                candidate_exogenous_processes  # type: ignore[arg-type]
                | endogenous_processes,
                option_specs=[])
            process_penalty = (
                _ExpectedNodesScoreFunction  # pylint: disable=protected-access
                ._get_operator_penalty(
                    candidate_exogenous_processes  # type: ignore[arg-type]
                ))
            return process_score + process_penalty

        # Set up the search.
        init_set: FrozenSet[ExogenousProcess] = frozenset()

        def _check_goal(s: FrozenSet[ExogenousProcess]) -> bool:
            del s  # unused
            return False

        def _get_successors(
            s: FrozenSet[ExogenousProcess]
        ) -> Iterator[Tuple[None, FrozenSet[ExogenousProcess], float]]:
            for process in sorted(exogenous_processes - s):
                yield (None, frozenset(s | {process}), 1.0)

        # Run the search.
        if CFG.grammar_search_search_algorithm == "hill_climbing":
            path, _, heuristics = utils.run_hill_climbing(
                init_set,
                _check_goal,
                _get_successors,
                _score_processes,
                enforced_depth=CFG.grammar_search_hill_climbing_depth,
                parallelize=CFG.grammar_search_parallelize_hill_climbing)
            logging.info("\nHill climbing summary:")
            for i in range(1, len(path)):  # pragma: no cover
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
                init_set,
                _check_goal,
                _get_successors,
                _score_processes,
                max_evals=CFG.grammar_search_gbfs_num_evals,
            )
        else:
            raise NotImplementedError(
                "Unrecognized grammar_search_search_algorithm: "
                f"{CFG.grammar_search_search_algorithm}.")

        selected_exogenous_processes = path[-1]
        logging.debug(f"Selected processes: "
                      f"{pformat(selected_exogenous_processes)}")

        return endogenous_processes | selected_exogenous_processes

    def _select_predicates_by_score_optimization(
        self,
        train_tasks: List[Task],
        candidates: Dict[Predicate, float],
        all_processes: Set[CausalProcess],
        all_trajs: List[LowLevelTrajectory],
        atom_dataset: List[GroundAtomTrajectory],
    ) -> Set[Predicate]:
        """Perform a greedy search over predicate sets."""
        endogenous_processes = {
            p
            for p in all_processes if isinstance(p, EndogenousProcess)
        }

        # Precompute stuff for scoring.
        segmented_trajs = [
            segment_trajectory(ll_traj, self._get_current_predicates(),
                               atom_seq)
            for (ll_traj, atom_seq) in atom_dataset
        ]
        score_func = _ExpectedNodesScoreFunction(
            _initial_predicates=set(),
            _atom_dataset=[],
            _candidates={},
            _train_tasks=train_tasks,
            _current_processes=set(),
            _use_processes=True,
            metric_name="num_nodes_expanded")

        def _filter_process(
                process: CausalProcess,
                candidate_predicates: FrozenSet[Predicate]) -> CausalProcess:
            """Filter a process to only keep atoms with candidate
            predicates."""
            proc_copy = process.copy()
            proc_copy.condition_at_start = {
                atom
                for atom in proc_copy.condition_at_start
                if atom.predicate in candidate_predicates
            }
            proc_copy.condition_overall = proc_copy.condition_at_start.copy()
            proc_copy.add_effects = {
                atom
                for atom in proc_copy.add_effects
                if atom.predicate in candidate_predicates
            }
            proc_copy.delete_effects = {
                atom
                for atom in proc_copy.delete_effects
                if atom.predicate in candidate_predicates
            }
            # Make sure the parameter only include variables that appear in the
            # conditions and effects
            remaining_variables = set()
            for atom in proc_copy.condition_at_start | proc_copy.add_effects |\
                        proc_copy.delete_effects:
                remaining_variables |= set(atom.variables)
            proc_copy.parameters = [
                v for v in proc_copy.parameters if v in remaining_variables
            ]
            return proc_copy

        def _get_best_compatible_exo_processes(
            candidate_predicates: FrozenSet[Predicate]
        ) -> Set[ExogenousProcess]:
            """Get the best compatible exogenous processes.

            # Get the processes compatible with the candidate
            predicates. # Look at all the scored conditions, find the
            top one that's a # subset of the candidate predicates; if
            none, remove the none # candidates from the top conditions.
            # Remove parts that are outside of candidates predicates
            """
            new_predicates = candidate_predicates - self._initial_predicates
            remaining_exogenous_processes = set()
            for _, results in self._proc_name_to_results.items():
                best_compatible_process = results[0][3]
                effect_pred = {
                    atom.predicate
                    for atom in best_compatible_process.add_effects
                    | best_compatible_process.delete_effects
                }
                if any(effect_p in candidate_predicates
                       for effect_p in effect_pred):
                    for _, (_, condition, _, proc) in enumerate(results):
                        condition_pred = {atom.predicate for atom in condition}
                        if new_predicates.issubset(condition_pred):
                            best_compatible_process = proc
                            break
                        if condition_pred.issubset(candidate_predicates):
                            # If the condition is a subset of the candidate
                            # predicates, then we can use this process.
                            best_compatible_process = proc
                            # logging.debug(f"Found compatible condition for "
                            #                 f"{proc.name}")
                            break
                    # else:
                    #     logging.debug(f"No compatible condition found for "
                    #                  f"{best_compatible_process.name}, "
                    #                  f"filtering out non-candidate atoms.")
                    # Haven't found a condition that is a subset of the
                    # candidate predicates, so we filter out the non-candidate
                    # condition
                    proc_copy = _filter_process(best_compatible_process,
                                                candidate_predicates)
                    if proc_copy.add_effects | proc_copy.delete_effects:
                        remaining_exogenous_processes.add(proc_copy)
            logging.debug(f"Remaining exogenous processes:\n"
                          f"{pformat(remaining_exogenous_processes)}")
            return remaining_exogenous_processes  # type: ignore[return-value]

        def _score_predicates(
                candidate_predicates: FrozenSet[Predicate]) -> float:
            new_preds = candidate_predicates - self._initial_predicates
            logging.debug(f"Evaluating predicates: {sorted(set(new_preds))}")
            remaining_exogenous_processes = _get_best_compatible_exo_processes(
                candidate_predicates)
            # Score processes with the score function.
            process_score = score_func.evaluate_with_operators(
                candidate_predicates=candidate_predicates,
                low_level_trajs=all_trajs,
                segmented_trajs=segmented_trajs,
                strips_ops=
                remaining_exogenous_processes  # type: ignore[arg-type]
                | endogenous_processes,
                option_specs=[])
            process_penalty = (
                _ExpectedNodesScoreFunction  # pylint: disable=protected-access
                ._get_operator_penalty(
                    remaining_exogenous_processes  # type: ignore[arg-type]
                ))
            final_score = process_score + process_penalty
            logging.debug(f"Candidate scores: {final_score:.4f}")
            return final_score

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
                yield (None, frozenset(s | {predicate}), 1.0)

        # Start the search with no candidates.
        # Don't need to include the initial predicates here because its
        init: FrozenSet[Predicate] = frozenset(self._initial_predicates)

        # Greedy local hill climbing search.
        if CFG.grammar_search_search_algorithm == "hill_climbing":
            path, _, heuristics = utils.run_hill_climbing(
                init,
                _check_goal,
                _get_successors,
                _score_predicates,
                enforced_depth=CFG.grammar_search_hill_climbing_depth,
                parallelize=CFG.grammar_search_parallelize_hill_climbing,
                exhaustive_lookahead=True)
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
                _score_predicates,
                max_evals=CFG.grammar_search_gbfs_num_evals,
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
        self._metrics["total_num_predicate_evaluations"] = len(path) * len(
            candidates)

        # # Filter out predicates that don't appear in some operator
        # # preconditions.
        # logging.info("\nFiltering out predicates that don't appear in "
        #              "preconditions...")
        # preds = kept_predicates | initial_predicates
        # pruned_atom_data = utils.prune_ground_atom_dataset(
        #     atom_dataset, preds)
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

        newly_selected = kept_predicates - self._initial_predicates
        new_candidates = set(candidates) - self._initial_predicates
        logging.info(f"\n[ite {self._online_learning_cycle}] Selected "
                     f"{len(newly_selected)} predicates"
                     f" out of {len(new_candidates)} candidates:")
        for pred in newly_selected:
            logging.info(f"\t{pred}")
        _score_predicates(kept_predicates)  # log useful numbers
        self._processes = endogenous_processes |\
                            _get_best_compatible_exo_processes(kept_predicates)

        return set(kept_predicates)


def get_false_positive_states_from_seg_trajs(
    segmented_trajs: List[List[Segment]],
    exogenous_processes: List[ExogenousProcess],
) -> Dict[_GroundExogenousProcess, List[State]]:
    """Get false positive states from seg trajs."""

    # Map from ground_exogenous_process to a list of init states where the
    # condition is satisfied.
    false_positive_process_state: Dict[_GroundExogenousProcess, List[State]] = \
        defaultdict(list)

    # Cache for ground_exogenous_processes to avoid recomputation
    objects_to_ground_processes = {}

    for segmented_traj in segmented_trajs:
        # Checking each segmented trajectory
        objects = frozenset(segmented_traj[0].trajectory.states[0])
        # Only recompute if objects are different
        if objects not in objects_to_ground_processes:
            ground_exogenous_processes, _ = process_task_plan_grounding(
                set(),
                objects,  # type: ignore[arg-type]
                exogenous_processes,
                allow_waits=True,
                compute_reachable_atoms=False)
            objects_to_ground_processes[objects] = ground_exogenous_processes
        else:
            ground_exogenous_processes = objects_to_ground_processes[objects]

        # Pre-compute segment init_atoms for efficiency
        segment_init_atoms = [segment.init_atoms for segment in segmented_traj]

        for g_exo_process in ground_exogenous_processes:
            condition = g_exo_process.condition_at_start  # Cache reference
            add_effects = g_exo_process.add_effects
            delete_effects = g_exo_process.delete_effects

            for i, segment in enumerate(segmented_traj):
                satisfy_condition = condition.issubset(segment_init_atoms[i])
                prev_doesnt = (
                    i == 0
                    or not condition.issubset(segment_init_atoms[i - 1]))

                if satisfy_condition and prev_doesnt:
                    false_positive_process_state[
                        g_exo_process].append(  # type: ignore[index]
                            # segment.trajectory.states[0])
                            segment.init_atoms)  # type: ignore[arg-type]

                # Check for removal condition
                if (add_effects.issubset(segment.add_effects)
                        and delete_effects.issubset(segment.delete_effects)):
                    if false_positive_process_state[
                            g_exo_process]:  # type: ignore[index]
                        # Note: we don't really know which one
                        # to remove; popping first is a bias.
                        fp_list = false_positive_process_state[
                            g_exo_process]  # type: ignore[index]
                        fp_list.pop(0)
    return false_positive_process_state


def get_false_positive_states(
    trajectories: List[LowLevelTrajectory],
    predicates: Set[Predicate],
    exogenous_processes: List[ExogenousProcess],
) -> Dict[_GroundExogenousProcess, List[State]]:
    """Get the false positive states for each exogenous process.

    Return:
        ground_exogenous_process ->
            Tuple[List[State], List[GroundAtom], List[GroundAtom]] per
            trajectory where List[State] is the list of states where the
            process is activated in the trajectory.
    """
    initial_segmenter_method = CFG.segmenter
    # Note: option_changes creates a segment for the noop
    # option at the end, but would cause problems if the
    # start/end of option execution doesn't satisfy the
    # condition but somewhere in the middle does. The same
    # problem exists for the effects.
    #
    # The fix for the atom_changes segmenter would be to
    # create a segment at the end if there are still states
    # after the last atom change.
    CFG.segmenter = "atom_changes"
    segmented_trajs = [
        segment_trajectory(traj, predicates, verbose=False)
        for traj in trajectories
    ]
    CFG.segmenter = initial_segmenter_method

    return get_false_positive_states_from_seg_trajs(segmented_trajs,
                                                    exogenous_processes)


def get_true_positive_process_states(
    predicates: Set[Predicate],
    exogenous_processes: List[ExogenousProcess],
    ground_exogenous_processes: List[_GroundExogenousProcess],
    trajectories: List[LowLevelTrajectory],
) -> Dict[_GroundExogenousProcess, List[State]]:
    """Get the true positive states for each exogenous process."""
    initial_segmenter_method = CFG.segmenter
    CFG.segmenter = "atom_changes"
    segmented_trajs = [
        segment_trajectory(traj, predicates) for traj in trajectories
    ]
    CFG.segmenter = initial_segmenter_method

    # Filter out segments explained by endogenous processes.
    filtered_segmented_trajs = filter_explained_segment(
        segmented_trajs,
        exogenous_processes,  # type: ignore[arg-type]
        remove_options=True)
    true_positive_process_state: Dict[_GroundExogenousProcess,
                                      List[State]] = defaultdict(list)
    for g_exo_process in ground_exogenous_processes:
        for segmented_traj in filtered_segmented_trajs:
            # Checking each segmented trajectory
            for segment in segmented_traj:
                # Check if the segment is a positive example for any
                # exogenous process
                if g_exo_process.condition_at_start.issubset(
                        segment.init_atoms) and \
                    g_exo_process.add_effects.issubset(
                        segment.add_effects) and \
                    g_exo_process.delete_effects.issubset(
                        segment.delete_effects):
                    true_positive_process_state[g_exo_process].append(
                        segment.trajectory.states[0])
    return true_positive_process_state


def _get_predicates_str(predicates: Set[Predicate],
                        include_primitive_preds: bool = True,
                        include_derived_preds: bool = True) -> str:

    init_pred_str = []
    for p in predicates:
        if include_primitive_preds and not isinstance(p, DerivedPredicate):
            init_pred_str.append(p.pretty_str_with_assertion())
        elif include_derived_preds and isinstance(p, DerivedPredicate):
            init_pred_str.append(p.pretty_str_with_assertion())
    logging.debug(f"Current predicate str: {init_pred_str}")
    init_pred_str = sorted(init_pred_str)
    return "\n".join(init_pred_str)


def _get_types_str(types: Set[Type],
                   _include_features: bool = True,
                   use_python_def_str: bool = False) -> str:
    """Get the types string."""
    excluded_types = []
    if CFG.excluded_objects_in_state_str:
        excluded_types = CFG.excluded_objects_in_state_str.split(",")

    if use_python_def_str:
        type_str = [
            t.python_definition_str() for t in types
            if t.name not in excluded_types
        ]
    else:
        type_str = [
            t.pretty_str() for t in types if t.name not in excluded_types
        ]
    type_str = sorted(type_str)
    return "\n".join(type_str)


def _get_transition_str(
    trajectories: List[LowLevelTrajectory],
    train_tasks: List[Task],
    predicates: Set[Predicate],
    ite: int,
    max_num_trajs: int = 1,
    only_use_successful_trajs: bool = False,
    use_abstract_state_str: bool = False,
) -> Tuple[str, str]:
    """Get the state before and after some actions.

    Prioritize successful trajectories.
    Future: save images of the states.
    """
    if CFG.rgb_observation:
        obs_dir = CFG.log_file + f"ite{ite}_obs/"
        os.makedirs(obs_dir, exist_ok=True)

    if only_use_successful_trajs:
        successful_trajs = [
            traj for traj in trajectories
            if traj_is_successful(traj, train_tasks)
        ]
        if successful_trajs:
            trajectories = successful_trajs
    trajectories = trajectories[:max_num_trajs]

    # Segment the trajectories and get states before and after the actions.
    segmented_trajs = [
        segment_trajectory(ll_traj, predicates) for ll_traj in trajectories
    ]
    result_str, state_str_set = [], []
    state_hash_to_id: Dict[int, int] = {}
    for seg_traj in segmented_trajs:
        for i, segment in enumerate(seg_traj):
            # Get state cache and observation name
            init_state_hash = hash(segment.states[0])
            if init_state_hash not in state_hash_to_id:
                state_hash_to_id[init_state_hash] = len(state_hash_to_id)
            init_state_id = state_hash_to_id[init_state_hash]
            obs_name = "state_" + str(init_state_id)

            # Append state
            if i == 0:
                result_str.append(
                    f"Starting at {obs_name} with additional info:")
            state = segment.states[0]
            assert isinstance(state, utils.PyBulletState)
            if use_abstract_state_str:
                state_str = sorted(utils.abstract(state, predicates))
            else:
                state_str = state.dict_str(
                    indent=2,  # type: ignore[assignment]
                    use_object_id=CFG.rgb_observation)

            result_str.append(f"{state_str}")
            str_for_this_state = [f"  {obs_name} with additional info:"]
            str_for_this_state.append(f"{state_str}")
            state_str_set.append("\n".join(str_for_this_state))
            if CFG.rgb_observation:
                assert state.labeled_image is not None
                save_image_with_label(
                    state.labeled_image.copy(),  # type: ignore[arg-type]
                    obs_name,
                    obs_dir)

            # Append action
            action_str = segment.actions[0].get_option().simple_str(
                use_object_id=CFG.rgb_observation)
            result_str.append(
                f"\nAction {action_str} was executed in {obs_name}")

            # Get state cache and observation name
            end_state_hash = hash(segment.states[-1])
            if end_state_hash not in state_hash_to_id:
                state_hash_to_id[end_state_hash] = len(state_hash_to_id)
            end_state_id = state_hash_to_id[end_state_hash]
            obs_name = "state_" + str(end_state_id)
            result_str.append(f"\nThis action results in {obs_name} "
                              "with additional info:")
        # Append final state
        if not seg_traj:
            continue
        segment = seg_traj[-1]
        state = segment.states[-1]
        if use_abstract_state_str:
            state_str = sorted(utils.abstract(state, predicates))
        else:
            state_str = state.dict_str(
                indent=2,  # type: ignore[assignment]
                use_object_id=CFG.rgb_observation)
        result_str.append(f"{state_str}")
        str_for_this_state = [f"  {obs_name} with additional info:"]
        str_for_this_state.append(f"{state_str}")
        state_str_set.append("\n".join(str_for_this_state))
        if CFG.rgb_observation:
            save_image_with_label(
                state.labeled_image.copy(),  # type: ignore[attr-defined]
                obs_name,
                obs_dir)

    return "\n".join(result_str), "\n\n".join(state_str_set)


def save_image_with_label(img_copy: Image,
                          s_name: str,
                          obs_dir: str,
                          f_suffix: str = ".png") -> None:
    """Save image with label."""
    draw = ImageDraw.Draw(img_copy)  # type: ignore[arg-type]
    font = ImageFont.load_default()
    font = font.font_variant(size=50)  # type: ignore[union-attr]
    text_color = (0, 0, 0)  # white
    draw.text((0, 0), s_name, fill=text_color, font=font)
    img_copy.save(  # type: ignore[attr-defined]
        os.path.join(obs_dir, s_name + f_suffix))
    logging.debug(f"Saved image {s_name}")


def load_images_from_directory(directory: str) -> List[PIL.Image.Image]:
    """Load images from directory."""
    images = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.lower().endswith(('.png', '.jpg')):
            img = PIL.Image.open(file_path)
            images.append(img)
    return images


def traj_is_successful(traj: LowLevelTrajectory,
                       train_tasks: List[Task]) -> bool:
    """Check if the trajectory is successful for any of the train tasks."""
    task_idx = traj._train_task_idx  # pylint: disable=protected-access
    goal_atoms = train_tasks[task_idx].goal  # type: ignore[index]
    goal_predicates = {atom.predicate for atom in goal_atoms}
    abstract_state = utils.abstract(traj.states[-1], goal_predicates)
    return goal_atoms.issubset(abstract_state)


def add_python_quote(text: str) -> str:
    """Add python quote."""
    return f"```python\n{text}\n```"


def _parse_predicates_predictions(
    response: str,
    initial_predicates: Set[Predicate],
    candidate_predicates: Set[Predicate],
    # existing_primitive_candidates: Set[Predicate],
    # existing_derived_candidates: Set[DerivedPredicate],
    types: Set[Type],
    example_state: State,
) -> Tuple[List[Predicate], List[DerivedPredicate]]:
    # Regular expression to match Python code blocks
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    python_blocks = []
    # Find all Python code blocks in the text
    for match in pattern.finditer(response):
        # Extract the Python code block and add it to the list
        python_blocks.append(match.group(1).strip())

    existing_primitive_candidates: Set[Predicate] = set(
        p for p in candidate_predicates if not isinstance(p, DerivedPredicate))
    existing_derived_candidates: Set[DerivedPredicate] = set(
        p for p in candidate_predicates if isinstance(p, DerivedPredicate))
    primitive_preds: Set[Predicate] = set()
    context: Dict[str, Any] = {}
    untranslated_derived_pred_str: List[str] = []
    # --- Existing predicates and their classifiers
    for p in initial_predicates:
        context[f"_{p.name}_NSP_holds"] = (p._classifier)  # pylint: disable=protected-access

    for p in existing_derived_candidates:
        context[f"_{p.name}_CP_holds"] = (p._classifier)  # pylint: disable=protected-access

    for p in existing_primitive_candidates | existing_derived_candidates:
        context[f"{p.name}"] = p

    # --- Types ---
    for t in types:
        context[f"_{t.name}_type"] = t

    # --- Imports ---
    exec(import_str, context)  # pylint: disable=exec-used

    # --- Interpret the Python blocks ---
    for code_str in python_blocks:
        # Extract name from code block
        name_match = re.search(r'(\w+)\s*=\s*(NS)?Predicate', code_str)
        if name_match is None:
            logging.warning("No predicate name found in the code block")
            continue
        pred_name = name_match.group(1)
        logging.info(f"Found definition for predicate {pred_name}")
        vlm_invention_use_concept_predicates = False
        if vlm_invention_use_concept_predicates:
            is_concept_predicate = check_is_derived_predicate(code_str)
            logging.info(f"\t it's a derived predicate: "
                         f"{is_concept_predicate}")
        else:
            is_concept_predicate = False
            # logging.info(f"\t derived predicate disabled")

        # Recognize that it's a derived predicate
        if is_concept_predicate:
            untranslated_derived_pred_str.append(add_python_quote(code_str))
        else:
            # Type check the code
            # passed = False
            # while not passed:
            #     result, passed = self.type_check_proposed_predicates(
            #                                                     pred_name,
            #                                                     code_str)
            #     if not passed:
            #         # Ask the LLM or the User to fix the code
            #         pass
            #     else:
            #         break

            # Instantiate the primitive predicates
            #   check if it's roughly runable, and add it to list if it is.
            try:
                exec(code_str, context)  # pylint: disable=exec-used
                logging.debug(f"Testing predicate {pred_name}")
                # Check1: Make sure it uses types present in the environment
                proposed_pred = context[pred_name]
                for t in proposed_pred.types:
                    if t not in types:
                        logging.warning(f"Type {t} not in the environment")
                        raise Exception(f"Type {t} not in the environment")
                utils.abstract(example_state, [context[pred_name]])
            except Exception as e:  # pylint: disable=broad-except
                error_trace = traceback.format_exc()
                logging.warning(f"Test failed: {e}\n{error_trace}")
                continue
            else:
                logging.debug("Test passed!")
                primitive_preds.add(context[pred_name])

    # Future: convert derived predicates to DerivedPredicate
    derived_predicates: Set[DerivedPredicate] = set()

    return primitive_preds, derived_predicates  # type: ignore[return-value]


import_str = """
import numpy as np
from typing import Sequence, Set, List
from predicators.structs import State, Object, Type, GroundAtom, Predicate, \
    NSPredicate, DerivedPredicate
from predicators.settings import CFG
"""


def check_is_derived_predicate(code_str: str) -> bool:
    """Check if the predicate is a derived predicate by looking for `get` or
    `evaluate_simple` in the code block."""
    if "state.get(" in code_str or\
        "state.evaluate_simple_assertion" in code_str:
        return False
    return True
