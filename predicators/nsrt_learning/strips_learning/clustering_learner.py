"""Algorithms for STRIPS learning that rely on clustering to obtain effects."""
import abc
import bisect
import copy
import functools
import itertools
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pprint import pformat
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, \
    Tuple, Union, cast

import multiprocess as mp  # type: ignore[import-untyped]
import psutil  # type: ignore[import-untyped]
from pathos.multiprocessing import ProcessingPool as Pool

from predicators import utils
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.planning import PlanningFailure, PlanningTimeout
from predicators.planning_with_processes import \
    task_plan_from_task as task_plan_with_processes
from predicators.settings import CFG
from predicators.structs import PNAD, CausalProcess, Datastore, \
    DerivedPredicate, DummyOption, EndogenousProcess, ExogenousProcess, \
    GroundAtom, LiftedAtom, Object, ParameterizedOption, Predicate, Segment, \
    STRIPSOperator, Variable, VarToObjSub, _TypedEntity

if sys.platform == "darwin":
    # Set this when using macOS, to avoid issues with forked processes.
    mp.set_start_method("spawn", force=True)  # pylint: disable=no-member


def _flat_pnad_scoring_worker(
    args: Tuple[int, int, ExogenousProcess, Set[LiftedAtom], List[Any],
                Set[Predicate], int, int, float, Optional[str], Optional[str],
                int]
) -> Tuple[int, int, float, Set[LiftedAtom], Tuple[float, ...],
           ExogenousProcess]:
    """Utility for flat multiprocessing: evaluates one condition candidate for
    one PNAD under the data-likelihood scoring regime.

    Returns (pnad_idx, condition_idx, cost, condition_candidate,
    scores_tuple, process).
    """
    (pnad_idx, condition_idx, base_process, condition_candidate, trajectories,
     predicates, seed, num_it, complexity_weight, _load_dir, _save_dir,
     early_stopping_patience) = args

    # Set the conditions on the process object.
    base_process.condition_at_start = condition_candidate
    base_process.condition_overall = condition_candidate

    # Calculate complexity penalty.
    complexity_penalty = complexity_weight * len(condition_candidate)

    # Local import avoids pickling issues with bound methods.
    from predicators.approaches.pp_param_learning_approach import \
        learn_process_parameters  # pylint: disable=import-outside-toplevel

    # Perform the expensive part: learning and scoring.
    process, scores = learn_process_parameters(
        trajectories,
        predicates,
        [base_process],  # The list now contains just the one process to score.
        use_lbfgs=False,
        plot_training_curve=False,
        lbfgs_max_iter=num_it,
        adam_num_steps=num_it,
        seed=seed,
        display_progress=False,
        early_stopping_patience=early_stopping_patience,
        batch_size=CFG.process_param_learning_batch_size,
        use_empirical=CFG.process_learning_use_empirical,
    )

    # Cost is negative log-likelihood plus penalty.
    cost = -scores[0] + complexity_penalty

    # Return the identifier, condition index, cost, candidate, and the full
    # scores tuple for logging.
    result_proc = process[0]  # type: ignore[index]
    return (
        pnad_idx,
        condition_idx,
        cost,
        condition_candidate,
        scores,  # type: ignore[return-value]
        result_proc)


class ClusteringSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a clustering-based STRIPS learner."""

    def _learn(self) -> List[PNAD]:
        segments = [seg for segs in self._segmented_trajs for seg in segs]
        # Cluster the segments according to common option and effects.
        pnads: List[PNAD] = []
        for segment in segments:
            if segment.has_option():
                segment_option = segment.get_option()
                segment_param_option = segment_option.parent
                segment_option_objs = tuple(segment_option.objects)
            else:
                segment_param_option = DummyOption.parent
                segment_option_objs = tuple()
            for pnad in pnads:
                # Try to unify this transition with existing effects.
                # Note that both add and delete effects must unify,
                # and also the objects that are arguments to the options.
                (pnad_param_option, pnad_option_vars) = pnad.option_spec
                suc, ent_to_ent_sub = utils.unify_preconds_effects_options(
                    frozenset(),  # no preconditions
                    frozenset(),  # no preconditions
                    frozenset(segment.add_effects),
                    frozenset(pnad.op.add_effects),
                    frozenset(segment.delete_effects),
                    frozenset(pnad.op.delete_effects),
                    segment_param_option,
                    pnad_param_option,
                    segment_option_objs,
                    tuple(pnad_option_vars))
                sub = cast(VarToObjSub,
                           {v: o
                            for o, v in ent_to_ent_sub.items()})
                if suc:
                    # Add to this PNAD.
                    assert set(sub.keys()) == set(pnad.op.parameters)
                    pnad.add_to_datastore(
                        (segment, sub),
                        check_effect_equality=CFG.
                        clustering_learner_check_effect_equality)
                    break
            else:
                # Otherwise, create a new PNAD.
                objects = {o for atom in segment.add_effects |
                           segment.delete_effects for o in atom.objects} | \
                          set(segment_option_objs)
                objects_lst = sorted(objects)
                params = utils.create_new_variables(
                    [o.type for o in objects_lst])
                preconds: Set[LiftedAtom] = set()  # will be learned later
                obj_to_var = dict(zip(objects_lst, params))
                var_to_obj = dict(zip(params, objects_lst))
                add_effects = {
                    atom.lift(obj_to_var)
                    for atom in segment.add_effects
                }
                delete_effects = {
                    atom.lift(obj_to_var)
                    for atom in segment.delete_effects
                }
                ignore_effects: Set[Predicate] = set()  # will be learned later
                op = STRIPSOperator(f"Op{len(pnads)}", params, preconds,
                                    add_effects, delete_effects,
                                    ignore_effects)
                datastore = [(segment, var_to_obj)]
                option_vars = [obj_to_var[o] for o in segment_option_objs]
                option_spec = (segment_param_option, option_vars)
                pnads.append(PNAD(op, datastore, option_spec))

        # Learn the preconditions of the operators in the PNADs. This part
        # is flexible; subclasses choose how to implement it.
        pnads = self._learn_pnad_preconditions(pnads)

        # Handle optional postprocessing to learn ignore effects.
        pnads = self._postprocessing_learn_ignore_effects(pnads)

        # Log and return the PNADs.
        if self._verbose:
            logging.info("Learned operators (before option learning):")
            for pnad in pnads:
                logging.info(pnad)
        return pnads

    @abc.abstractmethod
    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        """Subclass-specific algorithm for learning PNAD preconditions.

        Returns a list of new PNADs. Should NOT modify the given PNADs.
        """
        raise NotImplementedError("Override me!")

    def _postprocessing_learn_ignore_effects(self,
                                             pnads: List[PNAD]) -> List[PNAD]:
        """Optionally postprocess to learn ignore effects."""
        _ = self  # unused, but may be used in subclasses
        return pnads


class ClusterAndIntersectSTRIPSLearner(ClusteringSTRIPSLearner):
    """A clustering STRIPS learner that learns preconditions via
    intersection."""

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        new_pnads = []
        for pnad in pnads:
            if CFG.cluster_and_intersect_soft_intersection_for_preconditions:
                preconditions = \
                    self._induce_preconditions_via_soft_intersection(pnad)
            else:
                preconditions = self._induce_preconditions_via_intersection(
                    pnad)
            # Since we are taking an intersection, we're guaranteed that the
            # datastore can't change, so we can safely use pnad.datastore here.
            new_pnads.append(
                PNAD(pnad.op.copy_with(preconditions=preconditions),
                     pnad.datastore, pnad.option_spec))
        return new_pnads

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_intersect"

    def _postprocessing_learn_ignore_effects(self,
                                             pnads: List[PNAD]) -> List[PNAD]:
        """Prune PNADs whose datastores are too small.

        Specifically, keep PNADs that have at least
        CFG.cluster_and_intersect_min_datastore_fraction fraction of the
        segments produced by the option in their NSRT.
        """
        if not CFG.cluster_and_intersect_prune_low_data_pnads:
            return pnads
        option_to_dataset_size: Dict[ParameterizedOption,
                                     int] = defaultdict(int)
        for pnad in pnads:
            option = pnad.option_spec[0]
            option_to_dataset_size[option] += len(pnad.datastore)
        ret_pnads: List[PNAD] = []
        for pnad in pnads:
            option = pnad.option_spec[0]
            fraction = len(pnad.datastore) / option_to_dataset_size[option]
            if fraction >= CFG.cluster_and_intersect_min_datastore_fraction:
                ret_pnads.append(pnad)
        return ret_pnads


class ClusterAndLLMSelectSTRIPSLearner(ClusteringSTRIPSLearner):
    """Learn preconditions via LLM selection.

    Note: The current prompt are tailored for exogenous processes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the LLM and load the prompt template."""
        super().__init__(*args, **kwargs)
        self._llm = utils.create_llm_by_name(CFG.llm_model_name)
        prompt_file = utils.get_path_to_predicators_root() + \
            "/predicators/nsrt_learning/strips_learning/" + \
            "llm_op_learning_prompts/condition_selection.prompt"
        with open(prompt_file, "r", encoding='utf-8') as f:
            self.base_prompt = f.read()
        # pylint: disable-next=import-outside-toplevel
        from predicators.approaches import \
            pp_online_predicate_invention_approach as pp_inv
        self._get_false_positive_process_states = \
            pp_inv.get_false_positive_states

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_llm_select"

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        """Assume there is one segment per PNAD We can either do lifting first
        and selection second, or the other way around.

        If we have multiple segments per PNAD, lifting requires us to
        find a subset of atoms that unifies the segments. We'd have to
        do this if we want to learn a single condition. But we could
        also learn more than one.
        """
        # Add var_to_obj for objects in the init state of the segment
        new_pnads = []
        for pnad in pnads:
            # Removing this assumption because we're now making sure that
            # all the init_atoms in the PNAD are the same up to unification.
            # assert len(pnad.datastore) == 1
            seg, var_to_obj = pnad.datastore[0]
            existing_objs = set(var_to_obj.values())
            # Get the init atoms of the segment
            init_atoms = seg.init_atoms
            # Get the objects in the init atoms
            additional_objects = {
                o
                for atom in init_atoms for o in atom.objects
                if o not in existing_objs
            }
            # Create a new var_to_obj mapping for the objects
            objects_lst = sorted(additional_objects)
            params = utils.create_new_variables([o.type for o in objects_lst],
                                                existing_vars=list(var_to_obj))
            var_to_obj.update(dict(zip(params, objects_lst)))
            new_pnads.append(
                PNAD(pnad.op, [(seg, var_to_obj)],
                     pnad.option_spec))  # dummy option

        seperate_llm_query_per_pnad = True
        effect_and_conditions = ""
        proposed_conditions: List[str] = []
        for i, pnad in enumerate(new_pnads):
            if seperate_llm_query_per_pnad:
                effect_and_conditions += "Process 0:\n"
            else:
                effect_and_conditions += f"Process {i}:\n"
            add_effects = pnad.op.add_effects
            delete_effects = pnad.op.delete_effects
            effect_and_conditions += "Add effects: ("
            if add_effects:
                effect_and_conditions += "and " + " ".join(f"({str(atom)})" for\
                                                           atom in add_effects)
            effect_and_conditions += ")\n"
            effect_and_conditions += "Delete effects: ("
            if delete_effects:
                del_str = " ".join(f"({str(atom)})" for atom in delete_effects)
                effect_and_conditions += "and " + del_str
            effect_and_conditions += ")\n"
            segment_init_atoms = pnad.datastore[0][0].init_atoms
            segment_var_to_obj = pnad.datastore[0][1]
            obj_to_var = {v: k for k, v in segment_var_to_obj.items()}
            conditions_to_choose_from = pformat(
                {a.lift(obj_to_var)
                 for a in segment_init_atoms})
            effect_and_conditions += "Conditions to choose from:\n" +\
                conditions_to_choose_from + "\n\n"

            if seperate_llm_query_per_pnad:
                prompt = self.base_prompt.format(
                    EFFECTS_AND_CONDITIONS=effect_and_conditions)
                proposals = self._llm.sample_completions(
                    prompt, None, 0.0, CFG.seed)[0]
                pattern = r'```\n(.*?)\n```'
                matches = re.findall(pattern, proposals, re.DOTALL)
                proposed_conditions.append(matches[0])
                effect_and_conditions = ""

        if not seperate_llm_query_per_pnad:
            prompt = self.base_prompt.format(
                EFFECTS_AND_CONDITIONS=effect_and_conditions)
            proposals = self._llm.sample_completions(prompt, None, 0.0,
                                                     CFG.seed)[0]
            pattern = r'```\n(.*?)\n```'
            matches = re.findall(pattern, proposals, re.DOTALL)
            proposed_conditions = matches[0].split("\n\n")

        def atom_in_llm_selection(
                atom: LiftedAtom,
                conditions: List[Tuple[str, List[Tuple[str, str]]]]) -> bool:
            for condition in conditions:
                atom_name = condition[0]
                atom_variables = condition[1]
                if atom.predicate.name == atom_name and \
                        all(var_type[0] == var.name for (var_type, var) in
                            zip(atom_variables, atom.variables)):
                    return True
            return False

        # Assumes the same number of PNADs and response chunks
        assert len(new_pnads) == len(proposed_conditions)
        final_pnads: List[PNAD] = []
        for proposed_condition, corresponding_pnad in zip(
                proposed_conditions, new_pnads):
            # Get the effect atoms
            # Get the condition atoms
            lines = proposed_condition.split("\n")
            # add_effects = self.parse_effects_or_conditions(lines[0])
            # delete_effects = self.parse_effects_or_conditions(lines[1])
            conditions = self.parse_effects_or_conditions(lines[2])

            segment_init_atoms = corresponding_pnad.datastore[0][0].init_atoms
            segment_var_to_obj = corresponding_pnad.datastore[0][1]
            obj_to_var = {v: k for k, v in segment_var_to_obj.items()}
            lifted_conditions_to_choose_from: Set[LiftedAtom] = {
                a.lift(obj_to_var)
                for a in segment_init_atoms
            }
            new_conditions = set(atom
                                 for atom in lifted_conditions_to_choose_from
                                 if atom_in_llm_selection(atom, conditions))
            add_eff = corresponding_pnad.op.add_effects
            del_eff = corresponding_pnad.op.delete_effects
            # the variable might also just in the effects
            new_parameters = set(
                var for atom in new_conditions | add_eff | del_eff
                for var in atom.variables)  # type: ignore[union-attr]
            # Only append if it's unique
            for final_pnad in final_pnads:
                suc, _ = utils.unify_preconds_effects_options(
                    frozenset(new_conditions),
                    frozenset(final_pnad.op.preconditions),
                    frozenset(corresponding_pnad.op.add_effects),
                    frozenset(final_pnad.op.add_effects),
                    frozenset(corresponding_pnad.op.delete_effects),
                    frozenset(final_pnad.op.delete_effects),
                    corresponding_pnad.option_spec[0],
                    final_pnad.option_spec[0],
                    tuple(corresponding_pnad.option_spec[1]),
                    tuple(final_pnad.option_spec[1]),
                )
                if suc:
                    break
            else:
                # We have a new process!
                # Create a new PNAD with the new parameters and conditions
                # and add it to the final list
                pnad = PNAD(
                    corresponding_pnad.op.copy_with(
                        parameters=new_parameters,
                        preconditions=new_conditions),
                    corresponding_pnad.datastore,
                    corresponding_pnad.option_spec)
                final_pnads.append(pnad)

                # if CFG.process_learner_check_false_positives:
                #     # Go through the trajectories and check if this process
                #     # leads to false positive effect predications.
                #     false_positive_process_state = \
                #         self._get_false_positive_process_states(
                #             self._trajectories,
                #             self._predicates,
                #             [pnad.make_exogenous_process()])

                #     for _, states in false_positive_process_state.items():
                #         if len(states) > 0:
                #             # initial_segmenter_method = CFG.segmenter
                #             # CFG.segmenter = "atom_changes"
                # # segments = [segment_trajectory(
                # #     traj, self._predicates)
                # #     for traj in self._trajectories]
                #             # CFG.segmenter = initial_segmenter_method
        return final_pnads

    def parse_effects_or_conditions(
            self, line: str) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """Parse a line containing effects or conditions into a list of tuples.
        For example, when given: 'Conditions: (and (FaucetOn(?x1:faucet))
        (JugUnderFaucet(?x2:jug, ?x1:faucet)))'.

        Each returned tuple has:
        - An atom name (e.g., "JugFilled")
        - A list of (variable_name, type_name) pairs
        (e.g., [("?x0", "jug"), ("?x1", "faucet")]).

        Example Return:
        [
            ("FaucetOn", [("?x1", "faucet")]),
            ("JugUnderFaucet", [("?x2", "jug"), ("?x1", "faucet")])
        ]
        """

        # Remove the top-level (and ...) if present.
        # This way, we won't accidentally capture "and" as an atom.
        line = re.sub(r"\(\s*and\s+", "(", line)

        # Match an atom name and the entire content inside its parentheses.
        pattern = r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\)"
        atom_matches = re.findall(pattern, line)

        var_type_pattern = r"(\?[a-zA-Z0-9]+):([a-zA-Z0-9_]+)"
        parsed_atoms: List[Tuple[str, List[Tuple[str, str]]]] = []

        for atom_name, vars_str in atom_matches:
            # Find all variable:type pairs in the string
            var_type_pairs = re.findall(var_type_pattern, vars_str)
            parsed_atoms.append((atom_name, var_type_pairs))

        return parsed_atoms


class ClusteringProcessLearner(ClusteringSTRIPSLearner):
    """ClusteringProcessLearner class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.online_learning_cycle = kwargs.get("online_learning_cycle", None)
        self._endogenous_processes = kwargs["endogenous_processes"]
        # pylint: disable-next=import-outside-toplevel
        from predicators.approaches import \
            pp_online_predicate_invention_approach as pp_inv2
        self._get_false_positive_states_from_seg_trajs = (
            pp_inv2.get_false_positive_states_from_seg_trajs)

        from predicators.approaches.pp_param_learning_approach import \
            learn_process_parameters  # pylint: disable=import-outside-toplevel
        self._get_data_likelihood_and_learn_params = \
            learn_process_parameters

        self._atom_change_segmented_trajs: List[List[Segment]] = []

        if CFG.cluster_and_search_process_learner_llm_propose_top_conditions or\
            CFG.cluster_and_search_process_learner_llm_rank_atoms:
            self._llm = utils.create_llm_by_name(CFG.llm_model_name)
        else:
            self._llm = None  # type: ignore[assignment]

    def _learn(self) -> List[PNAD]:
        segments = [seg for segs in self._segmented_trajs for seg in segs]
        # Cluster the segments according to common option and effects.
        pnads: List[PNAD] = []
        for segment in segments:
            if segment.has_option():
                segment_option = segment.get_option()
                segment_param_option = segment_option.parent
                segment_option_objs = tuple(segment_option.objects)
            else:
                segment_param_option = DummyOption.parent
                segment_option_objs = tuple()
            if self.get_name() not in [
                    "cluster_and_llm_select",
                    "cluster_and_search_process_learner",
                    "cluster_and_inverse_planning"
            ] or CFG.exogenous_process_learner_do_intersect:
                preconds1: FrozenSet = frozenset()  # no preconditions
                segment_param_option = DummyOption.parent
                segment_option_objs = tuple()
            else:
                # Ground
                preconds1 = frozenset(segment.init_atoms)

            # ent_to_ent_sub here is obj_to_var
            seg_add_effects = frozenset(
                a for a in segment.add_effects
                if not isinstance(a.predicate, DerivedPredicate))
            seg_del_effects = frozenset(
                a for a in segment.delete_effects
                if not isinstance(a.predicate, DerivedPredicate))
            if self.get_name() in ["cluster_and_search_process_learner"]:
                # Remove atoms explained by endogenous processes
                filtered_add: Set[Union[LiftedAtom, GroundAtom]]
                filtered_del: Set[Union[LiftedAtom, GroundAtom]]
                filtered_add, filtered_del = \
                    self.remove_atoms_explained_by_endogenous_processes(
                        segment, self._endogenous_processes,
                        set(seg_add_effects), set(seg_del_effects))
                seg_add_effects = frozenset(cast(Set[GroundAtom],
                                                 filtered_add))
                seg_del_effects = frozenset(cast(Set[GroundAtom],
                                                 filtered_del))

            suc, ent_to_ent_sub, pnad = \
                self._unify_segment_with_pnads(  # type: ignore[misc]
                preconds1, seg_add_effects, seg_del_effects,
                segment_param_option, segment_option_objs, pnads)

            if suc:
                sub = cast(VarToObjSub,
                           {v: o
                            for o, v in ent_to_ent_sub.items()})
                # Add to this PNAD.
                if CFG.exogenous_process_learner_do_intersect:
                    # Find the largest conditions that unifies the init
                    # atoms of the segment and another segment in the PNAD.
                    # and add that segment and sub to the datastore.
                    # Doing this sequentially ensures one of the
                    # substitutions has the objects we care about with
                    # intersection. Hence it can fall out later in
                    # `induce_preconditions_via_intersection`.
                    (pnad_param_option, pnad_option_vars) = pnad.option_spec
                    sub = self._find_best_segment_unification(
                        segment, seg_add_effects, seg_del_effects, pnad,
                        ent_to_ent_sub, segment_param_option,
                        pnad_param_option, segment_option_objs,
                        tuple(pnad_option_vars), self._endogenous_processes)
                else:
                    assert set(sub.keys()) == set(pnad.op.parameters)
                pnad.add_to_datastore(
                    (segment, sub),
                    check_effect_equality=not self.get_name()
                    in ["cluster_and_search_process_learner"],
                    check_option_equality=not self.get_name()
                    in ["cluster_and_search_process_learner"])
            else:
                # Otherwise, create a new PNAD.
                objects = {o for atom in segment.add_effects |
                           segment.delete_effects for o in atom.objects} | \
                          set(segment_option_objs)

                if self.get_name() in [
                        "cluster_and_llm_select",
                        "cluster_and_search_process_learner",
                        "cluster_and_inverse_planning"
                ]:
                    # With cluster_and_llm_select, the param may include
                    # anything in the init atoms of the segment.
                    objects |= {
                        o
                        for atom in segment.init_atoms for o in atom.objects
                    }

                objects_lst = sorted(objects)
                params = utils.create_new_variables(
                    [o.type for o in objects_lst])
                preconds: Set[LiftedAtom] = set()  # will be learned later
                obj_to_var = dict(zip(objects_lst, params))
                var_to_obj = dict(zip(params, objects_lst))
                grd_add_effects = {
                    atom
                    for atom in segment.add_effects
                    if not isinstance(atom.predicate, DerivedPredicate)
                }
                grd_delete_effects = {
                    atom
                    for atom in segment.delete_effects
                    if not isinstance(atom.predicate, DerivedPredicate)
                }
                lfd_add_effects = {
                    atom.lift(obj_to_var)
                    for atom in grd_add_effects
                }
                lfd_delete_effects = {
                    atom.lift(obj_to_var)
                    for atom in grd_delete_effects
                }
                ignore_effects: Set[Predicate] = set()  # will be learned later
                if self.get_name() in ["cluster_and_search_process_learner"]:
                    # Remove atoms explained by endogenous processes
                    lfd_add_eff_union: Set[Union[LiftedAtom, GroundAtom]]
                    lfd_del_eff_union: Set[Union[LiftedAtom, GroundAtom]]
                    lfd_add_eff_union, lfd_del_eff_union = \
                        self.remove_atoms_explained_by_endogenous_processes(
                        segment, self._endogenous_processes,
                        cast(Set[Union[LiftedAtom, GroundAtom]],
                             lfd_add_effects),
                        cast(Set[Union[LiftedAtom, GroundAtom]],
                             lfd_delete_effects),
                        obj_to_var)
                    lfd_add_effects = cast(Set[LiftedAtom], lfd_add_eff_union)
                    lfd_delete_effects = cast(Set[LiftedAtom],
                                              lfd_del_eff_union)
                    remove_fn = (
                        self.remove_atoms_explained_by_endogenous_processes)
                    grd_add_eff_union: Set[Union[LiftedAtom, GroundAtom]]
                    grd_del_eff_union: Set[Union[LiftedAtom, GroundAtom]]
                    grd_add_eff_union, grd_del_eff_union = \
                        remove_fn(
                            segment,
                            self._endogenous_processes,
                            cast(Set[Union[LiftedAtom, GroundAtom]],
                                 grd_add_effects),
                            cast(Set[Union[LiftedAtom, GroundAtom]],
                                 grd_delete_effects))
                    grd_add_effects = cast(Set[GroundAtom], grd_add_eff_union)
                    grd_delete_effects = cast(Set[GroundAtom],
                                              grd_del_eff_union)

                    # ---- Single effect bias ----
                    if CFG.cluster_learning_one_effect_per_process:
                        #   If there are still processes with multiple effects,
                        #   add multiple PNAD here; after checking
                        #   such pnad don't
                        #   already exists.
                        for atom in grd_add_effects | grd_delete_effects:
                            neg_atom = atom.get_negated_atom()
                            if atom in grd_add_effects:
                                add_effect_set = frozenset({atom})
                                # Check if the negated atom is in the delete
                                # effects
                                if neg_atom in grd_delete_effects:
                                    del_effect_set = frozenset({neg_atom})
                                else:
                                    del_effect_set = frozenset()
                            else:
                                del_effect_set = frozenset({atom})
                                if neg_atom in grd_add_effects:
                                    add_effect_set = frozenset({neg_atom})
                                else:
                                    add_effect_set = frozenset()
                            # Check if the pnad already exists
                            result = self._unify_segment_with_pnads(
                                frozenset(), add_effect_set, del_effect_set,
                                segment_param_option, segment_option_objs,
                                pnads)
                            suc, ent_to_ent_sub, pnad = (
                                result  # type: ignore[misc]
                            )
                            if suc:
                                sub = cast(
                                    VarToObjSub,
                                    {v: o
                                     for o, v in ent_to_ent_sub.items()})
                                # Add to this PNAD.
                                if CFG.exogenous_process_learner_do_intersect:
                                    # Find the largest conditions that
                                    # unifies the init atoms of the segment
                                    # and another segment in the PNAD, and
                                    # add that segment+sub to the datastore.
                                    # Doing this sequentially ensures one of the
                                    # substitutions has the objects we
                                    # care about with intersection. Hence
                                    # it can fall out later in
                                    # `induce_preconditions_via_intersection`.
                                    (pnad_param_option,
                                     pnad_option_vars) = pnad.option_spec
                                    sub = self._find_best_segment_unification(
                                        segment, add_effect_set,
                                        del_effect_set, pnad, ent_to_ent_sub,
                                        segment_param_option,
                                        pnad_param_option, segment_option_objs,
                                        tuple(pnad_option_vars),
                                        self._endogenous_processes)
                                else:
                                    assert set(sub.keys()) == set(
                                        pnad.op.parameters)
                                pnad.add_to_datastore(
                                    (segment, sub),
                                    check_effect_equality=False,
                                    check_option_equality=False)
                            else:
                                add_effect_set = frozenset({
                                    atom.lift(obj_to_var)  # type: ignore[misc]
                                    for atom in add_effect_set
                                })
                                del_effect_set = frozenset({
                                    atom.lift(obj_to_var)  # type: ignore[misc]
                                    for atom in del_effect_set
                                })
                                # Create a new pnad with this atom
                                op = STRIPSOperator(
                                    f"Op{len(pnads)}",
                                    params,
                                    preconds,
                                    add_effect_set,  # type: ignore[arg-type]
                                    del_effect_set,  # type: ignore[arg-type]
                                    ignore_effects)
                                datastore = [(segment, var_to_obj)]
                                option_vars = [
                                    obj_to_var[o] for o in segment_option_objs
                                ]
                                option_spec = (segment_param_option,
                                               option_vars)
                                pnads.append(PNAD(op, datastore, option_spec))
                        continue
                op = STRIPSOperator(f"Op{len(pnads)}", params, preconds,
                                    lfd_add_effects, lfd_delete_effects,
                                    ignore_effects)
                datastore = [(segment, var_to_obj)]
                option_vars = [obj_to_var[o] for o in segment_option_objs]
                option_spec = (segment_param_option, option_vars)
                pnads.append(PNAD(op, datastore, option_spec))

        if self.get_name() in ["cluster_and_search_process_learner"]:
            # Do this extra step for this learner
            initial_segmenter_method = CFG.segmenter
            CFG.segmenter = "atom_changes"
            self._atom_change_segmented_trajs = [
                segment_trajectory(traj, self._predicates, verbose=False)
                for traj in self._trajectories
            ]
            CFG.segmenter = initial_segmenter_method
        # Learn the preconditions of the operators in the PNADs. This part
        # is flexible; subclasses choose how to implement it.
        pnads = self._learn_pnad_preconditions(pnads)

        # Handle optional postprocessing to learn ignore effects.
        pnads = self._postprocessing_learn_ignore_effects(pnads)

        # Log and return the PNADs.
        if self._verbose:
            logging.info("Learned operators (before option learning):")
            for pnad in pnads:
                logging.info(pnad)
        return pnads

    def _unify_segment_with_pnads(  # type: ignore[no-untyped-def]
            self, seg_preconds, seg_add_effects,
                                  seg_del_effects, seg_param_option,
                                  seg_option_objs, pnads: List[PNAD]) -> \
                                  Tuple[bool, VarToObjSub]:
        """Try to unify the segment with the PNADs."""
        for pnad in pnads:
            # Try to unify this transition with existing effects.
            # Note that both add and delete effects must unify,
            # and also the objects that are arguments to the options.
            (pnad_param_option, pnad_option_vars) = pnad.option_spec
            if self.get_name() not in [
                    "cluster_and_llm_select",
                    "cluster_and_search_process_learner",
                    "cluster_and_inverse_planning"
            ] or CFG.exogenous_process_learner_do_intersect:
                preconds2: FrozenSet = frozenset()  # no preconditions
            else:
                # Lifted
                obj_to_var = {v: k for k, v in pnad.datastore[-1][1].items()}
                preconds2 = frozenset({
                    atom.lift(obj_to_var)
                    for atom in pnad.datastore[-1][0].init_atoms
                })
            suc, ent_to_ent_sub = utils.unify_preconds_effects_options(
                seg_preconds, preconds2, seg_add_effects,
                frozenset(pnad.op.add_effects), seg_del_effects,
                frozenset(pnad.op.delete_effects), seg_param_option,
                pnad_param_option, seg_option_objs, tuple(pnad_option_vars))
            if suc:
                return True, ent_to_ent_sub, pnad  # type: ignore[return-value]
        return False, {}, None  # type: ignore[return-value]

    @staticmethod
    def _find_best_segment_unification(
            segment: Segment, seg_add_eff: FrozenSet[GroundAtom],
            seg_del_eff: FrozenSet[GroundAtom], pnad: PNAD,
            obj_to_var: Dict[Object, Variable],
            _segment_param_option: ParameterizedOption,
            _pnad_param_option: ParameterizedOption,
            _segment_option_objs: Tuple[Object, ...],
            _pnad_option_vars: Tuple[Variable, ...],
            endogenous_processes: List[EndogenousProcess]) -> VarToObjSub:
        """Try to unify and find the *best* set of matching init atoms between
        the given segment and the *last* segment in the PNAD's datastore, then
        return the resulting Var->Obj substitution.

        Prioritizes atoms involving effect variables to ensure critical
        atoms like SideOf(dest, source, direction) are preserved.
        """
        # ---------- 0) Gather init atoms (ground vs. lifted) ----------
        seg_init_atoms_full = set(segment.init_atoms)

        # The last segment in the PNAD's datastore and its variable mapping.
        last_seg, last_var_to_obj = pnad.datastore[-1]
        last_obj_to_var = {o: v for v, o in last_var_to_obj.items()}
        objects_in_last = set(last_obj_to_var)
        lifted_last_init_atoms = {
            atom.lift(last_obj_to_var)
            for atom in last_seg.init_atoms
            if all(o in objects_in_last for o in atom.objects)
        }

        # Identify effect variables for prioritization
        effect_vars = set()
        for atom in pnad.op.add_effects | pnad.op.delete_effects:
            effect_vars.update(atom.variables)

        # Identify critical ground objects from segment effects
        effect_objects = set()
        for atom in seg_add_eff | seg_del_eff:  # type: ignore[assignment]
            effect_objects.update(atom.objects)  # type: ignore[attr-defined]

        # Restrict to predicates shared between the two sides.
        common_preds = {a.predicate for a in seg_init_atoms_full} & \
                    {b.predicate for b in lifted_last_init_atoms}
        remove_ignore_atoms = True
        if remove_ignore_atoms:
            relevant_procs = [
                p for p in endogenous_processes
                if segment.get_option().parent == p.option
            ]
            for endo_proc in relevant_procs:
                common_preds -= endo_proc.ignore_effects

        seg_pre_list: List[GroundAtom] = sorted(
            [a for a in seg_init_atoms_full if a.predicate in common_preds],
            key=str,
        )
        pnad_pre_list: List[LiftedAtom] = sorted(
            [b for b in lifted_last_init_atoms if b.predicate in common_preds],
            key=str,
        )

        # Quick exits: nothing to match or no shared predicates.
        if not seg_pre_list or not pnad_pre_list:
            return cast(VarToObjSub, dict(
                (v, o) for o, v in obj_to_var.items()))

        # ---------- 1) Start from the mapping returned by effects+options -----
        current_map: Dict[_TypedEntity, Variable] = dict(obj_to_var.items())

        # Try to extend current_map with as many precondition
        # matches as possible.
        # Use weighted scoring that prioritizes effect-related atoms
        best_map: Dict[_TypedEntity, Variable] = {}
        best_map.update(current_map)
        best_score: float = 0.0  # Changed to float for weighted scoring

        # ---------- 2) Organize atoms by predicate for bounds & candidate searc
        idx_pnad_by_pred: Dict[Predicate, List[int]] = defaultdict(list)
        for j, b in enumerate(pnad_pre_list):
            idx_pnad_by_pred[b.predicate].append(j)

        # Compute atom weights based on involvement with effects
        def compute_atom_weight(ground_atom: GroundAtom,
                                lifted_atom: LiftedAtom) -> float:
            """Compute weight for matching this atom pair."""
            weight = 1.0  # Base weight

            # High priority for atoms involving effect objects/variables
            involves_effect_ground = any(obj in effect_objects
                                         for obj in ground_atom.objects)
            involves_effect_lifted = any(var in effect_vars
                                         for var in lifted_atom.variables)

            if involves_effect_ground and involves_effect_lifted:
                # Critical atoms like SideOf connecting source and dest
                if ground_atom.predicate.name == "SideOf":
                    # Check if it connects effect locations
                    if len(effect_objects.intersection(
                            ground_atom.objects)) >= 2:
                        weight = 100.0  # Highest priority
                    else:
                        weight = 10.0
                else:
                    weight = 5.0
            elif involves_effect_ground or involves_effect_lifted:
                weight = 2.0

            return weight

        # Upper bound helper with weighted scoring
        def weighted_upper_bound(seg_idxs: Set[int],
                                 pnad_unused: Set[int]) -> float:
            """Compute weighted upper bound on possible score."""
            bound = 0.0
            seg_by_pred = defaultdict(list)
            for i in seg_idxs:
                seg_by_pred[seg_pre_list[i].predicate].append(i)

            for pred, seg_indices in seg_by_pred.items():
                pnad_indices = [
                    j for j in idx_pnad_by_pred[pred] if j in pnad_unused
                ]
                # For each predicate, we can match at most min(seg_count,
                # pnad_count)
                max_matches = min(len(seg_indices), len(pnad_indices))
                if max_matches > 0:
                    # Use maximum possible weight for this predicate
                    max_weight = max(
                        compute_atom_weight(seg_pre_list[si],
                                            pnad_pre_list[pi])
                        for si in seg_indices[:max_matches]
                        for pi in pnad_indices[:max_matches]
                    ) if seg_indices and pnad_indices else 1.0
                    bound += max_matches * max_weight
            return bound

        # Compatibility check for a single (ground, lifted) atom pair
        def compatible_extension(
            a: GroundAtom, b: LiftedAtom, mapping: Dict[_TypedEntity, Variable]
        ) -> Optional[List[Tuple[_TypedEntity, Variable]]]:
            if a.predicate != b.predicate:
                return None
            new_pairs: List[Tuple[_TypedEntity, Variable]] = []
            inv = {v: k for k, v in mapping.items()}
            for obj_ent, var_ent in zip(a.entities, b.entities):
                # Types must match
                if obj_ent.type != var_ent.type:
                    return None
                # b side should be a Variable (usually), but handle if lifted
                # constant
                if isinstance(var_ent, Variable):
                    # mapping consistency: obj -> var one-to-one
                    if obj_ent in mapping:
                        if mapping[obj_ent] != var_ent:
                            return None
                    elif var_ent in inv:
                        if inv[var_ent] != obj_ent:
                            return None
                    else:
                        new_pairs.append((obj_ent, var_ent))
                else:
                    # If b side is a constant-typed entity, require equality
                    if obj_ent != var_ent:
                        return None
            return new_pairs

        def search(mapping: Dict[_TypedEntity, Variable], seg_left: Set[int],
                   pnad_unused: Set[int], score: float) -> None:
            nonlocal best_score, best_map

            # Upper bound pruning with weighted scoring
            ub = score + weighted_upper_bound(seg_left, pnad_unused)
            if ub <= best_score:
                return

            if not seg_left:
                if score > best_score:
                    best_score = score
                    best_map = dict(mapping)
                return

            # Choose next atom: prioritize high-weight atoms with few candidates
            best_i = None
            best_candidates: List[Tuple[int, List, float]] = []
            best_priority = -float('inf')

            for i in list(seg_left):
                a = seg_pre_list[i]
                candidates = []
                for j in idx_pnad_by_pred[a.predicate]:
                    if j not in pnad_unused:
                        continue
                    ext = compatible_extension(a, pnad_pre_list[j], mapping)
                    if ext is not None:
                        weight = compute_atom_weight(a, pnad_pre_list[j])
                        candidates.append((j, ext, weight))

                if not candidates:
                    # This atom cannot be matched; continue without it
                    seg_left_minus_i = set(seg_left)
                    seg_left_minus_i.remove(i)
                    search(mapping, seg_left_minus_i, pnad_unused, score)
                    return

                # Priority: high weight atoms with few candidates (more
                # constrained)
                max_weight = max(c[2] for c in candidates)
                priority = max_weight / (len(candidates) + 1
                                         )  # Favor constrained, high-weight

                if priority > best_priority:
                    best_i = i
                    best_candidates = candidates
                    best_priority = priority

            assert best_i is not None

            # Try candidates, ordered by weight (highest first)
            for j, ext_pairs, weight in sorted(best_candidates,
                                               key=lambda x: (-x[2], x[0])):
                # Apply extension
                for k, v in ext_pairs:
                    mapping[k] = v
                pnad_unused.remove(j)
                seg_left.remove(best_i)

                search(mapping, seg_left, pnad_unused, score + weight)

                # Revert
                seg_left.add(best_i)
                pnad_unused.add(j)
                for k, _ in ext_pairs:
                    try:
                        del mapping[k]
                    except KeyError:
                        pass

        # Run the weighted search
        search(dict(current_map), set(range(len(seg_pre_list))),
               set(range(len(pnad_pre_list))), 0.0)

        # Convert best map (Object->Variable) back to Var->Object for return
        sub = cast(VarToObjSub, {v: o for o, v in best_map.items()})
        return sub

    @staticmethod
    def remove_atoms_explained_by_endogenous_processes(
        segment: Segment,
        endogenous_processes: List[EndogenousProcess],
        add_effects: Set[Union[LiftedAtom, GroundAtom]],
        delete_effects: Set[Union[LiftedAtom, GroundAtom]],
        obj_to_var: Optional[Dict[Object, Variable]] = None
    ) -> Tuple[Set[Union[LiftedAtom, GroundAtom]], Set[Union[LiftedAtom,
                                                             GroundAtom]]]:
        """Remove effects explained by endogenous processes.

        If obj_to_var is None, operates on ground atoms; otherwise on
        lifted atoms.
        """
        process_lifted_atoms = bool(obj_to_var)
        objects = set(segment.states[0])
        seg_add_eff = segment.add_effects
        seg_del_eff = segment.delete_effects

        relevant_procs = [
            p for p in endogenous_processes
            if segment.get_option().parent == p.option
        ]
        for endo_proc in relevant_procs:
            if endo_proc.name == "Wait":
                continue
            add_effects = {
                a
                for a in add_effects
                if a.predicate not in endo_proc.ignore_effects
            }
            delete_effects = {
                a
                for a in delete_effects
                if a.predicate not in endo_proc.ignore_effects
            }
            var_to_obj = dict(
                zip(endo_proc.option_vars,
                    segment.get_option().objects))
            for g_proc in utils.all_ground_operators_given_partial(
                    endo_proc, objects, var_to_obj):  # type: ignore[arg-type]
                if g_proc.add_effects.issubset(seg_add_eff) and\
                    g_proc.delete_effects.issubset(seg_del_eff):
                    if process_lifted_atoms:
                        assert obj_to_var is not None
                        add_effects -= {
                            atom.lift(obj_to_var)
                            for atom in g_proc.add_effects
                        }
                        delete_effects -= {
                            atom.lift(obj_to_var)
                            for atom in g_proc.delete_effects
                        }
                    else:
                        add_effects -= g_proc.add_effects
                        delete_effects -= g_proc.delete_effects
                    # logging.debug(
                    #     f"Processing lifted atoms: {process_lifted_atoms}, "
                    #     f"Removed effects of {g_proc} \n from "
                    #     f"segment with \n add effect {seg_add_eff} "
                    #     f"and delete effect {seg_del_eff}\n"
                    # f"new add effects: {add_effects}, del effects:
                    # {delete_effects}")
        return add_effects, delete_effects

    @staticmethod
    def _get_top_candidates(
            candidates_with_scores: List, percentage: float,
            number: int) -> List[Tuple[float, Set[LiftedAtom]]]:
        assert percentage > 0 or number > 0, \
            "At least one of percentage or number must be greater than 0."
        n_candidates = len(candidates_with_scores)
        if percentage > 0:
            num_under_percentage = max(1,
                                       int(n_candidates * percentage / 100.0))
            score_at_threshold = candidates_with_scores[:num_under_percentage][
                -1][0]
            scores = [score for score, _ in candidates_with_scores]
            # Include all candidates with score_at_threshold
            position = bisect.bisect_right(scores, score_at_threshold)
            logging.info(
                f"Score threshold {score_at_threshold}; "
                f"Candidates under threshold: {position}/{n_candidates}")
        else:
            position = n_candidates

        # include at most top_n_candidates
        if number > 0:
            position = min(position, number)
        logging.debug(f"Returning {position}/{n_candidates} candidates:")
        num_to_log = 100
        for i, candidate in enumerate(candidates_with_scores[:num_to_log]):
            score, condition_candidate = candidate
            logging.debug(f"{i}: {condition_candidate}, Score: {score:.4f}")
        return candidates_with_scores[:position]

    def _get_top_consistent_conditions(self, initial_atom: Set[LiftedAtom],
                                       pnad: PNAD, method: str,
                                       seed: int) -> Iterator[Set[LiftedAtom]]:
        """Get the top consistent conditions for a PNAD."""
        exogenous_process = pnad.make_exogenous_process()
        logging.debug("For Process sketch:\n%s", exogenous_process)
        # pylint: disable=no-member
        score_fn = self.score_precondition_candidates  # type: ignore
        # pylint: enable=no-member
        candidates_with_scores = score_fn(exogenous_process, initial_atom,
                                          seed)

        if method == "top_p_percent":
            # Return top p% of candidates
            top_candidates = self._get_top_candidates(
                candidates_with_scores,
                CFG.cluster_process_learner_top_p_percent,
                CFG.cluster_process_learner_top_n_conditions)
            num_top = len(top_candidates)
            # Record the total number of candidates
            # pylint: disable=attribute-defined-outside-init,access-member-before-definition
            cnt = self._total_num_candidates  # type: ignore
            if cnt == 0:
                self._total_num_candidates = num_top  # type: ignore
            else:
                self._total_num_candidates = (  # type: ignore
                    cnt * num_top)
            # pylint: enable=attribute-defined-outside-init
        elif method == "top_n":
            # Return top n candidates
            n = CFG.cluster_process_learner_top_n_conditions
            top_candidates = candidates_with_scores[:n]
        else:
            raise NotImplementedError(
                f"Unknown top consistent method: {method}")

        # Yield the selected candidates
        for candidate in top_candidates:
            score, condition_candidate = candidate[0], candidate[1]
            logging.info(
                f"Selected condition: {condition_candidate}, Score: {score}")
            yield condition_candidate


class ClusterAndSearchProcessLearner(ClusteringProcessLearner):
    """ClusterAndSearchProcessLearner class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the process learner."""
        super().__init__(*args, **kwargs)
        self.proc_name_to_results: Dict[str, List[
            Tuple[float, FrozenSet[LiftedAtom], Tuple, ExogenousProcess]]] =\
                defaultdict(list)

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_search_process_learner"

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        """Learns preconditions for all PNADs.

        This implementation flattens the search for preconditions into a
        single multiprocessing pool. It supports an optional preliminary
        pruning step using a fast false-positive count metric to reduce
        the number of candidates that need to be scored with the more
        expensive data-likelihood metric.
        """
        cpu_cnt = self._determine_worker_count()
        use_parallel = (CFG.cluster_and_search_process_learner_parallel_pnad
                        and cpu_cnt > 1)

        logging.info(
            "Learning preconditions for %d PNADs "
            "using a flat parallel pool.", len(pnads))

        # Step 1: Generate candidate conditions
        (possible_atoms_per_pnad,
         condition_sets_per_pnad) = self._generate_candidate_conditions(pnads)

        # Step 2: Filter PNAD parameters
        pnads = self._filter_pnad_parameters(pnads, possible_atoms_per_pnad,
                                             condition_sets_per_pnad)

        # Step 2.5: Ablation - use top condition if flag is set
        if CFG.process_learner_ablate_bayes:
            logging.info("Using ablation: taking top condition "
                         "from condition_sets_per_pnad")
            best_conditions: Dict[int, FrozenSet[LiftedAtom]] = {}

            # Set up proc_name_to_results with placeholder values
            for i, pnad in enumerate(pnads):
                if (condition_sets_per_pnad is not None
                        and i < len(condition_sets_per_pnad)
                        and condition_sets_per_pnad[i]):
                    # Take the first (top) condition from condition_sets
                    best_condition = condition_sets_per_pnad[i][0]
                else:
                    # Fallback to empty condition if no condition sets available
                    best_condition = set()
                best_conditions[i] = best_condition  # type: ignore[assignment]

                # Create placeholder scored_conditions entry
                # for proc_name_to_results
                # Format: (cost, frozenset(condition), scores_tuple, process)
                placeholder_process = pnad.make_exogenous_process()
                placeholder_process.condition_at_start = best_condition.copy()
                placeholder_process.condition_overall = best_condition.copy()
                placeholder_scored_conditions = [
                    (0.0, frozenset(best_condition), (0.0, ),
                     placeholder_process)
                ]
                self.proc_name_to_results[
                    pnad.op.name] = placeholder_scored_conditions

            # Construct final PNADs with the top conditions
            return self._construct_final_pnads(best_conditions, pnads)

        # Step 3: Calculate candidate limits for CPU utilization
        min_candidates_to_keep = self._calculate_candidate_limits(
            possible_atoms_per_pnad, condition_sets_per_pnad, cpu_cnt)

        # Step 4: Generate final candidates with pruning
        final_candidates_for_pnad = \
            self._generate_final_candidates_with_pruning(
            pnads, possible_atoms_per_pnad, condition_sets_per_pnad,
            min_candidates_to_keep)

        # Step 5: Create work items for parallel scoring
        work_items = self._create_scoring_work_items(
            pnads, final_candidates_for_pnad)

        if not work_items:
            return []

        # Step 6: Execute parallel scoring
        start_time = time.time()
        logging.info(f"Scoring {len(work_items)} total conditions for "
                     f"{len(pnads)} PNADs using up to {cpu_cnt} workers.")
        logging.debug(f"Num vi steps: {CFG.cluster_and_search_vi_steps}, "
                      "Early stopping patience: "
                      f"{CFG.process_param_learning_patience}")

        if use_parallel:
            with Pool(nodes=min(len(work_items), cpu_cnt)) as pool:
                results = pool.map(_flat_pnad_scoring_worker, work_items)
        else:
            logging.info("Using sequential scoring as "
                         "alternative to parallel processing.")
            results = []
            for work_item in work_items:
                result = _flat_pnad_scoring_worker(work_item)
                results.append(result)

        logging.info(f"Finished scoring in {time.time() - start_time:.2f}s.")

        # Step 7: Process results and select best conditions
        best_conditions = self._process_scoring_results(
            results, final_candidates_for_pnad, pnads)

        # Step 8: Construct final PNADs
        return self._construct_final_pnads(best_conditions, pnads)

    def _generate_candidate_conditions(
        self, pnads: List[PNAD]
    ) -> Tuple[List[Set[LiftedAtom]], Optional[List[List[Set[LiftedAtom]]]]]:
        """Generate candidate conditions for PNADs using intersection or
        LLM."""
        possible_atoms_per_pnad = [
            self._induce_preconditions_via_intersection(pnad) for pnad in pnads
        ]

        if CFG.cluster_and_search_process_learner_llm_propose_top_conditions:
            condition_sets_per_pnad = self._llm_propose_condition_sets(
                possible_atoms_per_pnad,
                pnads,
                # batch_size=CFG.cluster_and_search_llm_propose_batch_size
            )
        elif CFG.cluster_and_search_process_learner_llm_rank_atoms:
            ranked_atoms_per_pnad = self._llm_rank_atoms(
                possible_atoms_per_pnad, pnads)
            possible_atoms_per_pnad = [
                set(atoms) for atoms in ranked_atoms_per_pnad
            ]
            condition_sets_per_pnad = None
        else:
            condition_sets_per_pnad = None

        return possible_atoms_per_pnad, condition_sets_per_pnad

    def _determine_worker_count(self) -> int:
        """Return number of worker processes to use based on config."""
        if CFG.process_learning_process_per_physical_core:
            return max(1, psutil.cpu_count(logical=False) - 1)
        return max(1, mp.cpu_count() - 1)  # pylint: disable=no-member

    def _build_process_descriptions(
        self,
        possible_atoms_per_pnad: List[Set[LiftedAtom]],
        pnads: Optional[List[PNAD]] = None
    ) -> List[Tuple[str, List[LiftedAtom]]]:
        """Build process descriptions for LLM prompts.

        Args:
            possible_atoms_per_pnad: List of sets of possible precondition atoms
            pnads: Optional list of PNADs to get effect information from

        Returns:
            List of (process_description, sorted_atoms) tuples
        """
        process_descriptions = []
        for i, poss_atoms in enumerate(possible_atoms_per_pnad):
            process_desc = f"Process {i}:\n"

            # Add effects information if PNADs are available
            if pnads and i < len(pnads):
                pnad = pnads[i]
                add_effects = pnad.op.add_effects
                delete_effects = pnad.op.delete_effects

                process_desc += "Add effects: "
                if add_effects:
                    process_desc += "(" + " ".join(
                        f"({str(atom)})" for atom in add_effects) + ")"
                else:
                    process_desc += "()"
                process_desc += "\n"

                process_desc += "Delete effects: "
                if delete_effects:
                    process_desc += "(" + " ".join(
                        f"({str(atom)})" for atom in delete_effects) + ")"
                else:
                    process_desc += "()"
                process_desc += "\n"

            # Add candidate atoms
            sorted_atoms = sorted(poss_atoms, key=str)
            process_desc += "Candidate atoms:\n"
            for j, atom in enumerate(sorted_atoms):
                process_desc += f"  {j}: {atom}\n"
            process_desc += "\n"

            process_descriptions.append((process_desc, sorted_atoms))

        return process_descriptions

    def _call_llm_with_template(self, template_path: str,
                                template_vars: Dict[str, Any],
                                debug_filename: str) -> str:
        """Call LLM with a template and save debug info.

        Args:
            template_path: Path to the prompt template file
            template_vars: Variables to substitute in template
            debug_filename: Name for debug output file

        Returns:
            LLM response text
        """
        if self._llm is None:
            raise ValueError("LLM not available")

        # Load the prompt template
        with open(template_path, "r", encoding='utf-8') as f:
            template = f.read()

        # Format the prompt
        prompt = template.format(**template_vars)

        # Get LLM response - use online_learning_cycle as seed if available
        seed = CFG.seed * 10 + self.online_learning_cycle if \
            self.online_learning_cycle is not None else CFG.seed
        response = self._llm.sample_completions(prompt,
                                                imgs=None,
                                                temperature=0.1,
                                                seed=seed)[0]

        # Save debug info
        with open(f"{CFG.log_file}/{debug_filename}", "w",
                  encoding='utf-8') as f:
            f.write(f"{prompt}\n=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*"
                    f"\n{response}")

        return response

    def _parse_llm_answer_block(self, response: str) -> Optional[str]:
        """Extract answer content from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Answer text or None if not found
        """
        answer_match = re.search(r'<answer>(.*?)</answer>', response,
                                 re.DOTALL)
        if not answer_match:
            return None
        return answer_match.group(1).strip()

    def _llm_rank_atoms(
            self,
            possible_atoms_per_pnad: List[Set[LiftedAtom]],
            pnads: Optional[List[PNAD]] = None,
            max_atoms: Optional[int] = None) -> List[List[LiftedAtom]]:
        """Rank the possible atoms by their likelihood of being
        relevant/necessary for the PNAD's effects.

        Args:
            possible_atoms_per_pnad: List of sets of possible
                precondition atoms, one set per PNAD
            pnads: Optional list of PNADs to get effect information from

        Returns:
            List of lists of ranked atoms, keeping only
            the most relevant ones based on LLM assessment
        """
        if not possible_atoms_per_pnad or self._llm is None:
            return [list(atoms) for atoms in possible_atoms_per_pnad]

        try:
            # Build process descriptions
            process_descriptions = self._build_process_descriptions(
                possible_atoms_per_pnad, pnads)

            # Call LLM with template
            template_path = (utils.get_path_to_predicators_root() +
                             "/predicators/nsrt_learning/strips_learning/" +
                             "llm_op_learning_prompts/atom_ranking.prompt")
            all_descriptions = "\n".join(
                [desc for desc, _ in process_descriptions])
            template_vars = {
                "PROCESS_EFFECTS_AND_CANDIDATES": all_descriptions
            }
            response = self._call_llm_with_template(
                template_path, template_vars, "atom_ranking_response.txt")

            # Parse the response
            answer_text = self._parse_llm_answer_block(response)
            if not answer_text:
                logging.warning("LLM failed to provide properly formatted "
                                "answer for atom ranking")
                return [list(atoms) for atoms in possible_atoms_per_pnad]
            lines = [
                line.strip() for line in answer_text.split('\n')
                if line.strip()
            ]

            # Parse rankings for each process
            ranked_atoms_per_pnad = []
            for i, (_, sorted_atoms) in enumerate(process_descriptions):
                # Find the line for this process
                process_line = None
                for line in lines:
                    if line.startswith(f"Process {i}:"):
                        process_line = line
                        break

                if process_line is None:
                    logging.warning(
                        f"No ranking found for process {i}, keeping all atoms")
                    ranked_atoms_per_pnad.append(list(sorted_atoms))
                    continue

                # Extract indices after the colon
                try:
                    indices_str = process_line.split(':', 1)[1].strip()
                    if indices_str:
                        indices = [
                            int(idx.strip()) for idx in indices_str.split(',')
                        ]
                        # Filter valid indices and get corresponding atoms
                        valid_indices = [
                            idx for idx in indices
                            if 0 <= idx < len(sorted_atoms)
                        ]
                        if valid_indices:
                            # Keep atoms in the order specified by LLM ranking
                            # But limit to top N atoms to avoid combinatorial
                            # explosion
                            if max_atoms is None:
                                max_atoms = len(valid_indices)
                            else:
                                max_atoms = min(max_atoms, len(valid_indices))
                            selected_atoms = [
                                sorted_atoms[idx]
                                for idx in valid_indices[:max_atoms]
                            ]
                            ranked_atoms_per_pnad.append(selected_atoms)
                        else:
                            # No valid indices, keep original list
                            ranked_atoms_per_pnad.append(list(sorted_atoms))
                    else:
                        # Empty ranking, keep original list
                        ranked_atoms_per_pnad.append(list(sorted_atoms))
                except (ValueError, IndexError) as e:
                    logging.warning(
                        f"Failed to parse ranking for process {i}: {e}")
                    ranked_atoms_per_pnad.append(list(sorted_atoms))

            # Log the results
            for i, ranked in enumerate(ranked_atoms_per_pnad):
                original = list(process_descriptions[i][1])
                logging.info(
                    f"Process {i}: Kept {len(ranked)}/{len(original)} atoms")
                logging.debug(f"  Kept atoms: {sorted(ranked, key=str)}")
                logging.debug("  Removed atoms: %s",
                              sorted(set(original) - set(ranked), key=str))

            return ranked_atoms_per_pnad

        except Exception as e:  # pylint: disable=broad-except
            logging.warning(
                f"LLM atom ranking failed: {e}, keeping original atoms")
            return [list(atoms) for atoms in possible_atoms_per_pnad]

    def _llm_propose_condition_sets(
            self,
            possible_atoms_per_pnad: List[Set[LiftedAtom]],
            pnads: Optional[List[PNAD]] = None,
            k: Optional[int] = None,
            batch_size: Optional[int] = None) -> List[List[Set[LiftedAtom]]]:
        """Propose top k condition sets for each PNAD using LLM.

        Args:
            possible_atoms_per_pnad: List of sets of possible
                precondition atoms, one set per PNAD
            pnads: Optional list of PNADs to get effect information from
            k: Number of condition sets to propose per PNAD
            batch_size: Maximum number of PNADs to process in each LLM call

        Returns:
            List of lists of condition sets, where each
            condition set is a set of atoms
        """
        if not possible_atoms_per_pnad or self._llm is None:
            return [[set(atoms)] for atoms in possible_atoms_per_pnad]

        if k is None:
            k = CFG.process_learner_llm_propose_conditions_k

        # If batch_size is not specified or if we have fewer
        # PNADs than the limit,
        # process all at once (original behavior)
        if (batch_size is None or len(possible_atoms_per_pnad) <= batch_size):
            return self._llm_propose_condition_sets_batch(
                possible_atoms_per_pnad, pnads, k)

        # Otherwise, process in batches
        all_condition_sets = []
        num_pnads = len(possible_atoms_per_pnad)

        for start_idx in range(0, num_pnads, batch_size):
            end_idx = min(start_idx + batch_size, num_pnads)

            # Extract batch data
            batch_atoms = possible_atoms_per_pnad[start_idx:end_idx]
            batch_pnads = pnads[start_idx:end_idx] if pnads else None

            # Process this batch
            batch_condition_sets = self._llm_propose_condition_sets_batch(
                batch_atoms, batch_pnads, k, batch_idx=start_idx // batch_size)

            all_condition_sets.extend(batch_condition_sets)

        return all_condition_sets

    def _llm_propose_condition_sets_batch(
            self,
            possible_atoms_per_pnad: List[Set[LiftedAtom]],
            pnads: Optional[List[PNAD]] = None,
            _k: Optional[int] = None,
            batch_idx: Optional[int] = None) -> List[List[Set[LiftedAtom]]]:
        """Process a batch of PNADs for condition set proposal."""
        try:
            # Build process descriptions
            process_descriptions = self._build_process_descriptions(
                possible_atoms_per_pnad, pnads)

            # Extract unique predicates from all candidate atoms
            all_predicates = set()
            for poss_atoms in possible_atoms_per_pnad:
                for atom in poss_atoms:
                    all_predicates.add(atom.predicate)

            # Create predicate listing string
            predicate_listing = "\n".join(
                predicate.pretty_str_with_assertion()
                for predicate in sorted(all_predicates, key=lambda p: p.name))

            # Call LLM with template
            template_path = (
                utils.get_path_to_predicators_root() +
                "/predicators/nsrt_learning/strips_learning/" +
                "llm_op_learning_prompts/condition_set_proposal.prompt")
            all_descriptions = "\n".join(
                [desc for desc, _ in process_descriptions])
            template_vars = {
                "PROCESS_EFFECTS_AND_CANDIDATES": all_descriptions,
                "PREDICATE_LISTING": predicate_listing,
                # "K": k
            }
            response = self._call_llm_with_template(
                template_path, template_vars,
                "condition_set_proposal_response_"\
                f"{self.online_learning_cycle}_{batch_idx}.txt")

            # Parse the response
            answer_text = self._parse_llm_answer_block(response)
            if not answer_text:
                logging.warning("LLM failed to provide properly formatted "
                                "answer for condition set proposal")
                return [[set(atoms)] for atoms in possible_atoms_per_pnad]
            lines = [
                line.strip() for line in answer_text.split('\n')
                if line.strip()
            ]

            # Parse condition sets for each process
            condition_sets_per_pnad = []
            for i, (_, sorted_atoms) in enumerate(process_descriptions):
                # Find lines for this process
                process_sets = []
                process_found = False

                for line in lines:
                    if line.startswith(f"Process {i}:"):
                        process_found = True
                        continue
                    if process_found and line.startswith("Process "):
                        # Start of next process, break
                        break
                    if process_found and line.startswith("Set "):
                        # Parse set line: "Set 1: [2,0,4]"
                        try:
                            set_part = line.split(":", 1)[1].strip()
                            # Remove brackets and split by comma
                            set_part = set_part.strip("[]")
                            if set_part:
                                indices = [
                                    int(idx.strip())
                                    for idx in set_part.split(',')
                                ]
                                # Filter valid indices and get corresponding
                                # atoms
                                valid_indices = [
                                    idx for idx in indices
                                    if 0 <= idx < len(sorted_atoms)
                                ]
                                if valid_indices:
                                    condition_set = {
                                        sorted_atoms[idx]
                                        for idx in valid_indices
                                    }
                                    process_sets.append(condition_set)
                        except (ValueError, IndexError) as e:
                            logging.warning("Failed to parse condition set "
                                            f"for process {i}: {e}")

                if not process_sets:
                    # No valid sets found, use original atoms as single set
                    process_sets.append(set(sorted_atoms))

                condition_sets_per_pnad.append(process_sets)

            # Log the results
            for i, sets in enumerate(condition_sets_per_pnad):
                if pnads:
                    logging.debug(f"Process {i}: {pformat(pnads[i])}\n"
                                  f"Proposed {len(sets)} condition sets")
                else:
                    logging.debug(
                        f"Process {i}: Proposed {len(sets)} condition sets")
                for j, condition_set in enumerate(sets):
                    logging.debug(
                        f"  Set {j+1}: {sorted(condition_set, key=str)}")

            return condition_sets_per_pnad

        except Exception as e:  # pylint: disable=broad-except
            logging.warning(
                f"LLM condition set proposal failed: {e}, using original atoms"
            )
            return [[set(atoms)] for atoms in possible_atoms_per_pnad]

    def _filter_pnad_parameters(
        self, pnads: List[PNAD],
        possible_atoms_per_pnad: List[Set[LiftedAtom]],
        condition_sets_per_pnad: Optional[List[List[Set[LiftedAtom]]]]
    ) -> List[PNAD]:
        """Filter PNAD parameters to only include variables used in
        preconditions or effects."""
        filtered_pnads: List[PNAD] = []
        for i, (pnad,
                poss_atoms) in enumerate(zip(pnads, possible_atoms_per_pnad)):
            if condition_sets_per_pnad is not None:
                poss_atoms = set.union(*condition_sets_per_pnad[i])
            eff_atoms = pnad.op.add_effects | pnad.op.delete_effects
            used_vars = {
                v
                for atom in (poss_atoms | eff_atoms) for v in atom.variables
            }
            if not used_vars:
                filtered_pnads.append(pnad)
                continue
            new_params = [p for p in pnad.op.parameters if p in used_vars]
            if list(pnad.op.parameters) == new_params:
                filtered_pnads.append(pnad)
                continue
            new_op = pnad.op.copy_with(parameters=new_params)
            filtered_pnads.append(
                PNAD(new_op, pnad.datastore, pnad.option_spec))
        return filtered_pnads

    def _calculate_candidate_limits(
            self, possible_atoms_per_pnad: List[Set[LiftedAtom]],
            condition_sets_per_pnad: Optional[List[List[Set[LiftedAtom]]]],
            cpu_cnt: int) -> int:
        """Calculate optimal candidate limits per PNAD to utilize available
        CPUs."""
        max_candidates_per_pnad = [
            2**len(possible_atoms)
            for possible_atoms in possible_atoms_per_pnad
        ]
        if condition_sets_per_pnad is not None:
            max_candidates_per_pnad = [
                len(condition_sets)
                for condition_sets in condition_sets_per_pnad
            ]
        max_candidates_across_pnads = min(max(max_candidates_per_pnad),
                                          cpu_cnt)
        min_candidates_to_keep = 1

        for i in range(max_candidates_across_pnads, 0, -1):
            total_candidates = sum(
                min(num, i) for num in max_candidates_per_pnad)
            if total_candidates <= cpu_cnt:
                logging.info(
                    "Setting candidate cap per PNAD "
                    "to %d to utilize %d CPUs "
                    "(total candidates: %d).", i, cpu_cnt, total_candidates)
                min_candidates_to_keep = i
                break
        return min_candidates_to_keep

    def _generate_final_candidates_with_pruning(
            self, pnads: List[PNAD],
            possible_atoms_per_pnad: List[Set[LiftedAtom]],
            condition_sets_per_pnad: Optional[List[List[Set[LiftedAtom]]]],
            min_candidates_to_keep: int) -> Dict[int, List[Set[LiftedAtom]]]:
        """Generate final candidates with optional false positive pruning."""
        final_candidates_for_pnad: Dict[int, List[Set[LiftedAtom]]] = {}
        indexed_pnads = dict(enumerate(pnads))

        fp_count_pruning = (
            CFG.process_scoring_method == 'data_likelihood'
            and CFG.process_condition_search_prune_with_fp_count and not CFG.
            cluster_and_search_process_learner_llm_propose_top_conditions)

        def _initial_lifted_atoms_for_index(idx: int,
                                            p: PNAD) -> Set[LiftedAtom]:
            if CFG.exogenous_process_learner_do_intersect:
                return possible_atoms_per_pnad[idx]
            init_ground_atoms = p.datastore[0][0].init_atoms
            var_to_obj = p.datastore[0][1]
            obj_to_var = {v: k for k, v in var_to_obj.items()}
            return {atom.lift(obj_to_var) for atom in init_ground_atoms}

        for i, pnad in indexed_pnads.items():
            initial_lift_atoms = _initial_lifted_atoms_for_index(i, pnad)

            if (condition_sets_per_pnad is not None
                    and i < len(condition_sets_per_pnad)):
                all_candidates = condition_sets_per_pnad[i]
            else:
                all_candidates = list(utils.all_subsets(initial_lift_atoms))

            if not all_candidates:
                final_candidates_for_pnad[i] = []
                continue

            if fp_count_pruning:
                pruned_candidates = self._prune_candidates_with_fp_count(
                    pnad, all_candidates, min_candidates_to_keep, i)
                final_candidates_for_pnad[i] = pruned_candidates
            else:
                final_candidates_for_pnad[
                    i] = all_candidates[:min_candidates_to_keep]

        return final_candidates_for_pnad

    def _prune_candidates_with_fp_count(
            self, pnad: PNAD, all_candidates: List[Set[LiftedAtom]],
            min_candidates_to_keep: int,
            pnad_idx: int) -> List[Set[LiftedAtom]]:
        """Prune candidates using false positive count metric."""
        base_process = pnad.make_exogenous_process()
        logging.debug("Pruning %d candidates for PNAD %d:\n%s",
                      len(all_candidates), pnad_idx, base_process)
        candidates_with_approx_scores = []
        for candidate in all_candidates:
            base_process.condition_at_start = candidate
            base_process.condition_overall = candidate
            complexity_penalty = (
                CFG.process_condition_search_complexity_weight *
                len(candidate))
            fp_fn = self._get_false_positive_states_from_seg_trajs
            false_positive_states = fp_fn(self._atom_change_segmented_trajs,
                                          [base_process])
            num_false_positives = sum(
                len(s) for s in false_positive_states.values())
            cost = num_false_positives + complexity_penalty
            candidates_with_approx_scores.append((cost, candidate))

        candidates_with_approx_scores.sort(key=lambda x: x[0])
        top_candidates = self._get_top_candidates(
            candidates_with_approx_scores,
            percentage=0,
            number=min_candidates_to_keep)
        pruned_candidates = [cand for _, cand in top_candidates]

        logging.debug("Pruned to %d candidates for PNAD %d.",
                      len(pruned_candidates), pnad_idx)

        return pruned_candidates

    def _create_scoring_work_items(
            self, pnads: List[PNAD],
            final_candidates_for_pnad: Dict[int,
                                            List[Set[LiftedAtom]]]) -> List:
        """Create work items for parallel scoring."""
        load_dir, save_dir = None, None
        if (self.online_learning_cycle is not None
                and CFG.process_learning_init_at_previous_results):
            load_save_dir = os.path.join(CFG.approach_dir,
                                         utils.get_config_path_str())
            load_dir = os.path.join(
                load_save_dir, f"online_cycle_{self.online_learning_cycle-1}")
            save_dir = os.path.join(
                load_save_dir, f"online_cycle_{self.online_learning_cycle}")

        indexed_pnads = dict(enumerate(pnads))
        work_items = []

        for i, pnad in indexed_pnads.items():
            base_process = pnad.make_exogenous_process()
            for condition_idx, condition in enumerate(
                    final_candidates_for_pnad[i]):
                item = (i, condition_idx, copy.deepcopy(base_process),
                        condition, self._trajectories, self._predicates,
                        CFG.seed, CFG.cluster_and_search_vi_steps,
                        CFG.process_condition_search_complexity_weight,
                        load_dir, save_dir,
                        CFG.process_param_learning_patience)
                work_items.append(item)

        return work_items

    def _process_scoring_results(
            self, results: List,
            final_candidates_for_pnad: Dict[int, List[Set[LiftedAtom]]],
            pnads: List[PNAD]) -> Dict[int, FrozenSet[LiftedAtom]]:
        """Process parallel scoring results and select best conditions."""
        indexed_pnads = dict(enumerate(pnads))
        pnad_scores: Dict[int,
                          List[Tuple[float, FrozenSet[LiftedAtom], Tuple[float,
                                                                         ...],
                                     ExogenousProcess]]] = defaultdict(list)

        for pnad_idx, condition_idx, cost, _, scores_tuple, process in results:
            original_condition = final_candidates_for_pnad[pnad_idx][
                condition_idx]
            process.condition_at_start = original_condition.copy()
            process.condition_overall = original_condition.copy()
            pnad_scores[pnad_idx].append(
                (cost, frozenset(original_condition), scores_tuple, process))

        best_conditions: Dict[int, FrozenSet[LiftedAtom]] = {}
        for pnad_idx, scored_conditions in pnad_scores.items():
            scored_conditions.sort(key=lambda x: x[0])
            self.proc_name_to_results[
                indexed_pnads[pnad_idx].op.name] = scored_conditions

            self._log_scored_conditions(pnad_idx, scored_conditions,
                                        indexed_pnads[pnad_idx])
            best_condition = self._select_best_condition(
                pnad_idx, scored_conditions, indexed_pnads[pnad_idx])
            best_conditions[pnad_idx] = best_condition
            logging.info(f"Selected best condition {best_condition}")

        return best_conditions

    def _log_scored_conditions(self, pnad_idx: int, scored_conditions: List,
                               pnad: PNAD) -> None:
        """Log the scored conditions for debugging."""
        logging.debug("Scored conditions for Process sketch "
                      "%d:\n%s", pnad_idx, pnad.make_exogenous_process())
        for rank, result in enumerate(scored_conditions):
            cost, condition_candidate, scores, process = result
            params = process._get_parameters()  # pylint: disable=protected-access
            process_param_str = ", ".join([f"{v:.4f}" for v in params])
            logging.debug(f"Conditions {rank}: "
                          f"{sorted(condition_candidate)}, "
                          f"Cost: {cost}, "
                          f"ELBO: {scores[0]:.4f}, "
                          f"Exp_state_prob: {scores[1]:.4f}, "
                          f"Exp_delay_prob: {scores[2]:.4f}, "
                          f"Entropy: {scores[3]:.4f}, "
                          f"Process params: {process_param_str}")

    def _select_best_condition(self, _pnad_idx: int, scored_conditions: List,
                               pnad: PNAD) -> FrozenSet[LiftedAtom]:
        """Select the best condition from scored candidates."""
        multiple_top_conditions = False
        best_ll = scored_conditions[0][2][0]
        num_top_conditions = len(
            list(
                itertools.takewhile(lambda x: x[2][0] == best_ll,
                                    scored_conditions)))
        if num_top_conditions > 1:
            multiple_top_conditions = True

        if (CFG.cluster_and_search_process_learner_llm_select_condition
                and multiple_top_conditions):
            best_condition = self._prompt_llm_to_select_from_top_conditions(
                pnad, scored_conditions[:num_top_conditions])
        else:
            _, best_condition, _, _ = scored_conditions[0]

        return best_condition  # type: ignore[return-value]

    def _construct_final_pnads(self,
                               best_conditions: Dict[int,
                                                     FrozenSet[LiftedAtom]],
                               pnads: List[PNAD]) -> List[PNAD]:
        """Construct the final unique PNADs with learned preconditions."""
        indexed_pnads = dict(enumerate(pnads))
        final_pnads: List[PNAD] = []

        for pnad_idx in sorted(best_conditions.keys()):
            cond_at_start = best_conditions[pnad_idx]
            base_pnad = indexed_pnads[pnad_idx]
            add_eff = base_pnad.op.add_effects
            del_eff = base_pnad.op.delete_effects
            new_params = {
                v
                for atom in cond_at_start | add_eff | del_eff
                for v in atom.variables
            }

            if self._is_unique_pnad(cond_at_start, base_pnad, final_pnads):
                final_pnads.append(
                    PNAD(
                        base_pnad.op.copy_with(preconditions=cond_at_start,
                                               parameters=new_params),
                        base_pnad.datastore, base_pnad.option_spec))

        return final_pnads

    def _is_unique_pnad(self, precon: FrozenSet[LiftedAtom], pnad: PNAD,
                        final_pnads: List[PNAD]) -> bool:
        """Check if a PNAD with given preconditions is unique."""
        for final_pnad in final_pnads:
            # Quick size checks first for efficiency
            if (len(precon) != len(final_pnad.op.preconditions) or
                    len(pnad.op.add_effects) != len(final_pnad.op.add_effects)
                    or len(pnad.op.delete_effects) != len(
                        final_pnad.op.delete_effects)):
                continue

            suc, _ = utils.unify_preconds_effects_options(
                frozenset(precon),
                frozenset(final_pnad.op.preconditions),
                frozenset(pnad.op.add_effects),
                frozenset(final_pnad.op.add_effects),
                frozenset(pnad.op.delete_effects),
                frozenset(final_pnad.op.delete_effects),
                pnad.option_spec[0],
                final_pnad.option_spec[0],
                tuple(pnad.option_spec[1]),
                tuple(final_pnad.option_spec[1]),
            )
            if suc:
                return False
        return True

    def _prompt_llm_to_select_from_top_conditions(
        self, pnad: PNAD, scored_conditions: List[Tuple[float,
                                                        FrozenSet[LiftedAtom],
                                                        Tuple, CausalProcess]]
    ) -> Set[LiftedAtom]:
        """Use the LLM to select the best condition from the top scored
        conditions for a PNAD."""
        assert self._llm is not None
        # 1. Load the prompt template.
        prompt_file = utils.get_path_to_predicators_root() + \
            "/predicators/nsrt_learning/strips_learning/" + \
            "llm_op_learning_prompts/"+\
            "cluster_and_search_process_learner_condition_select.prompt"
        with open(prompt_file, "r", encoding='utf-8') as f:
            # pylint: disable-next=attribute-defined-outside-init
            self.template = f.read()

        # 2. Fill the prompt template.
        proc = pnad.make_exogenous_process()
        proc_str = proc._str_wo_params  # pylint: disable=protected-access
        prompt = self.template.format(
            EXOGENOUS_PROCESS_SKETCH=proc_str,
            TOP_SCORING_CONDITIONS="\n".join(
                f"Conditions {i}: {sorted(condition)}"
                for i, (_, condition, _, _) in enumerate(scored_conditions)))

        # 3. Prompt the LLM.
        response = self._llm.sample_completions(prompt,
                                                imgs=None,
                                                temperature=0,
                                                seed=CFG.seed)[0]

        # Save the prompt and response for debugging
        with open(f"{CFG.log_file}/pnad_{pnad.op.name}_cond_select.txt",
                  "w",
                  encoding='utf-8') as f:
            f.write(f"{prompt}\n=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*\n"
                    f"{response}")

        # 4. Parse the answer.
        indices_str = re.findall(r"<answer>(.*?)</answer>", response)
        if indices_str:
            try:
                selected_idx = int(indices_str[0].strip())
                if 0 <= selected_idx < len(scored_conditions):
                    # The condition is the second element of the tuple.
                    _, best_condition, _, _ = scored_conditions[selected_idx]
                    return set(best_condition)
            except (ValueError, IndexError):
                # If parsing fails or index is out of bounds, fall back.
                logging.warning("LLM response parsing failed or index out of "
                                "bounds.")

        # Fallback: if LLM fails to produce a valid choice, pick the best one.
        logging.warning("LLM failed to select a condition, picking the best.")
        _, best_condition, _, _ = scored_conditions[0]
        return set(best_condition)


class ClusterAndInversePlanningProcessLearner(ClusteringProcessLearner):
    """ClusterAndInversePlanningProcessLearner class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # pylint: disable-next=import-outside-toplevel
        from predicators.predicate_search_score_functions import \
            _ExpectedNodesScoreFunction
        self._get_optimality_prob =\
            _ExpectedNodesScoreFunction._get_refinement_prob

        self._option_change_segmented_trajs: List[List[Segment]] = []
        self._demo_atoms_sequences: List[List[Set[LiftedAtom]]] = []
        self._total_num_candidates = 0

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_inverse_planning"

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        """Find the set of PNADs (with corresponding processes) that allows the
        agent make similar plans as the demonstrated/successful plans."""

        self._total_num_candidates = 0
        # --- Existing exogenous processes ---
        exogenous_process = [pnad.make_exogenous_process() for pnad in pnads]

        # Get the segmented trajectories for scoring the processes.
        initial_segmenter_method = CFG.segmenter
        CFG.segmenter = "atom_changes"
        self._atom_change_segmented_trajs = [
            segment_trajectory(traj, self._predicates, verbose=False)
            for traj in self._trajectories
        ]
        CFG.segmenter = "option_changes"
        self._option_change_segmented_trajs = [
            segment_trajectory(traj, self._predicates, verbose=False)
            for traj in self._trajectories
        ]
        CFG.segmenter = initial_segmenter_method
        self._demo_atoms_sequences = [
            utils.segment_trajectory_to_atoms_sequence(
                seg_traj)  # type: ignore[misc]
            for seg_traj in self._option_change_segmented_trajs
        ]
        # for i, seg_traj in enumerate(self._atom_change_segmented_trajs):
        #     logging.info(f"atom change trajectory {i}: {pformat(seg_traj)}")

        # --- Get the candidate preconditions ---
        # First option. Candidates are all possible subsets.
        conditions_at_start = []
        for pnad in pnads:
            if CFG.exogenous_process_learner_do_intersect:
                init_lift_atoms = self._induce_preconditions_via_intersection(
                    pnad)
            else:
                init_ground_atoms = pnad.datastore[0][0].init_atoms
                var_to_obj = pnad.datastore[0][1]
                obj_to_var = {v: k for k, v in var_to_obj.items()}
                init_lift_atoms = set(
                    atom.lift(obj_to_var) for atom in init_ground_atoms)

            if CFG.cluster_and_inverse_planning_candidates == "all":
                # 4 PNADS, with 7, 6, 7, 8 init atoms, possible combinations are
                # - 2^7 * 2^6 * 2^7 * 2^8 = 2^28 = 268,435,456
                # - 2^10 * 2^10 * 2^10 * 2^10 = 2^40 = 1,099,511,627,776
                # Get the initial conditions of the PNAD
                conditions_at_start.append(utils.all_subsets(init_lift_atoms))
            elif CFG.cluster_and_inverse_planning_candidates \
                    == "top_consistent":
                conditions_at_start.append(
                    self._get_top_consistent_conditions(
                        init_lift_atoms, pnad,
                        CFG.cluster_and_inverse_planning_top_consistent_method,
                        CFG.seed))
            else:
                raise NotImplementedError

        # --- Search for the best combination of preconditions ---
        best_cost = float("inf")
        best_conditions = []
        # Score all combinations of preconditions
        for i, combination in enumerate(
                itertools.product(*conditions_at_start)):
            # Set the conditions for each process
            for process, conditions in zip(exogenous_process, combination):
                process.condition_at_start = conditions
                process.condition_overall = conditions

            # Score this set of processes
            cost = self.compute_processes_score(set(exogenous_process))
            if cost < best_cost:
                best_cost = cost
                best_conditions = combination
            logging.debug("Combination %d/%d: cost = %s,"
                          " Best cost = %s", i + 1, self._total_num_candidates,
                          cost, best_cost)

        # --- Create new PNADs with the best conditions ---
        final_pnads: List[PNAD] = []
        for pnad, conditions in zip(pnads, best_conditions):
            # Check if this PNAD is unique
            for final_pnad in final_pnads:
                suc, _ = utils.unify_preconds_effects_options(
                    frozenset(conditions),
                    frozenset(final_pnad.op.preconditions),
                    frozenset(pnad.op.add_effects),
                    frozenset(final_pnad.op.add_effects),
                    frozenset(pnad.op.delete_effects),
                    frozenset(final_pnad.op.delete_effects),
                    pnad.option_spec[0],
                    final_pnad.option_spec[0],
                    tuple(pnad.option_spec[1]),
                    tuple(final_pnad.option_spec[1]),
                )
                if suc:
                    # Future: merge datastores if they are the same
                    break
            else:
                # If we reach here, it means the PNAD is unique
                # and we can add it to the final list
                new_pnad = PNAD(pnad.op.copy_with(preconditions=conditions),
                                pnad.datastore, pnad.option_spec)
                final_pnads.append(new_pnad)
        return final_pnads

    def compute_processes_score(
            self, exogenous_processes: Set[ExogenousProcess]) -> float:
        """Score the PNAD based on how well it allows the agent to make
        plans."""
        # Future: also incorporate number of nodes expanded
        cost = 0.0
        for i, traj in enumerate(self._trajectories):
            if not traj.is_demo:
                continue
            demo_atoms_sequence = self._demo_atoms_sequences[i]
            task = self._train_tasks[traj.train_task_idx]
            generator = task_plan_with_processes(
                task,
                self._predicates,
                exogenous_processes | self._endogenous_processes,
                CFG.seed,
                CFG.grammar_search_task_planning_timeout,
                # max_skeletons_optimized=CFG.sesame_max_skeletons_optimized,
                max_skeletons_optimized=1,
                use_visited_state_set=True)

            optimality_prob = 0.0
            try:
                for (_, plan_atoms_sequence, _metrics) in generator:
                    optimality_prob = self._get_optimality_prob(
                        demo_atoms_sequence,  # type: ignore[arg-type]
                        plan_atoms_sequence)
            except (PlanningTimeout, PlanningFailure):
                pass
            # low_quality_prob = 1.0 - optimality_prob
            cost += (1 - optimality_prob)  # * num_nodes

        return cost


class ClusterAndSearchSTRIPSLearner(ClusteringSTRIPSLearner):
    """A clustering STRIPS learner that learns preconditions via search,
    following the LOFT algorithm: https://arxiv.org/abs/2103.00589."""

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        new_pnads = []
        for i, pnad in enumerate(pnads):
            positive_data = pnad.datastore
            # Construct negative data by merging the datastores of all
            # other PNADs that have the same option.
            negative_data = []
            for j, other_pnad in enumerate(pnads):
                if i == j:
                    continue
                if pnad.option_spec[0] != other_pnad.option_spec[0]:
                    continue
                negative_data.extend(other_pnad.datastore)
            # Run the top-level search to find sets of precondition sets. This
            # also produces datastores, letting us avoid making a potentially
            # expensive call to recompute_datastores_from_segments().
            all_preconditions_to_datastores = self._run_outer_search(
                pnad, positive_data, negative_data)
            for j, preconditions in enumerate(all_preconditions_to_datastores):
                datastore = all_preconditions_to_datastores[preconditions]
                new_pnads.append(
                    PNAD(
                        pnad.op.copy_with(name=f"{pnad.op.name}-{j}",
                                          preconditions=preconditions),
                        datastore, pnad.option_spec))
        return new_pnads

    def _run_outer_search(
            self, pnad: PNAD, positive_data: Datastore,
            negative_data: Datastore
    ) -> Dict[FrozenSet[LiftedAtom], Datastore]:
        """Run outer-level search to find a set of precondition sets and
        associated datastores.

        Each precondition set will produce one operator.
        """
        all_preconditions_to_datastores = {}
        # We'll remove positives as they get covered.
        remaining_positives = list(positive_data)
        while remaining_positives:
            new_preconditions = self._run_inner_search(pnad,
                                                       remaining_positives,
                                                       negative_data)
            # Compute the datastore and update the remaining positives.
            datastore = []
            new_remaining_positives = []
            for seg, var_to_obj in remaining_positives:
                ground_pre = {a.ground(var_to_obj) for a in new_preconditions}
                if not ground_pre.issubset(seg.init_atoms):
                    # If the preconditions ground with this substitution don't
                    # hold in this segment's init_atoms, this segment has yet
                    # to be covered, so we keep it in the positives.
                    new_remaining_positives.append((seg, var_to_obj))
                else:
                    # Otherwise, we can add this segment to the datastore and
                    # also move it to negative_data, for any future
                    # preconditions that get learned.
                    datastore.append((seg, var_to_obj))
                    negative_data.append((seg, var_to_obj))
            # Special case: if the datastore is empty, that means these
            # new_preconditions don't cover any positives, so the search
            # failed to find preconditions that have a better score than inf.
            # Therefore we give up, without including these new_preconditions
            # into all_preconditions_to_datastores.
            if len(datastore) == 0:
                break
            assert len(new_remaining_positives) < len(remaining_positives)
            remaining_positives = new_remaining_positives
            # Update all_preconditions_to_datastores.
            assert new_preconditions not in all_preconditions_to_datastores
            all_preconditions_to_datastores[new_preconditions] = datastore
        if not all_preconditions_to_datastores:
            # If we couldn't find any preconditions, default to empty.
            assert len(remaining_positives) == len(positive_data)
            all_preconditions_to_datastores[frozenset()] = positive_data
        return all_preconditions_to_datastores

    def _run_inner_search(self, pnad: PNAD, positive_data: Datastore,
                          negative_data: Datastore) -> FrozenSet[LiftedAtom]:
        """Run inner-level search to find a single precondition set."""
        initial_state = self._get_initial_preconditions(positive_data)
        check_goal = lambda s: False
        heuristic = functools.partial(self._score_preconditions, pnad,
                                      positive_data, negative_data)
        max_expansions = CFG.cluster_and_search_inner_search_max_expansions
        timeout = CFG.cluster_and_search_inner_search_timeout
        path, _ = utils.run_gbfs(initial_state,
                                 check_goal,
                                 self._get_precondition_successors,
                                 heuristic,
                                 max_expansions=max_expansions,
                                 timeout=timeout)
        return path[-1]

    @staticmethod
    def _get_initial_preconditions(
            positive_data: Datastore) -> FrozenSet[LiftedAtom]:
        """The initial preconditions are a UNION over all lifted initial states
        in the data.

        We filter out atoms containing any object that doesn't have a
        binding to the PNAD parameters.
        """
        initial_preconditions = set()
        for seg, var_to_obj in positive_data:
            obj_to_var = {v: k for k, v in var_to_obj.items()}
            for atom in seg.init_atoms:
                if not all(obj in obj_to_var for obj in atom.objects):
                    continue
                initial_preconditions.add(atom.lift(obj_to_var))
        return frozenset(initial_preconditions)

    @staticmethod
    def _get_precondition_successors(
        preconditions: FrozenSet[LiftedAtom]
    ) -> Iterator[Tuple[int, FrozenSet[LiftedAtom], float]]:
        """The successors remove each atom in the preconditions."""
        preconditions_sorted = sorted(preconditions)
        for i in range(len(preconditions_sorted)):
            successor = preconditions_sorted[:i] + preconditions_sorted[i + 1:]
            yield i, frozenset(successor), 1.0

    @staticmethod
    def _score_preconditions(pnad: PNAD, positive_data: Datastore,
                             negative_data: Datastore,
                             preconditions: FrozenSet[LiftedAtom]) -> float:
        candidate_op = pnad.op.copy_with(preconditions=preconditions)
        option_spec = pnad.option_spec
        del pnad  # unused after this
        # Count up the number of true positives and false positives.
        num_true_positives = 0
        num_false_positives = 0
        for seg, var_to_obj in positive_data:
            ground_pre = {a.ground(var_to_obj) for a in preconditions}
            if ground_pre.issubset(seg.init_atoms):
                num_true_positives += 1
        if num_true_positives == 0:
            # As a special case, if the number of true positives is 0, we
            # never want to accept these preconditions, so we can give up.
            return float("inf")
        for seg, _ in negative_data:
            # We don't want to use the substitution in the datastore for
            # negative_data, because in general the variables could be totally
            # different. So we consider all possible groundings that are
            # consistent with the option_spec. If, for any such grounding, the
            # preconditions hold in the segment's init_atoms, then this is a
            # false positive.
            objects = list(seg.states[0])
            option = seg.get_option()
            assert option.parent == option_spec[0]
            option_objs = option.objects
            isub = dict(zip(option_spec[1], option_objs))
            for idx, ground_op in enumerate(
                    utils.all_ground_operators_given_partial(
                        candidate_op, objects, isub)):
                # If the maximum number of groundings is reached, treat this
                # as a false positive. Doesn't really matter in practice
                # because the GBFS is going to time out anyway -- we just
                # want the code to not hang in this score function.
                if idx >= CFG.cluster_and_search_score_func_max_groundings or \
                   ground_op.preconditions.issubset(seg.init_atoms):
                    num_false_positives += 1
                    break
        tp_w = CFG.clustering_learner_true_pos_weight
        fp_w = CFG.clustering_learner_false_pos_weight
        score = fp_w * num_false_positives + tp_w * (-num_true_positives)
        # Penalize the number of variables in the preconditions.
        all_vars = {v for atom in preconditions for v in atom.variables}
        score += CFG.cluster_and_search_var_count_weight * len(all_vars)
        # Penalize the number of preconditions.
        score += CFG.cluster_and_search_precon_size_weight * len(preconditions)
        return score

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_search"


class ClusterAndIntersectSidelineSTRIPSLearner(ClusterAndIntersectSTRIPSLearner
                                               ):
    """Base class for a clustering-based STRIPS learner that does sidelining
    via hill climbing, after operator learning."""

    def _postprocessing_learn_ignore_effects(self,
                                             pnads: List[PNAD]) -> List[PNAD]:
        # Run hill climbing search, starting from original PNADs.
        path, _, _ = utils.run_hill_climbing(
            tuple(pnads), self._check_goal, self._get_sidelining_successors,
            functools.partial(self._evaluate, pnads))
        # The last state in the search holds the final PNADs.
        pnads = list(path[-1])
        # Because the PNADs have been modified, recompute the datastores.
        self._recompute_datastores_from_segments(pnads)
        # Filter out PNADs that have an empty datastore.
        pnads = [pnad for pnad in pnads if pnad.datastore]
        return pnads

    @abc.abstractmethod
    def _evaluate(self, initial_pnads: List[PNAD], s: Tuple[PNAD,
                                                            ...]) -> float:
        """Abstract evaluation/score function for search.

        Lower is better.
        """
        raise NotImplementedError("Override me!")

    @staticmethod
    def _check_goal(s: Tuple[PNAD, ...]) -> bool:
        del s  # unused
        # There are no goal states for this search; run until exhausted.
        return False

    @staticmethod
    def _get_sidelining_successors(
        s: Tuple[PNAD,
                 ...], ) -> Iterator[Tuple[None, Tuple[PNAD, ...], float]]:
        # For each PNAD/operator...
        for i in range(len(s)):
            pnad = s[i]
            _, option_vars = pnad.option_spec
            # ...consider changing each of its add effects to an ignore effect.
            for effect in pnad.op.add_effects:
                if len(pnad.op.add_effects) > 1:
                    # We don't want sidelining to result in a noop.
                    new_pnad = PNAD(
                        pnad.op.effect_to_ignore_effect(
                            effect, option_vars, "add"), pnad.datastore,
                        pnad.option_spec)
                    sprime = list(s)
                    sprime[i] = new_pnad
                    yield (None, tuple(sprime), 1.0)

            # ...consider removing it.
            sprime = list(s)
            del sprime[i]
            yield (None, tuple(sprime), 1.0)


class ClusterAndIntersectSidelinePredictionErrorSTRIPSLearner(
        ClusterAndIntersectSidelineSTRIPSLearner):
    """A STRIPS learner that uses hill climbing with a prediction error score
    function for ignore effect learning."""

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_intersect_sideline_prederror"

    def _evaluate(self, initial_pnads: List[PNAD], s: Tuple[PNAD,
                                                            ...]) -> float:
        segments = [seg for traj in self._segmented_trajs for seg in traj]
        strips_ops = [pnad.op for pnad in s]
        option_specs = [pnad.option_spec for pnad in s]
        max_groundings = CFG.cluster_and_intersect_prederror_max_groundings
        num_true_positives, num_false_positives, _, _ = \
            utils.count_positives_for_ops(strips_ops, option_specs, segments,
                                          max_groundings=max_groundings)
        # Note: lower is better! We want more true positives and fewer
        # false positives.
        tp_w = CFG.clustering_learner_true_pos_weight
        fp_w = CFG.clustering_learner_false_pos_weight
        return fp_w * num_false_positives + tp_w * (-num_true_positives)


class ClusterAndIntersectSidelineHarmlessnessSTRIPSLearner(
        ClusterAndIntersectSidelineSTRIPSLearner):
    """A STRIPS learner that uses hill climbing with a harmlessness score
    function for ignore effect learning."""

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_intersect_sideline_harmlessness"

    def _evaluate(self, initial_pnads: List[PNAD], s: Tuple[PNAD,
                                                            ...]) -> float:
        preserves_harmlessness = self._check_harmlessness(list(s))
        if preserves_harmlessness:
            # If harmlessness is preserved, the score is the number of
            # operators that we have, minus the number of ignore effects.
            # This means we prefer fewer operators and more ignore effects.
            score = 2 * len(s)
            for pnad in s:
                score -= len(pnad.op.ignore_effects)
        else:
            # If harmlessness is not preserved, the score is an arbitrary
            # constant bigger than the total number of operators at the
            # start of the search. This is guaranteed to be worse (higher)
            # than any score that occurs if harmlessness is preserved.
            score = 10 * len(initial_pnads)
        return score
