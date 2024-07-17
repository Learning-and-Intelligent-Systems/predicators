"""Approaches that use an LLM to learn STRIPS operators instead of performing
symbolic learning of any kind."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from predicators import utils
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import PNAD, LiftedAtom, LowLevelTrajectory, \
    ParameterizedOption, Predicate, Segment, STRIPSOperator, Task, Type, \
    Variable


class LLMStripsLearner(BaseSTRIPSLearner):
    """Base class for all LLM-based learners."""

    def __init__(self,
                 trajectories: List[LowLevelTrajectory],
                 train_tasks: List[Task],
                 predicates: Set[Predicate],
                 segmented_trajs: List[List[Segment]],
                 verify_harmlessness: bool,
                 annotations: Optional[List[Any]],
                 verbose: bool = True) -> None:
        super().__init__(trajectories, train_tasks, predicates,
                         segmented_trajs, verify_harmlessness, annotations,
                         verbose)
        self._llm = utils.create_llm_by_name(CFG.llm_model_name)
        prompt_file = utils.get_path_to_predicators_root() + \
        "/predicators/nsrt_learning/strips_learning/" + \
        "llm_op_learning_prompts/naive_no_examples.txt"
        with open(prompt_file, "r", encoding="utf-8") as f:
            self._base_prompt = f.read()
        self._name_to_pred = {p.name: p for p in self._predicates}

    def _get_all_types_from_preds(self) -> Set[Type]:
        all_types: Set[Type] = set()
        for pred in self._predicates:
            all_types |= set(pred.types)
        return all_types

    def _get_all_options_from_segs(self) -> Set[ParameterizedOption]:
        # NOTE: this assumes all segments in self._segmented_trajs
        # have options associated with them.
        all_options: Set[ParameterizedOption] = set()
        for seg_traj in self._segmented_trajs:
            for seg in seg_traj:
                all_options.add(seg.get_option().parent)
        return all_options

    def _parse_line_into_pred_names_and_arg_names(
            self, line_str: str) -> List[Tuple[str, List[str]]]:
        pattern = r'\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([^\(\)]*)\)'
        matches = re.findall(pattern, line_str)
        predicates = []
        for match in matches:
            predicate = match[0]
            arguments = [
                arg.strip() for arg in match[1].split() if arg.strip()
            ]
            predicates.append((predicate, arguments))
        return predicates

    def _parse_option_str_into_opt_name_and_arg_names(
            self, line_str: str) -> Tuple[str, List[str]]:
        match = re.match(r'action: (\w+)\(([^)]*)\)', line_str)
        if not match:
            raise ValueError("The input string is not in the expected format.")
        skill_name = match.group(1)
        args = match.group(2).split(',') if match.group(2) else []
        args = [arg.strip() for arg in args]
        return skill_name, args

    def _parse_operator_str_into_structured_elems(
        self, op_str: str
    ) -> Tuple[str, Dict[str, str], List[Tuple[str, List[str]]], List[Tuple[
            str, List[str]]], List[Tuple[str, List[str]]], Tuple[str,
                                                                 List[str]]]:
        op_str_elems = op_str.split('\n')
        # Parse out operator name and args.
        name_and_args = op_str_elems[0]
        opening_paren_loc = name_and_args.find("(")
        closing_paren_loc = name_and_args.find(")")
        name_str = name_and_args[:opening_paren_loc]
        arg_str = name_and_args[opening_paren_loc + 1:closing_paren_loc]
        args = arg_str.split()
        arg_dict = {}
        for i in range(0, len(args), 3):
            arg_name = args[i]
            arg_type = args[i + 2]
            arg_dict[arg_name] = arg_type
        # Parse out structures for preconditions and effects.
        structured_precs = self._parse_line_into_pred_names_and_arg_names(
            op_str_elems[1])
        structured_add_effs = self._parse_line_into_pred_names_and_arg_names(
            op_str_elems[2])
        structured_del_effs = self._parse_line_into_pred_names_and_arg_names(
            op_str_elems[3])
        structured_action = self._parse_option_str_into_opt_name_and_arg_names(
            op_str_elems[4])
        return (name_str, arg_dict, structured_precs, structured_add_effs,
                structured_del_effs, structured_action)

    def _convert_structured_precs_or_effs_into_lifted_atom_set(
            self, structured_precs_or_effs: List[Tuple[str, List[str]]],
            pred_name_to_pred: Dict[str, Predicate],
            op_var_name_to_op_var: Dict[str, Variable]) -> Set[LiftedAtom]:
        ret_atoms = set()
        for prec_name, prec_args_list in structured_precs_or_effs:
            if prec_name not in pred_name_to_pred:
                continue
            all_args_valid = True
            prec_arg_vars = []
            for prec_arg in prec_args_list:
                if prec_arg not in op_var_name_to_op_var:
                    all_args_valid = False
                    break
                prec_arg_vars.append(op_var_name_to_op_var[prec_arg])
            if not all_args_valid:
                continue
            ret_atoms.add(
                LiftedAtom(pred_name_to_pred[prec_name], prec_arg_vars))
        return ret_atoms

    def _learn(self) -> List[PNAD]:
        """Overview of what's going on here:

        1. Form a prompt that contains the object types, predicates, and
        options in this env, as well as the training demonstration
        trajectories.
        2. Query an LLM to return operators given the above information
        from (1).
        3. Parse out symbolic operators from the LLM's response, with automatic
        pruning of nonsensical/hallucinated operators. This is sufficient
        info to make PNADs with empty datastores.
        4. Associate data with each PNAD via our
        _find_best_matching_pnad_and_sub procedure; this populates the
        PNAD's datastore.
        """
        prompt = self._base_prompt
        all_types = self._get_all_types_from_preds()
        all_options = sorted(self._get_all_options_from_segs())
        type_name_to_type = {t.name: t for t in all_types}
        pred_name_to_pred = {p.name: p for p in self._predicates}
        option_name_to_option = {o.name: o for o in all_options}
        prompt += "Types:\n"
        for t in sorted(all_types):
            prompt += f"- {t.name}\n"
        prompt += "\nPredicates:\n"
        for pred in sorted(self._predicates):
            prompt += f"- {pred.pddl_str()}\n"
        prompt += "\nActions:\n"
        for act in all_options:
            prompt += act.pddl_str() + "\n"
        prompt += "\nTrajectory data:\n"
        for i, seg_traj in enumerate(self._segmented_trajs):
            curr_goal = self._train_tasks[i].goal
            prompt += f"Trajectory {i} (Goal: {str(curr_goal)}):\n"
            for timestep, seg in enumerate(seg_traj):
                prompt += f"State {timestep}: {str(seg.init_atoms)}\n"
                curr_option = seg.get_option()
                action = curr_option.name + "(" + ", ".join(
                    obj.name for obj in curr_option.objects) + ")"
                prompt += f"Action {timestep}: {action}"
            prompt += f"State {timestep + 1}: {str(seg.final_atoms)}"  # pylint:disable=undefined-loop-variable
            prompt += "\n"
        # Query the LLM to see what kinds of operators it proposes.
        llm_response = self._llm.sample_completions(prompt, None, 0.0,
                                                    CFG.seed)
        op_predictions = llm_response[0]
        pattern = r'```\n(.*?)\n```'
        matches = re.findall(pattern, op_predictions, re.DOTALL)
        assert matches[0][:10] == "Operators:"
        specific_op_prediction = matches[0][11:].split('\n\n')
        # Now parse the operators into actual structured symbolic operators
        # with associated option specs (i.e. PNADs, but without the
        # datastores set).
        pnads = []
        for op_pred in specific_op_prediction:
            incorrect_op = False
            name_str, arg_dict, structured_precs, structured_add_effs, \
                structured_del_effs, structured_action = \
                self._parse_operator_str_into_structured_elems(op_pred)
            op_var_name_to_op_var = {}
            for arg_name_str, arg_type_str in arg_dict.items():
                if arg_type_str not in type_name_to_type:
                    incorrect_op = True
                    break
                op_var_name_to_op_var[arg_name_str] = Variable(
                    arg_name_str, type_name_to_type[arg_type_str])
            if incorrect_op:
                continue
            preconditions = \
                self._convert_structured_precs_or_effs_into_lifted_atom_set(
                structured_precs, pred_name_to_pred, op_var_name_to_op_var)
            add_effs = \
                self._convert_structured_precs_or_effs_into_lifted_atom_set(
                structured_add_effs, pred_name_to_pred, op_var_name_to_op_var)
            del_effs = \
                self._convert_structured_precs_or_effs_into_lifted_atom_set(
                structured_del_effs, pred_name_to_pred, op_var_name_to_op_var)
            option_name_str, option_args_strs_list = structured_action
            if option_name_str not in option_name_to_option:
                continue
            option = option_name_to_option[option_name_str]
            option_arg_vars = []
            for option_arg_str in option_args_strs_list:
                if option_arg_str not in op_var_name_to_op_var:
                    incorrect_op = True
                    break
                option_arg_vars.append(op_var_name_to_op_var[option_arg_str])
            if incorrect_op:
                continue
            # NOTE: for now, we do not create operators with ignore effects!
            # This could be done in the future by modifying the LLM query
            # prompt.
            symbolic_op = STRIPSOperator(name_str,
                                         list(op_var_name_to_op_var.values()),
                                         preconditions, add_effs, del_effs,
                                         set())
            pnad = PNAD(symbolic_op, [], (option, option_arg_vars))
            pnads.append(pnad)
        # Finally, populate the datastore of each PNAD.
        curr_pnad: Optional[PNAD] = None
        for seg_traj in self._segmented_trajs:
            for seg in seg_traj:
                objects = set(seg.states[0])
                curr_pnad, var_to_obj_sub = \
                    self._find_best_matching_pnad_and_sub(seg, objects, pnads)
                if curr_pnad is not None:
                    assert var_to_obj_sub is not None
                    curr_pnad.add_to_datastore((seg, var_to_obj_sub))
        return pnads

    @classmethod
    def get_name(cls) -> str:
        return "llm"
