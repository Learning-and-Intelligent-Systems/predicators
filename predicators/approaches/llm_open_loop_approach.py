"""Open-loop large language model (LLM) meta-controller approach.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python predicators/main.py --approach llm_open_loop --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug

Easier setting:
    python predicators/main.py --approach llm_open_loop --seed 0 \
        --strips_learner oracle \
        --env pddl_easy_delivery_procedural_tasks \
        --pddl_easy_delivery_procedural_train_min_num_locs 2 \
        --pddl_easy_delivery_procedural_train_max_num_locs 2 \
        --pddl_easy_delivery_procedural_train_min_want_locs 1 \
        --pddl_easy_delivery_procedural_train_max_want_locs 1 \
        --pddl_easy_delivery_procedural_train_min_extra_newspapers 0 \
        --pddl_easy_delivery_procedural_train_max_extra_newspapers 0 \
        --pddl_easy_delivery_procedural_test_min_num_locs 2 \
        --pddl_easy_delivery_procedural_test_max_num_locs 2 \
        --pddl_easy_delivery_procedural_test_min_want_locs 1 \
        --pddl_easy_delivery_procedural_test_max_want_locs 1 \
        --pddl_easy_delivery_procedural_test_min_extra_newspapers 0 \
        --pddl_easy_delivery_procedural_test_max_extra_newspapers 0 \
        --num_train_tasks 5 \
        --num_test_tasks 10 \
        --debug
"""
from __future__ import annotations

import logging
from typing import Collection, Dict, Iterator, List, Optional, Sequence, Set, \
    Tuple

from predicators.approaches import ApproachFailure
from predicators.approaches.nsrt_metacontroller_approach import \
    NSRTMetacontrollerApproach
from predicators.planning import task_plan_with_option_plan_constraint
from predicators.pretrained_model_interface import OpenAILLM
from predicators.settings import CFG
from predicators.structs import Box, Dataset, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT, _Option


class LLMOpenLoopApproach(NSRTMetacontrollerApproach):
    """LLMOpenLoopApproach definition."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Set up the LLM.
        self._llm = OpenAILLM(CFG.llm_model_name)
        # Set after learning.
        self._prompt_prefix = ""

    @classmethod
    def get_name(cls) -> str:
        return "llm_open_loop"

    def _predict(self, state: State, atoms: Set[GroundAtom],
                 goal: Set[GroundAtom], memory: Dict) -> _GroundNSRT:
        # If we already have an abstract plan, execute the next step.
        if "abstract_plan" in memory and memory["abstract_plan"]:
            return memory["abstract_plan"].pop(0)
        # Otherwise, we need to make a new abstract plan.
        action_seq = self._get_llm_based_plan(state, atoms, goal)
        if action_seq is not None:
            # If valid plan, add plan to memory so it can be refined!
            memory["abstract_plan"] = action_seq
            return memory["abstract_plan"].pop(0)
        raise ApproachFailure("No LLM predicted plan achieves the goal.")

    def _get_llm_based_plan(
            self, state: State, atoms: Set[GroundAtom],
            goal: Set[GroundAtom]) -> Optional[List[_GroundNSRT]]:
        # Try to convert each output into an abstract plan.
        # Return the first abstract plan that is found this way.
        objects = set(state)
        for option_plan in self._get_llm_based_option_plans(
                atoms, objects, goal):
            ground_nsrt_plan = self._option_plan_to_nsrt_plan(
                option_plan, atoms, objects, goal)
            if ground_nsrt_plan is not None:
                return ground_nsrt_plan
        return None

    def _get_llm_based_option_plans(
        self, atoms: Set[GroundAtom], objects: Set[Object],
        goal: Set[GroundAtom]
    ) -> Iterator[List[Tuple[ParameterizedOption, Sequence[Object]]]]:
        new_prompt = self._create_prompt(atoms, goal, [])
        prompt = self._prompt_prefix + new_prompt
        # Query the LLM.
        llm_predictions = self._llm.sample_completions(
            prompt=prompt,
            imgs=None,
            temperature=CFG.llm_temperature,
            seed=CFG.seed,
            num_completions=CFG.llm_num_completions)
        for pred in llm_predictions:
            option_plan = self._llm_prediction_to_option_plan(pred, objects)
            yield option_plan

    def _option_plan_to_nsrt_plan(
            self, option_plan: List[Tuple[ParameterizedOption,
                                          Sequence[Object]]],
            atoms: Set[GroundAtom], objects: Set[Object],
            goal: Set[GroundAtom]) -> Optional[List[_GroundNSRT]]:
        nsrts = self._get_current_nsrts()
        predicates = self._initial_predicates
        strips_ops = [n.op for n in nsrts]
        option_specs = [(n.option, list(n.option_vars)) for n in nsrts]
        return task_plan_with_option_plan_constraint(objects, predicates,
                                                     strips_ops, option_specs,
                                                     atoms, goal, option_plan)

    def _llm_prediction_to_option_plan(
        self, llm_prediction: str, objects: Collection[Object]
    ) -> List[Tuple[ParameterizedOption, Sequence[Object]]]:
        """Convert the output of the LLM into a sequence of
        ParameterizedOptions coupled with a list of objects that will be used
        to ground the ParameterizedOption."""
        option_plan: List[Tuple[ParameterizedOption, Sequence[Object]]] = []
        # Setup dictionaries enabling us to easily map names to specific
        # Python objects during parsing.
        option_name_to_option = {op.name: op for op in self._initial_options}
        obj_name_to_obj = {o.name: o for o in objects}
        # We assume the LLM's output is such that each line contains a
        # option_name(obj0:type0, obj1:type1,...).
        options_str_list = llm_prediction.split('\n')
        for option_str in options_str_list:
            option_str_stripped = option_str.strip()
            option_name = option_str_stripped.split('(')[0]
            # Skip empty option strs.
            if not option_str:
                continue
            if option_name not in option_name_to_option.keys() or \
                "(" not in option_str:
                logging.info(
                    f"Line {option_str} output by LLM doesn't "
                    "contain a valid option name. Terminating option plan "
                    "parsing.")
                break
            option = option_name_to_option[option_name]
            # Now that we have the option, we need to parse out the objects
            # along with specified types.
            typed_objects_str_list = option_str_stripped.split('(')[1].strip(
                ')').split(',')
            objs_list = []
            malformed = False
            for i, type_object_string in enumerate(typed_objects_str_list):
                object_type_str_list = type_object_string.strip().split(':')
                # We expect this list to be [object_name, type_name].
                if len(object_type_str_list) != 2:
                    logging.info(f"Line {option_str} output by LLM has a "
                                 "malformed object-type list.")
                    malformed = True
                    break
                object_name = object_type_str_list[0]
                type_name = object_type_str_list[1]
                if object_name not in obj_name_to_obj.keys():
                    logging.info(f"Line {option_str} output by LLM has an "
                                 "invalid object name.")
                    malformed = True
                    break
                obj = obj_name_to_obj[object_name]
                # Check that the type of this object agrees
                # with what's expected given the ParameterizedOption.
                if type_name != option.types[i].name:
                    logging.info(f"Line {option_str} output by LLM has an "
                                 "invalid type name.")
                    malformed = True
                    break
                objs_list.append(obj)
            # The types of the objects match, but we haven't yet checked if
            # all arguments of the option have an associated object.
            if len(objs_list) != len(option.types):
                malformed = True
            if not malformed:
                option_plan.append((option, objs_list))
        return option_plan

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # First, learn NSRTs.
        super().learn_from_offline_dataset(dataset)
        # Then, parse the data into the prompting format expected by the LLM.
        self._prompt_prefix = self._data_to_prompt_prefix(dataset)

    def _data_to_prompt_prefix(self, dataset: Dataset) -> str:
        # In this approach, we learned NSRTs, so we just use the segmented
        # trajectories that NSRT learning returned to us.
        prompts = []
        assert len(self._segmented_trajs) == len(dataset.trajectories)
        for segment_traj, ll_traj in zip(self._segmented_trajs,
                                         dataset.trajectories):
            if not ll_traj.is_demo:
                continue
            init = segment_traj[0].init_atoms
            goal = self._train_tasks[ll_traj.train_task_idx].goal
            seg_options = []
            for segment in segment_traj:
                assert segment.has_option()
                seg_options.append(segment.get_option())
            prompt = self._create_prompt(init, goal, seg_options)
            prompts.append(prompt)
        return "\n\n".join(prompts) + "\n\n"

    def _create_prompt(self, init: Set[GroundAtom], goal: Set[GroundAtom],
                       options: Sequence[_Option]) -> str:
        init_str = "\n  ".join(map(str, sorted(init)))
        goal_str = "\n  ".join(map(str, sorted(goal)))
        options_str = "\n  ".join(map(self._option_to_str, options))
        prompt = f"""
(:init
  {init_str}
)
(:goal
  {goal_str}
)
Solution:
  {options_str}"""
        return prompt

    @staticmethod
    def _option_to_str(option: _Option) -> str:
        objects_str = ", ".join(map(str, option.objects))
        return f"{option.name}({objects_str})"
