"""Open-loop large language model (LLM) meta-controller approach.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python src/main.py --approach open_loop_llm --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug

Easier setting:
    python src/main.py --approach open_loop_llm --seed 0 \
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
from typing import Collection, Dict, List, Optional, Sequence, Set, Tuple

from predicators.src.approaches import ApproachFailure
from predicators.src.approaches.nsrt_metacontroller_approach import \
    NSRTMetacontrollerApproach
from predicators.src.llm_interface import OpenAILLM
from predicators.src.planning import task_plan_with_option_plan_constraint
from predicators.src.settings import CFG
from predicators.src.structs import Box, Dataset, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT, _Option


class OpenLoopLLMApproach(NSRTMetacontrollerApproach):
    """OpenLoopLLMApproach definition."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Set up the LLM.
        self._llm = OpenAILLM(CFG.open_loop_llm_model_name)
        # Set after learning.
        self._prompt_prefix = ""

    @classmethod
    def get_name(cls) -> str:
        return "open_loop_llm"

    def _predict(self, state: State, atoms: Set[GroundAtom],
                 goal: Set[GroundAtom], memory: Dict) -> _GroundNSRT:
        # If we already have an abstract plan, execute the next step.
        if "abstract_plan" in memory and memory["abstract_plan"]:
            return memory["abstract_plan"].pop(0)
        # Otherwise, we need to make a new abstract plan.
        new_prompt = self._create_prompt(atoms, goal, [])
        prompt = self._prompt_prefix + new_prompt
        # Query the LLM.
        llm_predictions = self._llm.sample_completions(
            prompt=prompt,
            temperature=CFG.open_loop_llm_temperature,
            seed=CFG.seed,
            num_completions=CFG.open_loop_llm_num_completions)
        # Try to convert the output into an abstract plan.
        for llm_prediction in llm_predictions:
            ground_nsrt_plan = self._process_single_prediction(
                llm_prediction, state, atoms, goal)
            if ground_nsrt_plan is not None:
                # If valid plan, add plan to memory so it can be refined!
                memory["abstract_plan"] = ground_nsrt_plan
                return memory["abstract_plan"].pop(0)
        # Give up if none of the predictions work out.
        raise ApproachFailure("No LLM predicted plan achieves the goal.")

    def _process_single_prediction(
            self, llm_prediction: str, state: State, atoms: Set[GroundAtom],
            goal: Set[GroundAtom]) -> Optional[List[_GroundNSRT]]:
        objects = set(state)
        option_plan = self._llm_prediction_to_option_plan(
            llm_prediction, objects)
        # If we failed to find a nontrivial plan with this prediction,
        # continue on to next prediction.
        if len(option_plan) == 0:
            return None
        # Attempt to turn the plan into a sequence of ground NSRTs.
        nsrts = self._get_current_nsrts()
        predicates = self._initial_predicates
        strips_ops = [n.op for n in nsrts]
        option_specs = [(n.option, list(n.option_vars)) for n in nsrts]
        ground_nsrt_plan = task_plan_with_option_plan_constraint(
            objects, predicates, strips_ops, option_specs, atoms, goal,
            option_plan)
        # If we can't find an NSRT plan that achieves the goal,
        # continue on to next prediction.
        if not ground_nsrt_plan:
            return None
        return ground_nsrt_plan

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
