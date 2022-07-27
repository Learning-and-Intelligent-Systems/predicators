"""Open-loop large language model (LLM) meta-controller approach with.

prompt modification where operator names are randomly generated.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python src/main.py --approach llm_operator_renaming --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug

Easier setting:
    python src/main.py --approach llm_operator_renaming --seed 0 \
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

import string
from typing import Collection, List, Sequence, Set, Tuple

from predicators.src import utils
from predicators.src.approaches.llm_open_loop_approach import \
    LLMOpenLoopApproach
from predicators.src.structs import Box, GroundAtom, Object, \
    ParameterizedOption, Predicate, Task, Type, _Option


class LLMOperatorRenamingApproach(LLMOpenLoopApproach):
    """LLMOperatorRenamingApproach definition."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._original_to_random_dictionary: dict = {}
        self._alphabet: List[str] = list(string.ascii_lowercase)

    @classmethod
    def get_name(cls) -> str:
        return "llm_operator_renaming"

    def _llm_prediction_to_option_plan(
        self, llm_prediction: str, objects: Collection[Object]
    ) -> List[Tuple[ParameterizedOption, Sequence[Object]]]:
        # We assume the LLM's output is such that each line contains a
        # option_name(obj0:type0, obj1:type1,...).
        option_str_list = llm_prediction.split('\n')
        for option_str in option_str_list:
            for random_operator in self._original_to_random_dictionary.values(
            ):
                if random_operator in option_str:
                    key_list = list(self._original_to_random_dictionary.keys())
                    val_list = list(
                        self._original_to_random_dictionary.values())
                    option_str_list[option_str_list.index(
                        option_str)] = option_str.replace(
                            random_operator,
                            key_list[val_list.index(random_operator)])
                    break
        unmodified_prediction = "\n".join(option_str_list)
        option_plan = super()._llm_prediction_to_option_plan(
            unmodified_prediction, objects)
        return option_plan

    def _create_prompt(self, init: Set[GroundAtom], goal: Set[GroundAtom],
                       options: Sequence[_Option]) -> str:
        for option in options:
            sub = option.name
            if sub in self._original_to_random_dictionary:
                option.name = self._original_to_random_dictionary[sub]
            else:
                sub_random = ''.join(
                    utils.generate_random_string(sub, self._alphabet,
                                                 self._rng))
                self._original_to_random_dictionary[sub] = sub_random
                option.name = sub_random
        prompt = super()._create_prompt(init, goal, options)
        return prompt
