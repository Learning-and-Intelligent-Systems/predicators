"""Open-loop large language model (LLM) meta-controller approach.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python src/main.py --approach syntax_modifications --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug

Easier setting:
    python src/main.py --approach syntax_modifications --seed 0 \
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

from typing import List, Sequence, Set

from predicators.src.approaches.llm_open_loop_approach import \
    LLMOpenLoopApproach
from predicators.src.structs import Box, GroundAtom, ParameterizedOption, \
    Predicate, Task, Type, _Option


class SyntaxModifications(LLMOpenLoopApproach):
    """SyntaxModifications definition."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._modification_one = "".join(self._generate_random_string(")"))
        self._modification_two = "".join(self._generate_random_string("("))
        self._modification_three = "".join(self._generate_random_string(":"))

    @classmethod
    def get_name(cls) -> str:
        return "syntax_modifications"

    def _generate_random_string(self, sentence: str) -> List[str]:
        alphabet = ['^', '$', '#', '!', '*']
        return [self._rng.choice(alphabet) for i in range(len(sentence))]

    def _create_prompt(self, init: Set[GroundAtom], goal: Set[GroundAtom],
                       options: Sequence[_Option]) -> str:
        prompt = super()._create_prompt(init, goal, options)
        prompt = self._modify_syntax(prompt)
        return prompt

    def _modify_syntax(self, prompt: str) -> str:
        prompt_list: List[str] = []
        prompt_list[:0] = prompt
        # Iterate through prompt and replace with modifications.
        for ind in range(len(prompt_list)):
            if prompt_list[ind] == ")":
                prompt_list[ind] = self._modification_one
            elif prompt_list[ind] == "(":
                prompt_list[ind] = self._modification_two
            elif prompt_list[ind] == ":":
                prompt_list[ind] = self._modification_three
        prompt = "".join(prompt_list)
        return prompt
