"""Open-loop large language model (LLM) meta-controller approach with.

prompt modification where predicate names are randomly generated.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python src/main.py --approach llm_predicate_renaming --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug

Easier setting:
    python src/main.py --approach llm_predicate_naming --seed 0 \
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
from typing import List, Sequence, Set

from predicators.src import utils
from predicators.src.approaches.llm_open_loop_approach import \
    LLMOpenLoopApproach
from predicators.src.structs import Box, GroundAtom, ParameterizedOption, \
    Predicate, Task, Type, _Option


class LLMPredicateRenamingApproach(LLMOpenLoopApproach):
    """LLMPredicateRenamingApproach definition."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._original_to_random_dictionary: dict = {}
        self._alphabet: List[str] = list(string.ascii_lowercase)

    @classmethod
    def get_name(cls) -> str:
        return "llm_predicate_renaming"

    def _create_prompt(self, init: Set[GroundAtom], goal: Set[GroundAtom],
                       options: Sequence[_Option]) -> str:
        init_arr = self._replace_with_random_predicate_names(sorted(init))
        init_str = "\n  ".join(init_arr)
        goal_arr = self._replace_with_random_predicate_names(sorted(goal))
        goal_str = "\n  ".join(goal_arr)
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

    def _replace_with_random_predicate_names(
            self, original_set: List[GroundAtom]) -> List[str]:
        list_with_random_predicate_names = []
        for ground_atom in original_set:
            ground_atom_str = str(ground_atom)
            original_predicate_name = ground_atom_str[0:ground_atom_str.
                                                      index('(')]
            if original_predicate_name in self._original_to_random_dictionary:
                list_with_random_predicate_names.append(
                    self.
                    _original_to_random_dictionary[original_predicate_name] +
                    ground_atom_str[ground_atom_str.index('('):])
            else:
                new_predicate_name = ''.join(
                    utils.generate_random_string(original_predicate_name,
                                                 self._alphabet, self._rng))
                self._original_to_random_dictionary[
                    original_predicate_name] = new_predicate_name
                ground_atom_str = ground_atom_str.replace(
                    original_predicate_name, new_predicate_name)
                list_with_random_predicate_names.append(ground_atom_str)
        return list_with_random_predicate_names
