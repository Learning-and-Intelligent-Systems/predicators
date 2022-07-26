"""Open-loop large language model (LLM) meta-controller approach with
prompt modification where predicate names are randomly generated.
Note: Predicate and Task classes must be set to frozen=False
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

from typing import List, Sequence, Set

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
        self.original_predicates: List[str] = []
        self.random_predicates: List[str] = []

    @classmethod
    def get_name(cls) -> str:
        return "llm_predicate_renaming"

    def _generate_guess(self, sentence: str) -> List[str]:
        alphabet = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]
        return [self._rng.choice(alphabet) for i in range(len(sentence))]

    def _create_prompt(self, init: Set[GroundAtom], goal: Set[GroundAtom],
                       options: Sequence[_Option]) -> str:
        init_arr = self._replace_with_random_predicate_names(init)
        init_str = "\n  ".join(init_arr)
        goal_arr = self._replace_with_random_predicate_names(goal)
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
            self, original_set: Set[GroundAtom]) -> List[str]:
        modified_set = []
        for i in original_set:
            name = str(i)
            sub = name[0:name.index('(')]
            if sub in self.original_predicates:
                modified_set.append(self.random_predicates[
                    self.original_predicates.index(sub)] +
                                    name[name.index('('):])
            else:
                self.original_predicates.append(sub)
                sub_random = ''.join(self._generate_guess(sub))
                self.random_predicates.append(sub_random)
                name = name.replace(sub, sub_random)
                modified_set.append(name)
        return modified_set
