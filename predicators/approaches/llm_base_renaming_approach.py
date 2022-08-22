"""An abstract base class for LLM approaches that perform some renaming, where
strings are substituted before querying the LLM and then inversely substituted
in the outputs.

The subclasses are designed to test the extent to which the approaches
are aware of English words and/or PDDL syntax.
"""

import abc
from typing import Collection, Dict, List, Sequence, Set, Tuple

from predicators.approaches.llm_open_loop_approach import LLMOpenLoopApproach
from predicators.structs import Box, GroundAtom, Object, ParameterizedOption, \
    Predicate, Task, Type, _Option


class LLMBaseRenamingApproach(LLMOpenLoopApproach):
    """LLMBaseRenamingApproach definition."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._orig_to_replace = self._create_replacements()

    @abc.abstractmethod
    def _create_replacements(self) -> Dict[str, str]:
        """Create a map from original strings to replacement strings."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def _renaming_prefixes(self) -> List[str]:
        """When performing replacements, loop through each prefix and suffix
        and wrap the words before substituting.

        This is designed to avoid cases like at(...) and bat(...)
        colliding.
        """
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def _renaming_suffixes(self) -> List[str]:
        """See _renaming_prefixes().'."""
        raise NotImplementedError("Override me!")

    def _create_prompt(self, init: Set[GroundAtom], goal: Set[GroundAtom],
                       options: Sequence[_Option]) -> str:
        prompt = super()._create_prompt(init, goal, options)
        # Substitute the originals with the replacements.
        for orig, repl in self._orig_to_replace.items():
            for pre in self._renaming_prefixes:
                for suf in self._renaming_suffixes:
                    prompt = prompt.replace(pre + orig + suf, pre + repl + suf)
        return prompt

    def _llm_prediction_to_option_plan(
        self, llm_prediction: str, objects: Collection[Object]
    ) -> List[Tuple[ParameterizedOption, Sequence[Object]]]:
        # Substitute the replacements with the originals.
        for orig, repl in self._orig_to_replace.items():
            for pre in self._renaming_prefixes:
                for suf in self._renaming_suffixes:
                    llm_prediction = llm_prediction.replace(
                        pre + repl + suf, pre + orig + suf)
        return super()._llm_prediction_to_option_plan(llm_prediction, objects)
