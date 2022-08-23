"""Open-loop large language model (LLM) meta-controller approach with prompt
modification where certain PDDL syntax is replaced with random characters.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python predicators/main.py --approach llm_syntax_renaming --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug
"""
from typing import Dict, List

from predicators.approaches.llm_base_renaming_approach import \
    LLMBaseRenamingApproach

ORIGINAL_CHARS = ['(', ')', ':']
REPLACEMENT_CHARS = ['^', '$', '#', '!', '*']


class LLMSyntaxRenamingApproach(LLMBaseRenamingApproach):
    """LLMSyntaxRenamingApproach definition."""

    @classmethod
    def get_name(cls) -> str:
        return "llm_syntax_renaming"

    @property
    def _renaming_prefixes(self) -> List[str]:
        # Since we're replacing single characters, we don't need to worry about
        # the possibility that one string is a substring of another.
        return [""]

    @property
    def _renaming_suffixes(self) -> List[str]:
        # Since we're replacing single characters, we don't need to worry about
        # the possibility that one string is a substring of another.
        return [""]

    def _create_replacements(self) -> Dict[str, str]:
        # Without replacement because if multiple original characters mapped
        # to the same replacement character, the inverse substitution would be
        # not well defined and the parsing of the option plan would fail.
        replacement_chars = self._rng.choice(REPLACEMENT_CHARS,
                                             size=len(ORIGINAL_CHARS),
                                             replace=False)
        return dict(zip(ORIGINAL_CHARS, replacement_chars))
