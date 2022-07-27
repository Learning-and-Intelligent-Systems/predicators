"""Open-loop large language model (LLM) meta-controller approach with prompt
modification where certain PDDL syntax is replaced with random characters.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python src/main.py --approach llm_syntax_renaming --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug
"""
from typing import Dict

from predicators.src.approaches.llm_renaming_base_approach import \
    LLMBaseRenamingApproach

ORIGINAL_CHARS = ['(', ')', ':']
REPLACEMENT_CHARS = ['^', '$', '#', '!', '*']


class LLMSyntaxRenamingApproach(LLMBaseRenamingApproach):
    """LLMSyntaxRenamingApproach definition."""

    @classmethod
    def get_name(cls) -> str:
        return "llm_syntax_renaming"

    def _create_replacements(self) -> Dict[str, str]:
        # Without replacement!
        replacement_chars = self._rng.choice(REPLACEMENT_CHARS,
                                             size=len(ORIGINAL_CHARS),
                                             replace=False)
        return dict(zip(ORIGINAL_CHARS, replacement_chars))
