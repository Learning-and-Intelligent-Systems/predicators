"""Open-loop large language model (LLM) meta-controller approach with prompt
modification where option names are replaced with random strings.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python predicators/main.py --approach llm_option_renaming --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug
"""
import string
from typing import Dict, List

from predicators import utils
from predicators.approaches.llm_base_renaming_approach import \
    LLMBaseRenamingApproach


class LLMOptionRenamingApproach(LLMBaseRenamingApproach):
    """LLMOptionRenamingApproach definition."""

    @classmethod
    def get_name(cls) -> str:
        return "llm_option_renaming"

    @property
    def _renaming_prefixes(self) -> List[str]:
        # Options start with either a new line or a white space.
        return [" ", "\n"]

    @property
    def _renaming_suffixes(self) -> List[str]:
        # Option names end with a left parenthesis.
        return ["("]

    def _create_replacements(self) -> Dict[str, str]:
        return {
            o.name:
            utils.generate_random_string(len(o.name),
                                         list(string.ascii_lowercase),
                                         self._rng)
            for o in self._initial_options
        }
