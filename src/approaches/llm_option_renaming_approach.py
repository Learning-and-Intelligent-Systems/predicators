"""Open-loop large language model (LLM) meta-controller approach with prompt
modification where option names are replaced with random strings.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python src/main.py --approach llm_option_renaming --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug
"""
import string
from typing import Dict

from predicators.src import utils
from predicators.src.approaches.llm_renaming_base import \
    LLMBaseRenamingApproach


class LLMOptionRenamingApproach(LLMBaseRenamingApproach):
    """LLMOptionRenamingApproach definition."""

    @classmethod
    def get_name(cls) -> str:
        return "llm_option_renaming"

    def _create_replacements(self) -> Dict[str, str]:
        return {
            o.name: utils.generate_random_string(len(o.name),
                                                 list(string.ascii_lowercase),
                                                 self._rng)
            for o in self._initial_options
        }
