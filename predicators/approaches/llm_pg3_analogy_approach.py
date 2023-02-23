"""Use LLMs for cross-domain policy learning in PG3.

Command for testing gripper / ferry:
    python predicators/main.py --approach llm_pg3 --seed 0  \
        --env pddl_ferry_procedural_tasks --strips_learner oracle \
        --num_train_tasks 20 --pg3_init_policy gripper_ldl_policy.txt \
        --pg3_init_base_env pddl_gripper_procedural_tasks \
        --pg3_add_condition_allow_new_vars False \
        --llm_model_name code-davinci-002 \
        --llm_temperature 0.0 \
        --llm_openai_max_response_tokens 2000
"""
from __future__ import annotations

import logging
from typing import List, Set

from predicators import utils
from predicators.approaches.pg3_analogy_approach import PG3AnalogyApproach
from predicators.envs.base_env import BaseEnv
from predicators.llm_interface import OpenAILLM
from predicators.settings import CFG
from predicators.structs import NSRT, Box, LiftedDecisionList, \
    ParameterizedOption, Predicate, Task, Type


class LLMPG3AnalogyApproach(PG3AnalogyApproach):
    """Use LLM for cross-domain policy learning in PG3."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Set up the LLM.
        self._llm = OpenAILLM(CFG.llm_model_name)

    @classmethod
    def get_name(cls) -> str:
        return "llm_pg3"

    def _induce_policies_by_analogy(
            self, base_policy: LiftedDecisionList, base_env: BaseEnv,
            target_env: BaseEnv, base_nsrts: Set[NSRT],
            target_nsrts: Set[NSRT]) -> List[LiftedDecisionList]:

        base_domain_str = utils.create_pddl_domain(base_nsrts,
                                                   base_env.predicates,
                                                   base_env.types,
                                                   base_env.get_name())

        target_domain_str = utils.create_pddl_domain(target_nsrts,
                                                     target_env.predicates,
                                                     target_env.types,
                                                     target_env.get_name())

        # Create a prompt for the LLM.
        stop_token = "(end policy)"
        prompt = f"""We are going to find an analogy between PDDL domains.

Here is a PDDL domain for {base_env.get_name()}:
{base_domain_str}

Here is a PDDL domain for {target_env.get_name()}:
{target_domain_str}

Here is the policy for {base_env.get_name()}:
{base_policy}
{stop_token}

Here is the policy for {target_env.get_name()}:"""

        logging.info(f"Prompting LLM with: {prompt}")

        # Query the LLM.
        llm_predictions = self._llm.sample_completions(
            prompt=prompt,
            temperature=CFG.llm_temperature,
            seed=CFG.seed,
            num_completions=CFG.llm_num_completions,
            stop_token=stop_token)

        # Parse the responses into init policies.
        init_policies = []
        for ldl_str in llm_predictions:
            logging.info(f"Processing LLM response: {ldl_str}")

            init_policy = utils.parse_ldl_from_str(ldl_str, target_env.types,
                                                   target_env.predicates,
                                                   target_nsrts)
            init_policies.append(init_policy)

        return init_policies
