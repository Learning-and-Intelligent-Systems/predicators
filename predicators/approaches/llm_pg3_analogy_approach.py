"""Use LLMs for cross-domain policy learning in PG3.

Command for testing gripper / ferry:
    python predicators/main.py --approach llm_pg3 --seed 0  \
        --env pddl_ferry_procedural_tasks --strips_learner oracle \
        --num_train_tasks 20 --pg3_init_policy gripper_ldl_policy.txt \
        --pg3_init_base_env pddl_gripper_procedural_tasks \
        --pg3_add_condition_allow_new_vars False \
        --llm_model_name text-davinci-003 \
        --llm_temperature 0.5 \
        --llm_openai_max_response_tokens 1000 \
        --llm_pg3_num_completions 10
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

        base_goal_predicates = ", ".join([str(p) for p in base_env.goal_predicates])
        target_goal_predicates = ", ".join([str(p) for p in target_env.goal_predicates])

        # Create a prompt for the LLM.
        stop_token = "(end policy)"
        prompt = f"""We are going to find an analogy between PDDL domains.

PDDL domain for {base_env.get_name()}:
{base_domain_str}

Goal predicates for {base_env.get_name()}:
{base_goal_predicates}

Policy for {base_env.get_name()}:
{base_policy}
{stop_token}

PDDL domain for {target_env.get_name()}:
{target_domain_str}

Goal predicates for {target_env.get_name()}:
{target_goal_predicates}

Policy for {target_env.get_name()}:"""

        logging.info(f"Prompting LLM with: {prompt}")

        # Query the LLM.
        llm_predictions = []
        for i in range(CFG.llm_pg3_num_completions):
            single_predictions = self._llm.sample_completions(
                prompt=prompt,
                temperature=CFG.llm_temperature,
                seed=(CFG.seed + i),
                num_completions=1,
                stop_token=stop_token)
            llm_predictions.extend(single_predictions)

        # Parse the responses into init policies.
        # Always start with an empty policy in case the output policies are
        # all garbage.
        init_policies = [LiftedDecisionList([])]
        for ldl_str in llm_predictions:
            logging.info(f"Processing LLM response: {ldl_str}")

            # In the future, we should parse more gracefully, but for now just
            # skip any policy that is not perfectly formed.
            try:
                init_policy = utils.parse_ldl_from_str(ldl_str, target_env.types,
                                                    target_env.predicates,
                                                    target_env.goal_predicates,
                                                    target_nsrts,
                                                    add_missing_preconditions=True,
                                                    remove_invalid_goal_preconditions=True)
            except (AssertionError, ValueError, IndexError, KeyError):
                continue

            init_policies.append(init_policy)

        return init_policies
