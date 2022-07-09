"""Test cases for the open-loop LLM approach."""

import pytest

from predicators.src import utils
from predicators.src.approaches import ApproachFailure
from predicators.src.approaches.open_loop_llm_approach import OpenLoopLLMApproach
from predicators.src.approaches.oracle_approach import OracleApproach
from predicators.src.datasets import create_dataset
from predicators.src.envs import create_new_env
from predicators.src.llm_interface import LargeLanguageModel


def test_open_loop_llm_approach():
    """Tests for OpenLoopLLMApproach()."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({
        "env": env_name,
        "approach": "open_loop_llm",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
    })
    env = create_new_env(env_name)
    train_tasks = env.get_train_tasks()
    approach = OpenLoopLLMApproach(env.predicates, env.options, env.types,
                                    env.action_space, train_tasks)
    assert approach.get_name() == "open_loop_llm"
    # Test "learning", i.e., constructing the prompt prefix.
    dataset = create_dataset(env, train_tasks, env.options)
    assert not approach._prompt_prefix  # pylint: disable=protected-access
    approach.learn_from_offline_dataset(dataset)
    assert approach._prompt_prefix  # pylint: disable=protected-access
    # Create a mock LLM so that we can control the outputs.

    class _MockLLM(LargeLanguageModel):

        def __init__(self):
            self.response = None

        def get_id(self):
            return "dummy"

        def _sample_completions(self,
                                prompt,
                                temperature,
                                seed,
                                num_completions=1):
            return [self.response]

    llm = _MockLLM()
    approach._llm = llm # pylint: disable=protected-access

    # Test successful usage, where the LLM output corresponds to a plan.
    task_idx = 0
    task = train_tasks[task_idx]
    oracle = OracleApproach(env.predicates, env.options, env.types,
                                    env.action_space, train_tasks)
    oracle.solve(task, timeout=500)
    last_plan = oracle.get_last_plan()
    option_to_str = approach._option_to_str  # pylint: disable=protected-access
    # Options and NSRTs are 1:1 for this test / environment.
    ideal_response = "\n".join(map(option_to_str, last_plan))
    llm.response = ideal_response
    # Run the approach.
    policy = approach.solve(task, timeout=500)
    traj, _ = utils.run_policy(policy, env, "train", task_idx,
        task.goal_holds,
        max_num_steps=1000)
    assert task.goal_holds(traj.states[-1])

    


