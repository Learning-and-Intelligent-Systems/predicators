"""Test cases for the open-loop LLM approach."""
import shutil

import pytest

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.llm_open_loop_approach import LLMOpenLoopApproach
from predicators.approaches.oracle_approach import OracleApproach
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.pretrained_model_interface import LargeLanguageModel


def test_llm_open_loop_approach():
    """Tests for LLMOpenLoopApproach()."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    cache_dir = "_fake_llm_cache_dir"
    utils.reset_config({
        "env": env_name,
        "pretrained_model_prompt_cache_dir": cache_dir,
        "approach": "llm_open_loop",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
        "offline_data_method": "demo+replay",
        "offline_data_num_replays": 3,
    })
    env = create_new_env(env_name)
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = LLMOpenLoopApproach(env.predicates,
                                   get_gt_options(env.get_name()), env.types,
                                   env.action_space, train_tasks)
    assert approach.get_name() == "llm_open_loop"
    # Test "learning", i.e., constructing the prompt prefix.
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    assert not approach._prompt_prefix  # pylint: disable=protected-access
    approach.learn_from_offline_dataset(dataset)
    assert approach._prompt_prefix  # pylint: disable=protected-access

    # Create a mock LLM so that we can control the outputs.

    class _MockLLM(LargeLanguageModel):

        def __init__(self):
            self.response = None

        def get_id(self):
            return f"dummy-{hash(self.response)}"

        def _sample_completions(self,
                                prompt,
                                imgs,
                                temperature,
                                seed,
                                stop_token=None,
                                num_completions=1):
            del prompt, temperature, seed, stop_token, num_completions, imgs
            return [self.response]

    llm = _MockLLM()
    approach._llm = llm  # pylint: disable=protected-access

    # Test successful usage, where the LLM output corresponds to a plan.
    task_idx = 0
    task = train_tasks[task_idx]
    oracle = OracleApproach(env.predicates, get_gt_options(env.get_name()),
                            env.types, env.action_space, train_tasks)
    oracle.solve(task, timeout=500)
    last_plan = oracle.get_last_plan()
    option_to_str = approach._option_to_str  # pylint: disable=protected-access
    # Options and NSRTs are 1:1 for this test / environment.
    ideal_response = "\n".join(map(option_to_str, last_plan))
    # Add an empty line to the ideal response, should be no problem.
    ideal_response = "\n" + ideal_response
    llm.response = ideal_response
    # Run the approach.
    policy = approach.solve(task, timeout=500)
    traj, _ = utils.run_policy(policy,
                               env,
                               "train",
                               task_idx,
                               task.goal_holds,
                               max_num_steps=1000)
    assert task.goal_holds(traj.states[-1])

    # Test general approach failures.
    llm.response = "garbage"
    policy = approach.solve(task, timeout=500)
    with pytest.raises(ApproachFailure) as e:
        utils.run_policy(policy,
                         env,
                         "train",
                         task_idx,
                         task.goal_holds,
                         max_num_steps=1000)
    assert "No LLM predicted plan achieves the goal." in str(e)

    llm.response = ideal_response
    original_nsrts = approach._nsrts  # pylint: disable=protected-access
    approach._nsrts = set()  # pylint: disable=protected-access
    policy = approach.solve(task, timeout=500)
    with pytest.raises(ApproachFailure) as e:
        utils.run_policy(policy,
                         env,
                         "train",
                         task_idx,
                         task.goal_holds,
                         max_num_steps=1000)
    assert "No LLM predicted plan achieves the goal." in str(e)
    approach._nsrts = original_nsrts  # pylint: disable=protected-access

    # Test failure cases of _llm_prediction_to_option_plan().
    objects = set(task.init)
    assert approach._llm_prediction_to_option_plan(ideal_response, objects)  # pylint: disable=protected-access
    # Case where a line does not contain a valid option.
    response = "garbage\n" + ideal_response
    option_plan = approach._llm_prediction_to_option_plan(response, objects)  # pylint: disable=protected-access
    assert not option_plan
    # Case where object types are malformed.
    response = ideal_response.replace(":", "-")
    option_plan = approach._llm_prediction_to_option_plan(response, objects)  # pylint: disable=protected-access
    assert not option_plan
    # Case where object names are incorrect.
    response = ideal_response.replace(":", "-dummy:")
    option_plan = approach._llm_prediction_to_option_plan(response, objects)  # pylint: disable=protected-access
    assert not option_plan
    # Case where type names are incorrect.
    response = ideal_response.replace(":", ":dummy-")
    option_plan = approach._llm_prediction_to_option_plan(response, objects)  # pylint: disable=protected-access
    assert not option_plan
    # Case where types are correct, but the number of objects is wrong.
    assert ideal_response.startswith("\npick-up(paper-0:paper, loc-0:loc)")
    response = "\npick-up(paper-0:paper)"
    option_plan = approach._llm_prediction_to_option_plan(response, objects)  # pylint: disable=protected-access
    assert not option_plan

    shutil.rmtree(cache_dir)
