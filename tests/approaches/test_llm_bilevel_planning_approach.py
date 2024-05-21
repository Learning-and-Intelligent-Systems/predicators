"""Test cases for the LLM bilevel_planning approach."""
import shutil

from predicators import utils
from predicators.approaches.llm_bilevel_planning_approach import \
    LLMBilevelPlanningApproach
from predicators.approaches.oracle_approach import OracleApproach
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.pretrained_model_interface import LargeLanguageModel


def test_llm_bilevel_planning_approach():
    """Tests for LLMBilevelPlanningApproach()."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    cache_dir = "_fake_llm_cache_dir"
    utils.reset_config({
        "env": env_name,
        "pretrained_model_prompt_cache_dir": cache_dir,
        "approach": "llm_bilevel_planning",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
    })
    env = create_new_env(env_name)
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = LLMBilevelPlanningApproach(env.predicates,
                                          get_gt_options(env.get_name()),
                                          env.types, env.action_space,
                                          train_tasks)
    assert approach.get_name() == "llm_bilevel_planning"
    # Test "learning", i.e., constructing the prompt prefix.
    predicates, _ = utils.parse_config_excluded_predicates(env)
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()),
                             predicates)
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
    ideal_metrics = approach.metrics
    approach.reset_metrics()

    # If the LLM response is garbage, we should still find a plan that achieves
    # the goal, because we will just fall back to regular planning.
    llm.response = "garbage"
    policy = approach.solve(task, timeout=500)
    traj, _ = utils.run_policy(policy,
                               env,
                               "train",
                               task_idx,
                               task.goal_holds,
                               max_num_steps=1000)
    assert task.goal_holds(traj.states[-1])
    worst_case_metrics = approach.metrics
    approach.reset_metrics()

    # If the LLM response is suggests an invalid action, the plan should not
    # be used after that. In this example, the plan will just be to deliver
    # to a location that we're not yet at.
    llm.response = "\n".join(ideal_response.split("\n")[-1:])
    policy = approach.solve(task, timeout=500)
    traj, _ = utils.run_policy(policy,
                               env,
                               "train",
                               task_idx,
                               task.goal_holds,
                               max_num_steps=1000)
    assert task.goal_holds(traj.states[-1])
    worst_case_metrics2 = approach.metrics
    assert worst_case_metrics2["total_num_nodes_created"] == \
        worst_case_metrics["total_num_nodes_created"]
    approach.reset_metrics()

    # If the LLM response is almost perfect, it should be very helpful for
    # planning guidance.
    llm.response = "\n".join(ideal_response.split("\n")[:-1])
    policy = approach.solve(task, timeout=500)
    traj, _ = utils.run_policy(policy,
                               env,
                               "train",
                               task_idx,
                               task.goal_holds,
                               max_num_steps=1000)
    assert task.goal_holds(traj.states[-1])
    almost_ideal_metrics = approach.metrics
    worst_case_nodes = worst_case_metrics["total_num_nodes_created"]
    almost_ideal_nodes = almost_ideal_metrics["total_num_nodes_created"]
    ideal_nodes = ideal_metrics["total_num_nodes_created"]
    assert worst_case_nodes > almost_ideal_nodes
    assert almost_ideal_nodes > ideal_nodes
    approach.reset_metrics()

    shutil.rmtree(cache_dir)
