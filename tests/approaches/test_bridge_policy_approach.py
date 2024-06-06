"""Test cases for the BridgePolicyApproach class."""

from unittest.mock import patch

import numpy as np
import pytest

import predicators.approaches.bridge_policy_approach
import predicators.bridge_policies.oracle_bridge_policy
import predicators.teacher
from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.bridge_policy_approach import \
    BridgePolicyApproach, RLBridgePolicyApproach
from predicators.bridge_policies import BridgePolicyDone
from predicators.cogman import CogMan
from predicators.envs import get_or_create_env
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.main import _generate_interaction_results, _run_testing
from predicators.perception import create_perceiver
from predicators.settings import CFG
from predicators.structs import Action, DemonstrationResponse, DummyOption, \
    InteractionResult, LowLevelTrajectory, STRIPSOperator
from predicators.teacher import Teacher

_APPROACH_PATH = predicators.approaches.bridge_policy_approach.__name__
_ORACLE_PATH = predicators.bridge_policies.oracle_bridge_policy.__name__
_TEACHER_PATH = predicators.teacher.__name__


def test_bridge_policy_approach():
    """Tests for BridgePolicyApproach class."""
    # Test oracle bridge policy in painting.
    args = {
        "approach": "bridge_policy",
        "bridge_policy": "oracle",
        "env": "painting",
        "painting_lid_open_prob": 0.0,
        "painting_raise_environment_failure": False,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
    }
    utils.reset_config(args)
    env = get_or_create_env(CFG.env)
    train_tasks = [t.task for t in env.get_train_tasks()]
    test_tasks = [t.task for t in env.get_test_tasks()]
    approach = BridgePolicyApproach(env.predicates,
                                    get_gt_options(env.get_name()), env.types,
                                    env.action_space, train_tasks)
    assert approach.get_name() == "bridge_policy"
    assert not approach.is_learning_based
    task = test_tasks[0]
    policy = approach.solve(task, timeout=500)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           task.goal_holds,
                                           max_num_steps=CFG.horizon)
    assert task.goal_holds(traj.states[-1])

    # Test bridge policy timeout.
    with patch("time.perf_counter") as m:
        m.return_value = float("inf")
        with pytest.raises(ApproachTimeout) as e:
            policy(task.init)
        assert "Bridge policy timed out" in str(e)

    # Test case where bridge policy hands back control to planner immediately.
    # The policy should get stuck and detect a loop.
    def done_option_policy(s):
        del s  # ununsed
        raise BridgePolicyDone()

    with patch(f"{_ORACLE_PATH}.OracleBridgePolicy.get_option_policy") as m:
        m.return_value = done_option_policy
        policy = approach.solve(task, timeout=500)
        with pytest.raises(ApproachFailure) as e:
            traj = utils.run_policy_with_simulator(policy,
                                                   env.simulate,
                                                   task.init,
                                                   task.goal_holds,
                                                   max_num_steps=25)
        assert "Loop detected" in str(e)

    # Test case where the second time that the planner is called, it returns
    # an invalid option.
    first_policy = approach._get_option_policy_by_planning(task, timeout=500)  # pylint: disable=protected-access

    def second_policy(s):
        del s  # unused
        raise utils.OptionExecutionFailure("Second planning failed.")

    p = f"{_APPROACH_PATH}.BridgePolicyApproach._get_option_policy_by_planning"
    with patch(p) as m:
        m.side_effect = [first_policy, second_policy]
        policy = approach.solve(task, timeout=500)
        with pytest.raises(ApproachFailure) as e:
            utils.run_policy_with_simulator(policy,
                                            env.simulate,
                                            task.init,
                                            task.goal_holds,
                                            max_num_steps=CFG.horizon)
        assert "Second planning failed" in str(e)

    # Test case where task planning returns a non-initiable option.
    op = STRIPSOperator("Dummy", [], set(), set(), set(), set())
    dummy_nsrt = op.make_nsrt(
        DummyOption.parent,
        [],  # dummy sampler
        lambda s, g, rng, o: np.zeros(1, dtype=np.float32))

    path = f"{_APPROACH_PATH}.BridgePolicyApproach._run_task_plan"
    with patch(path) as m:
        m.return_value = ([dummy_nsrt.ground([])], [set(), set()], {})
        policy = approach.solve(task, timeout=500)
        with pytest.raises(ApproachFailure) as e:
            utils.run_policy_with_simulator(policy,
                                            env.simulate,
                                            task.init,
                                            task.goal_holds,
                                            max_num_steps=CFG.horizon)
        assert "Planning failed on init state" in str(e)

    # Test oracle bridge policy in stick button.
    args = {
        "approach": "bridge_policy",
        "bridge_policy": "oracle",
        "env": "stick_button",
        "num_train_tasks": 0,
        "num_test_tasks": 1,
    }
    utils.reset_config(args)
    env = get_or_create_env(CFG.env)
    train_tasks = [t.task for t in env.get_train_tasks()]
    test_tasks = [t.task for t in env.get_test_tasks()]
    approach = BridgePolicyApproach(env.predicates,
                                    get_gt_options(env.get_name()), env.types,
                                    env.action_space, train_tasks)
    assert approach.get_name() == "bridge_policy"
    task = test_tasks[0]
    policy = approach.solve(task, timeout=500)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           task.goal_holds,
                                           max_num_steps=CFG.horizon)
    assert task.goal_holds(traj.states[-1])

    # Test bridge policy LDL learning in painting.
    args = {
        "approach": "bridge_policy",
        "bridge_policy": "learned_ldl",
        "env": "painting",
        "painting_lid_open_prob": 0.0,
        "painting_raise_environment_failure": False,
        "max_initial_demos": 0,
        "interactive_num_requests_per_cycle": 1,
        "num_online_learning_cycles": 1,
        "segmenter": "oracle",
        "demonstrator": "human",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    }
    utils.reset_config(args)
    env = get_or_create_env(CFG.env)
    train_tasks = [t.task for t in env.get_train_tasks()]
    test_tasks = [t.task for t in env.get_test_tasks()]
    approach = BridgePolicyApproach(env.predicates,
                                    get_gt_options(env.get_name()), env.types,
                                    env.action_space, train_tasks)
    assert approach.get_name() == "bridge_policy"
    assert approach.is_learning_based
    assert len(train_tasks) == 1
    place_action = Action(
        np.array([env.obj_x, -4.0, env.obj_z, 0.5, -1.0, 0.0, 0.0, 0.0],
                 dtype=np.float32))
    box_open_action = Action(
        np.array([
            env.obj_x,
            (env.box_lb + env.box_ub) / 2, env.obj_z, 0.0, 1.0, 0.0, 0.0, 0.0
        ],
                 dtype=np.float32))
    pick_top_action = Action(
        np.array([env.obj_x, -4.0, env.obj_z, 1.0, 1.0, 0.0, 0.0, 0.0],
                 dtype=np.float32))
    action_queue = [place_action, box_open_action, pick_top_action]

    def _mock_human_demonstratory_policy(*args, **kwargs):
        del args, kwargs
        if not action_queue:
            raise utils.HumanDemonstrationFailure("Done demonstrating.")
        return action_queue.pop(0)

    with patch(f"{_TEACHER_PATH}.human_demonstrator_policy") as m:
        m.side_effect = _mock_human_demonstratory_policy
        interaction_requests = approach.get_interaction_requests()
        teacher = Teacher(train_tasks)
        perceiver = create_perceiver("trivial")
        exec_monitor = create_execution_monitor("trivial")
        cogman = CogMan(approach, perceiver, exec_monitor)
        interaction_results, _ = _generate_interaction_results(
            cogman, env, teacher, interaction_requests)
    real_result = interaction_results[0]
    # Add additional interaction result with no queries.
    interaction_results.append(
        InteractionResult(states=real_result.states[:1],
                          actions=[],
                          responses=[None]))
    # Add additional interaction result where the demonstration has just one
    # timestep in it (so segmentation is trivial).
    real_query = real_result.responses[-1].query
    real_teacher_traj = real_result.responses[-1].teacher_traj
    one_step_teacher_traj = LowLevelTrajectory(
        _states=real_teacher_traj.states[:1],
        _actions=[],
        _is_demo=True,
        _train_task_idx=0)
    interaction_results.append(
        InteractionResult(states=real_result.states[-1:],
                          actions=[],
                          responses=[
                              DemonstrationResponse(real_query,
                                                    one_step_teacher_traj)
                          ]))
    approach.learn_from_interaction_results(interaction_results)
    # Cover learning from no additional interaction results.
    approach.learn_from_interaction_results([])
    task = test_tasks[0]
    policy = approach.solve(task, timeout=500)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           task.goal_holds,
                                           max_num_steps=CFG.horizon)
    assert task.goal_holds(traj.states[-1])


def test_rl_bridge_policy_approach():
    """Tests for RLBridgePolicyApproach class."""
    # Test oracle bridge policy in grid_row_door.
    args = {
        "approach": "rl_bridge_policy",
        "env": "grid_row_door",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "grid_row_num_cells": 3,
        "strips_learner": "oracle",
        "sampler_learner": "oracle",
        "explorer": "maple_q",
        "interactive_num_requests_per_cycle": 10,
        "online_nsrt_learning_requests_per_cycle": 10,
        "max_initial_demos": 0
    }
    utils.reset_config(args)
    env = get_or_create_env(CFG.env)
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = RLBridgePolicyApproach(env.predicates,
                                      get_gt_options(env.get_name()),
                                      env.types, env.action_space, train_tasks)
    assert approach.get_name() == "rl_bridge_policy"
    assert approach.is_learning_based
    # interaction_requests = approach.get_interaction_requests()
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, exec_monitor)
    num_online_transitions = 0
    total_query_cost = 0
    # Run online interaction once.
    interaction_requests = cogman.get_interaction_requests()
    interaction_results, query_cost = _generate_interaction_results(
        cogman, env, teacher=None, requests=interaction_requests, cycle_num=0)
    num_online_transitions += sum(
        len(result.actions) for result in interaction_results)
    total_query_cost += query_cost
    # Learn from online interaction results, unless we are loading
    # and not restarting learning.
    if not CFG.load_approach or CFG.restart_learning:
        cogman.learn_from_interaction_results(interaction_results)
    # We should be adding more data to replay buffer
    assert len(approach.mapleq._q_function._replay_buffer) > 0  # pylint: disable=protected-access
    # Test that reward is positive for some trial
    gets_reward = False
    for (_, _, _, _, reward, _) in approach.mapleq._q_function._replay_buffer:  # pylint: disable=protected-access
        if reward > 0:
            gets_reward = True
    assert gets_reward

    # Evaluate approach after 5 online learning cycles.
    # We should have learned correct policy by now
    results = _run_testing(env, cogman)
    results["num_online_transitions"] = num_online_transitions
    #more test cases for solve??
    #ig u can straight up just run 3 grid cells !?
    #like get interaction requests from cogman and stuff

    #test Make sure learn_from_interaction_results
    #correctly segments trajectories
    #so like confirm that there are an increasing
    #number of trajectories each cycle
