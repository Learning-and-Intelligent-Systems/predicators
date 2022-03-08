"""Test cases for the GNN policy approach."""

import pytest
import numpy as np
from gym.spaces import Box
from predicators.src.structs import Type, ParameterizedOption, State, Action, \
    Task, Predicate, Dataset, LowLevelTrajectory, GroundAtom
from predicators.src.envs import create_new_env
from predicators.src.approaches import create_approach, ApproachFailure
from predicators.src.datasets import create_dataset
from predicators.src.settings import CFG
from predicators.src import utils


@pytest.mark.parametrize("env_name", ["cover", "cover_typed_options"])
def test_gnn_policy_approach_with_envs(env_name: str):
    """Tests for GNNPolicyApproach class on environments."""
    utils.reset_config({
        "env": env_name,
        "num_train_tasks": 3,
        "num_test_tasks": 3,
        "gnn_policy_num_epochs": 20,
        "gnn_policy_do_normalization": True,
        "max_num_steps_check_policy": 10
    })
    env = create_new_env(env_name)
    train_tasks = env.get_train_tasks()
    approach = create_approach("gnn_policy", env.predicates, env.options,
                               env.types, env.action_space, train_tasks)
    dataset = create_dataset(env, train_tasks)
    assert approach.is_learning_based
    task = env.get_test_tasks()[0]
    with pytest.raises(AssertionError):  # haven't learned yet!
        approach.solve(task, timeout=CFG.timeout)
    approach.learn_from_offline_dataset(dataset)
    policy = approach.solve(task, timeout=CFG.timeout)
    # Test predictions by executing policy.
    utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        task.goal_holds,
        max_num_steps=CFG.max_num_steps_check_policy)
    # Test loading.
    approach2 = create_approach("gnn_policy", env.predicates, env.options,
                                env.types, env.action_space, train_tasks)
    approach2.load(online_learning_cycle=None)


def test_gnn_policy_approach_special_cases():
    """Tests for special cases of the GNNPolicyApproach class."""
    utils.reset_config({
        "env": "cover",
        "gnn_policy_num_epochs": 20,
        "gnn_policy_use_validation_set": False,
        "max_num_steps_check_policy": 10
    })
    cup_type = Type("cup_type", ["feat1"])
    bowl_type = Type("bowl_type", ["feat1"])
    cup = cup_type("cup")

    def _simulator(s, a):
        ns = s.copy()
        assert a.arr.shape == (1, )
        ns[cup][0] += a.arr.item()
        return ns

    action_space = Box(0, 1, (1, ))
    params_space1 = Box(0, 1, (1, ))
    params_space2 = Box(0, 1, (0, ))

    def _policy(_1, _2, _3, p):
        return Action(p)

    def _initiable(s, _1, _2, _3):
        return s[cup][0] > 0.25

    def _solved_classifier(s, o):
        return s[o[0]][0] > 0.5

    Move = ParameterizedOption("Move", [cup_type], params_space1, _policy,
                               _initiable, lambda _1, _2, _3, _4: True)
    Dump = ParameterizedOption("Dump", [], params_space2, _policy, _initiable,
                               lambda _1, _2, _3, _4: True)
    Solved = Predicate("Solved", [cup_type], _solved_classifier)
    Solved2 = Predicate("Solved2", [], lambda s, o: False)
    state = State({cup: [0.3]})
    action1 = Move.ground([cup], np.array([0.5])).policy(state)
    next_state = _simulator(state, action1)
    action2 = Dump.ground([], np.array([])).policy(state)
    train_tasks = [
        Task(state, {Solved([cup])}),
        Task(state, {GroundAtom(Solved2, [])})
    ]
    # Note: from this initial state, the Move option is always non-initiable.
    test_task = Task(State({cup: [0.0]}), train_tasks[0].goal)

    approach = create_approach("gnn_policy", {Solved}, {Move, Dump},
                               {cup_type}, action_space, train_tasks)
    # Test a dataset where max_option_params is 0.
    approach.learn_from_offline_dataset(
        Dataset([
            LowLevelTrajectory([state, next_state], [action2],
                               _is_demo=True,
                               _train_task_idx=1)
        ]))
    approach.learn_from_offline_dataset(
        Dataset([
            LowLevelTrajectory([state, next_state], [action1],
                               _is_demo=True,
                               _train_task_idx=0),
            # For coverage, this is not a demo, so it will be ignored.
            LowLevelTrajectory([state, next_state], [action1])
        ]))

    policy = approach.solve(test_task, timeout=CFG.timeout)
    # Executing the policy should raise an ApproachFailure.
    with pytest.raises(ApproachFailure) as e:
        utils.run_policy_with_simulator(
            policy,
            _simulator,
            test_task.init,
            test_task.goal_holds,
            max_num_steps=CFG.max_num_steps_check_policy)
    assert "GNN policy chose a non-initiable option" in str(e)
    # Hackily change the type of the option so that the policy fails.
    Move.types[0] = bowl_type
    policy = approach.solve(train_tasks[0], timeout=CFG.timeout)
    with pytest.raises(ApproachFailure) as e:
        utils.run_policy_with_simulator(
            policy,
            _simulator,
            test_task.init,
            test_task.goal_holds,
            max_num_steps=CFG.max_num_steps_check_policy)
    assert "GNN policy could not select an object" in str(e)
