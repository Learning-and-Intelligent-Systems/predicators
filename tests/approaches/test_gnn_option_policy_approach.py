"""Test cases for the GNN option policy approach."""
import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    create_approach
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import Action, Dataset, GroundAtom, \
    LowLevelTrajectory, ParameterizedOption, Predicate, State, Task, Type


class _MockOptionModel1(_OptionModelBase):

    def __init__(self, simulator):
        self._simulator = simulator

    def get_next_state_and_num_actions(self, state, option):
        assert option.initiable(state)
        return self._simulator(state, option.policy(state)), 1


class _MockOptionModel2(_OptionModelBase):

    def __init__(self, simulator):
        self._simulator = simulator

    def get_next_state_and_num_actions(self, state, option):
        raise utils.EnvironmentFailure("Mock env failure")


class _MockOptionModel3(_OptionModelBase):

    def __init__(self, simulator):
        self._simulator = simulator

    def get_next_state_and_num_actions(self, state, option):
        return state.copy(), 0


@pytest.mark.parametrize("env_name", ["cover", "cover_typed_options"])
def test_gnn_option_policy_approach_with_envs(env_name):
    """Tests for GNNOptionPolicyApproach class on environments."""
    utils.reset_config({
        "env": env_name,
        "num_train_tasks": 3,
        "num_test_tasks": 3,
        "gnn_option_policy_solve_with_shooting": False,
        "gnn_num_epochs": 20,
        "gnn_do_normalization": True,
        "horizon": 10
    })
    env = create_new_env(env_name)
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = create_approach("gnn_option_policy", env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space, train_tasks)
    predicates, _ = utils.parse_config_excluded_predicates(env)
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()),
                             predicates)
    assert approach.is_learning_based
    task = env.get_test_tasks()[0].task
    with pytest.raises(AssertionError):  # haven't learned yet!
        approach.solve(task, timeout=CFG.timeout)
    approach.learn_from_offline_dataset(dataset)
    policy = approach.solve(task, timeout=CFG.timeout)
    # Test predictions by executing policy.
    utils.run_policy_with_simulator(policy,
                                    env.simulate,
                                    task.init,
                                    task.goal_holds,
                                    max_num_steps=CFG.horizon)
    # Test loading.
    approach2 = create_approach("gnn_option_policy", env.predicates,
                                get_gt_options(env.get_name()), env.types,
                                env.action_space, train_tasks)
    approach2.load(online_learning_cycle=None)


def test_gnn_option_policy_approach_special_cases():
    """Tests for special cases of the GNNOptionPolicyApproach class."""
    utils.reset_config({
        "env": "cover",
        "gnn_num_epochs": 20,
        "gnn_use_validation_set": False,
        "gnn_option_policy_solve_with_shooting": False,
        "horizon": 10
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
    # Note: from test_task.init, both Move and Dump are always non-initiable.
    test_task = Task(State({cup: [0.0]}), train_tasks[0].goal)

    approach = create_approach("gnn_option_policy", {Solved}, {Move, Dump},
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
        utils.run_policy_with_simulator(policy,
                                        _simulator,
                                        test_task.init,
                                        test_task.goal_holds,
                                        max_num_steps=CFG.horizon)
    assert "GNN option policy chose a non-initiable option" in str(e)
    # Hackily change the type of the option so that the policy fails.
    Move.types[0] = bowl_type
    policy = approach.solve(test_task, timeout=CFG.timeout)
    with pytest.raises(ApproachFailure) as e:
        utils.run_policy_with_simulator(policy,
                                        _simulator,
                                        test_task.init,
                                        test_task.goal_holds,
                                        max_num_steps=CFG.horizon)
    assert "GNN option policy could not select an object" in str(e)
    Move.types[0] = cup_type  # revert
    # Now test shooting.
    utils.reset_config({
        "env": "cover",
        "gnn_num_epochs": 20,
        "gnn_use_validation_set": False,
        "gnn_option_policy_solve_with_shooting": True,
        "gnn_option_policy_shooting_max_samples": 1,
        "timeout": 0.1,
        "horizon": 10
    })
    approach._option_model = _MockOptionModel1(_simulator)  # pylint: disable=protected-access
    policy = approach.solve(train_tasks[0], timeout=CFG.timeout)
    traj = utils.run_policy_with_simulator(policy,
                                           _simulator,
                                           train_tasks[0].init,
                                           train_tasks[0].goal_holds,
                                           max_num_steps=CFG.horizon)
    assert train_tasks[0].goal_holds(traj.states[-1])
    approach._option_model = _MockOptionModel2(_simulator)  # pylint: disable=protected-access
    with pytest.raises(ApproachTimeout) as e:
        policy = approach.solve(train_tasks[0], timeout=CFG.timeout)
    assert "Shooting timed out" in str(e)
    with pytest.raises(ApproachTimeout) as e:
        policy = approach.solve(test_task, timeout=CFG.timeout)
    assert "Shooting timed out" in str(e)
    trivial_task = Task(test_task.init, set())
    policy = approach.solve(trivial_task, timeout=CFG.timeout)
    traj = utils.run_policy_with_simulator(policy,
                                           _simulator,
                                           trivial_task.init,
                                           trivial_task.goal_holds,
                                           max_num_steps=CFG.horizon)
    assert trivial_task.goal_holds(traj.states[-1])
    assert len(traj.actions) == 0
    # Now test what happens if we solve the trivial task but roll out
    # in a non-trivial task. We should get an ApproachFailure because
    # the option plan should get exhausted.
    policy = approach.solve(trivial_task, timeout=CFG.timeout)
    with pytest.raises(ApproachFailure) as e:
        utils.run_policy_with_simulator(policy,
                                        _simulator,
                                        test_task.init,
                                        test_task.goal_holds,
                                        max_num_steps=CFG.horizon)
    assert "Option plan exhausted" in str(e)
    # Test that shooting does not infinitely hang in the case where the
    # option model noops.
    approach._option_model = _MockOptionModel3(_simulator)  # pylint: disable=protected-access
    with pytest.raises(ApproachTimeout) as e:
        policy = approach.solve(train_tasks[0], timeout=CFG.timeout)
    assert "Shooting timed out" in str(e)
