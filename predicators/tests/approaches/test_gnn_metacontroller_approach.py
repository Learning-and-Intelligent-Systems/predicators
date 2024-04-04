"""Test cases for the GNN metacontroller approach."""

import numpy as np
import pytest
from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, create_approach
from predicators.src.approaches.gnn_metacontroller_approach import \
    GNNMetacontrollerApproach
from predicators.src.datasets import create_dataset
from predicators.src.envs import create_new_env
from predicators.src.gnn.gnn_utils import get_single_model_prediction
from predicators.src.option_model import _OptionModelBase
from predicators.src.settings import CFG
from predicators.src.structs import Action, Dataset, LowLevelTrajectory, \
    ParameterizedOption, Predicate, State, Task, Type


class _MockGNNMetacontrollerApproach(GNNMetacontrollerApproach):
    """A mock approach that exposes some methods and fields for testing."""

    def graphify_single_input(self, state, atoms, goal):
        """Expose self._graphify_single_input()."""
        return self._graphify_single_input(state, atoms, goal)

    def extract_output_from_graph(self, graph_output, object_to_node):
        """Expose self._extract_output_from_graph()."""
        return self._extract_output_from_graph(graph_output, object_to_node)

    @property
    def gnn(self):
        """Expose self._gnn."""
        return self._gnn


class _MockOptionModel(_OptionModelBase):
    """A mock option model that raises an EnvironmentFailure."""

    def __init__(self, simulator):
        self._simulator = simulator

    def get_next_state_and_num_actions(self, state, option):
        raise utils.EnvironmentFailure("Mock env failure")


@pytest.mark.parametrize("env_name,num_epochs", [("cover", 100),
                                                 ("cover_typed_options", 20)])
def test_gnn_metacontroller_approach_with_envs(env_name, num_epochs):
    """Tests for GNNMetacontrollerApproach class on environments."""
    utils.reset_config({
        "env": env_name,
        "num_train_tasks": 3,
        "num_test_tasks": 3,
        "gnn_num_epochs": num_epochs,
        "gnn_do_normalization": True,
        "sampler_learner": "oracle",
        "horizon": 10
    })
    env = create_new_env(env_name)
    train_tasks = env.get_train_tasks()
    approach = create_approach("gnn_metacontroller", env.predicates,
                               env.options, env.types, env.action_space,
                               train_tasks)
    dataset = create_dataset(env, train_tasks, env.options)
    assert approach.is_learning_based
    task = env.get_test_tasks()[0]
    with pytest.raises(AssertionError):  # haven't learned yet!
        approach.solve(task, timeout=CFG.timeout)
    approach.learn_from_offline_dataset(dataset)
    policy = approach.solve(task, timeout=CFG.timeout)
    # Try executing the policy.
    if num_epochs > 50:
        policy(task.init)  # should be able to sample an option
        # Cover the case where a sampled option leads to an EnvironmentFailure.
        approach._option_model = _MockOptionModel(env.simulate)  # pylint: disable=protected-access
        policy = approach.solve(task, timeout=CFG.timeout)
        with pytest.raises(ApproachFailure) as e:
            policy(task.init)
        assert "GNN metacontroller could not sample an option" in str(e)
    else:
        with pytest.raises(ApproachFailure) as e:
            policy(task.init)  # should NOT be able to sample an option
        assert "GNN metacontroller could not sample an option" in str(e)
    # Test loading.
    approach2 = create_approach("gnn_metacontroller", env.predicates,
                                env.options, env.types, env.action_space,
                                train_tasks)
    approach2.load(online_learning_cycle=None)


@pytest.mark.parametrize("min_data_for_nsrt", [0, 2])
def test_gnn_metacontroller_approach_special_cases(min_data_for_nsrt):
    """Tests for special cases of the GNNMetacontrollerApproach class."""
    utils.reset_config({
        "env": "cover",
        "gnn_num_epochs": 20,
        "gnn_use_validation_set": False,
        "horizon": 10,
        "min_data_for_nsrt": min_data_for_nsrt,
    })
    cup_type = Type("cup_type", ["feat1"])
    bowl_type = Type("bowl_type", ["feat1"])
    cup = cup_type("cup")
    bowl = bowl_type("bowl")

    action_space = Box(0, 1, (1, ))
    params_space = Box(0, 1, (0, ))

    def _policy(_1, _2, _3, p):
        return Action(p)

    def _initiable(s, _1, _2, _3):
        return s[cup][0] > 0.25

    def _solved_classifier(s, o):
        return s[o[0]][0] < 0.5

    Move = ParameterizedOption("Move", [cup_type], params_space, _policy,
                               _initiable, lambda _1, _2, _3, _4: True)
    Dump = ParameterizedOption("Dump", [cup_type], params_space, _policy,
                               _initiable, lambda _1, _2, _3, _4: True)
    Solved = Predicate("Solved", [cup_type], _solved_classifier)
    state = State({cup: [0.3]})
    action1 = Move.ground([cup], np.array([])).policy(state)
    next_state1 = State({cup: [0.3]})
    action2 = Dump.ground([cup], np.array([])).policy(state)
    next_state2 = State({cup: [0.8]})
    action3 = Dump.ground([cup], np.array([])).policy(state)
    next_state3 = State({cup: [0.3]})
    train_tasks = [Task(state, set())]
    # Note: from test_task.init, both Move and Dump are always non-initiable.
    test_task = Task(State({cup: [0.0]}), set())

    approach = _MockGNNMetacontrollerApproach({Solved}, {Move, Dump},
                                              {cup_type}, action_space,
                                              train_tasks)
    approach.learn_from_offline_dataset(
        Dataset([
            LowLevelTrajectory([state, next_state1], [action1],
                               _is_demo=True,
                               _train_task_idx=0),
            LowLevelTrajectory([state, next_state2], [action2],
                               _is_demo=True,
                               _train_task_idx=0),
            LowLevelTrajectory([state, next_state3], [action3],
                               _is_demo=True,
                               _train_task_idx=0),
            # For coverage, this is not a demo, so it will be ignored.
            LowLevelTrajectory([state, next_state1], [action1]),
        ]))

    # Cover the case where the approach can't sample an initiable option.
    policy = approach.solve(test_task, timeout=CFG.timeout)
    with pytest.raises(ApproachFailure) as e:
        policy(test_task.init)
    assert "GNN metacontroller could not sample an option" in str(e)
    # Cover the case where the approach can't select an object.
    in_graph, _ = approach.graphify_single_input(test_task.init, set(), set())
    hacked_object_to_node = {bowl: 0}
    out_graph = get_single_model_prediction(approach.gnn, in_graph)
    with pytest.raises(ApproachFailure) as e:
        approach.extract_output_from_graph(out_graph, hacked_object_to_node)
    assert "GNN metacontroller could not select an object" in str(e)
