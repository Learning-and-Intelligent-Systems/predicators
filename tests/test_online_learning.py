"""Test cases for online learning / interaction with the environment."""

import pytest
from predicators.src.approaches import BaseApproach
from predicators.src.structs import Action, InteractionRequest, \
    InteractionResult
from predicators.src.main import _run_pipeline
from predicators.src.envs import create_env
from predicators.src.teacher import GroundAtomHoldsQuery
from predicators.src import utils
from predicators.src.settings import CFG


class _MockApproach(BaseApproach):
    """Mock approach that generates interaction requests for testing."""

    @property
    def is_learning_based(self):
        return True

    def _solve(self, task, timeout):
        return lambda s: Action(self._action_space.sample())

    def get_interaction_requests(self):
        act_policy = lambda s: Action(self._action_space.sample())
        query_policy1 = lambda s: GroundAtomHoldsQuery("HandEmpty", [])
        termination_function1 = lambda s: True  # terminate immediately
        request1 = InteractionRequest(0, act_policy, query_policy1,
                                      termination_function1)
        query_policy2 = lambda s: None  # no queries
        termination_function2 = lambda s: False  # go until max steps
        request2 = InteractionRequest(1, act_policy, query_policy2,
                                      termination_function2)
        return [request1, request2]

    def learn_from_interaction_results(self, results):
        max_steps = CFG.max_num_steps_interaction_request
        assert len(results) == 2
        result1, result2 = results
        assert isinstance(result1, InteractionResult)
        assert isinstance(result2, InteractionResult)
        assert len(result1.states) == 1
        assert len(result2.states) == max_steps + 1
        response1 = result1.responses[0]
        assert response1.query.predicate_name == "HandEmpty"
        assert response1.query.objects == []
        assert response1.holds
        # request2's queries were all None, so the responses should be too.
        assert result2.responses == [None for _ in range(max_steps + 1)]


def test_interaction():
    """Tests for sending InteractionRequest objects to main.py and receiving
    InteractionResult objects in return."""
    utils.update_config({
        "env": "cover",
        "cover_initial_holding_prob": 0.0,
        "approach": "unittest",
        "excluded_predicates": "",
        "experiment_id": "",
        "load_data": False,
        "load_approach": False,
        "timeout": 1,
        "make_videos": False,
        "num_train_tasks": 2,
        "num_test_tasks": 1
    })
    env = create_env("cover")
    train_tasks = env.get_train_tasks()
    approach = _MockApproach(env.predicates, env.options, env.types,
                             env.action_space, train_tasks)
    _run_pipeline(env, approach, train_tasks)
    utils.update_config({"approach": "nsrt_learning"})
    with pytest.raises(AssertionError) as e:
        _run_pipeline(env, approach, train_tasks)  # invalid query type
        assert "Disallowed query" in str(e)
