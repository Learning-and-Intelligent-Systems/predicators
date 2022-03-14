"""Test cases for online learning / interaction with the environment."""

import pytest

from predicators.src import utils
from predicators.src.approaches import BaseApproach
from predicators.src.datasets import create_dataset
from predicators.src.envs import create_new_env
from predicators.src.main import _run_pipeline
from predicators.src.settings import CFG
from predicators.src.structs import Action, GroundAtom, GroundAtomsHoldQuery, \
    InteractionRequest, InteractionResult, Predicate


class _MockApproach(BaseApproach):
    """Mock approach that generates interaction requests for testing."""

    def __init__(self, initial_predicates, initial_options, types,
                 action_space, train_tasks):
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._dummy_saved = []

    @classmethod
    def get_name(cls) -> str:
        return "dummy"

    @property
    def is_learning_based(self):
        return True

    def _solve(self, task, timeout):
        return lambda s: Action(self._action_space.sample())

    def learn_from_offline_dataset(self, dataset):
        self._dummy_saved.append(None)

    def get_interaction_requests(self):
        act_policy = lambda s: Action(self._action_space.sample())
        HandEmpty = Predicate("HandEmpty", [], lambda s, o: False)
        hand_empty_atom = GroundAtom(HandEmpty, [])
        query_policy1 = lambda s: GroundAtomsHoldQuery({hand_empty_atom})
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
        assert len(response1.query.ground_atoms) == 1
        query_atom = next(iter(response1.query.ground_atoms))
        assert query_atom.predicate.name == "HandEmpty"
        assert query_atom.objects == []
        assert len(response1.holds) == 1
        assert next(iter(response1.holds.values()))
        # request2's queries were all None, so the responses should be too.
        assert result2.responses == [None for _ in range(max_steps + 1)]
        assert self._dummy_saved
        next_saved = (0 if self._dummy_saved[-1] is None else
                      self._dummy_saved[-1] + 1)
        self._dummy_saved.append(next_saved)

    def load(self, online_learning_cycle):
        assert self._dummy_saved.pop(0) == online_learning_cycle


def test_interaction():
    """Tests for sending InteractionRequest objects to main.py and receiving
    InteractionResult objects in return."""
    utils.reset_config({
        "env": "cover",
        "cover_initial_holding_prob": 0.0,
        "approach": "unittest",
        "timeout": 1,
        "num_train_tasks": 2,
        "num_test_tasks": 1,
        "num_online_learning_cycles": 1,
        "make_interaction_videos": True,
        "max_num_steps_interaction_request": 3,
    })
    env = create_new_env("cover")
    train_tasks = env.get_train_tasks()
    approach = _MockApproach(env.predicates, env.options, env.types,
                             env.action_space, train_tasks)
    dataset = create_dataset(env, train_tasks)
    _run_pipeline(env, approach, train_tasks, dataset)
    utils.update_config({
        "approach": "nsrt_learning",
        "load_data": True,
        "make_interaction_videos": False,
    })
    # Invalid query type.
    with pytest.raises(AssertionError) as e:
        _run_pipeline(env, approach, train_tasks, dataset)
    assert "Disallowed query" in str(e)
    # Learning-based approaches expect a dataset.
    with pytest.raises(AssertionError) as e:
        _run_pipeline(env, approach, train_tasks)
    assert "Missing offline dataset" in str(e)
    # Test loading the approach.
    utils.update_config({
        "approach": "unittest",
        "load_data": True,
        "load_approach": True
    })
    _run_pipeline(env, approach, train_tasks, dataset)
