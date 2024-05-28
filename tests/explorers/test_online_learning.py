"""Test cases for online learning / interaction with the environment."""
import pytest

from predicators import utils
from predicators.approaches import BaseApproach
from predicators.cogman import CogMan
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.main import _run_pipeline
from predicators.perception import create_perceiver
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, GroundAtomsHoldQuery, \
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
        HandEmpty = Predicate("HandEmpty", [], lambda s, o: False)
        hand_empty_atom = GroundAtom(HandEmpty, [])
        act_policy1 = lambda s: Action(self._action_space.sample())
        query_policy1 = lambda s: GroundAtomsHoldQuery({hand_empty_atom})
        termination_function1 = lambda s: True  # terminate immediately
        request1 = InteractionRequest(1, act_policy1, query_policy1,
                                      termination_function1)
        act_policy2 = act_policy1
        query_policy2 = lambda s: None  # no queries
        termination_function2 = lambda s: False  # go until max steps
        request2 = InteractionRequest(2, act_policy2, query_policy2,
                                      termination_function2)

        def act_policy3(state):
            raise utils.RequestActPolicyFailure("mock failure")

        query_policy3 = lambda s: None  # no queries
        termination_function3 = lambda s: False  # go until max steps
        request3 = InteractionRequest(3, act_policy3, query_policy3,
                                      termination_function3)
        return [request1, request2, request3]

    def learn_from_interaction_results(self, results):
        max_steps = CFG.max_num_steps_interaction_request + 1
        assert len(results) == 3
        result1, result2, result3 = results
        assert isinstance(result1, InteractionResult)
        assert isinstance(result2, InteractionResult)
        assert isinstance(result3, InteractionResult)
        assert len(result1.states) == 1
        assert len(result2.states) == max_steps + 1
        assert len(result3.states) == 1
        response1 = result1.responses[0]
        assert len(response1.query.ground_atoms) == 1
        query_atom = next(iter(response1.query.ground_atoms))
        assert query_atom.predicate.name == "HandEmpty"
        assert query_atom.objects == []
        assert len(response1.holds) == 1
        assert next(iter(response1.holds.values()))
        # request2's queries were all None, so the responses should be too.
        assert result2.responses == [None for _ in range(max_steps + 1)]
        # In request3, the acting policy raised a RequestActPolicyFailure,
        # which ends the interaction immediately.
        assert len(result3.actions) == 0
        assert result3.responses == [None]
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
        "num_train_tasks": 4,
        "num_test_tasks": 1,
        "num_online_learning_cycles": 1,
        "make_interaction_videos": True,
        "max_num_steps_interaction_request": 3,
    })
    env = create_new_env("cover")
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = _MockApproach(env.predicates, get_gt_options(env.get_name()),
                             env.types, env.action_space, train_tasks)
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, exec_monitor)
    predicates, _ = utils.parse_config_excluded_predicates(env)
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()),
                             predicates)
    _run_pipeline(env, cogman, train_tasks, dataset)
    utils.update_config({
        "approach": "bridge_policy",
        "load_data": True,
        "make_interaction_videos": False,
    })
    env = create_new_env("cover")
    # Invalid query type.
    with pytest.raises(AssertionError) as e:
        _run_pipeline(env, cogman, train_tasks, dataset)
    assert "Disallowed query" in str(e)
    # Learning-based approaches expect a dataset.
    with pytest.raises(AssertionError) as e:
        _run_pipeline(env, cogman, train_tasks)
    assert "Missing offline dataset" in str(e)
    # Test loading the approach.
    utils.update_config({
        "approach": "unittest",
        "load_data": True,
        "load_approach": True
    })
    _run_pipeline(env, cogman, train_tasks, dataset)
    # Should succeed because all cycles are skipped. Note that we must
    # reset_config instead of update_config because of known issues with
    # update_config and default args.
    utils.reset_config({
        "num_online_learning_cycles": 10,
        "skip_until_cycle": 11,
        "env": "cover",
        "cover_initial_holding_prob": 0.0,
        "approach": "unittest",
        "timeout": 1,
        "num_train_tasks": 4,
        "num_test_tasks": 1,
        "make_interaction_videos": True,
        "max_num_steps_interaction_request": 3,
    })
    _run_pipeline(env, cogman, train_tasks, dataset)
    # Tests for CFG.allow_interaction_in_demo_tasks. An error should be raised
    # because the agent makes a request about a task where a demonstration was
    # generated.
    utils.reset_config({
        "max_initial_demos": 2,
        "allow_interaction_in_demo_tasks": False,
        "num_online_learning_cycles": 1,
        "env": "cover",
        "cover_initial_holding_prob": 0.0,
        "approach": "unittest",
        "timeout": 1,
        "num_train_tasks": 4,
        "num_test_tasks": 1,
        "make_interaction_videos": True,
        "max_num_steps_interaction_request": 3,
    })
    with pytest.raises(RuntimeError) as e:
        _run_pipeline(env, cogman, train_tasks, dataset)
    assert "Interaction requests cannot be on demo tasks" in str(e)
    # This should succeed since requests are about the last three train tasks.
    utils.reset_config({
        "max_initial_demos": 1,
        "allow_interaction_in_demo_tasks": False,
        "num_online_learning_cycles": 3,
        "env": "cover",
        "cover_initial_holding_prob": 0.0,
        "approach": "unittest",
        "timeout": 1,
        "num_train_tasks": 4,
        "num_test_tasks": 1,
        "make_interaction_videos": True,
        "max_num_steps_interaction_request": 3,
    })
    env = create_new_env("cover")
    _run_pipeline(env, cogman, train_tasks, dataset)
