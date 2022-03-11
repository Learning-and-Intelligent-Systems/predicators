"""Tests for main.py."""

from typing import Callable
import os
import shutil
import sys
import pytest
from predicators.src.approaches import BaseApproach, ApproachFailure, \
    create_approach
from predicators.src.envs import CoverEnv
from predicators.src.main import main, _run_testing
from predicators.src.structs import State, Task, Action
from predicators.src import utils


class _DummyApproach(BaseApproach):
    """Dummy approach that raises ApproachFailure for testing."""

    @property
    def is_learning_based(self):
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        def _policy(s: State) -> Action:
            raise ApproachFailure("Option plan exhausted.")

        return _policy


class _DummyCoverEnv(CoverEnv):
    """Dummy cover environment that raises EnvironmentFailure for testing."""

    def simulate(self, state, action):
        raise utils.EnvironmentFailure("", {"offending_objects": set()})


def test_main():
    """Tests for main.py."""
    sys.argv = [
        "dummy", "--env", "my_env", "--approach", "my_approach", "--seed",
        "123", "--num_test_tasks", "3"
    ]
    with pytest.raises(NotImplementedError):
        main()  # invalid env
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "my_approach", "--seed",
        "123", "--num_test_tasks", "3"
    ]
    with pytest.raises(NotImplementedError):
        main()  # invalid approach
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "random_actions", "--seed",
        "123", "--not-a-real-flag", "0"
    ]
    with pytest.raises(ValueError):
        main()  # invalid flag
    video_dir = os.path.join(os.path.dirname(__file__), "_fake_videos")
    results_dir = os.path.join(os.path.dirname(__file__), "_fake_results")
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "oracle", "--seed", "123",
        "--make_test_videos", "--num_test_tasks", "1", "--video_dir",
        video_dir, "--results_dir", results_dir
    ]
    main()
    # Test making videos of failures.
    sys.argv = [
        "dummy", "--env", "painting", "--approach", "oracle", "--seed", "123",
        "--num_test_tasks", "1", "--video_dir", video_dir, "--results_dir",
        results_dir, "--sesame_max_skeletons_optimized", "1",
        "--painting_lid_open_prob", "0.0", "--make_failure_videos"
    ]
    main()
    shutil.rmtree(video_dir)
    shutil.rmtree(results_dir)
    # Run actual main approach, but without sampler learning.
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "nsrt_learning", "--seed",
        "123", "--sampler_learner", "random", "--cover_initial_holding_prob",
        "0.0", "--num_train_tasks", "3", "--num_test_tasks", "3"
    ]
    main()
    # Try loading approaches.
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "nsrt_learning", "--seed",
        "123", "--load_approach", "--cover_initial_holding_prob", "0.0"
    ]
    main()
    # Try remaking data (this is the default).
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "nsrt_learning", "--seed",
        "123", "--cover_initial_holding_prob", "0.0", "--num_train_tasks", "3",
        "--num_test_tasks", "3"
    ]
    main()
    # Try loading the data.
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "nsrt_learning", "--seed",
        "123", "--load_data", "--cover_initial_holding_prob", "0.0"
    ]
    main()
    # Try running interactive approach with no online learning, to make sure
    # it doesn't crash. This is also an important test of the full pipeline
    # in the case where a goal predicate is excluded. No online learning occurs
    # because max number of transitions is set.
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "interactive_learning",
        "--seed", "123", "--num_online_learning_cycles", "1",
        "--online_learning_max_transitions", "3", "--excluded_predicates",
        "Covers", "--interactive_num_ensemble_members", "1",
        "--num_train_tasks", "3", "--num_test_tasks", "3",
        "--predicate_mlp_classifier_max_itr", "100"
    ]
    main()


def test_bilevel_planning_approach_failure():
    """Test coverage for ApproachFailure in run_testing()."""
    utils.reset_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "timeout": 10,
        "make_test_videos": False,
        "num_test_tasks": 1,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    approach = _DummyApproach(env.predicates, env.options, env.types,
                              env.action_space, train_tasks)
    assert not approach.is_learning_based
    task = train_tasks[0]
    approach.solve(task, timeout=500)
    _run_testing(env, approach)


def test_env_failure():
    """Test coverage for EnvironmentFailure in run_testing()."""
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "timeout": 10,
        "make_test_videos": False,
        "cover_initial_holding_prob": 0.0,
        "num_test_tasks": 1,
    })
    env = _DummyCoverEnv()
    train_tasks = env.get_train_tasks()
    approach = create_approach("random_actions", env.predicates, env.options,
                               env.types, env.action_space, train_tasks)
    assert not approach.is_learning_based
    task = train_tasks[0]
    approach.solve(task, timeout=500)
    _run_testing(env, approach)
