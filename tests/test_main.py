"""Tests for main.py."""

from typing import Callable
import os
import shutil
import sys
import pytest
from predicators.src.approaches import BaseApproach, ApproachFailure, \
    create_approach
from predicators.src.envs import CoverEnv, EnvironmentFailure
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
        raise EnvironmentFailure("", set())


def test_main():
    """Tests for main.py."""
    sys.argv = [
        "dummy", "--env", "my_env", "--approach", "my_approach", "--seed",
        "123", "--num_test_tasks", "5"
    ]
    with pytest.raises(NotImplementedError):
        main()  # invalid env
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "my_approach", "--seed",
        "123", "--num_test_tasks", "5"
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
        "--make_videos", "--num_test_tasks", "1", "--video_dir", video_dir,
        "--results_dir", results_dir
    ]
    main()
    # Test making videos of failures.
    sys.argv = [
        "dummy", "--env", "painting", "--approach", "oracle", "--seed", "123",
        "--num_test_tasks", "1", "--video_dir", video_dir, "--results_dir",
        results_dir, "--max_skeletons_optimized", "1",
        "--painting_lid_open_prob", "0.0", "--make_failure_videos"
    ]
    main()
    shutil.rmtree(video_dir)
    shutil.rmtree(results_dir)
    # Run actual main approach, but without sampler learning.
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "nsrt_learning", "--seed",
        "123", "--sampler_learner", "random", "--cover_initial_holding_prob",
        "0.0"
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
        "123", "--cover_initial_holding_prob", "0.0"
    ]
    main()
    # Try loading the data.
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "nsrt_learning", "--seed",
        "123", "--load_data", "--cover_initial_holding_prob", "0.0"
    ]
    main()
    # Try predicate exclusion.
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "random_options", "--seed",
        "123", "--excluded_predicates", "NotARealPredicate"
    ]
    with pytest.raises(AssertionError):
        main()  # can't exclude a non-existent predicate
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "random_options", "--seed",
        "123", "--excluded_predicates", "Covers"
    ]
    with pytest.raises(AssertionError):
        main()  # can't exclude a goal predicate
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "random_options", "--seed",
        "123", "--excluded_predicates", "all", "--num_test_tasks", "5",
        "--cover_initial_holding_prob", "0.0"
    ]
    main()


def test_tamp_approach_failure():
    """Test coverage for ApproachFailure in run_testing()."""
    utils.update_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "seed": 123,
        "timeout": 10,
        "make_videos": False,
    })
    env = CoverEnv()
    approach = _DummyApproach(env.predicates, env.options, env.types,
                              env.action_space)
    assert not approach.is_learning_based
    task = env.get_train_tasks()[0]
    approach.solve(task, timeout=500)
    _run_testing(env, approach)


def test_env_failure():
    """Test coverage for EnvironmentFailure in run_testing()."""
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "seed": 123,
        "timeout": 10,
        "make_videos": False,
        "cover_initial_holding_prob": 0.0,
    })
    env = _DummyCoverEnv()
    approach = create_approach("random_actions", env.predicates, env.options,
                               env.types, env.action_space)
    assert not approach.is_learning_based
    task = env.get_train_tasks()[0]
    approach.solve(task, timeout=500)
    _run_testing(env, approach)
