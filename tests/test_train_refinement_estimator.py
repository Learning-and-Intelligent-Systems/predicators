"""Tests for train_refinement_estimator.py."""

import os
import shutil
import sys
import tempfile

import pytest

from predicators import utils
from predicators.envs.narrow_passage import NarrowPassageEnv
from predicators.option_model import create_option_model
from predicators.planning import PlanningFailure
from predicators.train_refinement_estimator import \
    _collect_refinement_data_for_task, _get_refinement_estimation_parser, \
    _train_refinement_estimation_approach


def test_train_refinement_estimator():
    """Tests for train_refinement_estimator.py."""
    parser = _get_refinement_estimation_parser()
    utils.reset_config_with_parser(
        parser, {
            "env": "narrow_passage",
            "narrow_passage_door_width_padding_lb": 0.05,
            "narrow_passage_door_width_padding_ub": 0.05,
            "narrow_passage_passage_width_padding_lb": 0.05,
            "narrow_passage_passage_width_padding_ub": 0.05,
            "num_train_tasks": 1,
        })
    sys.argv = [
        "dummy", "--env", "narrow_passage", "--approach", "oracle", "--seed",
        "123", "--num_train_tasks", "3"
    ]
    with pytest.raises(AssertionError):
        _train_refinement_estimation_approach()  # invalid approach
    sys.argv = [
        "dummy", "--env", "narrow_passage", "--approach",
        "refinement_estimation", "--seed", "123"
    ]
    with pytest.raises(AssertionError):
        _train_refinement_estimation_approach()  # invalid refinement estimator
    sys.argv = [
        "dummy", "--env", "narrow_passage", "--approach",
        "refinement_estimation", "--seed", "123", "--not-a-real-flag", "0"
    ]
    with pytest.raises(ValueError):
        _train_refinement_estimation_approach()  # invalid flag
    parent_dir = os.path.dirname(__file__)
    data_dir = os.path.join(parent_dir, "_fake_data")
    approach_dir = os.path.join(parent_dir, "_fake_approach")

    # Test successful data generation and training
    temp_log_file = tempfile.NamedTemporaryFile(delete=False).name
    train_sys_argv = [
        "dummy",
        "--env",
        "narrow_passage",
        "--approach",
        "refinement_estimation",
        "--refinement_estimator",
        "tabular",
        "--seed",
        "123",
        "--num_train_tasks",
        "1",
        "--approach_dir",
        approach_dir,
        "--data_dir",
        data_dir,
        "--refinement_data_file_name",
        "test.data",
        "--refinement_data_save_every",
        "1",
        "--log_file",
        temp_log_file,
        "--refinement_train_with_frac_data",
        "1.1",
    ]
    sys.argv = train_sys_argv
    _train_refinement_estimation_approach()

    # Test training from loaded data
    sys.argv = train_sys_argv + ["--load_data"]
    _train_refinement_estimation_approach()

    # Test skipping training
    sys.argv = train_sys_argv + ["--skip_refinement_estimator_training"]
    _train_refinement_estimation_approach()

    # Test that PlanningTimeout is handled properly
    sys.argv = train_sys_argv + [
        "--skip_refinement_estimator_training", "--timeout", "0"
    ]
    _train_refinement_estimation_approach()

    # Test _MaxSkeletonsFailure is handled properly
    sys.argv = train_sys_argv + [
        "--skip_refinement_estimator_training",
        "--refinement_data_num_skeletons", "1"
    ]
    _train_refinement_estimation_approach()

    # Test for different sesame_grounder
    sys.argv = train_sys_argv + [
        "--skip_refinement_estimator_training", "--timeout", "0",
        "--sesame_grounder", "fd_translator"
    ]
    _train_refinement_estimation_approach()
    sys.argv = train_sys_argv + [
        "--skip_refinement_estimator_training", "--timeout", "0",
        "--sesame_grounder", "doesn't exist"
    ]
    with pytest.raises(ValueError):
        _train_refinement_estimation_approach()  # invalid sesame grounder

    shutil.rmtree(data_dir)
    shutil.rmtree(approach_dir)

    sys.argv = [
        "dummy", "--env", "narrow_passage", "--approach",
        "refinement_estimation", "--refinement_estimator", "tabular", "--seed",
        "123", "--num_train_tasks", "1", "--approach_dir", approach_dir,
        "--data_dir", data_dir, "--load_data"
    ]
    with pytest.raises(FileNotFoundError):
        _train_refinement_estimation_approach()  # load non-existent file

    # Test PlanningFailure if goal is not dr-reachable
    sample_env = NarrowPassageEnv()
    sample_task = sample_env.get_train_tasks()[0].task
    sample_option_model = create_option_model("oracle")
    utils.reset_config_with_parser(parser)
    with pytest.raises(PlanningFailure):
        _collect_refinement_data_for_task(sample_env, sample_task,
                                          sample_option_model, set(), set(),
                                          set(), 0, [])
