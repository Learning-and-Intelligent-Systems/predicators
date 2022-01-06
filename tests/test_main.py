"""Tests for main.py.
"""

import os
import shutil
import sys
import pytest
from predicators.src.main import main


def test_main():
    """Tests for main.py.
    """
    sys.argv = ["dummy", "--env", "my_env", "--approach", "my_approach",
                "--seed", "123", "--num_test_tasks", "5"]
    with pytest.raises(NotImplementedError):
        main()  # invalid env
    sys.argv = ["dummy", "--env", "cover", "--approach", "my_approach",
                "--seed", "123", "--num_test_tasks", "5"]
    with pytest.raises(NotImplementedError):
        main()  # invalid approach
    sys.argv = ["dummy", "--env", "cover", "--approach", "random_actions",
                "--seed", "123", "--not-a-real-flag", "0"]
    with pytest.raises(ValueError):
        main()  # invalid flag
    sys.argv = ["dummy", "--env", "cover", "--approach", "random_actions",
                "--seed", "123", "--num_test_tasks", "5"]
    main()
    sys.argv = ["dummy", "--env", "cover", "--approach", "random_options",
                "--seed", "123", "--num_test_tasks", "5"]
    main()
    sys.argv = ["dummy", "--env", "cover", "--approach", "oracle",
                "--seed", "123", "--num_test_tasks", "5"]
    main()
    sys.argv = ["dummy", "--env", "cluttered_table", "--approach",
                "random_actions", "--seed", "123", "--num_test_tasks", "20"]
    main()
    sys.argv = ["dummy", "--env", "blocks", "--approach",
                "random_actions", "--seed", "123", "--num_test_tasks", "5"]
    main()
    sys.argv = ["dummy", "--env", "blocks", "--approach",
                "random_options", "--seed", "123", "--num_test_tasks", "5"]
    main()
    video_dir = os.path.join(os.path.dirname(__file__), "_fake_videos")
    sys.argv = ["dummy", "--env", "cover", "--approach", "oracle",
                "--seed", "123", "--make_videos", "--num_test_tasks", "1",
                "--video_dir", video_dir]
    main()
    shutil.rmtree(video_dir)
    # Try running main with a strong timeout.
    sys.argv = ["dummy", "--env", "cover", "--approach", "oracle",
                "--seed", "123", "--timeout", "0.001", "--num_test_tasks", "5"]
    main()
    # Run actual main approach, but without sampler learning.
    sys.argv = ["dummy", "--env", "cover", "--approach", "nsrt_learning",
                "--seed", "123", "--do_sampler_learning", "0"]
    main()
    # Try loading.
    sys.argv = ["dummy", "--env", "cover", "--approach", "nsrt_learning",
                "--seed", "123", "--load"]
    main()
    # Try learning (with too low hyperparameters to actually work).
    sys.argv = ["dummy", "--env", "cover", "--approach",
                "nsrt_learning", "--seed", "123",
                "--do_sampler_learning", "1",
                "--classifier_max_itr_sampler", "10",
                "--regressor_max_itr", "10",
                "--timeout", "0.01"]
    main()  # correct usage
    # Try predicate exclusion.
    sys.argv = ["dummy", "--env", "cover", "--approach",
                "random_options", "--seed", "123",
                "--excluded_predicates", "NotARealPredicate"]
    with pytest.raises(AssertionError):
        main()  # can't exclude a non-existent predicate
    sys.argv = ["dummy", "--env", "cover", "--approach",
                "random_options", "--seed", "123",
                "--excluded_predicates", "Covers"]
    with pytest.raises(AssertionError):
        main()  # can't exclude a goal predicate
    sys.argv = ["dummy", "--env", "cover", "--approach",
                "random_options", "--seed", "123",
                "--excluded_predicates", "Holding,HandEmpty"]
    main()  # correct usage
    sys.argv = ["dummy", "--env", "cover", "--approach",
                "random_options", "--seed", "123",
                "--excluded_predicates", "HandEmpty",
                "--num_test_tasks", "5"]
    main()  # correct usage
    sys.argv = ["dummy", "--env", "cover", "--approach",
                "random_options", "--seed", "123",
                "--excluded_predicates", "all",
                "--num_test_tasks", "5"]
    main()  # correct usage
    results_dir = os.path.join(os.path.dirname(__file__), "_fake_results")
    sys.argv = ["dummy", "--env", "cover", "--approach", "oracle",
                "--seed", "123", "--num_test_tasks", "1",
                "--results_dir", results_dir]
    main()
    assert os.path.exists(results_dir)
    shutil.rmtree(results_dir)
