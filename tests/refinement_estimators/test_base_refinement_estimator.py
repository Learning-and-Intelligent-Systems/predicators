"""Test cases for the base refinement estimator class."""

import os
import shutil
from pathlib import Path

import pytest

from predicators import utils
from predicators.refinement_estimators import BaseRefinementEstimator, \
    create_refinement_estimator

# We don't run these tests for gnn because training on an empty dataset is
# not possible
ESTIMATOR_NAMES = ["oracle", "tabular", "cnn"]


def test_refinement_estimator_creation():
    """Tests for create_refinement_estimator()."""
    utils.reset_config({"env": "narrow_passage"})
    # Create fake directory to test saving and loading model
    parent_dir = os.path.dirname(__file__)
    approach_dir = os.path.join(parent_dir, "_fake_approach")
    os.makedirs(approach_dir, exist_ok=True)
    test_approach_path = Path(approach_dir) / "test.estimator"
    for est_name in ESTIMATOR_NAMES:
        estimator = create_refinement_estimator(est_name)
        assert isinstance(estimator, BaseRefinementEstimator)
        assert estimator.get_name() == est_name
        estimator.train([])
        estimator.save_model(test_approach_path)
        estimator.load_model(test_approach_path)
    shutil.rmtree(approach_dir)
    with pytest.raises(NotImplementedError):
        create_refinement_estimator("non-existent refinement estimator")
