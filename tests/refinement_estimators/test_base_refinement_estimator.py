"""Test cases for the base refinement estimator class."""

import pytest

from predicators.refinement_estimators import BaseRefinementEstimator, \
    create_refinement_estimator

ESTIMATOR_NAMES = ["oracle", "tabular"]


def test_refinement_estimator_creation():
    """Tests for create_refinement_estimator()."""
    for est_name in ESTIMATOR_NAMES:
        estimator = create_refinement_estimator(est_name)
        assert isinstance(estimator, BaseRefinementEstimator)
        assert estimator.get_name() == est_name
    with pytest.raises(NotImplementedError):
        create_refinement_estimator("non-existent refinement estimator")
