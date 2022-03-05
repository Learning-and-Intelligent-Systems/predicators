"""Test cases for some parts of the settings.py file."""

from predicators.src.settings import get_allowed_query_type_names
from predicators.src import utils


def test_get_allowed_query_type_names():
    """Test the get_allowed_query_type_names method."""
    utils.reset_config({
        "option_learner": "neural",
    })
    assert get_allowed_query_type_names() == {
        "DemonstrationQuery", "StateBasedDemonstrationQuery"
    }
    utils.reset_config({
        "option_learner": "no_learning",
        "approach": "interactive_learning"
    })
    assert get_allowed_query_type_names() == {"GroundAtomsHoldQuery"}
    utils.reset_config({
        "option_learner": "no_learning",
        "approach": "unittest"
    })
    assert get_allowed_query_type_names() == {
        "GroundAtomsHoldQuery", "DemonstrationQuery",
        "StateBasedDemonstrationQuery"
    }
