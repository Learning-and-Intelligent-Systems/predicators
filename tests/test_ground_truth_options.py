"""Tests for ground_truth_models.py."""

import pytest

from predicators import utils
from predicators.envs.cover import CoverMultistepOptions
from predicators.ground_truth_models import parse_config_included_options


def test_parse_config_included_options():
    """Tests for parse_config_included_options()."""
    # Test including nothing.
    utils.reset_config({
        "env": "cover_multistep_options",
        "included_options": "",
    })
    env = CoverMultistepOptions()
    included = parse_config_included_options(env)
    assert not included
    # Test including specific options.
    utils.reset_config({
        "included_options": "Pick",
    })
    Pick, Place = sorted(env.options)
    assert Pick.name == "Pick"
    assert Place.name == "Place"
    included = parse_config_included_options(env)
    assert included == {Pick}
    utils.reset_config({
        "included_options": "Place",
    })
    included = parse_config_included_options(env)
    assert included == {Place}
    utils.reset_config({
        "included_options": "Pick,Place",
    })
    included = parse_config_included_options(env)
    assert included == {Pick, Place}
    # Test including an unknown option.
    utils.reset_config({
        "included_options": "Pick,NotReal",
    })
    with pytest.raises(AssertionError) as e:
        parse_config_included_options(env)
    assert "Unrecognized option in included_options!" in str(e)
