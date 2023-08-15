"""Tests for competence_models.py."""

import numpy as np
import pytest

from predicators import utils
from predicators.competence_models import LatentVariableSkillCompetenceModel, \
    LegacySkillCompetenceModel, create_competence_model
from predicators.settings import CFG

longrun = pytest.mark.skipif("not config.getoption('longrun')")


def test_legacy_skill_competence_model():
    """Tests for LegacySkillCompetenceModel()."""
    model = create_competence_model("legacy", "test")
    assert isinstance(model, LegacySkillCompetenceModel)
    assert np.isclose(model.get_current_competence(), 0.5)
    assert np.isclose(model.predict_competence(1), 0.5 + 1e-2)
    model.observe(True)
    assert model.get_current_competence() > 0.5
    assert model.predict_competence(1) > 0.5 + 1e-2
    model.observe(False)
    assert np.isclose(model.get_current_competence(), 0.5)
    assert np.isclose(model.predict_competence(1), 0.5 + 1e-2)
    model.advance_cycle()
    assert np.isclose(model.get_current_competence(), 0.5)
    assert np.isclose(model.predict_competence(1), 0.5 + 1e-2)


@longrun
def test_latent_variable_skill_competence_model_long():
    """Long tests for LatentVariableSkillCompetenceModel()."""
    utils.reset_config()
    h = CFG.skill_competence_model_lookahead
    model = create_competence_model("latent_variable", "test")
    assert isinstance(model, LatentVariableSkillCompetenceModel)
    assert np.isclose(model.get_current_competence(), 0.5)
    assert np.isclose(model.predict_competence(h), 0.5 + 1e-2)

    # Test impossible skill.
    model = create_competence_model("latent_variable", "impossible-skill")
    model.observe(False)
    assert model.get_current_competence() < 0.5
    model.advance_cycle()
    assert model.get_current_competence() < 0.5
    assert model.get_current_competence() < model.predict_competence(h)
    model.advance_cycle()
    assert model.get_current_competence() < 0.5
    assert model.get_current_competence() < model.predict_competence(h)
    model.observe(False)
    model.observe(False)
    model.observe(False)
    assert model.get_current_competence() < 0.01

    # Test perfect skill.
    model = create_competence_model("latent_variable", "perfect-skill")
    model.observe(True)
    assert model.get_current_competence() > 0.5
    model.advance_cycle()
    assert model.get_current_competence() > 0.5
    assert model.get_current_competence() < model.predict_competence(h)
    model.advance_cycle()
    assert model.get_current_competence() > 0.5
    assert model.get_current_competence() < model.predict_competence(h)
    model.observe(True)
    model.observe(True)
    model.observe(True)
    assert model.get_current_competence() > 0.99

    # Test noisy skill with gradual improvements.
    model = create_competence_model("latent_variable", "gradual-improve")
    model.observe(False)
    model.observe(True)
    model.advance_cycle()
    assert model.predict_competence(h) > 0.5  # should be optimistic
    model.observe(False)
    model.observe(True)
    model.observe(True)
    model.advance_cycle()
    assert model.get_current_competence() > 0.5
    assert model.get_current_competence() < model.predict_competence(h)
    model.observe(True)
    model.observe(False)
    model.observe(True)
    model.observe(True)
    model.observe(True)
    model.advance_cycle()
    assert model.get_current_competence() > 0.8
    model.observe(True)
    model.observe(True)
    model.observe(True)
    model.observe(True)
    model.observe(True)
    model.advance_cycle()
    assert model.get_current_competence() > 0.9

    # Test noisy skill with no improvements.
    model = create_competence_model("latent_variable", "noisy-no-improve")
    model.observe(False)
    model.observe(True)
    model.advance_cycle()
    model.observe(True)
    model.observe(False)
    model.observe(True)
    model.observe(False)
    model.advance_cycle()
    assert 0.4 < model.get_current_competence() < 0.6
