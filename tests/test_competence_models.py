"""Tests for competence_models.py."""

from predicators.competence_models import create_competence_model, LegacySkillCompetenceModel, LatentVariableSkillCompetenceModel
from predicators import utils

import numpy as np
import pytest

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
    model = create_competence_model("latent_variable", "test")
    assert isinstance(model, LatentVariableSkillCompetenceModel)
    assert np.isclose(model.get_current_competence(), 0.5)
    assert np.isclose(model.predict_competence(1), 0.5 + 1e-2)
    
    # Test impossible skill.
    model = create_competence_model("latent_variable", "impossible-skill")
    model.observe(False)
    assert model.get_current_competence() < 0.5
    model.advance_cycle()
    assert model.get_current_competence() < 0.5
    assert model.get_current_competence() < model.predict_competence(1)
    model.advance_cycle()
    assert model.get_current_competence() < 0.5
    assert model.get_current_competence() < model.predict_competence(1)
    model.observe(False)
    model.observe(False)
    model.observe(False)
    assert model.get_current_competence() < 0.5
    assert model.get_current_competence() < model.predict_competence(1)
    
    # Test perfect skill.
    model = create_competence_model("latent_variable", "perfect-skill")
    model.observe(True)
    assert model.get_current_competence() > 0.5
    model.advance_cycle()
    assert model.get_current_competence() > 0.5
    assert model.get_current_competence() < model.predict_competence(1)
    model.advance_cycle()
    assert model.get_current_competence() > 0.5
    assert model.get_current_competence() < model.predict_competence(1)
    model.observe(True)
    model.observe(True)
    model.observe(True)
    assert model.get_current_competence() > 0.5
    assert model.get_current_competence() < model.predict_competence(1)
