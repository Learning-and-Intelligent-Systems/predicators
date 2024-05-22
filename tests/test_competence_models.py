"""Tests for competence_models.py."""

import numpy as np
import pytest

from predicators import utils
from predicators.competence_models import LatentVariableSkillCompetenceModel, \
    LegacySkillCompetenceModel, OptimisticSkillCompetenceModel, \
    create_competence_model
from predicators.settings import CFG

longrun = pytest.mark.skipif("not config.getoption('longrun')")


def test_create_competence_model():
    """Tests for create_competence_model()."""
    model = create_competence_model("legacy", "test")
    assert isinstance(model, LegacySkillCompetenceModel)
    model = create_competence_model("latent_variable", "test")
    assert isinstance(model, LatentVariableSkillCompetenceModel)
    model = create_competence_model("optimistic", "test")
    assert isinstance(model, OptimisticSkillCompetenceModel)
    with pytest.raises(NotImplementedError) as e:
        create_competence_model("not a real competence model", "test")
    assert "Unknown competence model" in str(e)


def test_legacy_skill_competence_model():
    """Tests for LegacySkillCompetenceModel()."""
    utils.reset_config({
        "skill_competence_default_alpha_beta": (1.0, 1.0),
        "skill_competence_initial_prediction_bonus": 1e-2,
    })
    model = create_competence_model("legacy", "test")
    assert isinstance(model, LegacySkillCompetenceModel)
    assert np.isclose(model.get_current_competence(), 0.5)
    assert np.isclose(model.predict_competence(1),
                      0.5 + CFG.skill_competence_initial_prediction_bonus)
    model.observe(True)
    assert model.get_current_competence() > 0.5
    assert model.predict_competence(
        1) > 0.5 + CFG.skill_competence_initial_prediction_bonus
    model.observe(False)
    assert np.isclose(model.get_current_competence(), 0.5)
    assert np.isclose(model.predict_competence(1),
                      0.5 + CFG.skill_competence_initial_prediction_bonus)
    model.advance_cycle()
    assert np.isclose(model.get_current_competence(), 0.5)
    assert np.isclose(model.predict_competence(1),
                      0.5 + CFG.skill_competence_initial_prediction_bonus)
    model.observe(True)
    assert model.get_current_competence() > 0.5


def test_latent_variable_skill_competence_model_short():
    """Quick tests for LatentVariableSkillCompetenceModel()."""
    utils.reset_config({
        "skill_competence_model_num_em_iters": 1,
        "skill_competence_model_max_train_iters": 10,
        "skill_competence_default_alpha_beta": (1.0, 1.0),
        "skill_competence_initial_prediction_bonus": 1e-2,
    })
    model = create_competence_model("latent_variable", "test")
    assert np.isclose(model.get_current_competence(), 0.5)
    assert np.isclose(model.predict_competence(1),
                      0.5 + CFG.skill_competence_initial_prediction_bonus)
    model.observe(True)
    assert model.get_current_competence() > 0.5
    assert model.predict_competence(1) > model.get_current_competence()
    model.observe(False)
    assert np.isclose(model.get_current_competence(), 0.5)
    model.advance_cycle()
    assert model.predict_competence(1) > model.get_current_competence()
    model.observe(True)
    assert model.get_current_competence() > 0.5


def test_optimistic_skill_competence_model():
    """Tests for OptimisticSkillCompetenceModel()."""
    utils.reset_config({
        "skill_competence_default_alpha_beta": (1.0, 1.0),
        "skill_competence_initial_prediction_bonus": 1e-2,
    })
    h = CFG.skill_competence_model_lookahead

    model = create_competence_model("optimistic", "test")
    assert np.isclose(model.get_current_competence(), 0.5)
    assert np.isclose(model.predict_competence(h),
                      0.5 + CFG.skill_competence_initial_prediction_bonus)

    # Test impossible skill.
    model = create_competence_model("optimistic", "impossible-skill")
    model.observe(False)
    assert model.get_current_competence() < 0.5
    model.advance_cycle()
    assert model.get_current_competence() < 0.5
    assert model.get_current_competence() <= model.predict_competence(h)
    model.advance_cycle()
    assert model.get_current_competence() < 0.5
    assert model.get_current_competence() <= model.predict_competence(h)
    for _ in range(100):
        model.observe(False)
    assert model.get_current_competence() <= 0.01

    # Test perfect skill.
    model = create_competence_model("optimistic", "perfect-skill")
    model.observe(True)
    assert model.get_current_competence() > 0.5
    model.advance_cycle()
    assert model.get_current_competence() > 0.5
    assert model.get_current_competence() <= model.predict_competence(h)
    model.advance_cycle()
    assert model.get_current_competence() > 0.5
    assert model.get_current_competence() <= model.predict_competence(h)
    for _ in range(100):
        model.observe(True)
    assert model.get_current_competence() > 0.99

    # Test noisy skill with gradual improvements.
    model = create_competence_model("optimistic", "gradual-improve")
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
    model.observe(False)
    for _ in range(50):
        model.observe(True)
    model.advance_cycle()
    assert model.get_current_competence() > 0.8

    # Test noisy skill with no improvements.
    model = create_competence_model("optimistic", "noisy-no-improve")
    model.observe(False)
    model.observe(True)
    model.advance_cycle()
    model.observe(True)
    model.observe(False)
    model.observe(True)
    model.observe(False)
    model.advance_cycle()
    assert 0.4 < model.get_current_competence() < 0.6

    # Custom tests based on PickFromBumpy in PickPlace1D.
    model = create_competence_model("optimistic", "pickplace1d-custom1")
    model.observe(False)
    model.observe(True)
    model.advance_cycle()
    model.observe(False)
    model.observe(False)
    assert model.get_current_competence() < model.predict_competence(h)


@longrun
def test_latent_variable_skill_competence_model_long():
    """Long tests for LatentVariableSkillCompetenceModel()."""
    utils.reset_config({
        "skill_competence_default_alpha_beta": (1.0, 1.0),
        "skill_competence_initial_prediction_bonus": 1e-2,
    })
    h = CFG.skill_competence_model_lookahead

    model = create_competence_model("latent_variable", "test")
    assert np.isclose(model.get_current_competence(), 0.5)
    assert np.isclose(model.predict_competence(h),
                      0.5 + CFG.skill_competence_initial_prediction_bonus)

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
    for _ in range(100):
        model.observe(False)
    assert model.get_current_competence() < 0.05

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
    for _ in range(100):
        model.observe(True)
    assert model.get_current_competence() > 0.95

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

    # Custom tests based on PickFromBumpy in PickPlace1D.
    model = create_competence_model("latent_variable", "pickplace1d-custom1")
    model.observe(False)
    model.observe(True)
    model.advance_cycle()
    model.observe(False)
    model.observe(False)
    assert model.get_current_competence() < model.predict_competence(h)
