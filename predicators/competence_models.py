"""Models for estimating and predicting skill competence."""
import abc
from typing import List, Optional
from typing import Type as TypingType

from scipy.stats import beta as BetaRV

from predicators import utils
from predicators.ml_models import BetaRegressor


class SkillCompetenceModel(abc.ABC):
    """A model that tracks and predicts competence for a single skill based on
    the history of outcomes and re-learning cycles."""

    def __init__(self, skill_name: str) -> None:
        self._skill_name = skill_name  # just for reference
        # Each list contains outcome for one cycle.
        self._cycle_observations: List[List[bool]] = [[]]

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this skill competence model."""

    @property
    def _current_cycle(self) -> int:
        """The current cycle."""
        return len(self._cycle_observations) - 1

    def observe(self, skill_outcome: bool) -> None:
        """Record a success or failure from running the skill."""
        self._cycle_observations[-1].append(skill_outcome)

    def advance_cycle(self) -> None:
        """Called after re-learning is performed."""
        self._cycle_observations.append([])

    @abc.abstractmethod
    def get_current_competence(self) -> float:
        """An estimate of the current competence."""

    @abc.abstractmethod
    def predict_competence(self, num_additional_data: int) -> float:
        """Predict what the competence for the next cycle would be if we were
        to collect num_additional_data outcomes during this cycle."""


class LegacySkillCompetenceModel(SkillCompetenceModel):
    """Our first un-principled implementation of competence modeling."""

    @classmethod
    def get_name(cls) -> str:
        return "legacy"

    def get_current_competence(self) -> float:
        # Highly naive: group together all outcomes.
        all_outcomes = [o for co in self._cycle_observations for o in co]
        return utils.beta_bernoulli_posterior(all_outcomes)

    def predict_competence(self, num_additional_data: int) -> float:
        # Highly naive: predict a constant improvement in competence.
        del num_additional_data  # unused
        current_competence = self.get_current_competence()
        return min(1.0, current_competence + 1e-2)


class LatentVariableSkillCompetenceModel(SkillCompetenceModel):
    """Uses expectation-maximization for learning."""

    def __init__(self, skill_name: str) -> None:
        super().__init__(skill_name)
        # Update competence estimate after every observation.
        self._posterior_competence = BetaRV(1.0, 1.0)
        # Model that maps number of data to competence.
        self._competence_regressor: Optional[BetaRegressor] = None

    @classmethod
    def get_name(cls) -> str:
        return "latent_variable"

    def get_current_competence(self) -> float:
        return self._posterior_competence.mean

    def predict_competence(self, num_additional_data: int) -> float:
        # If we haven't yet learned a regressor, default to an optimistic
        # naive model that assumes competence will improve slightly, like
        # the LegacySkillCompetenceModel.
        if self._competence_regressor is None:
            current_competence = self.get_current_competence()
            return min(1.0, current_competence + 1e-2)
        # Use the regressor to predict future competence.
        current_num_data = self._get_current_num_data()
        future_num_data = current_num_data + num_additional_data
        rv = self._competence_regressor.predict_beta(future_num_data)
        return rv.mean

    def observe(self, skill_outcome: bool) -> None:
        # Update the posterior competence after every observation.
        super().observe(skill_outcome)
        # Get the prior from the competence regressor.
        if self._competence_regressor is None:
            alpha0, beta0 = 1.0, 1.0
        else:
            current_num_data = self._get_current_num_data()
            rv = self._competence_regressor.predict_beta(current_num_data)
            alpha0, beta0 = rv.alpha, rv.alpha
        current_cycle_outcomes = self._cycle_observations[-1]
        self._posterior_competence = utils.beta_bernoulli_posterior(
            current_cycle_outcomes, prior_alpha=alpha0, prior_beta=beta0)

    def advance_cycle(self) -> None:
        super().advance_cycle()
        self._fit_competence_regressor()

    def _fit_competence_regressor(self) -> None:
        # Re-learn the competence regressor using EM.
        import ipdb
        ipdb.set_trace()

    def _get_current_num_data(self) -> int:
        return sum(len(o) for o in self._cycle_observations)


def _get_competence_model_cls_from_name(
        name: str) -> TypingType[SkillCompetenceModel]:
    for cls in utils.get_all_subclasses(SkillCompetenceModel):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            return cls
    raise NotImplementedError(f"Unknown competence model: {name}")


def create_competence_model(model_name: str,
                            skill_name: str) -> SkillCompetenceModel:
    """Create a competence model given its name."""

    cls = _get_competence_model_cls_from_name(model_name)
    return cls(skill_name)
