"""Models for estimating and predicting skill competence."""
import abc
from typing import List
from typing import Type as TypingType

from predicators import utils


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

    @classmethod
    def get_name(cls) -> str:
        return "latent_variable"

    def get_current_competence(self) -> float:
        # Highly naive: group together all outcomes.
        all_outcomes = [o for co in self._cycle_observations for o in co]
        return utils.beta_bernoulli_posterior(all_outcomes)

    def predict_competence(self, num_additional_data: int) -> float:
        # Highly naive: predict a constant improvement in competence.
        del num_additional_data  # unused
        current_competence = self.get_current_competence()
        return min(1.0, current_competence + 1e-2)


def _get_competence_model_cls_from_name(name: str) -> TypingType[SkillCompetenceModel]:
    for cls in utils.get_all_subclasses(SkillCompetenceModel):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            return cls
    raise NotImplementedError(f"Unknown competence model: {name}")


def create_competence_model(model_name: str, skill_name: str) -> SkillCompetenceModel:
    """Create a competence model given its name."""

    cls = _get_competence_model_cls_from_name(model_name)
    return cls(skill_name)
