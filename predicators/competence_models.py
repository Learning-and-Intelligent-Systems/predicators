"""Models for estimating and predicting skill competence."""
import abc
from typing import List

from predicators import utils


class SkillCompetenceModel(abc.ABC):
    """A model that tracks and predicts competence for a single skill based on
    the history of outcomes and re-learning cycles."""

    def __init__(self, name: str) -> None:
        self._name = name  # just for reference
        # Each list contains outcome for one cycle.
        self._cycle_observations: List[List[bool]] = [[]]

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

    def get_current_competence(self) -> float:
        # Highly naive: group together all outcomes.
        all_outcomes = [o for co in self._cycle_observations for o in co]
        return utils.beta_bernoulli_posterior(all_outcomes)

    def predict_competence(self, num_additional_data: int) -> float:
        # Highly naive: predict a constant improvement in competence.
        del num_additional_data  # unused
        current_competence = self.get_current_competence()
        return min(1.0, current_competence + 1e-2)
