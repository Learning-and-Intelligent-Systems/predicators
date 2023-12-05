"""Models for estimating and predicting skill competence."""
import abc
import logging
from typing import List, Optional
from typing import Type as TypingType

import numpy as np
from scipy.stats import beta as BetaRV

from predicators import utils
from predicators.ml_models import MonotonicBetaRegressor
from predicators.settings import CFG
from predicators.structs import Array


class SkillCompetenceModel(abc.ABC):
    """A model that tracks and predicts competence for a single skill based on
    the history of outcomes and re-learning cycles."""

    def __init__(self, skill_name: str) -> None:
        self._skill_name = skill_name  # just for reference
        # Each list contains outcome for one cycle.
        self._cycle_observations: List[List[bool]] = [[]]
        # For competence prior.
        self._default_alpha, self._default_beta = \
            CFG.skill_competence_default_alpha_beta

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this skill competence model."""

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
        return utils.beta_bernoulli_posterior_mean(all_outcomes,
                                                   alpha=self._default_alpha,
                                                   beta=self._default_beta)

    def predict_competence(self, num_additional_data: int) -> float:
        # Highly naive: predict a constant improvement in competence.
        del num_additional_data  # unused
        current_competence = self.get_current_competence()
        # Use a highly optimistic initial competence until the second cycle.
        return min(
            1.0,
            current_competence + CFG.skill_competence_initial_prediction_bonus)


class OptimisticSkillCompetenceModel(SkillCompetenceModel):
    """A simple and fast competence model."""

    @classmethod
    def get_name(cls) -> str:
        return "optimistic"

    def _get_nonempty_cycle_observations(self) -> List[List[bool]]:
        return [co for co in self._cycle_observations if co]

    def get_current_competence(self) -> float:
        # Use sliding window to estimate competence.
        nonempty_cycle_obs = self._get_nonempty_cycle_observations()
        if not nonempty_cycle_obs:
            return utils.beta_bernoulli_posterior_mean(
                [], alpha=self._default_alpha,
                beta=self._default_beta)  # default
        window = min(len(nonempty_cycle_obs),
                     CFG.skill_competence_model_optimistic_window_size)
        recent_cycle_obs = nonempty_cycle_obs[-window:]
        all_outcomes = [o for co in recent_cycle_obs for o in co]
        return utils.beta_bernoulli_posterior_mean(all_outcomes,
                                                   alpha=self._default_alpha,
                                                   beta=self._default_beta)

    def predict_competence(self, num_additional_data: int) -> float:
        # Look for maximum change in competence and optimistically assume that
        # we'll repeat that change.
        nonempty_cycle_obs = self._get_nonempty_cycle_observations()
        current_competence = self.get_current_competence()
        if len(nonempty_cycle_obs) < 2:
            return min(
                1.0, current_competence +
                CFG.skill_competence_initial_prediction_bonus)  # default
        # Look at changes between individual cycles.
        inference_window = 1
        recency_size = CFG.skill_competence_model_optimistic_recency_size
        competences: List[float] = []
        start = max(0, len(nonempty_cycle_obs) - recency_size)
        end = len(nonempty_cycle_obs) - inference_window + 1
        for i in range(start, end):
            sub_history = nonempty_cycle_obs[i:i + inference_window]
            sub_outcomes = [o for co in sub_history for o in co]
            competence = float(np.mean(sub_outcomes))
            competences.append(competence)
        best_change = max(competences) - min(competences)
        gain = best_change * num_additional_data
        return np.clip(current_competence + gain, 1e-6, 1.0)


class LatentVariableSkillCompetenceModel(SkillCompetenceModel):
    """Uses expectation-maximization for learning."""

    def __init__(self, skill_name: str) -> None:
        super().__init__(skill_name)
        self._log_prefix = f"[Competence] [{self._skill_name}]"
        # Update competence estimate after every observation.
        self._posterior_competences = [
            BetaRV(self._default_alpha, self._default_beta)
        ]
        # Model that maps number of data to competence.
        self._competence_regressor: Optional[MonotonicBetaRegressor] = None

    @classmethod
    def get_name(cls) -> str:
        return "latent_variable"

    def get_current_competence(self) -> float:
        return self._posterior_competences[-1].mean()

    def predict_competence(self, num_additional_data: int) -> float:
        # If we haven't yet learned a regressor, default to an optimistic
        # naive model that assumes competence will improve slightly, like
        # the LegacySkillCompetenceModel.
        if self._competence_regressor is None:
            current_competence = self.get_current_competence()
            return min(
                1.0, current_competence +
                CFG.skill_competence_initial_prediction_bonus)
        # Use the regressor to predict future competence.
        current_num_data = self._get_current_num_data()
        current_rv = self._competence_regressor.predict_beta(current_num_data)
        future_num_data = current_num_data + num_additional_data
        future_rv = self._competence_regressor.predict_beta(future_num_data)
        gain = future_rv.mean() - current_rv.mean()
        assert gain >= -1e-6
        return np.clip(self.get_current_competence() + gain, 0.0, 1.0)

    def observe(self, skill_outcome: bool) -> None:
        # Update the posterior competence after every observation.
        super().observe(skill_outcome)
        # Get the prior from the competence regressor.
        if self._competence_regressor is None:
            alpha0, beta0 = self._default_alpha, self._default_beta
        else:
            current_num_data = self._get_current_num_data()
            rv = self._competence_regressor.predict_beta(current_num_data)
            alpha0, beta0 = rv.args
        current_cycle_outcomes = self._cycle_observations[-1]
        self._posterior_competences[-1] = utils.beta_bernoulli_posterior(
            current_cycle_outcomes, alpha=alpha0, beta=beta0)

    def advance_cycle(self) -> None:
        # Re-learn before advancing the cycle.
        self._run_expectation_maximization()
        super().advance_cycle()

    def _run_expectation_maximization(self) -> None:
        # Re-learn the competence regressor using EM.
        inputs = self._get_regressor_inputs()
        # Warm-start from last inference cycle.
        for it in range(CFG.skill_competence_model_num_em_iters):
            logging.info(f"{self._log_prefix} EM iter {it}")
            # Run inference.
            map_comp = self._run_map_inference(self._posterior_competences)
            logging.info(f"{self._log_prefix}   Competences: {map_comp}")
            # Run learning.
            self._competence_regressor = MonotonicBetaRegressor(
                seed=CFG.seed,
                max_train_iters=CFG.skill_competence_model_max_train_iters,
                clip_gradients=CFG.mlp_regressor_max_itr,
                clip_value=CFG.mlp_regressor_gradient_clip_value,
                learning_rate=CFG.skill_competence_model_learning_rate)
            targets = np.array(map_comp, dtype=np.float32)
            targets = np.reshape(targets, (-1, 1))
            self._competence_regressor.fit(inputs, targets)
            # Update posteriors by evaluating the model.
            self._posterior_competences = [
                self._competence_regressor.predict_beta(x) for x in inputs
            ]
            means = [b.mean() for b in self._posterior_competences]
            variances = [b.var() for b in self._posterior_competences]
            ctheta = self._competence_regressor.get_transformed_params()
            logging.info(f"{self._log_prefix}   Params: {ctheta}")
            logging.info(f"{self._log_prefix}   Beta means: {means}")
            logging.info(f"{self._log_prefix}   Beta variances: {variances}")
        # Update the posterior after learning for the new cycle (for which
        # we have no data).
        assert self._competence_regressor is not None
        n = self._get_current_num_data()
        # Add new posterior competence for next cycle.
        self._posterior_competences.append(
            self._competence_regressor.predict_beta(n))

    def _get_current_num_data(self) -> int:
        return sum(len(o) for o in self._cycle_observations)

    def _get_regressor_inputs(self) -> Array:
        history = self._cycle_observations
        num_data_after_cycle = list(np.cumsum([len(h) for h in history]))
        num_data_before_cycle = np.array([0] + num_data_after_cycle[:-1],
                                         dtype=np.float32)
        inputs = np.reshape(num_data_before_cycle, (-1, 1))
        return inputs

    def _run_map_inference(self, betas: List[BetaRV]) -> List[float]:
        """Compute the MAP competences given the input beta priors."""
        assert len(betas) == len(self._cycle_observations)
        map_competences: List[float] = []
        for o, rv in zip(self._cycle_observations, betas):
            alpha, beta = rv.args
            prv = utils.beta_bernoulli_posterior(o, alpha=alpha, beta=beta)
            map_competences.append(prv.mean())
        return map_competences


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
