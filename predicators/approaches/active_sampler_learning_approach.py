"""An approach that performs active sampler learning.

The current implementation assumes for convenience that NSRTs and options are
1:1 and share the same parameters (like a PDDL environment). It is
straightforward conceptually to remove this assumption, because the approach
uses its own NSRTs to select options, but it is difficult implementation-wise,
so we're punting for now.


Example commands
----------------

Bumpy cover easy:
    python predicators/main.py --approach active_sampler_learning \
        --env bumpy_cover \
        --seed 0 \
        --strips_learner oracle \
        --sampler_learner oracle \
        --sampler_disable_classifier True \
        --bilevel_plan_without_sim True \
        --offline_data_bilevel_plan_without_sim False \
        --explorer random_nsrts \
        --max_initial_demos 1 \
        --num_train_tasks 1000 \
        --num_test_tasks 10 \
        --max_num_steps_interaction_request 4 \
        --bumpy_cover_num_bumps 2 \
        --bumpy_cover_spaces_per_bump 1 \
        --mlp_regressor_max_itr 100000 \
        --pytorch_train_print_every 10000


Bumpy cover with shifted targets:
    python predicators/main.py --approach active_sampler_learning \
        --env bumpy_cover \
        --seed 0 \
        --strips_learner oracle \
        --sampler_learner oracle \
        --sampler_disable_classifier True \
        --bilevel_plan_without_sim True \
        --offline_data_bilevel_plan_without_sim False \
        --explorer random_nsrts \
        --max_initial_demos 1 \
        --num_train_tasks 1000 \
        --num_test_tasks 10 \
        --max_num_steps_interaction_request 4 \
        --bumpy_cover_num_bumps 2 \
        --bumpy_cover_spaces_per_bump 1 \
        --mlp_regressor_max_itr 1000000 \
        --pytorch_train_print_every 10000 \
        --bumpy_cover_right_targets True
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import dill as pkl
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.ml_models import MLPRegressor
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LowLevelTrajectory, \
    NSRTSampler, Object, ParameterizedOption, Predicate, State, Task, Type, \
    Variable, _GroundNSRT, _Option, Segment

# Dataset for Q learning: includes (s, a, s', r).
_SamplerDataset = List[Tuple[State, _Option, State, float]]


class ActiveSamplerLearningApproach(OnlineNSRTLearningApproach):
    """Performs active sampler learning."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)

        assert CFG.sampler_disable_classifier
        assert CFG.strips_learner

        self._sampler_data: _SamplerDataset = []
        self._most_recent_nsrts: Optional[None] = None

    @classmethod
    def get_name(cls) -> str:
        return "active_sampler_learning"

    def _learn_nsrts(self, trajectories: List[LowLevelTrajectory],
                     online_learning_cycle: Optional[int],
                     annotations: Optional[List[Any]]) -> None:
        # Start by learning NSRTs in the usual way.
        super()._learn_nsrts(trajectories, online_learning_cycle, annotations)
        # Check the assumption that operators and options are 1:1.
        # This is just an implementation convenience.
        assert len({nsrt.option for nsrt in self._nsrts}) == len(self._nsrts)
        for nsrt in self._nsrts:
            assert nsrt.option_vars == nsrt.parameters
        # Update the sampler data using the updated self._segmented_trajs.
        self._update_sampler_data()
        # Re-learn sampler regressors. Updates the NSRTs.
        self._learn_sampler_regressors(online_learning_cycle)

    def _update_sampler_data(self) -> None:
        for segmented_traj in self._segmented_trajs:
            for segment in segmented_traj:
                state = segment.states[0]
                option = segment.get_option()
                next_state = segment.states[-1]
                if self._check_option_success(option, segment):
                    reward = 0.0
                else:
                    reward = -1.0
                self._sampler_data.append((state, option, next_state, reward))

    def _option_to_ground_nsrt(self, option: _Option) -> _GroundNSRT:
        nsrt_matches = [n for n in self._nsrts if n.option == option.parent]
        assert len(nsrt_matches) == 1
        nsrt = nsrt_matches[0]
        return nsrt.ground(option.objects)

    def _check_option_success(self, option: _Option,
                            segment: Segment) -> bool:
        ground_nsrt = self._option_to_ground_nsrt(option)
        return ground_nsrt.add_effects.issubset(
             segment.final_atoms) and not ground_nsrt.delete_effects.issubset(segment.final_atoms)

    def _learn_sampler_regressors(
            self, online_learning_cycle: Optional[int]) -> None:
        """Learn regressors to re-weight the base samplers.

        Update the NSRTs in place.
        """
        # Create a new Q value estimator.
        Q = _QValueEstimator(nsrts=self._get_most_recent_nsrts())
        # Fit with the current data.
        Q.run_fitted_q_iteration(self._sampler_data)
        # Update the NSRTs.
        new_nsrts = set()
        for old_nsrt, sampler in Q.get_samplers():
            # Create new NSRT.
            new_nsrt = NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions,
                            nsrt.add_effects, nsrt.delete_effects,
                            nsrt.ignore_effects, nsrt.option, nsrt.option_vars,
                            sampler.sample)
            new_nsrts.add(new_nsrt)
            # Save the sampler regressor for external analysis.
            regressor = sampler.get_regressor()
            approach_save_path = utils.get_approach_save_path_str()
            save_path = f"{approach_save_path}_{old_nsrt.name}_" + \
                f"{online_learning_cycle}.sampler_regressor"
            with open(save_path, "wb") as f:
                pkl.dump(regressor, f)
            logging.info(f"Saved sampler regressor to {save_path}.")
        self._nsrts = new_nsrts
        # The most_recent_nsrts are not changed by super()._learn_nsrts().
        self._most_recent_nsrts = new_nsrts

    def _get_most_recent_nsrts(self) -> Set[NSRT]:
        if self._most_recent_nsrts is None:
            return self._nsrts
        return self._most_recent_nsrts


@dataclass(frozen=True, eq=False, repr=False)
class _WrappedSampler:
    """Wraps a base sampler with a regressor.

    The outputs of the regressor are used to select among multiple
    candidate samples from the base sampler.
    """
    _base_sampler: NSRTSampler
    _regressor: MLPRegressor
    _variables: Sequence[Variable]
    _param_option: ParameterizedOption

    def get_regressor(self) -> MLPRegressor:
        """Expose the regressor."""
        return self._regressor

    def sample(self, state: State, goal: Set[GroundAtom],
                rng: np.random.Generator, objects: Sequence[Object]) -> Array:
        """The sampler corresponding to the given models.

        May be used as the _sampler field in an NSRT.
        """
        x_lst: List[Any] = [1.0]  # start with bias term
        sub = dict(zip(self._variables, objects))
        for var in self._variables:
            x_lst.extend(state[sub[var]])
        assert not CFG.sampler_learning_use_goals
        x = np.array(x_lst)

        samples = []
        scores = []
        for _ in range(CFG.active_sampler_learning_num_samples):
            params = self._base_sampler(state, goal, rng, objects)
            assert self._param_option.params_space.contains(params)
            score = self._regressor.predict(np.r_[x, params])[0]
            samples.append(params)
            scores.append(score)

        # For now, just pick the best scoring sample.
        idx = np.argmax(scores)
        return samples[idx]


@dataclass
class _QValueEstimator:
    """Convenience class for training all of the samplers."""

    # The most recent learned NSRTs.
    _nsrts: Set[NSRT]

    def run_fitted_q_iteration(self, data: _SamplerDataset) -> None:
        """Fit all of the samplers."""
        gamma = CFG.active_sampler_learning_score_gamma
        num_a_samp = CFG.active_sampler_learning_num_next_option_samples
        for it in range(CFG.active_sampler_learning_fitted_q_iters):
            logging.info(f"Starting fitted Q learning iter {it}")
            # Build inputs and outputs.
            inputs: List[Tuple[State, _Option]] = []
            outputs: List[float] = []
            for s, a, ns, r in data:
                inputs.append((s, a))
                # Sample actions to estimate Q in infinite action space.
                next_as = self._sample_options_from_state(ns, num=num_a_samp)
                next_q = np.mean(self._predict(ns, na) for na in next_as)
                output = r + gamma * next_q
                outputs.append(output)
            # Fit regressors for inputs and outputs.
            self._fit_regressors(inputs, outputs)

    def get_samplers(self) -> Dict[NSRT, _WrappedSampler]:
        """Expose the fitted samplers, organized by NSRTs."""
        import ipdb; ipdb.set_trace()

    def _predict(self, state: State, option: _Option) -> float:
        """Predict Q(s, a)."""
        import ipdb; ipdb.set_trace()

    def _sample_options_from_state(self, state: State, num: int = 1) -> List[_Option]:
        """Use NSRTs to sample options in the current state."""
        import ipdb; ipdb.set_trace()

    def _fit_regressors(self, inputs: List[Tuple[State, _Option]], outputs: List[float]) -> None:
        """Fit one regressor per NSRT."""
        import ipdb; ipdb.set_trace()


