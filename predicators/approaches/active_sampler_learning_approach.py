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
from predicators.ml_models import MLPRegressor, KNeighborsRegressor
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LowLevelTrajectory, \
    NSRTSampler, Object, ParameterizedOption, Predicate, Segment, State, \
    Task, Type, Variable, _GroundNSRT, _Option

# Dataset for Q learning: includes (s, a, s', r).
_SamplerDataset = List[Tuple[State, _Option, State, float]]
_NeuralSamplerInput = Tuple[State, Sequence[Object], Array]
_NeuralSamplerDataset = List[Tuple[_NeuralSamplerInput, float]]


# Helper function.
def _option_to_ground_nsrt(option: _Option, nsrts: Set[NSRT]) -> _GroundNSRT:
    nsrt_matches = [n for n in nsrts if n.option == option.parent]
    assert len(nsrt_matches) == 1
    nsrt = nsrt_matches[0]
    return nsrt.ground(option.objects)


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

    def _check_option_success(self, option: _Option, segment: Segment) -> bool:
        ground_nsrt = _option_to_ground_nsrt(option, self._nsrts)
        return ground_nsrt.add_effects.issubset(
            segment.final_atoms) and not ground_nsrt.delete_effects.issubset(
                segment.final_atoms)

    def _learn_sampler_regressors(
            self, online_learning_cycle: Optional[int]) -> None:
        """Learn regressors to re-weight the base samplers.

        Update the NSRTs in place.
        """
        # Create a new Q value estimator.
        Q = _QValueEstimator(self._get_current_nsrts(),
                             self._get_current_predicates(),
                             online_learning_cycle)
        # Fit with the current data.
        Q.run_fitted_q_iteration(self._sampler_data)
        # Update the NSRTs.
        new_nsrts = set()
        for nsrt, sampler in Q.get_samplers().items():
            # Create new NSRT.
            new_nsrt = NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions,
                            nsrt.add_effects, nsrt.delete_effects,
                            nsrt.ignore_effects, nsrt.option, nsrt.option_vars,
                            sampler.sample)
            new_nsrts.add(new_nsrt)
            # Save the sampler regressor for external analysis.
            regressor = sampler.get_regressor()
            approach_save_path = utils.get_approach_save_path_str()
            save_path = f"{approach_save_path}_{nsrt.name}_" + \
                f"{online_learning_cycle}.sampler_regressor"
            with open(save_path, "wb") as f:
                pkl.dump(regressor, f)
            logging.info(f"Saved sampler regressor to {save_path}.")
        self._nsrts = new_nsrts


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

    def predict_scores(self, state: State, objects: Sequence[Object],
                       sampled_params: List[Array]) -> List[float]:
        """Use the regressor to predict the score."""
        x_lst: List[Any] = [1.0]  # start with bias term
        sub = dict(zip(self._variables, objects))
        for var in self._variables:
            x_lst.extend(state[sub[var]])
        assert not CFG.sampler_learning_use_goals
        x = np.array(x_lst)
        return [
            self._regressor.predict(np.r_[x, p])[0] for p in sampled_params
        ]

    def sample(self, state: State, goal: Set[GroundAtom],
               rng: np.random.Generator, objects: Sequence[Object]) -> Array:
        """The sampler corresponding to the given models.

        May be used as the _sampler field in an NSRT.
        """
        samples = [
            self._base_sampler(state, goal, rng, objects)
            for _ in range(CFG.active_sampler_learning_num_samples)
        ]
        scores = self.predict_scores(state, objects, samples)
        # For now, just pick the best scoring sample.
        idx = np.argmax(scores)
        return samples[idx]


@dataclass
class _QValueEstimator:
    """Convenience class for training all of the samplers."""

    def __init__(self, nsrts: Set[NSRT], predicates: Set[Predicate],
                 online_learning_cycle: Optional[int]) -> None:
        self._nsrts = nsrts
        self._predicates = predicates
        self._online_learning_cycle = online_learning_cycle
        self._rng = np.random.default_rng(CFG.seed)
        self._learned_samplers: Optional[Dict[NSRT, _WrappedSampler]] = None
        self._initial_q_value = 0.0

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
                next_q = max(self._predict(ns, na) for na in next_as)
                # NOTE: there is no terminal state because we're in a lifelong
                # (reset-free) setup.
                output = r + gamma * next_q
                outputs.append(output)
            # Fit regressors for inputs and outputs.
            self._fit_regressors(inputs, outputs)

    def get_samplers(self) -> Dict[NSRT, _WrappedSampler]:
        """Expose the fitted samplers, organized by NSRTs."""
        assert self._learned_samplers is not None
        return self._learned_samplers

    def _predict(self, state: State, option: _Option) -> float:
        """Predict Q(s, a)."""
        if self._learned_samplers is None:
            return self._initial_q_value
        ground_nsrt = _option_to_ground_nsrt(option, self._nsrts)
        wrapped_sampler = self._learned_samplers[ground_nsrt.parent]
        return wrapped_sampler.predict_scores(state, ground_nsrt.objects,
                                              [option.params])[0]

    def _sample_options_from_state(self,
                                   state: State,
                                   num: int = 1) -> List[_Option]:
        """Use NSRTs to sample options in the current state."""
        # Create all applicable ground NSRTs.
        ground_nsrts: List[_GroundNSRT] = []
        for nsrt in sorted(self._nsrts):
            ground_nsrts.extend(utils.all_ground_nsrts(nsrt, list(state)))

        sampled_options: List[_Option] = []
        for _ in range(num):
            # Sample an applicable NSRT.
            ground_nsrt = utils.sample_applicable_ground_nsrt(
                state, ground_nsrts, self._predicates, self._rng)
            assert ground_nsrt is not None
            assert all(a.holds for a in ground_nsrt.preconditions)

            # Sample an option.
            option = ground_nsrt.sample_option(
                state,
                goal=set(),  # goal not used
                rng=self._rng)
            assert option.initiable(state)
            sampled_options.append(option)

        return sampled_options

    def _fit_regressors(self, inputs: List[Tuple[State, _Option]],
                        outputs: List[float]) -> None:
        """Fit one regressor per NSRT."""
        # Re-organize data by NSRT.
        regressor_data: Dict[NSRT, _NeuralSamplerDataset] = {
            n: []
            for n in self._nsrts
        }
        for (state, option), target in zip(inputs, outputs):
            ground_nsrt = _option_to_ground_nsrt(option, self._nsrts)
            regressor_input = (state, ground_nsrt.objects, option.params)
            nsrt = ground_nsrt.parent
            regressor_data[nsrt].append((regressor_input, target))
        # Fit each regressor.
        new_learned_samplers: Dict[NSRT, _WrappedSampler] = {}
        for nsrt, nsrt_data in regressor_data.items():
            logging.info(f"Fitting regressor for {nsrt.name}")
            regressor = self._fit_regressor(nsrt, nsrt_data)
            # This is the easiest way to access the oracle sampler.
            base_sampler = nsrt._sampler  # pylint: disable=protected-access
            wrapped_sampler = _WrappedSampler(base_sampler, regressor,
                                              nsrt.parameters, nsrt.option)
            new_learned_samplers[nsrt] = wrapped_sampler
        self._learned_samplers = new_learned_samplers

    def _fit_regressor(self, nsrt: NSRT,
                       data: _NeuralSamplerDataset) -> MLPRegressor:
        X_regressor: List[List[Array]] = []
        y_regressor: List[Array] = []
        for (state, objects, params), target in data:
            # input is state features and option parameters
            X_regressor.append([np.array(1.0)])  # start with bias term
            for obj in objects:
                X_regressor[-1].extend(state[obj])
            X_regressor[-1].extend(params)
            assert not CFG.sampler_learning_use_goals
            y_regressor.append(np.array([target]))
        X_arr_regressor = np.array(X_regressor)
        y_arr_regressor = np.array(y_regressor)
        # regressor = MLPRegressor(
        #     seed=CFG.seed,
        #     hid_sizes=CFG.mlp_regressor_hid_sizes,
        #     max_train_iters=CFG.mlp_regressor_max_itr,
        #     clip_gradients=CFG.mlp_regressor_clip_gradients,
        #     clip_value=CFG.mlp_regressor_gradient_clip_value,
        #     learning_rate=CFG.learning_rate,
        #     weight_decay=CFG.weight_decay,
        #     use_torch_gpu=CFG.use_torch_gpu,
        #     train_print_every=CFG.pytorch_train_print_every,
        #     n_iter_no_change=CFG.active_sampler_learning_n_iter_no_change)
        regressor = KNeighborsRegressor(seed=CFG.seed, n_neighbors=min(len(X_regressor), 5))
        regressor.fit(X_arr_regressor, y_arr_regressor)
        # Save the sampler regressor for external analysis.
        approach_save_path = utils.get_approach_save_path_str()
        save_path = f"{approach_save_path}_{nsrt.name}_" + \
            f"{self._online_learning_cycle}.sampler_regressor"
        with open(save_path, "wb") as f:
            pkl.dump(regressor, f)
        logging.info(f"Saved sampler regressor to {save_path}.")
        return regressor
